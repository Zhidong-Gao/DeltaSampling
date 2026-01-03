import comfy
import torch

import comfy.samplers
import latent_preview
import comfy.k_diffusion
import comfy.k_diffusion.sampling

import torch.nn as nn

from tqdm.auto import trange

from comfy.k_diffusion.sampling import to_d
from comfy.samplers import preprocess_conds_hooks, get_total_hook_groups_in_conds, process_conds
from comfy.samplers import Sampler, KSamplerX0Inpaint


class DELTAKSAMPLER(Sampler):
    def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options if extra_options is not None else {}
        self.inpaint_options = inpaint_options if inpaint_options is not None else {}

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        if self.inpaint_options.get("random", False): #TODO: Should this be the default?
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))

        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)
        
        # samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
        # samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        # return samples
        
        sampler_function_input = (self, model_k, noise, sigmas, extra_args, k_callback, disable_pbar)
        return sampler_function_input

def delta_inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed):
    if latent_image is not None and torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
        latent_image = self.inner_model.process_latent_in(latent_image)

    # Ensure conds is a dictionary
    if not hasattr(self, 'conds') or self.conds is None:
        self.conds = {}

    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed)

    # Ensure model_options is a dictionary
    if not hasattr(self, 'model_options') or self.model_options is None:
        self.model_options = {}
    
    extra_args = {"model_options": comfy.model_patcher.create_model_options_clone(self.model_options), "seed": seed}

    executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
        sampler.sample,
        sampler,
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, extra_args["model_options"], is_model_options=True)
    )
    # samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
    sampler_function_input = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
    return sampler_function_input # self.inner_model.process_latent_out(samples.to(torch.float32))

def delta_outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    # Ensure conds is a dictionary before passing to prepare_sampling
    if not hasattr(self, 'conds') or self.conds is None:
        self.conds = {}
    
    # Ensure model_options is a dictionary
    if not hasattr(self, 'model_options') or self.model_options is None:
        self.model_options = {}
    
    self.inner_model, self.conds, self.loaded_models = comfy.sampler_helpers.prepare_sampling(self.model_patcher, noise.shape, self.conds, self.model_options)
    device = self.model_patcher.load_device

    if denoise_mask is not None:
        denoise_mask = comfy.sampler_helpers.prepare_mask(denoise_mask, noise.shape, device)

    noise = noise.to(device)
    latent_image = latent_image.to(device)
    sigmas = sigmas.to(device)

    try:
        self.model_patcher.pre_run()
        # output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        sampler_function_input = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
    finally:
        pass;
        # self.model_patcher.cleanup()

    # comfy.sampler_helpers.cleanup_models(self.conds, self.loaded_models)
    # del self.inner_model
    # del self.loaded_models
    return sampler_function_input

def delta_cfg_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    if sigmas.shape[-1] == 0:
        return latent_image

    # Initialize conds safely
    if not hasattr(self, 'original_conds') or self.original_conds is None:
        self.original_conds = {}
    
    self.conds = {}
    for k in self.original_conds:
        self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))
    preprocess_conds_hooks(self.conds)

    try:
        orig_model_options = self.model_options
        # Ensure model_options is never None
        if self.model_options is None:
            self.model_options = {}
        self.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        # if one hook type (or just None), then don't bother caching weights for hooks (will never change after first step)
        orig_hook_mode = self.model_patcher.hook_mode
        if get_total_hook_groups_in_conds(self.conds) <= 1:
            self.model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
        comfy.sampler_helpers.prepare_model_patcher(self.model_patcher, self.conds, self.model_options)
        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self.outer_sample,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, self.model_options, is_model_options=True)
        )
        # output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        sampler_function_input = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
    finally:
        pass;
        # self.model_options = orig_model_options
        # self.model_patcher.hook_mode = orig_hook_mode
        # self.model_patcher.restore_hook_patches()

    # del self.conds
    post_obj = (self, orig_model_options, orig_hook_mode)
    return post_obj, sampler_function_input

def delta_sample(model, noise, steps, cfg, sampler_name, scheduler, 
             positive, negative, latent_image, denoise=1.0, 
             disable_noise=False, start_step=None, last_step=None, 
             force_full_denoise=False, noise_mask=None, 
             sigmas=None, callback=None, disable_pbar=False, seed=None):
    # replica of comfy.sample.sample()
    sampler = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, 
                                      sampler=sampler_name, scheduler=scheduler, 
                                      denoise=denoise, model_options=model.model_options)

    post_obj, sampler_function_input = sampler.sample(noise, 
                                                    positive, 
                                                    negative, 
                                                    cfg=cfg, 
                                                    latent_image=latent_image, 
                                                    start_step=start_step, 
                                                    last_step=last_step, 
                                                    force_full_denoise=force_full_denoise, 
                                                    denoise_mask=noise_mask, 
                                                    sigmas=sigmas, 
                                                    callback=callback, 
                                                    disable_pbar=disable_pbar,
                                                    seed=seed)
    # samples = samples.to(comfy.model_management.intermediate_device())
    return post_obj, sampler_function_input

def delta_common_ksampler(model, seed, steps, 
                          cfg, sampler_name, scheduler,
                          positive, negative, latent, 
                          denoise=1.0, disable_noise=False, 
                          start_step=None, last_step=None, 
                          force_full_denoise=False, sigmas=None,):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    
    # sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, 
    #        denoise=1.0, disable_noise=False, start_step=None, last_step=None, 
    #        force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None)
    
    post_obj, sampler_function_input = delta_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    # out["samples"] = samples
    return post_obj, sampler_function_input, out # (out, )
    
class Ensemble_Wrapper:
    def __init__(self, source_model_wrapper, adapted_model_wrapper, target_model_wrapper, delta_strength=1.0):
        self.source_model_wrapper = source_model_wrapper
        self.adapted_model_wrapper = adapted_model_wrapper
        self.target_model_wrapper = target_model_wrapper
        self.delta_strength = delta_strength
        self.inner_model = target_model_wrapper.inner_model

    def __call__(self, *args, **kwargs):
        x = args[0]
        s_in = args[1]
        n_channels = self.source_model_wrapper.inner_model.inner_model.latent_format.latent_channels
        source_x = x[:,:n_channels,:,:]
        target_x = x[:,n_channels:,:,:]
        
        source_latent = self.source_model_wrapper(*(source_x, s_in), **kwargs)
        delta_latent = self.adapted_model_wrapper(*(source_x, s_in), **kwargs)
        target_latent = self.target_model_wrapper(*(target_x, s_in), **kwargs)
        
        source_latent_real = self.source_model_wrapper.inner_model.inner_model.process_latent_out(source_latent.to(torch.float32))
        delta_latent_real  = self.adapted_model_wrapper.inner_model.inner_model.process_latent_out(delta_latent.to(torch.float32))
        target_latent_real = self.target_model_wrapper.inner_model.inner_model.process_latent_out(target_latent.to(torch.float32))

        target_latent_real = target_latent_real + self.delta_strength*(delta_latent_real - source_latent_real)
        
        target_latent = self.target_model_wrapper.inner_model.inner_model.process_latent_in(target_latent_real)
        source_latent = self.source_model_wrapper.inner_model.inner_model.process_latent_in(target_latent_real)


        return torch.cat([source_latent, target_latent], dim=1)

class VAE_Model_Wrapper_Combo:
    def __init__(self, source_model_wrapper, source_vae,
                 adapted_model_wrapper,  delta_vae,
                 target_model_wrapper, target_vae,
                 delta_strength=1.0):
        self.source_model_wrapper = source_model_wrapper
        self.source_vae = source_vae
        self.adapted_model_wrapper = adapted_model_wrapper
        self.delta_vae = delta_vae
        self.target_model_wrapper = target_model_wrapper
        self.target_vae = target_vae
        self.delta_strength = delta_strength

def clean_and_reset(source_cfg_guider_obj, delta_cfg_guider_obj, target_cfg_guider_obj,
                    source_orig_model_options, delta_orig_model_options, target_orig_model_options,
                    source_orig_hook_mode, delta_orig_hook_mode, target_orig_hook_mode):
    # clean models and conds
    source_cfg_guider_obj.model_patcher.cleanup()
    delta_cfg_guider_obj.model_patcher.cleanup()
    target_cfg_guider_obj.model_patcher.cleanup()
    
    comfy.sampler_helpers.cleanup_models(source_cfg_guider_obj.conds, source_cfg_guider_obj.loaded_models)
    comfy.sampler_helpers.cleanup_models(delta_cfg_guider_obj.conds, delta_cfg_guider_obj.loaded_models)
    comfy.sampler_helpers.cleanup_models(target_cfg_guider_obj.conds, target_cfg_guider_obj.loaded_models)
    
    del source_cfg_guider_obj.inner_model
    del delta_cfg_guider_obj.inner_model
    del target_cfg_guider_obj.inner_model
    
    del source_cfg_guider_obj.loaded_models
    del delta_cfg_guider_obj.loaded_models
    del target_cfg_guider_obj.loaded_models
    
    source_cfg_guider_obj.model_options = source_orig_model_options
    delta_cfg_guider_obj.model_options = delta_orig_model_options
    target_cfg_guider_obj.model_options = target_orig_model_options
    
    source_cfg_guider_obj.model_patcher.hook_mode = source_orig_hook_mode
    delta_cfg_guider_obj.model_patcher.hook_mode = delta_orig_hook_mode
    target_cfg_guider_obj.model_patcher.hook_mode = target_orig_hook_mode
    
    source_cfg_guider_obj.model_patcher.restore_hook_patches()
    delta_cfg_guider_obj.model_patcher.restore_hook_patches()
    target_cfg_guider_obj.model_patcher.restore_hook_patches()
    
    del source_cfg_guider_obj.conds
    del delta_cfg_guider_obj.conds
    del target_cfg_guider_obj.conds
    
    return

class LatentDeltaKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "source_positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image for source model."}),
                "source_negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image for source model."}),
                "adapted_model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "adapted_positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image for source model."}),
                "adapted_negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image for source model."}),
                "target_model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "target_positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image for target model."}),
                "target_negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image for target model."}),
                "delta_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1e9, "step":0.1, "round": 0.01, "tooltip": "Strength to control the delta guidence."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Delta KSampler working on latent space."

    def sample(self, source_model, source_positive, 
               source_negative, adapted_model, 
               adapted_positive, adapted_negative,
               target_model, target_positive, 
               target_negative, delta_strength, 
               seed, steps, cfg, sampler_name, 
               scheduler, latent_image, denoise):
        
        original_KSAMPLER = comfy.samplers.KSAMPLER
        original_inner_sample = comfy.samplers.CFGGuider.inner_sample 
        original_outer_sample = comfy.samplers.CFGGuider.outer_sample 
        original_sample = comfy.samplers.CFGGuider.sample 

        try:
            comfy.samplers.KSAMPLER = DELTAKSAMPLER
            comfy.samplers.CFGGuider.inner_sample = delta_inner_sample
            comfy.samplers.CFGGuider.outer_sample = delta_outer_sample
            comfy.samplers.CFGGuider.sample = delta_cfg_sample

            source_post_obj, source_sampler_function_input, _ = delta_common_ksampler(source_model, seed, steps, 
                                                                                    cfg, sampler_name, scheduler, 
                                                                                    source_positive, source_negative, 
                                                                                    latent_image, denoise)
            delta_post_obj, delta_sampler_function_input, _ = delta_common_ksampler(adapted_model, seed, steps, 
                                                                                    cfg, sampler_name, scheduler,
                                                                                    adapted_positive, adapted_negative, 
                                                                                    latent_image, denoise)
            target_post_obj, target_sampler_function_input, target_out = delta_common_ksampler(target_model, seed, steps, 
                                                                                            cfg, sampler_name, scheduler, 
                                                                                            target_positive, target_negative, 
                                                                                            latent_image, denoise)
            
            # Unpack sampler function inputs with better error handling
            if source_sampler_function_input is None or not isinstance(source_sampler_function_input, tuple):
                raise ValueError(f"source_sampler_function_input is invalid: {type(source_sampler_function_input)}")
            if delta_sampler_function_input is None or not isinstance(delta_sampler_function_input, tuple):
                raise ValueError(f"delta_sampler_function_input is invalid: {type(delta_sampler_function_input)}")
            if target_sampler_function_input is None or not isinstance(target_sampler_function_input, tuple):
                raise ValueError(f"target_sampler_function_input is invalid: {type(target_sampler_function_input)}")
            
            _, source_model_wrapper, source_noise, _, _, _, _ = source_sampler_function_input
            _, adapted_model_wrapper, delta_noise, _, _, _, _ = delta_sampler_function_input
            target_sampler, target_model_wrapper, target_noise, sigmas, extra_args, k_callback, disable_pbar = target_sampler_function_input
            
            source_cfg_guider_obj, source_orig_model_options, source_orig_hook_mode = source_post_obj
            delta_cfg_guider_obj, delta_orig_model_options, delta_orig_hook_mode = delta_post_obj
            target_cfg_guider_obj, target_orig_model_options, target_orig_hook_mode = target_post_obj
            
            ensemble_model = Ensemble_Wrapper(source_model_wrapper, 
                                            adapted_model_wrapper, 
                                            target_model_wrapper, 
                                            delta_strength=delta_strength)
            
            noise = torch.cat([source_noise, target_noise], dim=1)
            samples = target_sampler.sampler_function(ensemble_model, noise, sigmas, 
                                                      extra_args=extra_args, callback=k_callback, 
                                                      disable=disable_pbar, **target_sampler.extra_options)
            samples = ensemble_model.target_model_wrapper.inner_model.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
            n_channels = source_model_wrapper.inner_model.inner_model.latent_format.latent_channels

            samples = samples[:,n_channels:,:,:]
            samples = target_cfg_guider_obj.inner_model.process_latent_out(samples.to(torch.float32))

            clean_and_reset(source_cfg_guider_obj, delta_cfg_guider_obj, target_cfg_guider_obj,
                            source_orig_model_options, delta_orig_model_options, target_orig_model_options,
                            source_orig_hook_mode, delta_orig_hook_mode, target_orig_hook_mode)
                    
            samples = samples.to(comfy.model_management.intermediate_device())
            target_out["samples"] = samples
        finally:
            comfy.samplers.KSAMPLER =  original_KSAMPLER
            comfy.samplers.CFGGuider.inner_sample =  original_inner_sample
            comfy.samplers.CFGGuider.outer_sample =  original_outer_sample
            comfy.samplers.CFGGuider.sample =  original_sample
        return (target_out, )

NODE_CLASS_MAPPINGS = {
    "LatentDeltaKSampler": LatentDeltaKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentDeltaKSampler": "Latent Delta KSampler",
}