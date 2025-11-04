import torch
from diffusers.utils import load_image
from diffusers import StableVideoDiffusionPipeline
import cv2
import numpy as np
from PIL import Image
from equilib import equi2pers, equi2equi
from typing import List, Tuple
import copy
import gc
import os

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision.transforms.functional import to_tensor, gaussian_blur
from torch.nn import functional as F
from imageio import mimsave

from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from src import tensor_to_vae_latent, pers2equi, generate_mask, generate_mask_batch, get_rpy, pers2equi_batch, partial360_to_pers

def _append_dims(x, target_dims):
	"""Appends dimensions to the end of a tensor until it has target_dims dimensions."""
	dims_to_append = target_dims - x.ndim
	if dims_to_append < 0:
		raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
	return x[(...,) + (None,) * dims_to_append]


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
	scheduler,
	num_inference_steps: Optional[int] = None,
	device: Optional[Union[str, torch.device]] = None,
	timesteps: Optional[List[int]] = None,
	sigmas: Optional[List[float]] = None,
	**kwargs,
):
	"""
	Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
	custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

	Args:
		scheduler (`SchedulerMixin`):
			The scheduler to get timesteps from.
		num_inference_steps (`int`):
			The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
			must be `None`.
		device (`str` or `torch.device`, *optional*):
			The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
		timesteps (`List[int]`, *optional*):
			Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
			`num_inference_steps` and `sigmas` must be `None`.
		sigmas (`List[float]`, *optional*):
			Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
			`num_inference_steps` and `timesteps` must be `None`.

	Returns:
		`Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
		second element is the number of inference steps.
	"""
	if timesteps is not None and sigmas is not None:
		raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
	if timesteps is not None:
		accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
		if not accepts_timesteps:
			raise ValueError(
				f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
				f" timestep schedules. Please check whether you are using the correct scheduler."
			)
		scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
		timesteps = scheduler.timesteps
		num_inference_steps = len(timesteps)
	elif sigmas is not None:
		accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
		if not accept_sigmas:
			raise ValueError(
				f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
				f" sigmas schedules. Please check whether you are using the correct scheduler."
			)
		scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
		timesteps = scheduler.timesteps
		num_inference_steps = len(timesteps)
	else:
		scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
		timesteps = scheduler.timesteps
	return timesteps, num_inference_steps


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
	r"""
	Output class for Stable Video Diffusion pipeline.

	Args:
		frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.Tensor`]):
			List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
			num_frames, height, width, num_channels)`.
	"""

	frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


class StableVideoDiffusionPipelineCustom(DiffusionPipeline):
	r"""
	Pipeline to generate video from an input image using Stable Video Diffusion.

	This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
	implemented for all pipelines (downloading, saving, running on a particular device, etc.).

	Args:
		vae ([`AutoencoderKLTemporalDecoder`]):
			Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
		image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
			Frozen CLIP image-encoder
			([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
		unet ([`UNetSpatioTemporalConditionModel`]):
			A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
		scheduler ([`EulerDiscreteScheduler`]):
			A scheduler to be used in combination with `unet` to denoise the encoded image latents.
		feature_extractor ([`~transformers.CLIPImageProcessor`]):
			A `CLIPImageProcessor` to extract features from generated images.
	"""

	model_cpu_offload_seq = "image_encoder->unet->vae"
	_callback_tensor_inputs = ["latents"]

	def __init__(
		self,
		vae: AutoencoderKLTemporalDecoder,
		image_encoder: CLIPVisionModelWithProjection,
		unet: UNetSpatioTemporalConditionModel,
		scheduler: EulerDiscreteScheduler,
		feature_extractor: CLIPImageProcessor,
	):
		super().__init__()

		self.register_modules(
			vae=vae,
			image_encoder=image_encoder,
			unet=unet,
			scheduler=scheduler,
			feature_extractor=feature_extractor,
		)
		self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
		self.video_processor = VideoProcessor(do_resize=True, vae_scale_factor=self.vae_scale_factor)

	def _encode_image(
		self,
		image: torch.Tensor, # in (B, T, C, H, W) format, range [-1, 1], now B = 1
		device: Union[str, torch.device],
		num_videos_per_prompt: int,
		do_classifier_free_guidance: bool,
	) -> torch.Tensor:
		
		b, t, c, h, w = image.shape
		image = image.view(b*t, c, h, w)
		dtype = next(self.image_encoder.parameters()).dtype

		# We normalize the image before resizing to match with the original implementation.
		# Then we unnormalize it after resizing.
		# image = image * 2.0 - 1.0
		image = _resize_with_antialiasing(image, (224, 224))
		image = (image + 1.0) / 2.0

		# Normalize the image with for CLIP input
		image = self.feature_extractor(
			images=image.to(dtype=torch.float32),
			do_normalize=True,
			do_center_crop=False,
			do_resize=False,
			do_rescale=False,
			return_tensors="pt",
		).pixel_values.to(dtype=dtype)

		image = image.to(device=device, dtype=dtype)
		image_embeddings = self.image_encoder(image).image_embeds
		image_embeddings = image_embeddings.view(b, t, -1)

		# duplicate image embeddings for each generation per prompt, using mps friendly method
		bs_embed, seq_len, _ = image_embeddings.shape
		image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
		image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

		if do_classifier_free_guidance:
			negative_image_embeddings = torch.zeros_like(image_embeddings)
			# For classifier free guidance, we need to do two forward passes.
			# Here we concatenate the unconditional and text embeddings into a single batch
			# to avoid doing two forward passes
			image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

		return image_embeddings

	def _encode_vae_image(
		self,
		image: torch.Tensor,
		device: Union[str, torch.device],
		num_videos_per_prompt: int,
		do_classifier_free_guidance: bool = False, 
	):
		image = image.to(device=device)
		image_latents = self.vae.encode(image).latent_dist.mode()

		# duplicate image_latents for each generation per prompt, using mps friendly method
		image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

		if do_classifier_free_guidance:
			negative_image_latents = torch.zeros_like(image_latents)

			# For classifier free guidance, we need to do two forward passes.
			# Here we concatenate the unconditional and text embeddings into a single batch
			# to avoid doing two forward passes
			image_latents = torch.cat([negative_image_latents, image_latents])

		return image_latents
	
	def _encode_vae_video(self, video: torch.Tensor, device: Union[str, torch.device], 
					   num_videos_per_prompt: int, 
					   do_classifier_free_guidance: bool = False) -> torch.Tensor:
		# encode
		video = video.to(device=device)
		video_length = video.shape[1]
		video = video.view(-1, video.shape[2], video.shape[3], video.shape[4])
		video_latents = self.vae.encode(video).latent_dist.mode()
		video_latents = video_latents * self.vae.config.scaling_factor

		video_latents = video_latents.view(-1, video_length, video_latents.shape[1], video_latents.shape[2], video_latents.shape[3])

		# duplicate video_latents for each generation per prompt, using mps friendly method
		video_latents = video_latents.repeat(num_videos_per_prompt, 1, 1, 1, 1)

		if do_classifier_free_guidance:
			negative_video_latents = torch.zeros_like(video_latents)
			video_latents = torch.cat([negative_video_latents, video_latents])

		return video_latents # (B, T, C, H, W), B = original batch size * num_videos_per_prompt * 2 if do_classifier_free_guidance

	def _get_add_time_ids(
		self,
		fps: int,
		motion_bucket_id: int,
		noise_aug_strength: float,
		dtype: torch.dtype,
		batch_size: int,
		num_videos_per_prompt: int,
		do_classifier_free_guidance: bool,
	):
		add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

		passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
		expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

		if expected_add_embed_dim != passed_add_embed_dim:
			raise ValueError(
				f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
			)

		add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
		add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

		if do_classifier_free_guidance:
			add_time_ids = torch.cat([add_time_ids, add_time_ids])

		return add_time_ids

	def decode_latents(self, latents: torch.Tensor, num_frames: int, decode_chunk_size: int = 25, 
					   extended_decoding: bool = False,
					   blend_decoding_ratio: int = 4) -> torch.Tensor:
		# [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
		# if blend rotation, rotate the latents and blend the decoded frames
		decode_chunk_size = num_frames if decode_chunk_size is None else decode_chunk_size
		latents = latents.flatten(0, 1)

		latents = latents / self.vae.config.scaling_factor

		forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
		accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

		# decode decode_chunk_size frames at a time to avoid OOM
		frames = []
		for i in range(0, latents.shape[0], decode_chunk_size):
			num_frames_in = latents[i : i + decode_chunk_size].shape[0]
			decode_kwargs = {}
			if accepts_num_frames:
				# we only pass num_frames_in if it's expected
				decode_kwargs["num_frames"] = num_frames_in
			
			if extended_decoding:
				latent_width = latents.shape[-1]
				frame_left = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
				frame_right = self.vae.decode(torch.roll(latents[i : i + decode_chunk_size], latent_width // 2, dims=-1), **decode_kwargs).sample
				
				frame_width = frame_left.shape[-1]
				frame_right = torch.roll(frame_right, -frame_width // 2, dims=-1)

				blend_count = int(frame_width // blend_decoding_ratio)
				same_width = (frame_width // 2 - blend_count) // 2
				weight_left_half = torch.cat([torch.zeros(same_width), torch.linspace(0, 1, frame_width // 2 - same_width * 2), torch.ones(same_width)]).to(frame_left.device)
				weight_left = torch.cat([weight_left_half, weight_left_half.flip(0)])
				weight_right = 1 - weight_left
				frame = frame_left * weight_left + frame_right * weight_right
			else:
				frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
			frames.append(frame)
		frames = torch.cat(frames, dim=0)

		# [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
		frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

		# we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
		frames = frames.float()
		return frames

	def prepare_latents(
		self,
		batch_size: int,
		num_frames: int,
		num_channels_latents: int,
		height: int,
		width: int,
		dtype: torch.dtype,
		device: Union[str, torch.device],
		generator: torch.Generator,
		latents: Optional[torch.Tensor] = None,
		return_noise: bool = False,
	):
		shape = (
			batch_size,
			num_frames,
			num_channels_latents // 2,
			height // self.vae_scale_factor,
			width // self.vae_scale_factor,
		)
		if isinstance(generator, list) and len(generator) != batch_size:
			raise ValueError(
				f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
				f" size of {batch_size}. Make sure the batch size matches the length of the generators."
			)

		if latents is None:
			latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
		else:
			latents = latents.to(device)

		# scale the initial noise by the standard deviation required by the scheduler
		if return_noise:
			return latents * self.scheduler.init_noise_sigma, latents
		else:
			return latents * self.scheduler.init_noise_sigma

	@property
	def guidance_scale(self):
		return self._guidance_scale

	# here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
	# of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
	# corresponds to doing no classifier free guidance.
	@property
	def do_classifier_free_guidance(self):
		if isinstance(self.guidance_scale, (int, float)):
			return self.guidance_scale > 1
		return self.guidance_scale.max() > 1

	@property
	def num_timesteps(self):
		return self._num_timesteps

	@torch.no_grad()
	def __call__(
		self,
		# image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
		video: torch.Tensor,
		conditional_images: torch.Tensor,
		height: int = 576,
		width: int = 1024,
		num_frames: Optional[int] = None,
		num_inference_steps: int = 25,
		sigmas: Optional[List[float]] = None,
		min_guidance_scale: float = 1.0,
		max_guidance_scale: float = 3.0,
		fps: int = 7,
		motion_bucket_id: int = 127,
		noise_aug_strength: float = 0.02,
		decode_chunk_size: Optional[int] = None,
		num_videos_per_prompt: Optional[int] = 1,
		generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
		latents: Optional[torch.Tensor] = None,
		generated_latents: Optional[torch.Tensor] = None,
		# output_type: Optional[str] = "pil",
		callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
		callback_on_step_end_tensor_inputs: List[str] = ["latents"],
		inference_final_rotation: float = 0,
		blend_decoding_ratio: int = 4, 
		extended_decoding: bool = False,
		rotation_during_inference: bool = False,
		return_noise: bool = False,
		return_latents: bool = False,
	):
		r"""
		The call function to the pipeline for generation.

		Args:
			video (torch.Tensor):
				equirectangular video of shape (batch, frames, channels, height, width), currently batch = 1, 
				values in range [-1, 1]
			conditional_images (torch.Tensor): 
				normal image of shape (batch, frames, channels, height, width), currently batch = 1, values in range [-1, 1]
			mask (torch.Tensor):
				mask of shape [batch, 1, height, width], values in range {0, 1}
			height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
				The height in pixels of the generated image.
			width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
				The width in pixels of the generated image.
			num_frames (`int`, *optional*):
				The number of video frames to generate. Defaults to `self.unet.config.num_frames` (14 for
				`stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`).
			num_inference_steps (`int`, *optional*, defaults to 25):
				The number of denoising steps. More denoising steps usually lead to a higher quality video at the
				expense of slower inference. This parameter is modulated by `strength`.
			sigmas (`List[float]`, *optional*):
				Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
				their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
				will be used.
			min_guidance_scale (`float`, *optional*, defaults to 1.0):
				The minimum guidance scale. Used for the classifier free guidance with first frame.
			max_guidance_scale (`float`, *optional*, defaults to 3.0):
				The maximum guidance scale. Used for the classifier free guidance with last frame.
			fps (`int`, *optional*, defaults to 7):
				Frames per second. The rate at which the generated images shall be exported to a video after
				generation. Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
			motion_bucket_id (`int`, *optional*, defaults to 127):
				Used for conditioning the amount of motion for the generation. The higher the number the more motion
				will be in the video.
			noise_aug_strength (`float`, *optional*, defaults to 0.02):
				The amount of noise added to the init image, the higher it is the less the video will look like the
				init image. Increase it for more motion.
			decode_chunk_size (`int`, *optional*):
				The number of frames to decode at a time. Higher chunk size leads to better temporal consistency at the
				expense of more memory usage. By default, the decoder decodes all frames at once for maximal quality.
				For lower memory usage, reduce `decode_chunk_size`.
			num_videos_per_prompt (`int`, *optional*, defaults to 1):
				The number of videos to generate per prompt.
			generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
				A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
				generation deterministic.
			latents (`torch.Tensor`, *optional*):
				Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
				generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
				tensor is generated by sampling using the supplied random `generator`.
			generated_latents (`torch.Tensor`, *optional*):
				Pre-generated latents, will be used to replace the first generated_latents[1] frames of the latents 
				in each denoising step. This can be used to generate a longer video autoreressively.
			output_type (`str`, *optional*, defaults to `"pil"`):
				The output format of the generated image. Choose between `pil`, `np` or `pt`.
			callback_on_step_end (`Callable`, *optional*):
				A function that is called at the end of each denoising step during inference. The function is called
				with the following arguments:
					`callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.
				`callback_kwargs` will include a list of all tensors as specified by
				`callback_on_step_end_tensor_inputs`.
			callback_on_step_end_tensor_inputs (`List`, *optional*):
				The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
				will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
				`._callback_tensor_inputs` attribute of your pipeline class.

		Returns: frames: torch.Tensor
		"""
		# 0. Default height and width to unet
		height = height or self.unet.config.sample_size * self.vae_scale_factor
		width = width or self.unet.config.sample_size * self.vae_scale_factor

		num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
		decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

		# 1. Check inputs.
		assert video.shape[-2] == height and video.shape[-1] == width, (
			f"Input video height and width should be {height} and {width}, but got {video.shape[-2]} and {video.shape[-1]}."
		)

		# 2. Define call parameters
		batch_size = video.shape[0]
		device = self._execution_device
		# here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
		# of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
		# corresponds to doing no classifier free guidance.
		self._guidance_scale = max_guidance_scale

		# 3. Prepare inputs
		# Encode input image, for creating image embeddings (CLIPEncoder)
		image_embeddings = self._encode_image(conditional_images, device, num_videos_per_prompt, self.do_classifier_free_guidance)

		# NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
		# See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
		fps = fps - 1

		# 4. Encode input video using VAE, for creating latents
		# noise = randn_tensor(video.shape, generator=generator, device=device, dtype=video.dtype)
		# video_noisy = video + (noise_aug_strength * noise * mask)

		needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
		if needs_upcasting:
			self.vae.to(dtype=torch.float32)

		# video_latents_noisy = self._encode_vae_video(video_noisy, device, num_videos_per_prompt, self.do_classifier_free_guidance)
		video_latents_noisy = self._encode_vae_video(video, device, num_videos_per_prompt, self.do_classifier_free_guidance)
		video_latents_noisy = video_latents_noisy.to(video.dtype) # (batch, frames, channels, height, width), batch doubled if classifier free guidance
		video_latents_noisy = video_latents_noisy * (1 / self.vae.config.scaling_factor)

		# cast back to fp16 if needed
		if needs_upcasting:
			self.vae.to(dtype=torch.float16)

		# 5. Get Added Time IDs
		added_time_ids = self._get_add_time_ids(
			fps,
			motion_bucket_id,
			noise_aug_strength,
			video.dtype,
			batch_size,
			num_videos_per_prompt,
			self.do_classifier_free_guidance,
		)
		added_time_ids = added_time_ids.to(device)

		# 6. Prepare timesteps
		timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None, sigmas)

		# 7. Prepare latent variables
		num_channels_latents = self.unet.config.in_channels
		latents, noise = self.prepare_latents( # prepare the noise for the diffusion
			batch_size * num_videos_per_prompt,
			num_frames,
			num_channels_latents,
			height,
			width,
			video.dtype,
			device,
			generator,
			latents,
			return_noise=True,
		)

		# 8. Prepare guidance scale
		guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
		guidance_scale = guidance_scale.to(device, latents.dtype)
		guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
		guidance_scale = _append_dims(guidance_scale, latents.ndim)

		self._guidance_scale = guidance_scale

		# 9. Denoising loop
		num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
		self._num_timesteps = len(timesteps)

		total_rotation = 0.

		with self.progress_bar(total=num_inference_steps) as progress_bar:
			for i, t in enumerate(timesteps):

				# expand the latents if we are doing classifier free guidance
				latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
				latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

				# Concatenate latents over channels dimension
				latent_model_input = torch.cat([latent_model_input, video_latents_noisy], dim=2) # (B*2, T, C*2, H, W)

				# predict the noise residual
				noise_pred = self.unet(
					latent_model_input,
					t,
					encoder_hidden_states=image_embeddings,
					added_time_ids=added_time_ids,
					return_dict=False,
				)[0]

				# perform guidance
				if self.do_classifier_free_guidance:
					noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
					noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

				# compute the previous noisy sample x_t -> x_t-1
				latents = self.scheduler.step(noise_pred, t, latents).prev_sample

				if callback_on_step_end is not None:
					callback_kwargs = {}
					for k in callback_on_step_end_tensor_inputs:
						callback_kwargs[k] = locals()[k]
					callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

					latents = callback_outputs.pop("latents", latents)

				if generated_latents is not None:
					num_blend_frames = generated_latents.shape[1]
					latents[:, :num_blend_frames] = generated_latents
					
				if rotation_during_inference and i < len(timesteps) - 1: # do rotation
					latents = torch.roll(latents, shifts=latents.shape[-1]//4, dims=-1)
					video_latents_noisy = torch.roll(video_latents_noisy, shifts=video_latents_noisy.shape[-1]//4, dims=-1)
					if generated_latents is not None:
						generated_latents = torch.roll(generated_latents, shifts=generated_latents.shape[-1]//4, dims=-1)
					total_rotation += 90.

				if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
					progress_bar.update()

		# 11. Decode latents and rotate the final video
		rotation_angle_left = (inference_final_rotation + 360 - total_rotation % 360) % 360
		latents = torch.roll(latents, shifts=int((latents.shape[-1] * rotation_angle_left)//360), dims=-1)

		if return_latents:
			return latents

		if needs_upcasting:
			self.vae.to(dtype=torch.float16)
		frames = self.decode_latents(latents, num_frames, decode_chunk_size, extended_decoding, blend_decoding_ratio)

		self.maybe_free_model_hooks()

		if return_noise:
			return frames, noise
		else:
			return frames
	

# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
	h, w = input.shape[-2:]
	factors = (h / size[0], w / size[1])

	# First, we have to determine sigma
	# Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
	sigmas = (
		max((factors[0] - 1.0) / 2.0, 0.001),
		max((factors[1] - 1.0) / 2.0, 0.001),
	)

	# Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
	# https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
	# But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
	ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

	# Make sure it is odd
	if (ks[0] % 2) == 0:
		ks = ks[0] + 1, ks[1]

	if (ks[1] % 2) == 0:
		ks = ks[0], ks[1] + 1

	input = _gaussian_blur2d(input, ks, sigmas)

	output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
	return output


def _compute_padding(kernel_size):
	"""Compute padding tuple."""
	# 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
	# https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
	if len(kernel_size) < 2:
		raise AssertionError(kernel_size)
	computed = [k - 1 for k in kernel_size]

	# for even kernels we need to do asymmetric padding :(
	out_padding = 2 * len(kernel_size) * [0]

	for i in range(len(kernel_size)):
		computed_tmp = computed[-(i + 1)]

		pad_front = computed_tmp // 2
		pad_rear = computed_tmp - pad_front

		out_padding[2 * i + 0] = pad_front
		out_padding[2 * i + 1] = pad_rear

	return out_padding


def _filter2d(input, kernel):
	# prepare kernel
	b, c, h, w = input.shape
	tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

	tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

	height, width = tmp_kernel.shape[-2:]

	padding_shape: List[int] = _compute_padding([height, width])
	input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

	# kernel and input tensor reshape to align element-wise or batch-wise params
	tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
	input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

	# convolve the tensor with the kernel.
	output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

	out = output.view(b, c, h, w)
	return out


def _gaussian(window_size: int, sigma):
	if isinstance(sigma, float):
		sigma = torch.tensor([[sigma]])

	batch_size = sigma.shape[0]

	x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

	if window_size % 2 == 0:
		x = x + 0.5

	gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

	return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
	if isinstance(sigma, tuple):
		sigma = torch.tensor([sigma], dtype=input.dtype)
	else:
		sigma = sigma.to(dtype=input.dtype)

	ky, kx = int(kernel_size[0]), int(kernel_size[1])
	bs = sigma.shape[0]
	kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
	kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
	out_x = _filter2d(input, kernel_x[..., None, :])
	out = _filter2d(out_x, kernel_y[..., None])

	return out

@torch.no_grad()
def sample_svd(args, accelerator, pipeline,
			    weight_dtype,
				conditional_video: torch.Tensor, # in (T, C, H, W) format, range [-1, 1]
				out_file_path: str,
				roll: np.ndarray = None, pitch: np.ndarray = None, yaw: np.ndarray = None, # in radians, numpy array of length T
				width: int = None, height: int = None,
				fov_x=90,
				hw_ratio=2/3,
				guidance_scale: float = 5.0,
				num_inference_steps=25,
				fps=7,
				decode_chunk_size=None,
				noise_aug_strength=0.02,
				inference_final_rotation=0,
				blend_decoding_ratio=4,
				replacement_sampling=False,
				extended_decoding=False,
				noise_conditioning=False,
				noise_conditioning_strength=0.25,
				rotation_during_inference=False,
				equirectangular_input=True,
				return_for_metrics=False,
				post_rotation=False,
				narrow=False,
				height_narrow: int = None,
				width_narrow: int = None,
				num_known_frames=0,
				):

	T = conditional_video.shape[0]
	if equirectangular_input:
		height, width = conditional_video.shape[-2:]

		mask = generate_mask_batch(fov_x=fov_x, hw_ratio=hw_ratio, 
							 		roll=roll, pitch=pitch, yaw=yaw, 
							  		width=width, height=height, device=accelerator.device) # (T, 1, H, W)
		
		rots = [{"roll": roll[i], "pitch": pitch[i], "yaw": yaw[i]} for i in range(args.num_frames)]
		conditional_video_pers = equi2pers(conditional_video.to(torch.float32), fov_x=fov_x, 
											width=480, height=int(480*hw_ratio),
											rots=rots, z_down=True).to(weight_dtype) # in default height and width, will be sent to the CLIP
		
		if not post_rotation:
			conditional_video_equi = torch.where(mask == 1, conditional_video,
									torch.randn_like(conditional_video, device=accelerator.device) * noise_conditioning_strength \
									if noise_conditioning else -torch.ones_like(conditional_video, device=accelerator.device))
		else:
			zeros = torch.zeros(T, device=accelerator.device)
			conditional_video_equi = pers2equi_batch(conditional_video_pers, fov_x=fov_x, height=height, width=width, device=accelerator.device, 
													 roll=zeros, pitch=zeros, yaw=zeros)
			
		# conditional_video_equi = conditional_video_equi + torch.randn_like(conditional_video, device=accelerator.device) * noise_aug_strength
		conditional_video_equi[:num_known_frames] = conditional_video[:num_known_frames]
		conditional_video_equi = conditional_video_equi.unsqueeze(0)


	else:
		conditional_video_pers = copy.deepcopy(conditional_video)
		with torch.inference_mode(), torch.amp.autocast("cuda", dtype=weight_dtype):
			conditional_video_equi, mask = pers2equi_batch(conditional_video.to(device=accelerator.device,dtype=weight_dtype, non_blocking=True), fov_x=fov_x, 
															roll=roll, pitch=pitch, yaw=yaw,
															width=width, height=height, device=accelerator.device,
															return_mask=True) # (T, C, H, W)
		conditional_video_equi = conditional_video_equi.to(weight_dtype)
		# conditional_video_equi = conditional_video_equi + torch.randn_like(conditional_video_equi, device=accelerator.device) * noise_aug_strength
		# conditional_video_equi = conditional_video_equi + torch.randn_like(conditional_video_equi, device=accelerator.device) * noise_aug_strength * mask
		if noise_conditioning:
			conditional_video_equi = torch.where(mask == 1, conditional_video, torch.randn_like(conditional_video, device=accelerator.device))
		conditional_video_equi = conditional_video_equi.unsqueeze(0)
		hw_ratio = conditional_video_pers.shape[-2] / conditional_video_pers.shape[-1]
	
	if narrow:
		conditional_video_equi = conditional_video_equi[:, :, :, (height - height_narrow) // 2:(height + height_narrow) // 2, (width - width_narrow) // 2:(width + width_narrow) // 2]
		mask = mask[:, :, (height - height_narrow) // 2:(height + height_narrow) // 2, (width - width_narrow) // 2:(width + width_narrow) // 2]
		if equirectangular_input:
			conditional_video = conditional_video[:, :, (height - height_narrow) // 2:(height + height_narrow) // 2, (width - width_narrow) // 2:(width + width_narrow) // 2]
		height, width = height_narrow, width_narrow

	ext = '.'+out_file_path.split('.')[-1]
	return_file_path = out_file_path.replace(ext, f"_output_fov{fov_x:.0f}_hw{hw_ratio:.2f}.mp4")

	mask = mask.unsqueeze(0)
	conditional_video_pers = conditional_video_pers.unsqueeze(0)
	
	with torch.autocast(str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision != 'no', dtype=weight_dtype):
		#num_frames_remaining = args.num_frames
		num_frames_remaining = args.num_frames if args.num_frames is not None else conditional_video.shape[0]
		num_frames_batch = args.num_frames_batch if (hasattr(args, 'num_frames_batch') and args.num_frames_batch is not None) else args.num_frames
		generated_latents = None
		generated_latents_this = None
		generated_frames_this = None
		num_frames_processed = 0
		blend_frames = args.blend_frames if hasattr(args, 'blend_frames') else 0
		round = 0

		while num_frames_remaining > 0:
			num_frames_to_process = min(num_frames_batch, num_frames_remaining)
			conditional_video_input = conditional_video_equi[:, :num_frames_to_process] if round == 0 else \
				torch.cat([generated_frames_this[:, :, -blend_frames:].permute(0, 2, 1, 3, 4),
						conditional_video_equi[:, num_frames_processed+blend_frames: num_frames_processed+num_frames_to_process]], 
						dim=1)
			#if round == 0:
			#	conditional_video_input = conditional_video_equi[:, :num_frames_to_process]
			#elif blend_frames > 0:
			#	conditional_video_input = torch.cat([generated_frames_this[:, :, -blend_frames:].permute(0, 2, 1, 3, 4),
			#									conditional_video_equi[:, num_frames_processed+blend_frames: num_frames_processed+num_frames_to_process]], 
			#									dim=1)
			#else:
			#	conditional_video_input = conditional_video_equi[:, num_frames_processed: num_frames_processed+num_frames_to_process]

			mask_this = mask[:, num_frames_processed: num_frames_processed+num_frames_to_process]
			conditional_video_pers_this = conditional_video_pers[:, num_frames_processed: num_frames_processed+num_frames_to_process]
			conditional_video_input = conditional_video_input + torch.randn_like(conditional_video_input, device=accelerator.device) * noise_aug_strength * mask_this

			generated_latents_this = pipeline(
				conditional_video_input, # (1, T, C, H, W)
				conditional_images = conditional_video_pers_this, # (1, T, C, H, W)
				height=height,
				width=width,
				num_frames=num_frames_to_process,
				decode_chunk_size=decode_chunk_size,
				motion_bucket_id=127,
				fps=fps,
				num_inference_steps=num_inference_steps,
				noise_aug_strength=noise_aug_strength,
				min_guidance_scale=guidance_scale,
				max_guidance_scale=guidance_scale,
				inference_final_rotation=inference_final_rotation,
				blend_decoding_ratio=blend_decoding_ratio,
				extended_decoding=extended_decoding,
				rotation_during_inference=rotation_during_inference,
				return_latents=True,
			) # [B, T, C, H, W]
			if blend_frames != 0: # save current generation results into a file
				generated_frames_this = pipeline.decode_latents(generated_latents_this, num_frames_to_process, decode_chunk_size=decode_chunk_size, extended_decoding=extended_decoding, blend_decoding_ratio=blend_decoding_ratio)
				generated_frames_this_save = ((generated_frames_this.clamp(-1, 1) + 1) * 127.5).cpu().to(torch.float32).numpy().astype(np.uint8)
				generated_frames_this_save = generated_frames_this_save[0].transpose(1, 2, 3, 0) # (T, H, W, C)
				save_path = out_file_path.replace(ext, f"_output_fov{fov_x:.0f}_hw{hw_ratio:.2f}_round{round}.mp4")
				mimsave(save_path, generated_frames_this_save, fps=fps)

			if generated_latents is None:
				generated_latents = generated_latents_this
			else:
				blend_weight = torch.linspace(1, 0, blend_frames, device=accelerator.device, dtype=generated_latents.dtype).view(1, blend_frames, 1, 1, 1)
				generated_latents = torch.cat([generated_latents[:, :-blend_frames], 
									blend_weight * generated_latents[:, -blend_frames:] + (1 - blend_weight) * generated_latents_this[:, :blend_frames],
									generated_latents_this[:, blend_frames:]], dim=1)
				#if blend_frames > 0:
				#	bf = int(min(blend_frames, generated_latents.shape[1], generated_latents_this.shape[1]))
				#	if bf > 0:
				#		blend_weight = torch.linspace(1, 0, bf, device=accelerator.device, dtype=generated_latents.dtype).view(1, bf, 1, 1, 1)
				#		generated_latents = torch.cat([
				#			generated_latents[:, :-bf],
				#			blend_weight * generated_latents[:, -bf:] + (1 - blend_weight) * generated_latents_this[:, :bf],
				#			generated_latents_this[:, bf:]
				#		], dim=1)
				#	else:
				#		generated_latents = torch.cat([generated_latents, generated_latents_this], dim=1)
				#else:
				#	generated_latents = torch.cat([generated_latents, generated_latents_this], dim=1)

			if num_frames_remaining == num_frames_to_process:
				break

			num_frames_remaining -= (num_frames_to_process - blend_frames)
			num_frames_processed += (num_frames_to_process - blend_frames)
			round += 1

		generated_frames = pipeline.decode_latents(generated_latents, args.num_frames, decode_chunk_size=decode_chunk_size, extended_decoding=extended_decoding, blend_decoding_ratio=blend_decoding_ratio)

	if post_rotation:
		rots = [{"roll": -roll[i], "pitch": -pitch[i], "yaw": -yaw[i]} for i in range(args.num_frames)]
		generated_frames = equi2equi(generated_frames.permute(0, 2, 1, 3, 4).flatten(0, 1), rots=rots, z_down=True).permute(1, 0, 2, 3).view(generated_frames.shape)
		conditional_video_equi = equi2equi(conditional_video_equi.flatten(0, 1), rots=rots, z_down=True).view(conditional_video_equi.shape)

	# save the input video
	save_path = out_file_path.replace(ext, f"_input_fov{fov_x:.0f}_hw{hw_ratio:.2f}.mp4")
	conditional_video_equi = ((conditional_video_equi.clamp(-1, 1) + 1) * 127.5).cpu().to(torch.float32).numpy().astype(np.uint8)
	conditional_video_equi = conditional_video_equi[0].transpose(0, 2, 3, 1) # (T, H, W, C)
	mimsave(save_path, conditional_video_equi, fps=fps)

	# save the generated video, RGB -> BGR
	generated_frames = ((generated_frames.clamp(-1, 1) + 1) * 127.5).cpu().to(torch.float32).numpy().astype(np.uint8)
	generated_frames = generated_frames[0].transpose(1, 2, 3, 0) # (T, H, W, C)
	mimsave(return_file_path, generated_frames, fps=fps)

	# save the input video and the generated video side-by-side
	save_path = out_file_path.replace(ext, f"_stack_fov{fov_x:.0f}_hw{hw_ratio:.2f}.mp4")
	if equirectangular_input:
		conditional_video = ((conditional_video.clamp(-1, 1) + 1) * 127.5).cpu().to(torch.float32).numpy().astype(np.uint8)
		conditional_video = conditional_video.transpose(0, 2, 3, 1) # (T, H, W, C)
		concatenated_video = np.concatenate([conditional_video_equi, generated_frames, conditional_video], axis=1) # (T, 3H, W, C)
		mimsave(save_path, concatenated_video, fps=fps)
	else:
		concatenated_video = np.concatenate([conditional_video_equi, generated_frames], axis=1) # (T, 3H, W, C)
		mimsave(save_path, concatenated_video, fps=fps)

	if narrow: # transform the narrow video into perspective format in orignal fov
		save_path = out_file_path.replace(ext, f"_narrow_pers_fov{fov_x:.0f}_hw{hw_ratio:.2f}.mp4")
		partial360_to_pers(return_file_path, save_path, fov_x=fov_x, hw_ratio=hw_ratio, width=args.width, height=args.height)

		save_path = out_file_path.replace(ext, f"_narrow_pers_fov90_hw1.mp4")
		partial360_to_pers(return_file_path, save_path, fov_x=90, hw_ratio=1, width=args.width, height=args.height)

	del pipeline
	torch.cuda.empty_cache()
	gc.collect()

	if return_for_metrics:
		return mask, generated_frames

	return return_file_path
