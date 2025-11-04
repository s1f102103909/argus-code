from diffusers import UNetSpatioTemporalConditionModel, AutoencoderKLTemporalDecoder
import numpy as np
import argparse
import torch
from dataset.video_dataset import VideoDataset
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import os
from accelerate import Accelerator
from src import rotation_matrix_to_euler, sample_svd, focal2fov, StableVideoDiffusionPipelineCustom, get_rpy
from equilib import equi2pers
import cv2
import random
import math
import copy
import time
from mast3r.model import AsymmetricMASt3R
from src.calibrate_cameras import get_camera_params_from_frames

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# load model arguments
	parser.add_argument('--pretrained_model_name_or_path', type=str, default='stabilityai/stable-video-diffusion-img2vid', help='Base pipeline path.')
	parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model's checkpoint.")
	parser.add_argument("--val_base_folder", nargs='+', type=str, required=True, help="Path to the valiadtion dataset.")
	parser.add_argument("--val_clip_file", type=str, default=None, help="Path to the validation clip file.")
	parser.add_argument("--calibration_cache_path", type=str, default=None, help="Path to the calibration cache file.")
	parser.add_argument("--calibration_cache_path_add_prefix", action='store_true', default=None, help="Path to the calibration cache file.")
	parser.add_argument("--cached_motion_npz_path", type=str, default=None, help="Path to the cached motion npz file.")
	parser.add_argument('--val_save_folder', type=str, default='results', help='Path to save the generated videos.')
	parser.add_argument("--unet_path", type=str, required=True, help="Path to the pretrained U-Net model.")
	
	# data arguments
	parser.add_argument('--dataset_size', type=int, default=None, help='Number of videos to generate.')
	parser.add_argument("--width", type=int, default=1024, help="Width of the generated video.")
	parser.add_argument("--height", type=int, default=512, help="Height of the generated video.")

	parser.add_argument('--equirectangular_input', action='store_true', help='Input is equirectangular.')
	parser.add_argument('--crop_center_then_calibrate', action='store_true', help='Crop the center of the video and then calibrate.')
	parser.add_argument('--cached_motion_path', type=str, default=None, help='Path to the cached motion files.')
	parser.add_argument('--fov_x_min', type=float, default=90., help='Minimum width fov')
	parser.add_argument('--fov_x_max', type=float, default=90., help='Maximum width fov')
	parser.add_argument('--fov_y_min', type=float, default=90., help='Minimum height fov')
	parser.add_argument('--fov_y_max', type=float, default=90., help='Maximum height fov')
	parser.add_argument('--narrow', action='store_true', help='Use narrow image')
	parser.add_argument('--width_narrow', type=int, default=512, help='Width of the narrow image')
	parser.add_argument('--height_narrow', type=int, default=512, help='Height of the narrow image')

	parser.add_argument("--frame_rate", type=int, default=None, help="Frame rate of the video.")
	parser.add_argument('--frame_interval', type=int, default=None, help='Interval between frames.')
	parser.add_argument('--fixed_start_frame', action='store_true', help='for each video, start from the first frame, for debugging')
	parser.add_argument('--full_sampling', action='store_true', help='Sample all frames in the video.')

	# inference arguments
	parser.add_argument('--rotation_during_inference', action='store_true', help='Rotate the video during inference.')
	parser.add_argument('--post_rotation', action='store_true', help='Rotate the video after inference.')
	parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps.')
	parser.add_argument('--inference_final_rotation', type=int, default=0, help='Final rotation during inference.')
	parser.add_argument('--blend_decoding_ratio', type=int, default=4, help='Blend decoding ratio. typically 2 or 4')
	parser.add_argument('--extended_decoding', action='store_true', help='Use extended decoding.')
	parser.add_argument('--replacement_sampling', action='store_true', help='Use replacement sampling.')
	parser.add_argument('--noise_conditioning', action='store_true', help='Condition on noise.')
	
	# calibration arguments
	parser.add_argument('--predict_camera_motion', action='store_true', help='Predict camera motion.')
	parser.add_argument('--dense_calibration', action='store_true', help='Dense calibration, i.e., use all frames for calibration.')
	parser.add_argument('--calibration_img_size', type=int, default=512, help='long side of image for MASt3R.')

	parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to generate.")
	parser.add_argument('--blend_frames', type=int, default=0, help='Number of frames to blend.')
	parser.add_argument("--num_frames_batch", type=int, default=25, help="Number of frames to generate.")
	parser.add_argument("--decode_chunk_size", type=int, default=10, help="Decode chunk size.")
	parser.add_argument("--motion_bucket_id", type=int, default=127, help="Motion bucket ID.")
	parser.add_argument("--noise_aug_strength", type=float, default=0.02, help="Noise augmentation strength.")
	parser.add_argument("--guidance_scale", type=float, default=1., help="Minimum guidance scale.")
	parser.add_argument('--fixed_fov', type=float, default=None, help='Fixed fov for all videos.')
	parser.add_argument('--fixed_rpy', action='store_true', help='Fixed rpy for all videos.')
	parser.add_argument('--yaw_start', type=float, default=0., help='Start yaw., in degrees')
	parser.add_argument('--noisy_rpy', type=float, default=0, help='add noise to rpy with this std.')

	args = parser.parse_args()
	os.makedirs(args.val_save_folder, exist_ok=True)

	# set a random seed based on time
	random.seed(int(time.time()))
	np.random.seed(int(time.time()))
	torch.manual_seed(int(time.time()))

	if args.calibration_cache_path is not None:
		args.calibration_cache_path = args.calibration_cache_path + f'_{args.num_frames}'
		os.makedirs(args.calibration_cache_path, exist_ok=True)

	accelerator = Accelerator(mixed_precision='bf16')
	if accelerator.mixed_precision == "bf16":
		weight_dtype = torch.bfloat16
	elif accelerator.mixed_precision == "fp16":
		weight_dtype = torch.float16
	else:  # "no"
		weight_dtype = torch.float32


	unet = UNetSpatioTemporalConditionModel.from_pretrained(
		pretrained_model_name_or_path=args.unet_path,
		subfolder="unet",
	)

	feature_extractor = CLIPImageProcessor.from_pretrained(
		args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
	)
	image_encoder = CLIPVisionModelWithProjection.from_pretrained(
		args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
	)
	vae = AutoencoderKLTemporalDecoder.from_pretrained(
		args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

	val_dataset = VideoDataset(args.val_base_folder, sample_frames=args.num_frames,
							   clip_info_path=args.val_clip_file,
							   fixed_start_frame=args.fixed_start_frame,
							   dataset_size=args.dataset_size,
							   width=args.width if args.equirectangular_input else None,
							   height=args.height if args.equirectangular_input else None,
							   calibration_img_size=args.calibration_img_size,
							   frame_interval=args.frame_interval,
							   frame_rate=args.frame_rate,
							   dense_calibration=args.dense_calibration,
							   cached_motion_path=args.cached_motion_path,
							   cached_motion_npz_path=args.cached_motion_npz_path,
							   full_sampling=args.full_sampling,)

	calibration_model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(accelerator.device)
	pipeline = StableVideoDiffusionPipelineCustom.from_pretrained(
				args.pretrained_model_name_or_path,
				unet=accelerator.unwrap_model(unet),
				image_encoder=accelerator.unwrap_model(image_encoder),
				vae=accelerator.unwrap_model(vae),
				revision=args.revision,
				torch_dtype=weight_dtype,
				).to(accelerator.device)

	for idx, batch in enumerate(val_dataset):

		frame_rate = batch['frame_rate'] # int
		video = batch["video"].to(weight_dtype).to(accelerator.device, non_blocking=True) # (T, C, H, W)
		video_calibration = video if not args.dense_calibration else batch['frames_dense'].to(weight_dtype).to(accelerator.device, non_blocking=True) # (T, C, H, W)
		path = batch['path']
		
		ext = '.'+path.split('.')[-1]
		out_file_path = os.path.join(args.val_save_folder, path.split('/')[-1])
		
		rolls, pitches, yaws, fov_x, hw_ratio = None, None, None, None, None

		if args.crop_center_then_calibrate: # input is equirectangular, unwrapp the center then calibrate (disable equirectangular input)
			# randomly choose fov_x, fov_y
			fov_x = random.uniform(args.fov_x_min, args.fov_x_max)
			# set hw_ratio to be 9/16 or 3/4
			hw_ratio = random.choice([9/16])
			# unwrap the center of the video
			height_crop, width_crop = 360, int(360 / hw_ratio)
			pers_frames = equi2pers(video, fov_x=fov_x, width=width_crop, height=height_crop, rots=[{"roll": 0, "pitch": 0, "yaw": 0} for _ in range(args.num_frames)], z_down=True)
			args.equirectangular_input = False
			video = copy.deepcopy(pers_frames)
			video_calibration = copy.deepcopy(pers_frames)
			del pers_frames

		if 'fov' in batch and 'motion' in batch:
			motion = batch['motion']
			fov = batch['fov']
			pitches, yaws, rolls = motion[0], motion[1], motion[2]
			fov_x, fov_y = fov[0], fov[1]
			hw_ratio = math.tan(math.radians(fov_y) / 2) / math.tan(math.radians(fov_x) / 2)

		else:
			if not args.equirectangular_input:

				if args.fixed_fov is None:
					
					suffix = os.path.basename(path).replace(ext, '_dense.pth') if args.dense_calibration else os.path.basename(path).replace(ext, '.pth')
					suffix = f'{path.split("/")[-2]}_{fov_x:.0f}_{suffix}' if args.calibration_cache_path_add_prefix is not None else suffix
					calibration_save_path = os.path.join(args.calibration_cache_path, suffix) if args.calibration_cache_path is not None else None
					if calibration_save_path and os.path.exists(calibration_save_path):
						print(f'Loading calibration cache from {calibration_save_path}')
						poses, fov_x = torch.load(calibration_save_path)
					else:
						poses, intrinsics, width_resized = get_camera_params_from_frames(video_calibration, shared_intrinsics=True, 
																			img_size=args.calibration_img_size,
																			scene_graph='swin-2-noncyclic',  # swin-2-noncyclic, logwin-2-noncyclic, complete
																			model=calibration_model) # (T, 4, 4), (T, 3, 3), but intrinsics are the same for all frames

						if args.dense_calibration:
							interval = (len(video_calibration) - 1) // (len(video) - 1)
							assert (len(video_calibration) - 1) % (len(video) - 1) == 0, "Invalid interval"
							poses = poses[::interval]
							intrinsics = intrinsics[::interval]
							
						focal_length = intrinsics[0, 0, 0].cpu().item() # focal length in pixels
						fov_x = torch.tensor(focal2fov(focal_length, width_resized), dtype=torch.float32).item()
						calibration_model = calibration_model.to('cpu')
						poses = poses.to(weight_dtype).to(accelerator.device, non_blocking=True)

						if calibration_save_path:
							torch.save((poses, fov_x), calibration_save_path)

				else:
					fov_x = args.fixed_fov
					assert args.predict_camera_motion is False, "Predict camera motion is not supported with fixed fov."

				if args.predict_camera_motion:
					'''
					The calibrated pose are camera-to-world, with z forward, x right, y up convention. (right-handed)
					We use z top, x forward, y right convention for camera poses. (also right-handed)
					'''
					convention_rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
					convention_inverse = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

					rolls, pitches, yaws = np.zeros(len(poses)), np.zeros(len(poses)), np.zeros(len(poses))
					#R1 = poses[0, :3, :3].cpu().numpy()
					R1 = poses[0, :3, :3].float().cpu().numpy() #修正

					for i in range(1, len(poses)):
						#R2 = poses[i, :3, :3].cpu().numpy()
						R2 = poses[i, :3, :3].float().cpu().numpy() #修正
						roll, pitch, yaw = rotation_matrix_to_euler(convention_inverse @ R2.T @ R1 @ convention_rotation, z_down=True) # rotation matrix are camera-to-world, cam1 --> cam2 is R2.T @ R1
						rolls[i] = -roll
						pitches[i] = pitch
						yaws[i] = yaw
					print(rolls, pitches, yaws)
				
				else:
					pitches, yaws, rolls = np.zeros(len(video)), np.zeros(len(video)), np.zeros(len(video))

			else:
				if args.fixed_rpy:
					pitches, yaws, rolls = np.zeros(args.num_frames), np.zeros(args.num_frames), np.zeros(args.num_frames)
				else:
					pitches, yaws, rolls = get_rpy(frame_rate, timesteps=args.num_frames)
				fov_x = random.uniform(args.fov_x_min, args.fov_x_max) # (1,)
				hw_ratio = 9 / 16

		# apply the yaw_start
		yaws = yaws + np.radians(args.yaw_start)
		if args.noisy_rpy > 0:
			noise_strength = args.noisy_rpy
			pitches += np.random.normal(0, np.radians(noise_strength), len(pitches))
			yaws += np.random.normal(0, np.radians(noise_strength), len(yaws))
			rolls += np.random.normal(0, np.radians(noise_strength), len(rolls))
			# save the noisy rpy
			np.save(out_file_path.replace('.mp4', '_rpy.npy'), np.stack([rolls, pitches, yaws], axis=1))

		sample_svd(args, accelerator, pipeline, weight_dtype, 
							fov_x = fov_x, hw_ratio = hw_ratio,
							roll = rolls, pitch = pitches, yaw = yaws,
							out_file_path=out_file_path,
							conditional_video=video,
							noise_aug_strength=args.noise_aug_strength,
							fps = frame_rate,
							decode_chunk_size=args.decode_chunk_size,
							num_inference_steps=args.num_inference_steps,
							width=args.width, height=args.height,
							guidance_scale=args.guidance_scale,
							inference_final_rotation=args.inference_final_rotation,
							blend_decoding_ratio=args.blend_decoding_ratio,
							extended_decoding=args.extended_decoding,
							noise_conditioning=args.noise_conditioning,
							rotation_during_inference=args.rotation_during_inference,
							equirectangular_input=args.equirectangular_input,
							post_rotation=args.post_rotation,
							replacement_sampling=args.replacement_sampling,
							narrow=args.narrow,
							width_narrow=args.width_narrow, height_narrow=args.height_narrow,
							)	