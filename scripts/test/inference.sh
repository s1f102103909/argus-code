#!/bin/bash

# Set cache directories to the current working directory
export HF_HOME=$(pwd)/hf_cache
export TRITON_CACHE_DIR=$(pwd)/triton_cache
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
# 互換として併用するなら:
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
mkdir -p $HF_HOME $TRITON_CACHE_DIR # Ensure directories exist

unet_path=$1
val_folder_or_video_path=$2
val_save_folder=$3
guidance_scale=${4:-3}
num_inference_steps=${5:-25}

video_list=()
# if val_folder_or_video_path is a folder, then process all videos in the folder
if [ -d "$val_folder_or_video_path" ]; then
    for video_path in "$val_folder_or_video_path"/*; do
        video_list+=("$video_path")
    done
else
    video_list+=("$val_folder_or_video_path")
fi

echo "Processing ${#video_list[@]} videos"
accelerate launch --num_processes 2 --mixed_precision bf16 inference.py \
    --val_base_folder ${video_list} \
    --val_save_folder ${val_save_folder} \
    --unet_path $unet_path \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid \
    --decode_chunk_size 10 \
    --noise_aug_strength 0.02 \
    --guidance_scale $guidance_scale \
    --height 512 --width 1024 \
    --fixed_start_frame \
    --num_inference_steps $num_inference_steps \
    --inference_final_rotation 0 \
    --rotation_during_inference \
    --extended_decoding \
    --predict_camera_motion \
    --blend_decoding_ratio 16 \
    --frame_interval 1

    #--frame_rate 5 \
    #--num_frames 25 \
    #--full_sampling

    