#!/bin/bash

echo "Setting node-local cache dirs (avoid NFS)..."

export HF_HOME=/nfs1/s3f102500025/cache/hf_shared
mkdir -p "$HF_HOME"

CACHE_ROOT=/tmp/$USER/argus_cache/${HOSTNAME}/$$
mkdir -p "$CACHE_ROOT"
trap 'rm -rf "$CACHE_ROOT"' EXIT INT TERM

export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda_compute"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"


# allocator設定はそのままでOK
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

echo "HF_HOME=$HF_HOME"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
echo "CUDA_CACHE_PATH=$CUDA_CACHE_PATH"


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
accelerate launch --num_processes 1 --mixed_precision bf16 inference.py \
    --val_base_folder ${video_list} \
    --val_save_folder ${val_save_folder} \
    --unet_path $unet_path \
    --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid \
    --decode_chunk_size 40 \
    --noise_aug_strength 0.01 \
    --motion_bucket_id 50 \
    --guidance_scale $guidance_scale \
    --frame_rate 60 \
    --height 512 --width 1024 \
    --fixed_start_frame \
    --num_frames 90 \
    --num_inference_steps $num_inference_steps \
    --inference_final_rotation 0 \
    --rotation_during_inference \
    --extended_decoding \
    --blend_decoding_ratio 16 \
    --blend_frames 8 \
    --seed 42 \
    --num_frames_batch 35 \
    --fixed_fov 65
    #--dense_calibration \
    #--calibration_img_size 1024 \
    #--predict_camera_motion \

    