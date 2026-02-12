#!/bin/bash
set -euo pipefail

GPU_ID=$1
UNET_PATH=$2
VIDEO_PATH=$3
RESULT_FOLDER=$4
GUIDANCE_SCALE=${5:-3}
NUM_INFERENCE_STEPS=${6:-25}
FRAME_RATE=${7:-60}
NUM_FRAMES=${8:-90}
DECODE_CHUNK_SIZE=${9:-40}
NUM_FRAMES_BATCH=${10:-35}
BLEND_FRAMES=${11:-8}

export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "[inference.sh] GPU=$CUDA_VISIBLE_DEVICES"
echo "[inference.sh] video=$VIDEO_PATH"
echo "[inference.sh] out=$RESULT_FOLDER"
echo "[inference.sh] guidance=$GUIDANCE_SCALE steps=$NUM_INFERENCE_STEPS fps=$FRAME_RATE num_frames=$NUM_FRAMES"
echo "[inference.sh] decode_chunk=$DECODE_CHUNK_SIZE num_frames_batch=$NUM_FRAMES_BATCH blend_frames=$BLEND_FRAMES"

# HuggingFaceは永続キャッシュ（NFS）
export HF_HOME=/nfs1/s3f102500025/cache/hf_shared
mkdir -p "$HF_HOME"

# キャッシュはノードローカル推奨（/dev/shm が大きいので安全）
CACHE_ROOT=/dev/shm/$USER/argus_cache/${HOSTNAME}/$$
mkdir -p "$CACHE_ROOT"
trap 'rm -rf "$CACHE_ROOT"' EXIT INT TERM

export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda_compute"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"

# allocator設定
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

mkdir -p "$RESULT_FOLDER"

accelerate launch --num_processes 1 --mixed_precision bf16 inference.py \
  --val_base_folder "$VIDEO_PATH" \
  --val_save_folder "$RESULT_FOLDER" \
  --unet_path "$UNET_PATH" \
  --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid \
  --decode_chunk_size "$DECODE_CHUNK_SIZE" \
  --noise_aug_strength 0.01 \
  --motion_bucket_id 50 \
  --guidance_scale "$GUIDANCE_SCALE" \
  --frame_rate "$FRAME_RATE" \
  --height 512 --width 1024 \
  --fixed_start_frame \
  --num_frames "$NUM_FRAMES" \
  --num_inference_steps "$NUM_INFERENCE_STEPS" \
  --inference_final_rotation 0 \
  --rotation_during_inference \
  --extended_decoding \
  --blend_decoding_ratio 16 \
  --blend_frames "$BLEND_FRAMES" \
  --seed 42 \
  --num_frames_batch "$NUM_FRAMES_BATCH" \
  --fixed_fov 65
