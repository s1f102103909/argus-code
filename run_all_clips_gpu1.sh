#!/bin/bash
set -euo pipefail

GPU_ID=1
# 分割したクリップが入っているフォルダ
CLIP_FOLDER="movie3_clips"
# 結果保存先
RESULT_FOLDER="results_movie3_clips"
UNET_PATH="checkpoints"
GUIDANCE_SCALE=4.0
NUM_INFERENCE_STEPS=100
FRAME_RATE=24
NUM_FRAMES=96
DECODE_CHUNK_SIZE=40
NUM_FRAMES_BATCH=40
BLEND_FRAMES=15

mkdir -p $RESULT_FOLDER

for video in "$CLIP_FOLDER"/*.mp4; do
    echo "In $GPU_ID Generating for $video..."
    filename=$(basename "$video")
    
    # 短い動画用の高画質設定で実行
    bash scripts/test/inference.sh \
      "$GPU_ID" \
      "$UNET_PATH" \
      "$video" \
      "$RESULT_FOLDER" \
      "$GUIDANCE_SCALE" \
      "$NUM_INFERENCE_STEPS" \
      "$FRAME_RATE" \
      "$NUM_FRAMES" \
      "$DECODE_CHUNK_SIZE" \
      "$NUM_FRAMES_BATCH" \
      "$BLEND_FRAMES"
done