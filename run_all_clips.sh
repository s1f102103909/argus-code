#!/bin/bash
# 分割したクリップが入っているフォルダ
CLIP_FOLDER="movie2_clips"
# 結果保存先
RESULT_FOLDER="results_movie2_clips"

mkdir -p $RESULT_FOLDER

for video in "$CLIP_FOLDER"/*.mp4; do
    echo "Generating for $video..."
    filename=$(basename "$video")
    
    # 短い動画用の高画質設定で実行
    bash scripts/test/inference.sh \
      checkpoints \
      "$video" \
      "$RESULT_FOLDER" \
      2.5 \
      40
done