import sys
import subprocess
import re
import os

def detect_crop(input_file):
    """
    安全対策済みクロップ検出:
    1. 動画の開始30秒後をサンプルにする（フェードイン対策）
    2. 黒色の判定閾値を厳しくする（暗いシーンの誤削除対策）
    """
    cmd = [
        "ffmpeg",
        "-ss", "30",           # 【対策1】開始30秒後から解析開始（イントロの暗転を回避）
        "-i", input_file,
        "-t", "5",             # 5秒間だけ解析
        "-vf", "cropdetect=limit=10:round=2:reset=0", # 【対策2】limit=10 (真っ黒以外は無視), round=2 (偶数サイズ確保)
        "-f", "null",
        "-"
    ]
    
    # 実行
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    
    # ログから crop=w:h:x:y を抽出
    matches = re.findall(r"crop=(\d+:\d+:\d+:\d+)", result.stderr)
    
    if not matches:
        # 30秒後が存在しない短い動画の場合、冒頭から再トライ
        print("動画が短い、または検出できなかったため、冒頭から再スキャンします...")
        cmd[2] = "0" # -ss を 0 に戻す
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        matches = re.findall(r"crop=(\d+:\d+:\d+:\d+)", result.stderr)
        
        if not matches:
            return None
    
    # 最も多く検出された値を採用（ノイズ対策）
    most_common = max(set(matches), key=matches.count)
    return most_common

def crop_video(input_file, output_file):
    print(f"黒帯を解析中 (Safety Mode): {input_file} ...")
    crop_params = detect_crop(input_file)
    
    if not crop_params:
        print("黒帯は検出されませんでした。コピーのみ行います。")
        subprocess.run(["cp", input_file, output_file])
        return

    print(f"検出されたクロップ範囲: {crop_params}")
    
    # ここで元の解像度と比較して、極端に小さくなっていないかチェックも可能ですが
    # 今回はSVD用なのでそのまま進めます。

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-vf", f"crop={crop_params}",
        "-c:a", "copy",
        output_file
    ]
    
    subprocess.run(cmd)
    print(f"完了: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_crop.py <input_video.mp4>")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = os.path.join(
        os.path.dirname(input_video),
        "cropped_" + os.path.basename(input_video)
    )
    
    crop_video(input_video, output_video)