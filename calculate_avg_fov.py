import torch
import math
import glob
import os
from src.calibrate_cameras import get_camera_params_from_frames
from mast3r.model import AsymmetricMASt3R

# 焦点距離からFOV(x)に変換する関数 (Argus内部ロジック準拠)
def focal2fov(focal, width):
    return 2 * math.atan(width / (2 * focal)) * 180 / math.pi

def calculate_stable_fov(video_folder, img_size=512):
    # MASt3Rモデルのロード
    model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to("cuda")
    
    video_files = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))
    focal_lengths = []

    print(f"解析開始: {len(video_files)}個のクリップをスキャン中...")

    for video_path in video_files:
        # 動画からフレームを読み込む（簡易的な読み込み処理が必要）
        # ここでは get_camera_params_from_frames が受け取れるテンソル形式を想定
        # video_tensor = load_video_to_tensor(video_path) 
        
        # 解析実行 (shared_intrinsics=True で平均的な焦点距離を得る)
        # poses, intrinsics, width_resized = get_camera_params_from_frames(video_tensor, model=model, img_size=img_size)
        
        # ログにある数値を集計する場合（今回は手動ログから想定）
        # focal = intrinsics[0, 0, 0].item()
        # focal_lengths.append(focal)
        pass

    # 例: ユーザー様が提示した数値を使用
    sample_focals = [631.99, 781.77, 1068.39] 
    avg_focal = sum(sample_focals) / len(sample_focals)
    
    # 解析用サイズ 512 に対するFOVを計算
    stable_fov = focal2fov(avg_focal, 512)
    
    print(f"--- 解析結果 ---")
    print(f"平均焦点距離: {avg_focal:.2f}")
    print(f"推奨 fixed_fov: {stable_fov:.2f}")
    
    return stable_fov

# 使用例
final_fixed_fov = calculate_stable_fov("./movie2_clips/")