import cv2
import os
import glob
import numpy as np

def merge_videos_with_crossfade(clips_folder, output_filename, overlap_seconds=1.0):
    """
    分割生成された動画クリップを読み込み、オーバーラップ部分をクロスフェードして結合します。
    ※クリップ自体の長さ（2秒や5秒など）は自動取得するため、引数での指定は不要です。
    
    Args:
        clips_folder (str): 生成された動画クリップが入っているフォルダ
        output_filename (str): 完成した動画の保存名
        overlap_seconds (float): 分割時に設定した「のりしろ」の秒数
    """
    
    # クリップを取得してファイル名順にソート
    video_files = sorted(glob.glob(os.path.join(clips_folder, "*.mp4")))
    
    if not video_files:
        print(f"エラー: フォルダ {clips_folder} に動画が見つかりません。")
        return

    # 最初の動画を開いて仕様を取得
    cap = cv2.VideoCapture(video_files[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # 出力設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Macや一部環境でエラーが出る場合は 'avc1' に変更
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    print(f"結合開始: {len(video_files)}個のクリップ")
    print(f"・のりしろ設定: {overlap_seconds}秒")
    print(f"・出力ファイル: {output_filename} ({width}x{height}, {fps:.2f}fps)")

    # のりしろのフレーム数計算
    overlap_frames = int(overlap_seconds * fps)
    
    prev_cap = None
    
    for i, filename in enumerate(video_files):
        print(f"処理中 ({i+1}/{len(video_files)}): {os.path.basename(filename)}")
        curr_cap = cv2.VideoCapture(filename)
        
        # 【重要】動画の総フレーム数を自動取得
        # これにより、分割秒数が2秒でも5秒でも自動で対応できます
        curr_total_frames = int(curr_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # === 1. 最初のクリップの場合 ===
        if prev_cap is None:
            # 「全体の長さ」から「末尾ののりしろ」を引いた分だけ書く
            frames_to_write = curr_total_frames - overlap_frames
            
            # クリップが1つしかない場合は全部書く
            if i == len(video_files) - 1:
                frames_to_write = curr_total_frames

            count = 0
            while count < frames_to_write:
                ret, frame = curr_cap.read()
                if not ret: break
                out.write(frame)
                count += 1
            
            prev_cap = curr_cap
            continue

        # === 2. 2つ目以降のクリップ ===
        
        # A. クロスフェード部分（のりしろ）の処理
        for f in range(overlap_frames):
            ret_prev, frame_prev = prev_cap.read()
            ret_curr, frame_curr = curr_cap.read()
            
            if not ret_prev or not ret_curr:
                break
                
            # alpha: 0.0 -> 1.0 (徐々に今のクリップに切り替える)
            alpha = f / float(overlap_frames)
            blended = cv2.addWeighted(frame_curr, alpha, frame_prev, 1.0 - alpha, 0)
            out.write(blended)

        prev_cap.release() # 前のクリップは完了

        # B. 残りの部分の書き出し
        # 「現在の長さ」-「冒頭のりしろ」= 残り
        frames_remaining = curr_total_frames - overlap_frames 
        
        if i < len(video_files) - 1:
            # 次のクリップがあるなら、末尾ののりしろ分は書かずに残す
            frames_to_write = frames_remaining - overlap_frames
        else:
            # 最後のクリップなら最後まで書く
            frames_to_write = frames_remaining
            
        count = 0
        while count < frames_to_write:
            ret, frame = curr_cap.read()
            if not ret: break
            out.write(frame)
            count += 1
            
        prev_cap = curr_cap

    if prev_cap is not None:
        prev_cap.release()
        
    out.release()
    print("結合完了！")

if __name__ == "__main__":
    # ==========================================
    # ▼ 設定エリア ▼
    # ==========================================
    
    # 1. 生成されたクリップが入っているフォルダ名
    CLIPS_FOLDER = "results_movie2_clips" 
    
    # 2. 完成した動画のファイル名
    OUTPUT_FILE = "movie2_final_merged.mp4"
    
    # 3. 分割時に設定した「のりしろ（オーバーラップ）」の秒数
    # ※ split_video.py の overlap_seconds と同じ値にしてください
    OVERLAP_SEC = 1.0
    
    # ※ 分割秒数（2秒や5秒など）の設定は不要です（自動判定されます）
    # ==========================================

    merge_videos_with_crossfade(CLIPS_FOLDER, OUTPUT_FILE, overlap_seconds=OVERLAP_SEC)