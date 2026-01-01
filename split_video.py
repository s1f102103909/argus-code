import cv2
import os
import math

def split_video(input_path, output_folder, segment_seconds=3.0, overlap_seconds=1.0):
    """
    動画を指定した秒数ごとに分割し、のりしろ（オーバーラップ）を含めて保存します。
    
    Args:
        input_path (str): 入力動画のパス
        output_folder (str): 出力先フォルダ
        segment_seconds (int): 1つのクリップのメインの長さ（秒）
        overlap_seconds (int): 前後のダブらせる長さ（秒）
    """
    
    # 動画を読み込む
    if not os.path.exists(input_path):
        print(f"エラー: ファイルが見つかりません: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("エラー: 動画を開けませんでした。")
        return

    # 動画情報を取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 出力フォルダ作成
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"入力動画: {input_path}")
    print(f"FPS: {fps:.2f}, 総フレーム数: {total_frames}, 長さ: {total_frames/fps:.2f}秒")
    print(f"設定: {segment_seconds}秒刻み + {overlap_seconds}秒オーバーラップ")

    # フレーム単位での長さを計算
    step_frames = int(segment_seconds * fps)      # 5秒分のフレーム数 (進む幅)
    overlap_frames = int(overlap_seconds * fps)   # 1秒分のフレーム数 (のりしろ)
    
    # 分割処理ループ
    # 開始フレーム(start)を 0, 150, 300... と step_frames ずつずらしていく
    clip_idx = 0
    for start_frame in range(0, total_frames, step_frames):
        # 終了フレーム計算 (5秒先 + 1秒のりしろ)
        # ※最後のクリップなどで動画の長さを超えないように min をとる
        end_frame = min(start_frame + step_frames + overlap_frames, total_frames)
        
        # もし残りが短すぎてのりしろ分しかない場合はスキップ（あるいは結合）
        if end_frame - start_frame <= overlap_frames:
            break

        # 出力ファイル名
        output_filename = f"clip_{clip_idx:03d}_{start_frame}_{end_frame}.mp4"
        output_path = os.path.join(output_folder, output_filename)
        
        # 書き出し設定 (mp4v)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 指定範囲のフレームを読み込んで書き出す
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_pos = start_frame
        
        print(f"クリップ {clip_idx}: フレーム {start_frame} 〜 {end_frame} を書き出し中...")
        
        while current_pos < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_pos += 1
            
        out.release()
        clip_idx += 1

    cap.release()
    print("分割完了！")

if __name__ == "__main__":
    # ======= ここを設定してください =======
    INPUT_VIDEO = "video/movie2.mp4"       # 分割したい動画ファイル名
    OUTPUT_DIR = "movie2_clips"      # 保存先フォルダ名
    # ====================================

    split_video(INPUT_VIDEO, OUTPUT_DIR, segment_seconds=3.0, overlap_seconds=1.0)