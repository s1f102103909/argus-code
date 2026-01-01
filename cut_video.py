import cv2
import os

def cut_video(input_path, output_path, start_time_sec, duration_sec):
    """
    動画を指定した時間から指定した秒数だけ切り出します。
    FPSと解像度は元の動画のものを維持します。

    Args:
        input_path (str): 入力動画のパス
        output_path (str): 出力動画のパス
        start_time_sec (float): 切り出し開始時間（秒）
        duration_sec (float): 切り出す長さ（秒）
    """
    
    # 入力確認
    if not os.path.exists(input_path):
        print(f"エラー: ファイルが見つかりません: {input_path}")
        return

    # 動画を読み込む
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("エラー: 動画を開けませんでした。")
        return

    # 元動画のプロパティを取得（FPS、解像度など）
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_len_sec = total_frames / fps

    print(f"入力: {input_path}")
    print(f"情報: {width}x{height}, {fps:.2f}fps, 長さ: {video_len_sec:.2f}秒")

    # 開始フレームと終了フレームを計算
    start_frame = int(start_time_sec * fps)
    end_frame = start_frame + int(duration_sec * fps)

    # 範囲チェック
    if start_frame >= total_frames:
        print("エラー: 開始時間が動画の長さを超えています。")
        cap.release()
        return
    
    if end_frame > total_frames:
        print(f"警告: 指定された終了時間が動画の長さを超えています。動画の最後まで切り出します。")
        end_frame = total_frames

    print(f"処理: {start_time_sec}秒目から {duration_sec}秒間 ({start_frame}F 〜 {end_frame}F) を切り出します...")

    # 出力設定
    # 汎用性の高い H.264 (avc1) を使用
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 開始位置へシーク
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        current_frame += 1

    # 後始末
    cap.release()
    out.release()
    print(f"完了: {output_path} に保存しました。")

if __name__ == "__main__":
    # ==========================================
    # ▼ 設定エリア ▼
    # ==========================================
    
    INPUT_FILE = "videoplayback.mp4"   # 切り取りたい元動画
    OUTPUT_FILE = "kayaking.mp4"     # 保存するファイル名
    
    START_TIME = 10.0     # 開始時間 (秒)
    DURATION   = 60.00  # 切り出す長さ (秒)

    # ==========================================

    cut_video(INPUT_FILE, OUTPUT_FILE, START_TIME, DURATION)