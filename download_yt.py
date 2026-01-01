import yt_dlp
import os
from yt_dlp.utils import download_range_func

def download_youtube_segment(url, output_filename, start_time_sec, end_time_sec):
    """
    YouTube動画の指定範囲をダウンロードします。
    FPSは変換せず、元動画のまま保持します。
    解像度は1080pを優先します。
    
    Args:
        url (str): YouTubeのURL
        output_filename (str): 保存するファイル名
        start_time_sec (float): 開始時間
        end_time_sec (float): 終了時間
    """
    
    print(f"ダウンロード開始: {url}")
    print(f"範囲: {start_time_sec}秒 〜 {end_time_sec}秒")
    print("解像度: 1080p優先 (FPSは元動画のまま)")

    ydl_opts = {
        # 1080pのmp4があればそれを、なければ最高画質を取得
        'format': 'bestvideo[height=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height=1080]+bestaudio/best[height=1080]',
        
        'outtmpl': output_filename,
        'download_ranges': download_range_func(None, [(start_time_sec, end_time_sec)]),
        'force_keyframes_at_cuts': True,
        
        # 形式をmp4に整える（FPS変換は行わない）
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        
        'overwrites': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"完了しました: {output_filename}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    # ==========================================
    # ▼ 設定エリア ▼
    # ==========================================
    
    URL = "https://www.youtube.com/watch?v=D4JmMBC28x8" 
    OUTPUT_NAME = "movie3.mp4"
    
    START = 60
    END   = 120
    
    # ==========================================

    download_youtube_segment(URL, OUTPUT_NAME, START, END)