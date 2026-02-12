import subprocess
from pathlib import Path

def convert_mp4_to_30fps(input_mp4: str, output_mp4: str, crf: int = 18, preset: str = "medium"):
    """
    MP4動画を30fps(CFR)に変換する。
    - 映像: H.264(libx264)で再エンコード
    - fps: 30 に固定 (CFR)
    - 音声: そのままコピー (問題が出る場合は aac に変更)
    """
    in_path = Path(input_mp4)
    out_path = Path(output_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",                        # 上書き
        "-i", str(in_path),
        "-vf", "fps=30",             # フレームレートを30に（間引き）
        "-fps_mode", "cfr",          # CFRに固定（ffmpeg新しめ）
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",              # 音声コピー（ダメなら aac に）
        str(out_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        # ffmpegのエラーログを出して原因が分かるようにする
        raise RuntimeError(f"ffmpeg failed:\n{e.stderr}") from e


if __name__ == "__main__":
    convert_mp4_to_30fps("video/game2.mp4", "game2_30fps.mp4")
    print("done")