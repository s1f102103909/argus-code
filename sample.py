import cv2
import os

# 動画パスを指定（実際のファイル名に書き換えてください）
video_path = 'video/movie3.mp4' 

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = count / fps

print(f"動画パス: {video_path}")
print(f"FPS: {fps}")
print(f"総フレーム数: {count}")
print(f"動画の長さ: {duration:.2f}秒")

cap.release()


DIR = './game1_clips'

print(sum(os.path.isfile(os.path.join(DIR, name)) for name in os.listdir(DIR)))
