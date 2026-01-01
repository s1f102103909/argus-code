import cv2

# 動画パスを指定（実際のファイル名に書き換えてください）
video_path = 'movie2_clips/clip_001_71_165.mp4' 

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = count / fps

print(f"FPS: {fps}")
print(f"総フレーム数: {count}")
print(f"動画の長さ: {duration:.2f}秒")

cap.release()