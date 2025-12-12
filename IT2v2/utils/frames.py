import cv2
import os
import re

video_path = "D:/Downloads/video.mp4"
output_dir = "frames"

os.makedirs(output_dir, exist_ok=True)

pattern = re.compile(r"frame_(\d+)\.png")
existing_frames = [
    f for f in os.listdir(output_dir)
    if pattern.match(f)
]

if existing_frames:
    max_index = max(int(pattern.match(f).group(1)) for f in existing_frames)
    frame_index = max_index + 1
else:
    frame_index = 0

cap = cv2.VideoCapture(video_path)

k = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    k += 1
    if k%15 != 0: continue

    cv2.imwrite(f"{output_dir}/frame_{frame_index}.png", frame)
    frame_index += 1

cap.release()