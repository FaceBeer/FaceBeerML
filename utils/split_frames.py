import cv2
from pathlib import Path

video = cv2.VideoCapture("../videos/kylee.webm")

duration = video.get(cv2.CAP_PROP_POS_MSEC)
frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
fps = video.get(cv2.CAP_PROP_FPS)

print("duration:", duration)
print("frame_count:", frame_count)
print("fps:", fps)

success, image = video.read()
count = 0
while success:
    cv2.imwrite(f"../data/kylee/{str(count).zfill(5)}.jpg", image)  # save frame as JPEG file
    success, image = video.read()
    count += 1
