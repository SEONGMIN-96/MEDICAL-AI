import os
import sys
import numpy as np
import pandas as pd
import glob
import shutil
import cv2
import time

base_path = os.path.join('bin')

gastric_video_paths = glob.glob(os.path.join(base_path, 'data', 'video_gastroscopy', '*.mp4'))

print(gastric_video_paths)

cap = cv2.VideoCapture(gastric_video_paths[1])       

# 프레임 길이, 너비/높이, 초당 프레임 수 확인
video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
video_fps = cap.get(cv2.CAP_PROP_FPS)

print('프레임 길이: %d, 프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d' %(video_length, video_width, video_height, video_fps))
print('=='*50)

video_roi = [1198, 649, 532, 70]
playback_time = 3
max_count = int(playback_time * video_fps + 1)

count = 0
a = time.time()

while cap.isOpened():
    # 1sec = 60frame -> 10초 재생 
    if count == max_count:
        b = time.time()
        print("real time : %ssec" % (str(b - a)))
        print("%d초 재생 완!" % (playback_time))

        cap.release()
        sys.exit()

    ret, frame = cap.read()

    # 프레임이 읽히면 ret == True
    if not ret:
        print("프레임을 수신할 수 없습니다...")
        cap.release()

        sys.exit()
    elif ret:
        frame = frame[video_roi[3]:video_roi[1], video_roi[2]:video_roi[0]]
        
        # BGR 이미지를 HSV 이미지로 변환합니다.
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # HSV 이미지에서 각 채널을 분리합니다.
        h, s, v = cv2.split(hsv_image)
        
        # 분리된 각 채널을 표시하거나 다른 작업을 수행할 수 있습니다.
        # cv2.imshow('Hue (H) Channel', h)
        # cv2.imshow('Saturation (S) Channel', s)
        cv2.imshow('Value (V) Channel', v)
        cv2.waitKey(120)

    count += 1

cv2.destroyAllWindows()
    
    
