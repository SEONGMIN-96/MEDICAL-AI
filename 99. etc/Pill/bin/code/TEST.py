import cv2
import os
import glob
import numpy as np

# # paths = glob.glob(os.path.join('.', 'data', 'pill_list.v1.0.4', '*'))
# paths = glob.glob(os.path.join('bin', 'data', 'pill_list.v1.0.4', '*'))

# src = cv2.imread(paths[0])

# width = 512  # 원하는 너비
# height = 320  # 원하는 높이
# resized_image = cv2.resize(src, (width, height))  # 크기 조정

# fgbg = cv2.createBackgroundSubtractorMOG2()

# # 배경 제거
# fgmask = fgbg.apply(resized_image)

# # 이진화
# thresh = 150
# ret, threshed = cv2.threshold(fgmask, thresh, 255, cv2.THRESH_BINARY_INV)

# # 노이즈 제거
# kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel, iterations=2)

# # 물체 탐지
# contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# for contour in contours:
#     if cv2.contourArea(contour) > 100:  # 물체의 크기 조건을 설정
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# if x + w > width * 0.6 :
#     print("over center")

# cv2.imshow('Original Frame', resized_image)
# cv2.imshow('test', fgmask)

# cv2.waitKey()
# cv2.destroyAllWindows()

import asyncio
import datetime
import time

switch = True
switch_sleep_time = 5

print("switch:", switch)

while True:
    if switch == False:
        # 현재 시간 가져오기
        new_current_time = datetime.datetime.now()
        # 원하는 형식으로 시간을 포맷팅
        new_formatted_time = new_current_time.strftime("%Y-%m-%d %H:%M:%S")
        new_formatted_sec = int(new_current_time.strftime("%S"))
        
        if formatted_sec >= 60-switch_sleep_time:
            switch_time = formatted_sec + switch_sleep_time
            
            if switch_time >= 60:
                switch_time = switch_time - 60
            else:
                pass
            
        elif formatted_sec < 60-switch_sleep_time:
            switch_time = formatted_sec + switch_sleep_time
        
        if new_formatted_sec == switch_time:
            switch = True
        
    elif switch == True:
        # 현재 시간 가져오기
        current_time = datetime.datetime.now()
        # 원하는 형식으로 시간을 포맷팅
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_sec = int(current_time.strftime("%S"))
        # 현재 시간 출력
        print("현재 시간:", formatted_time)

        switch = False
    
    if 0xFF == ord('q'):
        break