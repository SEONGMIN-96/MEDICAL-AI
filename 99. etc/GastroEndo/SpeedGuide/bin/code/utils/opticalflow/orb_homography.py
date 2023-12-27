import numpy as np
import pandas as pd
import glob
import os
import cv2
import math


class OrbHomography(object):
    def __init__(self, previous_frame : np.ndarray, current_frame : np.ndarray):
        self.previous_frame = previous_frame
        self.current_frame = current_frame
    
    def measure_distance(self) -> float:
        # 앞프레임, 뒷프레임 순서맞춤
        # previous_frame, current_frame = self.specify_frames()

        # 프레임당 orb_homography 진행
        src1 = self.previous_frame
        gray1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
        src2 = self.current_frame
        gray2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(
            nfeatures=40000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20,
        )

        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        min_dist, max_dist = matches[0].distance, matches[-1].distance
        ratio = 0.05
        good_thresh = (max_dist - min_dist) * ratio + min_dist

        good_matches = [m for m in matches if m.distance < good_thresh]
        good_matches_len = len(good_matches)
        
        dist_total = 0
        
        for i in range(good_matches_len):
            f_pt = kp1[i].pt
            s_pt = kp2[i].pt

            a = int(f_pt[0] - s_pt[0])
            b = int(f_pt[1] - s_pt[1])

            dist = math.sqrt(a**2 + b**2)
            
            # print('dist :', dist)
            
            dist_total += round(dist, 1)
        
        if dist_total == 0:
            # 매칭된 특징점 그리기
            res = cv2.drawMatches(src1, kp1, src2, kp2, matches[:15], None, flags = 2)
        else:
            mean = round(dist_total / good_matches_len)
            
            # 매칭된 특징점 그리기
            res = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None, flags = 2)
            
            if mean == 0:
                text = '0'
            elif 0 < mean <= 20:
                text = '1'
            elif 20 < mean <= 40:
                text = '2'
            elif 40 < mean <= 60:
                text = '3'
            elif 60 < mean <= 80:
                text = '4'
            elif 80 < mean <= 100:
                text = '5'
            else:
                text = '<5'
            
            res = cv2.putText(res, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        print(res.shape)
        
        return res

def main():
    pass
    
if __name__ == '__main__':
    main()
