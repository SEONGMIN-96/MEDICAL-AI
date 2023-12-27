import numpy as np
import pandas as pd
import glob
import os
import cv2
import time
import math
import matplotlib.pyplot as plt


class OpticalFlow(object):
    def __init__(self, previous_frame : np.ndarray, current_frame : np.ndarray, size : int, flow_show : bool):
        self.previous_frame = previous_frame
        self.current_frame = current_frame
        self.size = size
        self.flow_show = flow_show
    
    def lucas_kanade(self):
        old_F = cv2.resize(self.previous_frame, (self.size, self.size))
        gray1 = cv2.cvtColor(old_F, cv2.COLOR_BGR2GRAY)
        new_F = cv2.resize(self.current_frame, (self.size, self.size))
        gray2 = cv2.cvtColor(new_F, cv2.COLOR_BGR2GRAY)

        # 코너점 찾는 함수, 그레이스케일 영상만 입력 가능
        pt1 = cv2.goodFeaturesToTrack(gray1, 5, 0.001, 0)
        
        # 찾은 코너점 정보를 옵티컬플로우 함수에 입력
        # src1, src2에서 움직임 정보를 찾아내고 pt1에 입력한 좌표가 어디로 이동했는지 파악
        pt2, status, err = cv2.calcOpticalFlowPyrLK(old_F, new_F, pt1, None)
        
        # 가중합으로 개체가 어느 정도 이동했는지 보기 위함
        dst = cv2.addWeighted(old_F, 0.5, new_F, 0.5, 0)
        
        dist_lst = []
        x_val_lst, y_val_lst = [], []
        
        # pt1과 pt2를 화면에 표시
        for i in range(pt2.shape[0]):
            if status[i,0] == 0: # status = 0인 것은 제외, 잘못 찾은 것을 의미
                continue
            
            a = int(pt1[i, 0][0] - pt2[i, 0][0])
            b = int(pt1[i, 0][1] - pt2[i, 0][1])
            
            x_val_lst.append(math.sqrt(a**2))
            y_val_lst.append(math.sqrt(b**2))
            
            dist = math.sqrt(a**2 + b**2)
            dist_lst.append(round(dist))
            
            if self.flow_show == True:
                # pt1과 pt2를 이어주는 선 그리기    
                cv2.circle(old_F, tuple(pt1[i, 0]), 4, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(old_F, tuple(pt2[i, 0]), 4, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.arrowedLine(old_F, tuple(pt1[i, 0]), tuple(pt2[i, 0]), (0, 255, 0), 2)
        
        # hist, bin = np.histogram(dist_lst, np.unique(dist_lst))
        # hist, bin = np.histogram(dist_lst, 10)
               
        if len(dist_lst) == 0:
            mean = 0
            x_mean = 0
            y_mean = 0
        else:
            mean = round(sum(dist_lst) / len(dist_lst))
            x_mean = round(sum(x_val_lst) / len(x_val_lst))
            y_mean = round(sum(y_val_lst) / len(y_val_lst))
        
        value = [x_mean, y_mean]
        
        # plt.hist(dist_lst)
        # plt.show()
        
        if mean == 0:
            mean = '0'
        elif 0 < mean <= 20:
            mean = '1'
        elif 20 < mean <= 40:
            mean = '2'
        elif 40 < mean <= 60:
            mean = '3'
        elif 60 < mean <= 80:
            mean = '4'
        elif 80 < mean <= 100:
            mean = '5'
        else:
            mean = '6'
        
        if int(mean) > 3:
            res = cv2.putText(old_F, 'warning', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 69, 255), 2, cv2.LINE_AA)
        else:
            res = cv2.putText(old_F, str(mean), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # cv2.imshow('dst', dst)
        # cv2.waitKey(0)
        
        return [res, value]
    
    def gunner_farneback(self):
        old_F = cv2.resize(self.previous_frame, (self.size, self.size))
        gray1 = cv2.cvtColor(old_F, cv2.COLOR_BGR2GRAY)
        new_F = cv2.resize(self.current_frame, (self.size, self.size))
        gray2 = cv2.cvtColor(new_F, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        
        mix_img = cv2.addWeighted(old_F, 0.5, new_F, 0.5, 0)
        
        h,w = mix_img.shape[:2]
        # idx_y,idx_x = np.mgrid[32/2:h:32,32/2:w:32].astype(np.int)
        idx_y,idx_x = np.mgrid[32/2:h:32,32/2:w:32].astype(int)
        indices =  np.stack((idx_x,idx_y), axis =-1).reshape(-1, 2)

        dist_lst = []
        x_val_lst, y_val_lst = [], []
        
        for x,y in indices:   # 인덱스 순회
            if self.flow_show == True:
                # 각 그리드 인덱스 위치에 점 그리기 ---③
                cv2.circle(mix_img, (x,y), 1, (0,255,0), -1)
                # 각 그리드 인덱스에 해당하는 플로우 결과 값 (이동 거리)  ---④
                # dx,dy = flow[y, x].astype(np.int)
                dx,dy = flow[y, x].astype(int)
                # 각 그리드 인덱스 위치에서 이동한 거리 만큼 선 그리기 ---⑤
                cv2.line(mix_img, (x,y), (x+dx, y+dy), (0,255, 0),2, cv2.LINE_AA )
            elif self.flow_show == False:
                # dx,dy = flow[y, x].astype(np.int)
                dx,dy = flow[y, x].astype(int)
            
            a = int(x - x+dx)
            b = int(y - y+dy)
            
            x_val_lst.append(math.sqrt(a**2))
            y_val_lst.append(math.sqrt(b**2))

            dist = math.sqrt(a**2 + b**2)
            
            dist_lst.append(round(dist))
        
        x_mean = round(sum(x_val_lst) / len(x_val_lst))
        y_mean = round(sum(y_val_lst) / len(y_val_lst))
        mean = str(round(sum(dist_lst) / len(dist_lst)))
        
        value = [x_mean, y_mean]    
        
        # 일정 임계값 이상으로 속도가 측정될 경우, warning 텍스트를 넣어줌
        # if int(mean) >= 13:
        #     res = cv2.putText(res, 'warning', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 69, 255), 2, cv2.LINE_AA) 
        # else:
        #     res = cv2.putText(res, mean, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 
        
        return [mix_img, value, mean]
    
    def rlof(self):
        old_F = cv2.resize(self.previous_frame, (self.size, self.size))
        gray1 = cv2.cvtColor(old_F, cv2.COLOR_BGR2GRAY)
        new_F = cv2.resize(self.current_frame, (self.size, self.size))
        gray2 = cv2.cvtColor(new_F, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.optflow.calcOpticalFlowDenseRLOF(old_F, new_F, None, *[])
        
        res = cv2.addWeighted(old_F, 0.5, new_F, 0.5, 0)
        
        h,w = res.shape[:2]
        # idx_y,idx_x = np.mgrid[32/2:h:32,32/2:w:32].astype(np.int)
        idx_y,idx_x = np.mgrid[32/2:h:32,32/2:w:32].astype(int)
        indices =  np.stack((idx_x,idx_y), axis=-1).reshape(-1,2)

        dist_lst = []
        x_val_lst, y_val_lst = [], []
        
        for x,y in indices:   # 인덱스 순회
            if self.flow_show == True:
                # 각 그리드 인덱스 위치에 점 그리기 ---③
                cv2.circle(res, (x,y), 1, (0,255,0), -1)
                # 각 그리드 인덱스에 해당하는 플로우 결과 값 (이동 거리)  ---④
                # dx,dy = flow[y, x].astype(np.int)
                dx,dy = flow[y, x].astype(int)
                # 각 그리드 인덱스 위치에서 이동한 거리 만큼 선 그리기 ---⑤
                cv2.line(res, (x,y), (x+dx, y+dy), (0,255, 0),2, cv2.LINE_AA )
            elif self.flow_show == False:
                # dx,dy = flow[y, x].astype(np.int)
                dx,dy = flow[y, x].astype(int)
            
            a = int(x - x+dx)
            b = int(y - y+dy)
            
            x_val_lst.append(math.sqrt(a**2))
            y_val_lst.append(math.sqrt(b**2))

            dist = math.sqrt(a**2 + b**2)
            
            dist_lst.append(round(dist))
            
        x_mean = round(sum(x_val_lst) / len(x_val_lst))
        y_mean = round(sum(y_val_lst) / len(y_val_lst))    
        mean = str(round(sum(dist_lst) / len(dist_lst)))
        
        value = [x_mean, y_mean]    
        
        if int(mean) >= 13:
            res = cv2.putText(res, 'warning', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 69, 255), 2, cv2.LINE_AA) 
        else:
            res = cv2.putText(res, mean, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 
        
        return [res, value]
    
    def frame_info(self):
        # old_F = cv2.resize(self.previous_frame, (self.size, self.size))
        # gray1 = cv2.cvtColor(old_F, cv2.COLOR_BGR2GRAY)
        # new_F = cv2.resize(self.current_frame, (self.size, self.size))
        # gray2 = cv2.cvtColor(new_F, cv2.COLOR_BGR2GRAY)
        old_F = self.previous_frame
        new_F = self.current_frame
        
        
        return [old_F]
    
    def draw_result(self, frame, mean_avg, res):
        # 기본으로 녹색 원이 왼쪽 상단에 표시되도록 설정
        # 일정 임계값 이상으로 속도가 측정될 경우, 녹색 원이 붉은 원으로 변환
        # 측정한 상대 속도를 프레임에 입력합니다.
        
        orange_color = (3,97,255) 
        green_color = (0,255,0)
        red_color = (0,0,255)
        white_color = (255,255,255)
        
        # text = str(mean_avg)
        text = str(res)
        
        if int(text) >= 13:
            frame = cv2.circle(frame, (20,20), 10, red_color, -1)
            # frame = cv2.putText(frame, 'warning', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, red_color, 2, cv2.LINE_AA) 
            frame = cv2.putText(frame, text, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, red_color, 2, cv2.LINE_AA) 
        # elif 9 < int(mean_avg) < 13: 
            # frame = cv2.circle(frame, (20,20), 10, orange_color, -1)
        else:
            frame = cv2.circle(frame, (20,20), 10, green_color, -1)
            frame = cv2.putText(frame, text, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, white_color, 2, cv2.LINE_AA) 

        # frame = cv2.putText(frame, text, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 
        
        return frame
    
    
def main():
    pass
    
if __name__ == '__main__':
    main()
