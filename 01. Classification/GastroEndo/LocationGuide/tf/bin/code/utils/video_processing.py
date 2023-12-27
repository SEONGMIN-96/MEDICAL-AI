from collections import Counter

import os
import numpy as np
import glob
import cv2
import time
import sys


#----------------------------------------------------------------------------

class VideoProcessing():
    def __init__(self) -> None:
        ...
    
    def video_load(self, path: str):
        """
            
        Args:
            ...

        Return:
            ...
        """
        video_path = glob.glob(os.path.join(path, '*.mp4'))
        
        cap = cv2.VideoCapture(video_path[0])   

        # 프레임 길이, 너비/높이, 초당 프레임 수 확인
        length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        playback_time = length // fps
        playback_time_M = int(playback_time // 60)
        playback_time_S = int(playback_time - (playback_time_M * 60))
        
        print('프레임 길이: %d, 프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d' %(length, width, height, fps))
        print('비디어 재생시간: %s:%s' %(str(playback_time_M).zfill(2), str(playback_time_S).zfill(2)))
        print('=='*50)
        
        return cap, {'length':length, 'width': width, 'height': height, 'fps': fps, 
                     'playback_time': playback_time, 'playback_time_M': playback_time_M, 'playback_time_S': playback_time_S}
    
    def video_load_x(self, path: str):
        """
            
        Args:
            ...

        Return:
            ...
        """
        
        cap = cv2.VideoCapture(path)   

        # 프레임 길이, 너비/높이, 초당 프레임 수 확인
        length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        playback_time = length // fps
        playback_time_M = int(playback_time // 60)
        playback_time_S = int(playback_time - (playback_time_M * 60))
        
        print('프레임 길이: %d, 프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d' %(length, width, height, fps))
        print('비디어 재생시간: %s:%s' %(str(playback_time_M).zfill(2), str(playback_time_S).zfill(2)))
        print('=='*50)
        
        return cap, {'length':length, 'width': width, 'height': height, 'fps': fps, 
                     'playback_time': playback_time, 'playback_time_M': playback_time_M, 'playback_time_S': playback_time_S}
    
    def crop_frame(self, frame: None, coord: list, new_width: int, new_height):
        """
            
        Args:
            ...

        Return:
            ...
        """
        cut_frame = frame[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
               
        croped_frame = cv2.resize(src=cut_frame, 
                                  dsize=(new_width, new_height))
        
        croped_frame = croped_frame.reshape(1, new_width, new_height, 3)
        
        return frame, croped_frame
    
    def crop_frame_x(self, frame: np.array, coord: list, new_width: int, new_height):
        """
            
        Args:
            ...

        Return:
            ...
        """
        
        cut_frame = frame[coord[3]:coord[1], coord[2]:coord[0]]

        croped_frame = cv2.resize(src=cut_frame, 
                                  dsize=(new_width, new_height))
        
        croped_frame = croped_frame.reshape(1, new_width, new_height, 3)
        
        return frame, croped_frame
    
    def video_save(self, exp_path: str, cap_info: dict, path: str):
        """
            
        Args:
            ...

        Return:
            ...
        """
        video_path = glob.glob(os.path.join(path, '*.mp4'))
        
        save_path = os.path.join('bin', 'exp', exp_path, 'video')
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        encode = cv2.VideoWriter(os.path.join(save_path, video_path[0].split('/')[-1]), fourcc, cap_info['fps'], (int(cap_info['width']), int(cap_info['height'])))
        
        return encode
    
    def video_save_to_exp_v(self, now_time: str, cap_info: dict, path: str):
        """
            
        Args:
            ...

        Return:
            ...
        """
        video_path = glob.glob(os.path.join(path, '*.mp4'))
        
        save_path = os.path.join('bin', 'exp_v', now_time)
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        encode = cv2.VideoWriter(os.path.join(save_path, video_path[0].split('/')[-1]), fourcc, cap_info['fps'], (int(cap_info['width']), int(cap_info['height'])))
        
        return encode
    
    def putText_in_frame(self, frame: None, pred_deq: list, MAX_CLS: str, GT: list, cap_info: dict):
        """
        Args:
            ...
        Return:
            ...
        """
        
        RED_COLOR = (0,0,255)
        BLUE_COLOR = (255,0,0)
        GREEN_COLOR = (0,255,0)
        WHITE_COLOR = (255,255,255)
        
        # 연속된 프레임의 예측 리스트
        frame = cv2.putText(frame, str(list(pred_deq)), (30, int(cap_info['height'])-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE_COLOR, 1, cv2.LINE_AA)      
        # 예측 리스트중 max 값
        frame = cv2.putText(frame, "PRED: "+MAX_CLS, (220, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED_COLOR, 2, cv2.LINE_AA)                                        
        # True 라벨값
        gt_str = str()
        for i, gt in enumerate(GT):
            if i == len(GT)-1:
                gt_str = gt_str+gt
            else:
                gt_str = gt_str+gt+' '
        frame = cv2.putText(frame, "TRUE: "+gt_str, (220, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN_COLOR, 2, cv2.LINE_AA)                                        
        
        return frame

    def draw_box(self,
                 img,
                 pt1,
                 pt2
    ):
        
        RED_COLOR = (0,0,255)
        BLUE_COLOR = (255,0,0)
        GREEN_COLOR = (0,255,0)
        WHITE_COLOR = (255,255,255)
        
        box_pt1 = pt1 # 좌측 상단 좌표
        box_pt2 = pt2 # 우측 하단 좌표
        box_thickness = 1 # 선 두께
        
        # text box
        img = cv2.rectangle(img, 
                            box_pt1, 
                            box_pt2, 
                            WHITE_COLOR, 
                            box_thickness
        )
        
        return img       
        

    def putText_in_frame_v2(self, 
                            frame: np.array, 
                            pred_deq: list, 
                            MAX_CLS: str, 
                            cap_info: dict,
                            pt1: int, pt2: int,
                            pt3: int, pt4: int,
    ):
        """
        for video predict
        Args:
            ...
        Return:
            ...
        """
        
        RED_COLOR = (0,0,255)
        BLUE_COLOR = (255,0,0)
        GREEN_COLOR = (0,255,0)
        WHITE_COLOR = (255,255,255)
        
        # text1 = "Location Guide : " + MAX_CLS # title & predict
        text1 = MAX_CLS # title & predict
        text2 = str(list(pred_deq)) # pred_list
        
        frame = cv2.putText(frame, text1, (pt1, pt2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE_COLOR, 1, cv2.LINE_AA)      
        frame = cv2.putText(frame, text2, (pt3, pt4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE_COLOR, 1, cv2.LINE_AA)                                        
        
        return frame
    
    def bit_operation_s0(self, 
                         board: np.array, 
                         logo: np.array,
                         pt1: int, pt2: int
    ):
        """
            
        Args:
            ...

        Return:
            ...
        """
        rows, cols, channels = logo.shape
        roi = board[pt1:pt1+rows, pt2:pt2+cols]

        sample_logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(sample_logo_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        board_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        sample_logo_fg = cv2.bitwise_and(logo, logo, mask=mask)

        dst = cv2.add(board_bg, sample_logo_fg)
        board[pt1:pt1+rows, pt2:pt2+cols] = dst
    
        return board

    def bit_operation_s1(self, frame: np.array, board: np.array, height: int, weight: int, size_scale: float):
        """
            
        Args:
            ...

        Return:
            ...
        """
        board = cv2.resize(board, (0,0), fx=size_scale, fy=size_scale, interpolation=cv2.INTER_AREA)
        
        rows, cols, channels = board.shape
        roi = frame[height:rows+height, weight:cols+weight]

        sample_logo_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(sample_logo_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        board_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        sample_logo_fg = cv2.bitwise_and(board, board, mask=mask)

        dst = cv2.add(board_bg, sample_logo_fg)
        frame[height:rows+height, weight:cols+weight] = dst
  
        return frame
    
    def figure_to_array(self, fig):
        """
            
        Args:
            ...

        Return:
            ...
        """
        fig.canvas.draw()
        
        return np.array(fig.canvas.renderer._renderer)
    
    def time2frame(self, time_zone):
        a = format(time_zone, '04')

        m = int(a[0:2])
        s = int(a[2:])

        frametime = ((m*60) + s) * 60

        return frametime
    
# def main():
    # ...
        
# if __name__ == "__main__":
    # main()