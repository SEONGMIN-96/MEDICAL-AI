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
        
        print('프레임 길이: %d, 프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d' %(length, width, height, fps))
        print('=='*50)
        
        return cap, {'length':length, 'width': width, 'height': height, 'fps': fps}
    
    def crop_frame(self, frame: None, coord: list):
        """
            
        Args:
            ...

        Return:
            ...
        """
        cut_frame = frame[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
               
        croped_frame = cv2.resize(src=cut_frame, 
                                  dsize=(256, 256))
        
        croped_frame = croped_frame.reshape(1, 256, 256, 3)
        
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
    
    def put_text_n_img(self, frame: None, pred_deq: list, MAX_CLS: str, cap_info: dict, status_lv: dict):
        """
            
        Args:
            ...

        Return:
            ...
        """
        frame = cv2.putText(frame, str(list(pred_deq)), (30, int(cap_info['height'])-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 69, 255), 1, cv2.LINE_AA)      # 연속된 프레임의 예측 리스트
        frame = cv2.putText(frame, MAX_CLS, (220, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 69, 255), 2, cv2.LINE_AA)                                        # 예측 리스트중 max 값
        # frame = cv2.putText(frame, str(status_lv), (int(cap_info['width'])-1300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 69, 255), 2, cv2.LINE_AA)          # 전체 누적 예측 클래스
        
        return frame
    
    def bit_operation_s0(self, board: np.array, logo: np.array):
        """
            
        Args:
            ...

        Return:
            ...
        """
        rows, cols, channels = logo.shape
        roi = board[0:rows, 0:cols]

        sample_logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(sample_logo_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        board_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        sample_logo_fg = cv2.bitwise_and(logo, logo, mask=mask)

        dst = cv2.add(board_bg, sample_logo_fg)
        board[0:rows, 0:cols] = dst
    
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
            
# def main():
    # ...
        
# if __name__ == "__main__":
    # main()