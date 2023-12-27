from .model.model import CreateGastricModel
from .utils.dataload import DataIOStream
from .utils.video_processing import VideoProcessing

from collections import deque, Counter

from tqdm import tqdm

import os
import sys
import yaml
import time
import datetime
import itertools
import shutil
import glob

import argparse

import numpy as np
import pandas as pd

import cv2
import tensorflow as tf

import matplotlib.pyplot as plt



#----------------------------------------------------------------------------

class GastricPositionClassificationMain(DataIOStream, VideoProcessing, CreateGastricModel):
    def __init__(self, main_conf: dict, args: None, learning_stage: bool, now_time: str) -> None:
        DataIOStream.__init__(self)    
        VideoProcessing.__init__(self)
        CreateGastricModel.__init__(self)
        
        # main_exp config 정의
        self.main_data_classes = main_conf['data_classes']
        self.main_model_name = main_conf['model_name']
        self.main_optimizer = main_conf['optimizer']
        self.main_exp_path = args.exp_main
        
        self.video_path = args.video_path
        self.frame_space = args.frame_space
        self.learning_stage = learning_stage
        self.now_time = now_time
    
    def run(self):
        # 적용될 동영상 로드, 프레임 정보 추출        
        video_path = os.path.join('bin', 'data', 'video', self.video_path)
        
        cap, cap_info = self.video_load(path=video_path)

        # 인코더 생성
        # encode = self.video_save(exp_path=self.exp_path,
                                #  cap_info=cap_info,
                                #  path=video_path)
        
        encode = self.video_save_to_exp_v(now_time=self.now_time,
                                          cap_info=cap_info,
                                          path=video_path)
        
        # 동영상 roi 로드
        coord_path = glob.glob(os.path.join(video_path, '*.txt'))
        
        with open(coord_path[0]) as f:
            coord = f.readline()
        
        coord = coord.split(',')
       
        # 적용될 위내시경 로고 이미지 로드
        logo_paths = glob.glob(os.path.join('bin', 'data', 'gastric_logo', '*'))

        logo_lst = []
    
        for i in range(len(logo_paths)):
            if os.path.splitext(logo_paths[i])[-1] != '.png':
                logo_png_path_lst = glob.glob(os.path.join(logo_paths[i], '*.png'))
                logos = []
                for logo_png_path in logo_png_path_lst:
                    logos.append(cv2.imread(logo_png_path))
                logo_lst.append(logos)
        
        # 위내시경 로고 이미지 background 생성
        board = np.zeros((410, 358, 3), dtype=np.uint8)
        
        # 메인 모델의 predict 저장 데크 생성
        main_deq = deque(maxlen=7)
        
        # 메인 데크의 과반수 이상 class의 카운트 dict 생성
        main_status_lv = {}
        # 메인 클래스 dict 생성
        main_class_dict = {}
        # 시간 흐름에 따른 MAX CLASS dict 생성
        main_playback_cls_list = []
        
        for i, CLS in enumerate(self.main_data_classes):
            main_status_lv[CLS[0]] = 0
            main_class_dict[CLS[0]] = i
        
        main_class_dict_reverse = {v:k for k,v in main_class_dict.items()}       
        
        # main_model의 img_shape 로드
        main_img_shape = self.load_imgShape(model_name=self.main_model_name)
        
        # 메인 모델 로드
        main_model = self.load_model(exp_path=self.main_exp_path)
        
        # compile & fit
        main_model.compile(optimizer=self.main_optimizer, metrics=['acc'],
                        loss='categorical_crossentropy')
        
        # 진행된 프레임 카운트
        count = 0
        
        # 적용될 동영상 길이의 tqdm 생성
        pbar = tqdm(total=cap_info['length'])
        
        # 프레임 진행
        while cap.isOpened():
            ret, frame = cap.read()
            
            # 더이상 프레임이 존재하지 않을경우 종료
            if not ret:
                pbar.close()
                
                # 동영상 평가를 위한 exp_v 폴더 생성
                if not os.path.exists(os.path.join('bin', 'exp_v')):
                    os.mkdir(os.path.join('bin', 'exp_v'))
                
                # exp_v에 현재 시간대의 폴더 생성
                if not os.path.exists(os.path.join('bin', 'exp_v', self.now_time)):
                    os.mkdir(os.path.join('bin', 'exp_v', self.now_time))
                
                # 시간 흐름에 따른 predict 값 저장
                frame_flow = [i for i, elem in enumerate(main_playback_cls_list)]
                
                playback_cls_dict = {'frame_flow': frame_flow,
                                     'main': main_playback_cls_list}
                
                playback_df = pd.DataFrame(playback_cls_dict)
                
                playback_df.to_excel(excel_writer=os.path.join('bin', 'exp_v', self.now_time, 'playback.xlsx'),
                                     sheet_name='Sheet1',
                                     na_rep='NaN',
                                     header=True,
                                     index=False,
                                     startrow=1,
                                     startcol=1,
                                     freeze_panes=(2,0)
                                     )
                
                sys.exit("Frame done...")
            else:
                # 동영상 roi에 맞게 crop & 모델 input size에 맞게 resize & normalization
                frame, croped_frame = self.crop_frame(frame=frame, coord=coord,
                                                      new_width=main_img_shape,
                                                      new_height=main_img_shape)
                croped_frame = croped_frame / 255.0

            # 모든 프레임을 사용하지 않고, 1초에 2~3개의 프레임만 활용
            if count % self.frame_space == 0:
                main_pred = np.argmax(main_model.predict(croped_frame), axis=1)
                
                main_deq.extend([main_class_dict_reverse[main_pred[0]]])
            
            # 메인 데크의 class 누적량 계산
            counter = Counter(main_deq)
        
            MAX_VALUE = 0
            
            # 메인 데크에서 가장많이 누적된 class 계산 
            # 해당 코드 for문 대신해서 수정 필요
            for CLS in counter.keys():
                VALUE = counter[CLS]
                if VALUE > MAX_VALUE:
                    MAX_VALUE = VALUE
                    MAX_CLS = CLS
            
            main_status_lv[MAX_CLS] += 1
            main_playback_cls_list.append(MAX_CLS)
            
            # predict 결과 및 Max_CLS를 프레임에 draw
            frame = self.put_text_n_img(frame=frame,
                                        pred_deq=main_deq,
                                        cap_info=cap_info,
                                        MAX_CLS=MAX_CLS,
                                        status_lv=main_status_lv
                                        )

            # # main model에서 가장 많이 추론되는 클래스를 시각화
            # for i in range(len(logo_lst)):
            #     lv_value = list(main_status_lv.values())[i] // 60
                
            #     if lv_value < 2:
            #         LV = 0
            #     elif 3 >= lv_value >= 2:
            #         LV = 1
            #     elif lv_value >= 4:
            #         LV = 2
                
            #     logo = logo_lst[i][LV]
            #     board = self.bit_operation_s0(board=board, logo=logo)
            
            # # 프레임마다 추론되는 클래스의 수치 총합 시각화
            # f = plt.figure(facecolor='dimgray')
            
            # plt_x = np.arange(len(list(main_status_lv)))
            # plt_height = list(main_status_lv.values())

            # plt.bar(x=plt_x, height=plt_height, color='blue')
            # plt.xticks(ticks=plt_x, labels=list(main_status_lv.keys()), fontsize=20)
            
            # plt.gca().set_facecolor('dimgray')

            # plt.gca().tick_params(axis='x', colors='green')
            # plt.gca().tick_params(axis='y', colors='green')
            
            # f_arr = self.figure_to_array(f)

            # # plt가 너무 많이 띄워지는것 방지
            # plt.close() 

            # new_arr = np.dsplit(f_arr, 4)

            # class_sum_img = np.concatenate((new_arr[1], new_arr[2], new_arr[0]), axis=2)

            # frame = self.bit_operation_s1(frame=frame, board=class_sum_img, 
            #                               height=450, weight=210,
            #                               size_scale=0.4)
            
            # # main model의 현재 예측되는 위치 포인터
            # if MAX_CLS == 'NO':
            #     pass
            # else:
            #     logo = logo_lst[main_class_dict_reverse[MAX_CLS]][-1]
            #     board = self.bit_operation_s0(board=board, logo=logo)
            
            # frame = self.bit_operation_s1(frame=frame, board=board, 
            #                               height=150, weight=210,
            #                               size_scale=0.7)
            
            encode.write(frame)
            
            pbar.update(1)
            
            count += 1
            

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_main", default=None, required=True, help='main exp folder', type=str)
    parser.add_argument("--exp_es", default=None, required=False, help='ES-GE exp folder', type=str)
    parser.add_argument("--exp_du", default=None, required=False, help='SD-BB exp folder', type=str)
    parser.add_argument("--video_path", default=None, required=True, help='select video folder ex)01', type=str)
    parser.add_argument("--frame_space", default=20, required=False, help='define space between frames', type=int)
    parser.add_argument("--learning_stage", default=False, required=False, help='to show the learning stage train(True) or test(False)', type=bool)
    args = parser.parse_args()
    
    with open(os.path.join('bin', 'exp', args.exp_main, 'train.yaml'), 'r') as f:
        main_conf = yaml.safe_load(f)

    os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"]=str(main_conf['gpu'])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    print('=='*50)
    for item in main_conf:
        print(f'{item}: {main_conf[item]}')
    print('=='*50)

    # now_time
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day}-{d.hour}-{d.minute}-{d.second}"
    
    GPCM = GastricPositionClassificationMain(main_conf, args, args.learning_stage, now_time)
    GPCM.run()

if __name__ == '__main__':
    main()
    sys.exit('save done..!')