from .model.model import CreateGastricModel
from .utils.dataload import DataIOStream
from .utils.video_processing import VideoProcessing
from .utils.roc import AnalyzeROC
from .utils.matrix import PerformanceMeasurement

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

from collections import deque, Counter

from tqdm import tqdm

from PIL import Image

import os
import sys
import yaml
import time
import datetime
import itertools
import shutil
import glob
import re

import argparse

import numpy as np
import pandas as pd

import cv2
import tensorflow as tf

import matplotlib.pyplot as plt

import tensorflow.keras.backend as K


#----------------------------------------------------------------------------

class GastricPositionClassificationMain(DataIOStream, VideoProcessing, 
                                        CreateGastricModel, AnalyzeROC, 
                                        PerformanceMeasurement):
    def __init__(self, main_conf: dict, sub0_conf: dict, sub1_conf: dict, sub2_conf: dict,
                 args: argparse.ArgumentParser(), f_name: str, now_time: str) -> None:
        DataIOStream.__init__(self)    
        VideoProcessing.__init__(self)
        CreateGastricModel.__init__(self)
        AnalyzeROC.__init__(self)
        PerformanceMeasurement.__init__(self)
        
        # main_exp config 정의
        self.main_data_classes = main_conf['data_classes']
        self.main_model_name = main_conf['model_name']
        self.main_optimizer = main_conf['optimizer']
        self.main_exp_path = args.exp_main
        
        # sub0_exp config  정의
        self.sub0_data_classes = sub0_conf['data_classes']
        self.sub0_model_name = sub0_conf['model_name']
        self.sub0_optimizer = sub0_conf['optimizer']
        self.sub0_exp_path = args.exp_sub0
        
        # sub1_exp config  정의
        self.sub1_data_classes = sub1_conf['data_classes']
        self.sub1_model_name = sub1_conf['model_name']
        self.sub1_optimizer = sub1_conf['optimizer']
        self.sub1_exp_path = args.exp_sub1
        
        # sub2_exp config  정의
        self.sub2_data_classes = sub2_conf['data_classes']
        self.sub2_model_name = sub2_conf['model_name']
        self.sub2_optimizer = sub2_conf['optimizer']
        self.sub2_exp_path = args.exp_sub2
        
        self.f_name = f_name
        self.frame_space = args.frame_space
        self.now_time = now_time
    
    def run(self):
        # 코드 실행 시간 확인용
        start = time.time()
        
        # 적용될 동영상 로드, 프레임 정보 추출        
        video_path = os.path.join('bin', 'data', '00_normal_video_roi', self.f_name+'.mp4')
        
        cap, cap_info = self.video_load_x(path=video_path)

        # 동영상 roi 로드
        coord_path = os.path.join('bin', 'data', '00_normal_video_roi', self.f_name+'.txt')

        with open(coord_path) as f:
            coord = f.readline()

        coord = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', '', coord).split(' ')
        coord = [int(i) for i in coord]

        # main_model의 img_shape 로드
        main_img_shape = self.load_imgShape(model_name=self.main_model_name)
        
        #--------------------------------------------------------------------
        
        # 메인 모델 로드
        main_model = self.load_model(exp_path=self.main_exp_path)
        
        # compile & fit
        main_model.compile(optimizer=self.main_optimizer, metrics=['acc'],
                           loss='categorical_crossentropy')
        
        # sub0 모델 로드
        sub0_model = self.load_model(exp_path=self.sub0_exp_path)
        
        # compile & fit
        sub0_model.compile(optimizer=self.sub0_optimizer, metrics=['acc'],
                           loss='categorical_crossentropy')
        
        # sub1 모델 로드
        sub1_model = self.load_model(exp_path=self.sub1_exp_path)
        
        # compile & fit
        sub1_model.compile(optimizer=self.sub1_optimizer, metrics=['acc'],
                           loss='categorical_crossentropy')
        
        # sub2 모델 로드
        sub2_model = self.load_model(exp_path=self.sub2_exp_path)
        
        # compile & fit
        sub2_model.compile(optimizer=self.sub2_optimizer, metrics=['acc'],
                           loss='categorical_crossentropy')
        
        #--------------------------------------------------------------------
        
        # 비디오 내 프레임 추론을 통한 이미지 데이터 생성
        video_extraction_path = os.path.join('bin', 'data', 'video_extraction', 'v1.0.1')
        
        if not os.path.exists(os.path.join(video_extraction_path, 'ES')):
            os.mkdir(os.path.join(video_extraction_path, 'ES'))
        if not os.path.exists(os.path.join(video_extraction_path, 'GE')):
            os.mkdir(os.path.join(video_extraction_path, 'GE'))
        if not os.path.exists(os.path.join(video_extraction_path, 'CR')):
            os.mkdir(os.path.join(video_extraction_path, 'CR'))
        if not os.path.exists(os.path.join(video_extraction_path, 'UB')):
            os.mkdir(os.path.join(video_extraction_path, 'UB'))
        if not os.path.exists(os.path.join(video_extraction_path, 'MB')):
            os.mkdir(os.path.join(video_extraction_path, 'MB'))
        if not os.path.exists(os.path.join(video_extraction_path, 'LB')):
            os.mkdir(os.path.join(video_extraction_path, 'LB'))
        if not os.path.exists(os.path.join(video_extraction_path, 'AG')):
            os.mkdir(os.path.join(video_extraction_path, 'AG'))
        if not os.path.exists(os.path.join(video_extraction_path, 'AT')):
            os.mkdir(os.path.join(video_extraction_path, 'AT'))
        if not os.path.exists(os.path.join(video_extraction_path, 'BB')):
            os.mkdir(os.path.join(video_extraction_path, 'BB'))
        if not os.path.exists(os.path.join(video_extraction_path, 'SD')):
            os.mkdir(os.path.join(video_extraction_path, 'SD'))
        if not os.path.exists(os.path.join(video_extraction_path, 'NO')):
            os.mkdir(os.path.join(video_extraction_path, 'NO'))
        
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
                cap.release()
            else:
                # 동영상 roi에 맞게 crop & 모델 input size에 맞게 resize & normalization
                frame, croped_frame = self.crop_frame_x(frame=frame, coord=coord,
                                                      new_width=main_img_shape,
                                                      new_height=main_img_shape)
                               
                c_frame = croped_frame.copy()
                c_frame = c_frame.reshape(c_frame.shape[1], croped_frame.shape[2], croped_frame.shape[3])
                c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2RGB)
                
                croped_frame = croped_frame / 255.0
                
                new_croped_frame = Image.fromarray(c_frame)
                                
            # 모든 프레임을 사용하지 않고, 1초에 2~3개의 프레임만 활용
            # 추론된 결과에서 Nonclear(NO)의외의 데이터는 전부 저장
            if count % (cap_info['fps'] // self.frame_space) == 0:
                main_pred = np.argmax(main_model.predict(croped_frame), axis=1)
                            
                # 추론된 class값이 sub모델을 통해 2차 추론이 필요할경우, sub모델 추론
                # 추론된 값이 0 == 'ES'일 경우
                if main_pred == [0]:
                    sub0_pred = np.argmax(sub0_model.predict(croped_frame), axis=1)

                    # save img
                    if sub0_pred == [0]:
                        new_croped_frame.save(os.path.join(video_extraction_path, 'ES', f'{self.f_name}_{count}_{self.sub0_data_classes[sub0_pred[0]][0]}.jpg'), dpi=(300, 300))
                    elif sub0_pred == [1]:
                        new_croped_frame.save(os.path.join(video_extraction_path, 'GE', f'{self.f_name}_{count}_{self.sub0_data_classes[sub0_pred[0]][0]}.jpg'), dpi=(300, 300))
                    
                # 추론된 값이 5 == 'DU'일 경우
                elif main_pred == [5]:
                    sub1_pred = np.argmax(sub1_model.predict(croped_frame), axis=1)

                    # save img
                    if sub1_pred == [0]:
                        new_croped_frame.save(os.path.join(video_extraction_path, 'BB', f'{self.f_name}_{count}_{self.sub1_data_classes[sub1_pred[0]][0]}.jpg'), dpi=(300, 300))
                    elif sub1_pred == [1]:
                        new_croped_frame.save(os.path.join(video_extraction_path, 'SD', f'{self.f_name}_{count}_{self.sub1_data_classes[sub1_pred[0]][0]}.jpg'), dpi=(300, 300))
                
                # 추론된 값이 2 == 'BODY'일 경우
                elif main_pred == [2]:
                    sub2_pred = np.argmax(sub2_model.predict(croped_frame), axis=1)
                
                    # save img
                    if sub2_pred == [0]:
                        new_croped_frame.save(os.path.join(video_extraction_path, 'UB', f'{self.f_name}_{count}_{self.sub2_data_classes[sub2_pred[0]][0]}.jpg'), dpi=(300, 300))
                    elif sub2_pred == [1]:
                        new_croped_frame.save(os.path.join(video_extraction_path, 'MB', f'{self.f_name}_{count}_{self.sub2_data_classes[sub2_pred[0]][0]}.jpg'), dpi=(300, 300))
                    elif sub2_pred == [2]:
                        new_croped_frame.save(os.path.join(video_extraction_path, 'LB', f'{self.f_name}_{count}_{self.sub2_data_classes[sub2_pred[0]][0]}.jpg'), dpi=(300, 300))
                
                else:
                    
                    if main_pred == [1]:
                        new_croped_frame.save(os.path.join(video_extraction_path, 'CR', f'{self.f_name}_{count}_{self.main_data_classes[main_pred[0]][0]}.jpg'), dpi=(300, 300))
                    elif main_pred == [3]:
                        new_croped_frame.save(os.path.join(video_extraction_path, 'AG', f'{self.f_name}_{count}_{self.main_data_classes[main_pred[0]][0]}.jpg'), dpi=(300, 300))
                    elif main_pred == [4]:
                        new_croped_frame.save(os.path.join(video_extraction_path, 'AT', f'{self.f_name}_{count}_{self.main_data_classes[main_pred[0]][0]}.jpg'), dpi=(300, 300))
                    elif main_pred == [6]:
                        new_croped_frame.save(os.path.join(video_extraction_path, 'NO', f'{self.f_name}_{count}_{self.main_data_classes[main_pred[0]][0]}.jpg'), dpi=(300, 300))
            
            pbar.update(1)
            
            count += 1
            

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_main", default=None, required=True, help='main exp folder', type=str)
    parser.add_argument("--exp_sub0", default=None, required=False, help='ES-GE exp folder', type=str)
    parser.add_argument("--exp_sub1", default=None, required=False, help='SD-BB exp folder', type=str)
    parser.add_argument("--exp_sub2", default=None, required=False, help='BODY exp folder', type=str)
    parser.add_argument("--video_path", default=None, required=True, help='select video folder ex)01', type=str)
    parser.add_argument("--frame_space", default=3, required=False, help='define space between frames', type=int)
    args = parser.parse_args()
    
    with open(os.path.join('bin', 'exp', args.exp_main, 'train.yaml'), 'r') as f:
        main_conf = yaml.safe_load(f)

    with open(os.path.join('bin', 'exp', args.exp_sub0, 'train.yaml'), 'r') as f:
        sub0_conf = yaml.safe_load(f)
    
    with open(os.path.join('bin', 'exp', args.exp_sub1, 'train.yaml'), 'r') as f:
        sub1_conf = yaml.safe_load(f)
        
    with open(os.path.join('bin', 'exp', args.exp_sub2, 'train.yaml'), 'r') as f:
        sub2_conf = yaml.safe_load(f)
    
    os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"]=str(main_conf['gpu'])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
       
    print('=='*50)
    for item in main_conf:
        print(f'{item}: {main_conf[item]}')
    print('=='*50)

    print('=='*50)
    for item in sub0_conf:
        print(f'{item}: {sub0_conf[item]}')
    print('=='*50)
    
    print('=='*50)
    for item in sub1_conf:
        print(f'{item}: {sub1_conf[item]}')
    print('=='*50)
    
    print('=='*50)
    for item in sub2_conf:
        print(f'{item}: {sub2_conf[item]}')
    print('=='*50)
    
    # now_time
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day}-{d.hour}-{d.minute}-{d.second}"
    
    # 폴더안에 있는 비디오 이름 분리
    video_paths = glob.glob(os.path.join('bin', 'data', args.video_path, '*.mp4'))
    
    for video_path in video_paths:
        f_name = video_path.split('/')[-1].replace('.mp4', '')
        
        GPCM = GastricPositionClassificationMain(main_conf, sub0_conf, sub1_conf, sub2_conf,
                                                args, f_name, now_time)
        GPCM.run()
        
        K.clear_session()
    
if __name__ == '__main__':
    main()
    sys.exit('save done..!')