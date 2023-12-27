from .utils.model import GastricModeling
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

class GastricPositionClassificationMain(DataIOStream, VideoProcessing, GastricModeling):
    def __init__(self, conf: dict, args: None, learning_stage: bool) -> None:
        DataIOStream.__init__(self)    
        VideoProcessing.__init__(self)
        GastricModeling.__init__(self)
        
        self.dataset = conf['data_classes']
        self.model_name = conf['model_name']
        self.batch = conf['batch']
        self.epoch = conf['epoch']
        self.optimizer = conf['optimizer']
        self.n_class = conf['n_class']
        self.es_patience = conf['es_patience']
        self.reduce_lr_patience = conf['reduce_lr_patience']
        self.reduce_lr_factor = conf['reduce_lr_factor']

        self.exp_path = args.exp
        self.video_path = args.video_path
        self.frame_space = args.frame_space
        self.learning_stage = learning_stage
    
    def run(self):
        # dataset
        name_lst = []
        for name in self.data_classes:
            name_lst.append(name[0])
        
        dataset_name = '_'.join(name_lst)
        
        category_fpath = os.path.join('bin', 'npy', dataset_name)
        
        img_shape = self.load_imgShape(model_name=self.model_name)
        
        # Create a folder based on the shape of the img
        shape_fpath = os.path.join(category_fpath, str(img_shape))
        
        class_dict = dict(class_lst[0])
        class_dict_reverse = {v:k for k,v in class_dict.items()}
        
        video_path = os.path.join('bin', 'data', 'video', self.video_path)
        
        # logo load
        logo_paths = glob.glob(os.path.join('bin', 'data', 'gastric_logo', '*'))

        logo_lst = []

        gastric_status_lv = {}
        
        for i in range(len(logo_paths)):
            if os.path.splitext(logo_paths[i])[-1] != '.png':
                logo_png_path_lst = glob.glob(os.path.join(logo_paths[i], '*.png'))
                logos = []
                for logo_png_path in logo_png_path_lst:
                    logos.append(cv2.imread(logo_png_path))
                logo_lst.append(logos)
        for CLS in class_lst[0]:
            gastric_status_lv[CLS] = 0

        # draw gastric logo board
        board = np.zeros((410, 358, 3), dtype=np.uint8)
                                  
        # frame load
        cap, cap_info = self.video_load(path=video_path)
        
        # call encoder
        encode = self.video_save(exp_path=self.exp_path,
                                 cap_info=cap_info,
                                 path=video_path)
        
        # load coord
        with open(os.path.join(video_path, 'crop.txt')) as f:
            coord = f.readline()
        
        coord = coord.split(',')
        
        # deque max length
        deq = deque(maxlen=7)
        
        # load_model
        model = self.load_model_(exp_path=self.exp_path)
        
        # compile & fit
        model.compile(optimizer=self.optimizer, metrics=['acc'],
                        loss='categorical_crossentropy')
        
        count = 0
        
        # tqdm
        pbar = tqdm(total=cap_info['length'])
        
        # define frame space
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                pbar.close()
                sys.exit("Frame done...")
            else:
                frame, croped_frame = self.crop_frame(frame=frame, coord=coord)
            
            croped_frame = croped_frame / 255.0

            # define frame space 
            if count % 20  == 0:
                pred = np.argmax(model.predict(croped_frame), axis=1)
                
                deq.extend(class_lst[0][pred])
            
            counter = Counter(deq)
        
            MAX_VALUE = 0
            
            for CLS in counter.keys():
                VALUE = counter[CLS]
                if VALUE > MAX_VALUE:
                    MAX_VALUE = VALUE
                    MAX_CLS = CLS
            
            gastric_status_lv[MAX_CLS] += 1
            
            # put text
            frame = self.put_text_n_img(frame=frame,
                                        pred_deq=deq,
                                        cap_info=cap_info,
                                        MAX_CLS=MAX_CLS,
                                        status_lv=gastric_status_lv
                                        )

            # 가장 많이 추론되는 클래스를 시각화
            for i in range(len(logo_lst)):
                lv_value = list(gastric_status_lv.values())[i] // 60
                
                if lv_value < 2:
                    LV = 0
                elif 3 >= lv_value >= 2:
                    LV = 1
                elif lv_value >= 4:
                    LV = 2
                
                logo = logo_lst[i][LV]
                board = self.bit_operation_s0(board=board, logo=logo)
            
            # 프레임마다 추론되는 클래스의 수치 총합 시각화
            f = plt.figure(facecolor='dimgray')
            
            plt_x = np.arange(len(list(gastric_status_lv)))
            plt_height = list(gastric_status_lv.values())

            plt.bar(x=plt_x, height=plt_height, color='blue')
            plt.xticks(ticks=plt_x, labels=list(gastric_status_lv.keys()), fontsize=20)
            
            plt.gca().set_facecolor('dimgray')

            plt.gca().tick_params(axis='x', colors='green')
            plt.gca().tick_params(axis='y', colors='green')
            
            f_arr = self.figure_to_array(f)

            # plt가 너무 많이 띄워지는것 방지
            plt.close() 

            new_arr = np.dsplit(f_arr, 4)

            class_sum_img = np.concatenate((new_arr[1], new_arr[2], new_arr[0]), axis=2)

            frame = self.bit_operation_s1(frame=frame, board=class_sum_img, 
                                          height=450, weight=210,
                                          size_scale=0.4)
            
            # 현재 예측되는 위치 포인터
            if MAX_CLS == 'NO':
                pass
            else:
                logo = logo_lst[class_dict_reverse[MAX_CLS]][-1]
                board = self.bit_operation_s0(board=board, logo=logo)
            
            frame = self.bit_operation_s1(frame=frame, board=board, 
                                          height=150, weight=210,
                                          size_scale=0.7)
            
            encode.write(frame)
            
            pbar.update(1)
            
            count += 1
            

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default=None, required=True, help='select exp folder ex)2022-12-22-15-55-46', type=str)
    parser.add_argument("--video_path", default=None, required=True, help='select video folder ex)01', type=str)
    parser.add_argument("--frame_space", default=20, required=False, help='define space between frames', type=int)
    parser.add_argument("--learning_stage", default=False, required=False, help='to show the learning stage train(True) or test(False)', type=bool)
    args = parser.parse_args()
    
    with open(os.path.join('bin', 'exp', args.exp, 'train.yaml'), 'r') as f:
        conf = yaml.safe_load(f)

    os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"]=str(conf['gpu'])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    print('=='*50)
    for item in conf:
        print(f'{item}: {conf[item]}')
    print('=='*50)

    GPCM = GastricPositionClassificationMain(conf, args, args.learning_stage)
    GPCM.run()

if __name__ == '__main__':
    main()
    sys.exit('save done..!')