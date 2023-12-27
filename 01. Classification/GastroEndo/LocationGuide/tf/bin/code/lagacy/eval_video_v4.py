from .model.model import CreateGastricModel
from .utils.dataload import DataIOStream
from .utils.video_processing import VideoProcessing
from .utils.roc import AnalyzeROC
from .utils.matrix import PerformanceMeasurement

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

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
import re

import argparse

import numpy as np
import pandas as pd

import cv2
import tensorflow as tf
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

#----------------------------------------------------------------------------

class GastricPositionClassificationMain(DataIOStream, VideoProcessing, 
                                        CreateGastricModel, AnalyzeROC, 
                                        PerformanceMeasurement):
    def __init__(self, main_conf: dict, sub0_conf: dict, sub1_conf: dict,
                 args: argparse.ArgumentParser(), now_time: str, video_path: str) -> None:
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
        
        self.video_path = os.path.join('bin', 'data', 'video', video_path)
        self.frame_space = args.frame_space
        self.now_time = now_time
        
        # 프레임 정보를 추출합니다.
        self.cap, self.cap_info = self.video_load(path=self.video_path)
        # 모델을 통해 추론된 해부학적 위치값을 저장할 deque를 생성합니다.
        self.main_deq = deque(maxlen=7)
        self.encode = self.video_save_to_exp_v(now_time=self.now_time,
                                               cap_info=self.cap_info,
                                               path=self.video_path)
        # 동영상 roi파일을 로드합니다.
        self.coord = self.coord_load(self.video_path)
        # main_model에 적용된 img_shape를 로드합니다.
        self.img_shape = self.load_imgShape(model_name=self.main_model_name)
        # 동영상 길이의 tqdm 생성합니다.
        self.pbar = tqdm(total=self.cap_info['length'])
        # 하드보팅 전/후 결과를 담기위한 list를 생성합니다.
        self.none_postprocess, self.postprocess = [], []
        
        self.data_classes = {'ES':0, 'GE':1, 'CR':2, 'BODY':3, 'AG':4, 'AT':5, 'BB':6, 'SD':7, 'NO':8}
        self.data_classesR = {v:k for k,v in self.data_classes.items()}
        
    def coord_load(self, v_path: str):
        # 동영상 roi파일을 로드합니다.
        coord_path = glob.glob(os.path.join(v_path, '*.txt'))
        
        with open(coord_path[0]) as f:
            coord = f.readline()
            
        coord = coord.split(',')
        
        return coord
    
    def run(self):
        # 모델별로 클래스가 추론된 양을 체크하기위한 dict를 생성합니다.
        main_class_dict, sub0_class_dict, sub1_class_dict = {}, {}, {}
        
        # main 모델의 클래스 정보를 dict로 정리
        for i, CLS in enumerate(self.main_data_classes):
            main_class_dict[CLS[0]] = i
        # sub0 모델의 클래스 정보를 dict로 정리
        for i, CLS in enumerate(self.sub0_data_classes):
            sub0_class_dict[CLS[0]] = i
        # sub1 모델의 클래스 정보를 dict로 정리
        for i, CLS in enumerate(self.sub1_data_classes):
            sub1_class_dict[CLS[0]] = i
        
        main_class_dict_reverse = {v:k for k,v in main_class_dict.items()}       
        sub0_class_dict_reverse = {v:k for k,v in sub0_class_dict.items()}       
        sub1_class_dict_reverse = {v:k for k,v in sub1_class_dict.items()}       
        
        #--------------------------------------------------------------------
        
        # main 모델을 로드한 뒤, 컴파일합니다.
        main_model = self.load_model(exp_path=self.main_exp_path)
        main_model.compile(optimizer=self.main_optimizer, metrics=['acc'],
                           loss='categorical_crossentropy')
        
        # sub0 모델을 로드한 뒤, 컴파일합니다.
        sub0_model = self.load_model(exp_path=self.sub0_exp_path)
        sub0_model.compile(optimizer=self.sub0_optimizer, metrics=['acc'],
                           loss='categorical_crossentropy')
        
        # sub1 모델을 로드한 뒤, 컴파일합니다.
        sub1_model = self.load_model(exp_path=self.sub1_exp_path)
        sub1_model.compile(optimizer=self.sub1_optimizer, metrics=['acc'],
                           loss='categorical_crossentropy')
        
        #--------------------------------------------------------------------
        
        # GT 데이터를 정리합니다.
        # GT 데이터 로드
        GT_xlsx_path = glob.glob(os.path.join(self.video_path, '*.xlsx'))
        GT_xlsx = pd.read_excel(GT_xlsx_path[0], engine='openpyxl')
        
        # GT 데이터는 분(min)으로 구성되어, 초(sec)로 변환한다.
        L_lst, S_lst, E_lst = [], [], []

        for i, (L, S, E) in enumerate(zip(GT_xlsx['Location'], GT_xlsx['Start time'], GT_xlsx['End time'])):
            S_frametime = self.time2frame(S)
            E_frametime = self.time2frame(E)
            
            L_lst.append(L)
            S_lst.append(S_frametime)
            E_lst.append(E_frametime)
        
        # GT 데이터상, 정확한 프레임 구간까지 시간이 정의되지않았음.
        # 따라서 동영상 데이터의 length값으로 GT의 마지막 구간을 재설정합니다.
        E_lst.pop(-1)
        E_lst.append(int(self.cap_info['length']))
        
        GT_info = pd.DataFrame({'Location': L_lst, 'Start_time': S_lst, 'End_time': E_lst})
        
        frame_flow, gt = [], []

        for (L, S, E) in zip(GT_info['Location'], GT_info['Start_time'], GT_info['End_time']):
            for i in range(S, E):
                if i not in frame_flow:
                    frame_flow.append(i)
                    gt.append([L])
                else:
                    gt[i].append(L)

        #--------------------------------------------------------------------
        # 프레임 카운트 확인용입니다.
        count = 0
        
        # 프레임 진행
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            
            # 더이상 프레임이 존재하지 않을경우 종료합니다.
            if not ret:
                self.pbar.close()
                
                # 동영상 평가를 위한 exp_v 폴더를 생성합니다.
                if not os.path.exists(os.path.join('bin', 'exp_v')):
                    os.mkdir(os.path.join('bin', 'exp_v'))
                
                # exp_v 내에 현재 시간대의 폴더를 생성합니다.
                if not os.path.exists(os.path.join('bin', 'exp_v', self.now_time)):
                    os.mkdir(os.path.join('bin', 'exp_v', self.now_time))
                
                # 정량적 평가 진행합니다.
                y_trueP, y_trueNP, y_predP, y_predNP = [], [], [], []
                
                for i, (true, postP, NpostP) in enumerate(zip(gt, self.postprocess, self.none_postprocess)):
                    if postP not in true:
                        y_trueP.append(self.data_classes[true[-1]])
                    elif postP in true:
                        y_trueP.append(self.data_classes[postP])
                    
                    if NpostP not in true:
                        y_trueNP.append(self.data_classes[true[-1]])
                    elif NpostP in true:
                        y_trueNP.append(self.data_classes[NpostP])
                        
                    y_predP.append(self.data_classes[postP])
                    y_predNP.append(self.data_classes[NpostP])

                del_idx0, del_idx1 = [], []
                
                # NO는 평가에서 제외하기위해 뺍니다.
                for i, (trueP, predP) in enumerate(zip(y_trueP, y_predP)):
                    if trueP == 8 or predP == 8:
                        del_idx0.append(i)
                for i in sorted(del_idx0, reverse=True):
                    del y_trueP[i]
                    del y_predP[i]
                for i, (trueNP, predNP) in enumerate(zip(y_trueNP, y_predNP)):
                    if trueNP == 8 or predNP == 8:
                        del_idx1.append(i)
                for i in sorted(del_idx1, reverse=True):
                    del y_trueNP[i]
                    del y_predNP[i]
                
                # 현재 클래스 개수를 정의합니다.
                del self.data_classes['NO']
                n_class = len(self.data_classes)

                # true, pred를 범주형 값으로 변환합니다.
                y_trueP2CA = tf.keras.utils.to_categorical(np.array(y_trueP), num_classes=n_class)
                y_predP2CA = tf.keras.utils.to_categorical(np.array(y_predP), num_classes=n_class)
                y_trueNP2CA = tf.keras.utils.to_categorical(np.array(y_trueNP), num_classes=n_class)
                y_predNP2CA = tf.keras.utils.to_categorical(np.array(y_predNP), num_classes=n_class)
                
                #  confusion_matrix를 저장합니다.
                self.plot_confusion_matrix_video(y_true=y_trueP, y_pred=y_predP, 
                                                 class_dict=self.data_classes, now_time=self.now_time,
                                                 plot_name='PP')
                self.plot_confusion_matrix_video(y_true=y_trueNP, y_pred=y_predNP, 
                                                 class_dict=self.data_classes, now_time=self.now_time,
                                                 plot_name='NPP')
                
                # classification_reports를 저장합니다.
                pp_report = pd.DataFrame(classification_report(y_true=y_trueP, y_pred=y_predP, 
                                               digits=self.data_classes.values(), 
                                               output_dict=True))
                pp_report.to_csv(os.path.join('bin', 'exp_v', self.now_time, 'PP_report.csv'))
                
                npp_report = pd.DataFrame(classification_report(y_true=y_trueNP, y_pred=y_predNP, 
                                               digits=self.data_classes.values(), 
                                               output_dict=True))
                npp_report.to_csv(os.path.join('bin', 'exp_v', self.now_time, 'NPP_report.csv'))
                
                # roc curve를 저장합니다.
                # sensitivity & specificity 생성합니다.
                self.ROC_multi_video(class_dict=self.data_classes, 
                                     y_true=y_trueP2CA, y_pred=y_predP2CA, 
                                     now_time=self.now_time, plot_name='PP')
                self.ROC_multi_video(class_dict=self.data_classes, 
                                     y_true=y_trueNP2CA, y_pred=y_predNP2CA, 
                                     now_time=self.now_time, plot_name='NPP')
                
                pp_sen_spe, npp_sen_spe = {}, {}
                
                for i in range(len(self.data_classes)):
                    sen, spe = self.sensitivity_specificity_per_class(y_true=y_trueP, y_pred=y_predP, y_class=i)
                    pp_sen_spe[list(self.data_classes.keys())[i]] = {'sensitivity': sen, 'specificity': spe}
                
                pd.DataFrame(pp_sen_spe).to_csv(os.path.join('bin', 'exp_v', self.now_time, 'PP_sen_spe.csv'))
                
                for i in range(len(self.data_classes)):
                    sen, spe = self.sensitivity_specificity_per_class(y_true=y_trueNP, y_pred=y_predNP, y_class=i)
                    npp_sen_spe[list(self.data_classes.keys())[i]] = {'sensitivity': sen, 'specificity': spe}
                
                pd.DataFrame(pp_sen_spe).to_csv(os.path.join('bin', 'exp_v', self.now_time, 'NPP_sen_spe.csv'))
                
                # sys.exit("Done...")
                print('EVAL Done..!')
                
                break
                
            else:
                # 동영상 roi에 맞게 crop & 모델 input size에 맞게 resize & normalization을 진행합니다.
                frame, croped_frame = self.crop_frame(frame=frame, coord=self.coord,
                                                      new_width=self.img_shape,
                                                      new_height=self.img_shape)
                croped_frame = croped_frame / 255.0

            # 모든 프레임을 사용하지 않고, 1초에 2~3개의 프레임만 활용합니다.
            if count % self.frame_space == 0:
                # main 모델을 이용하여 추론을 진행합니다.
                main_pred = np.argmax(main_model.predict(croped_frame), axis=1)
                
                # 추론된 값이 sub모델을 통해 2차 추론이 필요할경우, sub모델 추론을 진행합니다.
                # 추론된 값이 0 == 'ES'일 경우, sub0_model을 통해 추론합니다.
                if main_pred == [0]:
                    sub0_pred = np.argmax(sub0_model.predict(croped_frame), axis=1)
                    self.main_deq.extend([sub0_class_dict_reverse[sub0_pred[0]]])
                # 추론된 값이 5 == 'DU'일 경우, sub1_model을 통해 추론합니다.
                elif main_pred == [5]:
                    sub1_pred = np.argmax(sub1_model.predict(croped_frame), axis=1)
                    self.main_deq.extend([sub1_class_dict_reverse[sub1_pred[0]]])
                else:
                    self.main_deq.extend([main_class_dict_reverse[main_pred[0]]])
            
            # 메인 데크에서 가장많이 누적된 class 계산 
            counter = Counter(self.main_deq)
            MAX_VALUE = 0
            for CLS in counter.keys():
                VALUE = counter[CLS]
                if VALUE > MAX_VALUE:
                    MAX_VALUE = VALUE
                    MAX_CLS = CLS
            
            # 후처리를 적용하지 않은 추론 결과값 저장합니다.
            self.none_postprocess.append(self.main_deq[-1])
            # 후처리를 적용한 결과값 저장합니다.
            self.postprocess.append(MAX_CLS)
            
            # predict 결과 및 Max_CLS를 프레임에 draw
            frame = self.putText_in_frame(frame=frame,
                                          pred_deq=self.main_deq,
                                          MAX_CLS=MAX_CLS,
                                          GT=gt[count],
                                          cap_info=self.cap_info,
                                          )
            
            self.encode.write(frame)
            self.pbar.update(1)
            
            count += 1
            

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_main", default=None, required=True, help='main exp folder', type=str)
    parser.add_argument("--exp_sub0", default=None, required=False, help='ES-GE exp folder', type=str)
    parser.add_argument("--exp_sub1", default=None, required=False, help='SD-BB exp folder', type=str)
    # parser.add_argument("--video_path", default=None, required=True, help='select video folder ex)01', type=str)
    parser.add_argument("--frame_space", default=15, required=False, help='define space between frames', type=int)
    args = parser.parse_args()
    
    with open(os.path.join('bin', 'exp', args.exp_main, 'train.yaml'), 'r') as f:
        main_conf = yaml.safe_load(f)

    with open(os.path.join('bin', 'exp', args.exp_sub0, 'train.yaml'), 'r') as f:
        sub0_conf = yaml.safe_load(f)
    
    with open(os.path.join('bin', 'exp', args.exp_sub1, 'train.yaml'), 'r') as f:
        sub1_conf = yaml.safe_load(f)
    
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
    
    # 비디오 전부 불러오기
    v_paths = glob.glob(os.path.join('bin', 'data', 'video', '*.dcm'))
    
    for path in v_paths:
        video_path = path.split('/')[-1]
        print('video_path', video_path)
        # now_time
        d = datetime.datetime.now()
        now_time = f"{d.year}-{d.month}-{d.day}-{d.hour}-{d.minute}-{d.second}"
        
        GPCM = GastricPositionClassificationMain(main_conf, sub0_conf, sub1_conf,
                                                args, now_time, video_path)
        GPCM.run()
        
        # 다음 학습을 위해 케라스 세션을 종료합니다.
        K.clear_session()
        

    
if __name__ == '__main__':
    main()
    sys.exit('save done..!')