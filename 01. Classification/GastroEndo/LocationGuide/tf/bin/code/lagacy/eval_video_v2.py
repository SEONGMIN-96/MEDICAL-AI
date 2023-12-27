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

import matplotlib.pyplot as plt



#----------------------------------------------------------------------------

class GastricPositionClassificationMain(DataIOStream, VideoProcessing, 
                                        CreateGastricModel, AnalyzeROC, 
                                        PerformanceMeasurement):
    def __init__(self, main_conf: dict, sub0_conf: dict, sub1_conf: dict,
                 args: argparse.ArgumentParser(), now_time: str) -> None:
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
        
        self.video_path = os.path.join('bin', 'data', 'video', args.video_path)
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
        # 하드보팅 알고리즘을 통해 선정된 클래스들을 담기위한 list를 생성합니다.
        self.main_playback_cls_list = []
        
    def coord_load(self, v_path: str):
        # 동영상 roi파일을 로드합니다.
        coord_path = glob.glob(os.path.join(self.video_path, '*.txt'))
        
        with open(coord_path[0]) as f:
            coord = f.readline()
            
        coord = coord.split(',')
        
        return coord
    
    def run(self):
        # 코드의 전체 소요시간을 체크합니다.
        start = time.time()
       
        # 모델별로 클래스가 추론된 양을 체크하기위한 dict를 생성합니다.
        main_class_dict, sub0_class_dict, sub1_class_dict = {}, {}, {}
        
        
        # 하드보팅 전/후 결과를 담기위한 list를 생성합니다.
        none_post_process, post_process = [], []
        
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
        # GT 정리합니다.
        
        
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
                
                # 시간 흐름에 따른 predict 값 저장
                frame_flow = [i for i, elem in enumerate(self.main_playback_cls_list)]
                
                playback_cls_dict = {'frame_flow': frame_flow,
                                     'main': self.main_playback_cls_list,
                                     'none_post_process': none_post_process}
                
                playback_df = pd.DataFrame(playback_cls_dict)
                
                # GT데이터 로드
                GT_xlsx_path = glob.glob(os.path.join(self.video_path, '*.xlsx'))
                GT_xlsx = pd.read_excel(GT_xlsx_path[0], engine='openpyxl')
                
                # GT데이터 분 -> 초 변환
                L_lst, S_lst, E_lst = [], [], []

                for i, (L, S, E) in enumerate(zip(GT_xlsx['Location'], GT_xlsx['Start time'], GT_xlsx['End time'])):
                    S_frametime = self.time2frame(S)
                    E_frametime = self.time2frame(E)
                    
                    L_lst.append(L)
                    S_lst.append(S_frametime)
                    E_lst.append(E_frametime)

                # GT데이터의 경우 초 단위로 라벨링이 구성되어 마지막 0.xx초에 대한 라벨링이 없다.
                # 하지만 마지막 class와 동일한 라벨링으로 구성되어야 하기때문에 이 부분 수정합니다.
                E_lst.pop(-1)
                E_lst.append(len(playback_df))

                GT_info = {'Location': L_lst, 'Start_time': S_lst, 'End_time': E_lst}
                gt_df = pd.DataFrame(GT_info)
                
                new_frame_flow, gt = [], []

                # 영상 특성상 구간이 겹치는 클래스가 존재,
                # 따라서 GT를 중복으로 구성해 df 생성
                for df in gt_df.values:
                    for i in range(df[1], df[2]):
                        if not i in new_frame_flow:
                            new_frame_flow.append(i)
                            gt.append([df[0]])
                        else:
                            gt[i].append(df[0])
                
                new_playback_df = pd.DataFrame({'frame_flow':new_frame_flow, 
                                                'gt': gt, 
                                                'none_post_process': none_post_process, 
                                                'post_process':playback_df['main'],
                                                })

                # GT상 'NO'는 신체 외부를 뜻함, 이는 평가에 속하지 않음
                # 따라서 이부분 프레임 제거 val[1] == gt
                for i, val in enumerate(new_playback_df.values):
                    if 'NO' in val[1]:
                        new_playback_df.drop(i, inplace=True)

                class_NUM = {'ES': 0, 'GE': 1, 'CR': 2, 'BODY': 3, 'AG': 4, 'AT': 5, 'BB': 6, 'SD': 7}
                
                # ============================================================================================
                
                y_true, y_pred = [], []
                
                for i, (GT, PRED) in enumerate(zip(new_playback_df['gt'], new_playback_df['none_post_process'])):
                    try:
                        y_p = class_NUM[PRED]
                        
                        if y_p in GT:
                            y_true.append(y_p)
                            y_pred.append(y_p)    
                        else:
                            y_true.append(class_NUM[GT[0]])
                            y_pred.append(y_p)    
                    except:
                        ...
                
                new_y_true = tf.keras.utils.to_categorical(np.array(y_true), num_classes=len(class_NUM))
                new_y_pred = tf.keras.utils.to_categorical(np.array(y_pred), num_classes=len(class_NUM))
                
                # confusion_matrix 저장
                self.plot_confusion_matrix_video(y_true=y_true, y_pred=y_pred, 
                                                 class_dict=class_NUM, now_time=self.now_time,
                                                 plot_name='none_post_process')
        
                # classification_reports
                report = classification_report(y_true=y_true, y_pred=y_pred, 
                                               digits=class_NUM.values(), 
                                               output_dict=True)
                
                report_df = pd.DataFrame(report)
                
                report_df.to_csv(os.path.join('bin', 'exp_v', self.now_time, 'none_post_process_report.csv'))
                
                sen_spe_dict = {}
                
                # roc curve 생성 
                # sensitivity & specificity 생성
                self.ROC_multi_video(class_dict=class_NUM, 
                                     y_true=new_y_true, y_pred=new_y_pred, 
                                     now_time=self.now_time, plot_name='none_post_process')
                
                for i in range(len(class_NUM)):
                    sen, spe = self.sensitivity_specificity_per_class(y_true=y_true, y_pred=y_pred, y_class=i)
                    sen_spe_dict[list(class_NUM.keys())[i]] = {'sensitivity': sen, 'specificity': spe}
                
                sen_spe_df = pd.DataFrame(sen_spe_dict)
                sen_spe_df.to_csv(os.path.join('bin', 'exp_v', self.now_time, 'none_post_process_sen_spe.csv'))
                
                # ============================================================================================
                
                y_true, y_pred = [], []
                
                for i, (GT, PRED) in enumerate(zip(new_playback_df['gt'], new_playback_df['post_process'])):
                    try:
                        y_p = class_NUM[PRED]
                    
                        if y_p in GT:
                            y_true.append(y_p)
                            y_pred.append(y_p)    
                        else:
                            y_true.append(class_NUM[GT[0]])
                            y_pred.append(y_p)    
                    except:
                        ...
                
                new_y_true = tf.keras.utils.to_categorical(np.array(y_true), num_classes=len(class_NUM))
                new_y_pred = tf.keras.utils.to_categorical(np.array(y_pred), num_classes=len(class_NUM))
                
                # confusion_matrix 저장
                self.plot_confusion_matrix_video(y_true=y_true, y_pred=y_pred, 
                                                 class_dict=class_NUM, now_time=self.now_time,
                                                 plot_name='post_process')
        
                # classification_reports
                report = classification_report(y_true=y_true, y_pred=y_pred, 
                                               digits=class_NUM.values(), 
                                               output_dict=True)
                
                report_df = pd.DataFrame(report)
                
                report_df.to_csv(os.path.join('bin', 'exp_v', self.now_time, 'post_process_report.csv'))
                
                sen_spe_dict = {}
                
                # roc curve 생성 
                # sensitivity & specificity 생성
                self.ROC_multi_video(class_dict=class_NUM, 
                                     y_true=new_y_true, y_pred=new_y_pred, 
                                     now_time=self.now_time, plot_name='post_process')
                
                for i in range(len(class_NUM)):
                    sen, spe = self.sensitivity_specificity_per_class(y_true=y_true, y_pred=y_pred, y_class=i)
                    sen_spe_dict[list(class_NUM.keys())[i]] = {'sensitivity': sen, 'specificity': spe}
                
                sen_spe_df = pd.DataFrame(sen_spe_dict)
                sen_spe_df.to_csv(os.path.join('bin', 'exp_v', self.now_time, 'post_process_sen_spe.csv'))
                
                # ============================================================================================
                
                new_playback_df.to_excel(excel_writer=os.path.join('bin', 'exp_v', self.now_time, 'playback.xlsx'),
                                         sheet_name='Sheet1',
                                         na_rep='NaN',
                                         header=True,
                                         index=False,
                                         startrow=0,
                                         startcol=0,
                                         freeze_panes=(0,0)
                                         )
                
                # 코드 실행 총 시간 확인용
                end = int(time.time() - start)
                end_M = int(end // 60)
                end_S = int(end - (end_M * 60))
                e_time = f"{str(end_M).zfill(2)}:{str(end_S).zfill(2)}"
                
                # 비디오 재생 시간 확인용
                v_time = f"{str(self.cap_info['playback_time_M']).zfill(2)}:{str(self.cap_info['playback_time_S']).zfill(2)}"
                
                # 사용한 동영상, best_model 및 참고 가능한 정보 exp_v에 함께 저장 코드 추가 필요
                save_config = {'video': self.video_path,
                               'video_info': v_time,
                               'main_model': self.main_exp_path,
                               'sub0_model': self.sub0_exp_path,
                               'sub1_model': self.sub1_exp_path,
                               'total_time_required': e_time,
                               }
                
                # save parameter
                with open(os.path.join('bin', 'exp_v', self.now_time, 'config.yaml'), 'w') as f:
                    yaml.safe_dump(save_config, f)
                
                sys.exit("Frame done...")
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
            
            self.main_playback_cls_list.append(MAX_CLS)
            
            # 후처리를 적용하지 않은 추론 결과값 저장합니다.
            none_post_process.append(self.main_deq[-1])
            # 후처리를 적용한 결과값 저장합니다.
            post_process.append(MAX_CLS)
            
            # predict 결과 및 Max_CLS를 프레임에 draw
            frame = self.put_text_n_img(frame=frame,
                                        pred_deq=self.main_deq,
                                        cap_info=self.cap_info,
                                        MAX_CLS=MAX_CLS,
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
    parser.add_argument("--video_path", default=None, required=True, help='select video folder ex)01', type=str)
    parser.add_argument("--frame_space", default=20, required=False, help='define space between frames', type=int)
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
    
    # now_time
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day}-{d.hour}-{d.minute}-{d.second}"
    
    GPCM = GastricPositionClassificationMain(main_conf, sub0_conf, sub1_conf,
                                             args, now_time)
    GPCM.run()

    
if __name__ == '__main__':
    main()
    sys.exit('save done..!')