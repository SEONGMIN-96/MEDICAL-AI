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

class GastricPositionClassificationMain(DataIOStream, 
                                        VideoProcessing, 
                                        CreateGastricModel, 
                                        AnalyzeROC, 
                                        PerformanceMeasurement
):
    def __init__(self, 
                 main_conf: dict, 
                 sub0_conf: dict, 
                 sub1_conf: dict,
                 sub2_conf: dict,
                 args: argparse.ArgumentParser(), 
                 now_time: str,
                 video_path: str,
        ) -> None:
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
        
        self.video_path = video_path
        self.frame_space = args.frame_space
        self.now_time = now_time
        # 동영상 저장할지말지
        self.encoder_bool = False
    
    def run(self):
        # 코드 실행 시간 확인용
        start = time.time()
        # 적용될 동영상 로드, 프레임 정보 추출        
        # video_path = os.path.join('bin', 'data', 'video', self.video_path)
        video_path = self.video_path
        cap, cap_info = self.video_load(path=video_path)
        # 인코더 생성
        # exp_v 파일에 동영상 저장
        if self.encoder_bool == True:
            encode = self.video_save_to_exp_v(now_time=self.now_time,
                                            cap_info=cap_info,
                                            path=video_path)
        # 동영상 roi 로드
        coord_path = glob.glob(os.path.join(video_path, '*.txt'))
        with open(coord_path[0]) as f:
            coord = f.readline()
        coord = coord.split(',')
        # 적용될 위내시경 로고 이미지 로드
        logo_paths = glob.glob(os.path.join('bin', 'data', 'logo', 'new_logo', '*.png'))
        logo_box = {}
        for path in logo_paths:
            logo_fname = path.split('/')[-1].split('.')[0]
            img = cv2.imread(path)
            x, y, w, h = 0, 0, 1800, 1800
            img = img[y:y+h, x:x+w]
            img = cv2.resize(src=img, 
                             dsize=(320, 320))
            logo_box[logo_fname] = img
        # 메인 모델의 predict 저장 데크 생성
        main_deq = deque(maxlen=13)
        # Location Guide
        L_ALL_CLS = {"ES":0, 
                    "GE":1, 
                    "CR":2, 
                    "UB":3, 
                    "MB":4, 
                    "LB":5, 
                    "AG":6, 
                    "AT":7, 
                    "BB":8, 
                    "SD":9, 
                    "NO":10
        }
        L_00_CLS = {"ES":0, 
                    "CR":1, 
                    "BODY":2, 
                    "AG":3, 
                    "AT":4, 
                    "DU":5, 
                    "NO":6
        }
        L_01_CLS = {"ES":0, 
                    "GE":1
        }
        L_02_CLS = {"BB":0, 
                    "SD":1
        }
        L_03_CLS = {"UB":0, 
                    "MB":1, 
                    "LB":2
        }      
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
        # sub1 모델 로드
        sub2_model = self.load_model(exp_path=self.sub2_exp_path)
        # compile & fit
        sub2_model.compile(optimizer=self.sub2_optimizer, metrics=['acc'],
                           loss='categorical_crossentropy')
        #--------------------------------------------------------------------
        # GT 데이터 로드
        GT_xlsx_path = glob.glob(os.path.join(self.video_path, '*.xlsx'))
        GT_xlsx = pd.read_excel(GT_xlsx_path[0], engine='openpyxl')
        # GT 데이터는 분(min)으로 구성되어, 초(sec)로 변환합니다
        L_lst, S_lst, E_lst = [], [], []
        for i, (L, S, E) in enumerate(zip(GT_xlsx['Location'], GT_xlsx['Start time'], GT_xlsx['End time'])):
            S_frametime = self.time2frame(S)
            E_frametime = self.time2frame(E)
            L_lst.append(L)
            S_lst.append(S_frametime)
            E_lst.append(E_frametime)
        # GT 데이터상, 정확한 프레임 구간까지 시간이 정의되지않음
        # 동영상 데이터의 length값으로 GT의 마지막 구간을 재설정합니다
        E_lst.pop(-1)
        E_lst.append(int(cap_info['length']))
        GT_info = pd.DataFrame({'Location': L_lst, 'Start_time': S_lst, 'End_time': E_lst})
        frame_flow, gt = [], []
        for (L, S, E) in zip(GT_info['Location'], GT_info['Start_time'], GT_info['End_time']):
            for i in range(S, E):
                if i not in frame_flow:
                    frame_flow.append(i)
                    gt.append([L])
                else:
                    gt[i].append(L)
        # 진행된 프레임 카운트합니다
        count = 0
        # 적용될 동영상 길이의 tqdm 생성합니다
        pbar = tqdm(total=cap_info['length'])
        # 각 모델 probability 저장합니다
        frame_num_box = []
        m0_pred_box = []
        m1_pred_box = []
        m2_pred_box = []
        m3_pred_box = []
        voting_box = []
        non_pp_box = []
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
                probability_df = pd.DataFrame({"frame_NUM":frame_num_box,
                                               "primary":m0_pred_box,
                                               "secondary_ES":m1_pred_box,
                                               "secondary_DU":m2_pred_box,
                                               "secondary_BODY":m3_pred_box,
                                               }
                )       
                video_fname = video_path.split('/')[-1]
                probability_df.to_excel(os.path.join('bin', 'exp_v', self.now_time, f'{video_fname}_probability.xlsx'), 
                                        index=False,
                )
                # 정량적 평가 진행합니다.
                y_truePP, y_predPP, = [], []
                y_trueNPP, y_predNPP, = [], []
                for i, (yt_box, predPP, predNPP) in enumerate(zip(gt, voting_box, non_pp_box)):
                    if predPP not in yt_box:
                        y_truePP.append(L_ALL_CLS[yt_box[-1]])
                    elif predPP in yt_box:
                        y_truePP.append(L_ALL_CLS[predPP])
                    y_predPP.append(L_ALL_CLS[predPP])
                    if predNPP not in yt_box:
                        y_trueNPP.append(L_ALL_CLS[yt_box[-1]])
                    elif predNPP in yt_box:
                        y_trueNPP.append(L_ALL_CLS[predNPP])
                    y_predNPP.append(L_ALL_CLS[predNPP])
                del_idx0, del_idx1 = [], []
                # NO는 평가에서 제외하기위해 뺍니다.
                for i, (yt, yp) in enumerate(zip(y_truePP, y_predPP)):
                    if yt == 10 or yp == 10:
                        del_idx0.append(i)
                for i in sorted(del_idx0, reverse=True):
                    del y_truePP[i]
                    del y_predPP[i]
                for i, (yt, yp) in enumerate(zip(y_trueNPP, y_predNPP)):
                    if yt == 10 or yp == 10:
                        del_idx1.append(i)
                for i in sorted(del_idx1, reverse=True):
                    del y_trueNPP[i]
                    del y_predNPP[i]
                # NO가 제외된 CLS_list 정의
                L_ALL_CLS_without_NO = L_ALL_CLS
                del L_ALL_CLS_without_NO["NO"]
                # # true, pred를 범주형 값으로 변환합니다.
                # y_true2CA = tf.keras.utils.to_categorical(np.array(y_true), num_classes=len(L_ALL_CLS_without_NO))
                # y_pred2CA = tf.keras.utils.to_categorical(np.array(y_pred), num_classes=len(L_ALL_CLS_without_NO))
                #  confusion_matrix를 저장합니다.
                self.plot_confusion_matrix_video(y_true=y_truePP, 
                                                 y_pred=y_predPP, 
                                                 class_dict=L_ALL_CLS_without_NO, 
                                                 now_time=self.now_time,
                                                 plot_name='PP'
                )
                self.plot_confusion_matrix_video(y_true=y_trueNPP, 
                                                 y_pred=y_predNPP, 
                                                 class_dict=L_ALL_CLS_without_NO, 
                                                 now_time=self.now_time,
                                                 plot_name='NPP'
                )
                # classification_reports를 저장합니다.
                PP_report = pd.DataFrame(classification_report(y_true=y_truePP, 
                                                            y_pred=y_predPP, 
                                                            digits=L_ALL_CLS_without_NO.values(), 
                                                            output_dict=True
                ))
                PP_report.to_csv(os.path.join('bin', 'exp_v', self.now_time, 'PP_report.csv'))
                NPP_report = pd.DataFrame(classification_report(y_true=y_trueNPP, 
                                                            y_pred=y_predNPP, 
                                                            digits=L_ALL_CLS_without_NO.values(), 
                                                            output_dict=True
                ))
                NPP_report.to_csv(os.path.join('bin', 'exp_v', self.now_time, 'NPP_report.csv'))
                # sensitivity & specificity 생성합니다.
                PP_sen_spe, NPP_sen_spe = {}, {}
                for i in range(len(L_ALL_CLS_without_NO)):
                    sen, spe = self.sensitivity_specificity_per_class(y_true=y_truePP, 
                                                                      y_pred=y_predPP, 
                                                                      y_class=i
                    )
                    PP_sen_spe[list(L_ALL_CLS_without_NO.keys())[i]] = {'sensitivity': sen, 'specificity': spe}
                pd.DataFrame(PP_sen_spe).to_csv(os.path.join('bin', 'exp_v', self.now_time, 'PP_sen_spe.csv'))
                for i in range(len(L_ALL_CLS_without_NO)):
                    sen, spe = self.sensitivity_specificity_per_class(y_true=y_trueNPP, 
                                                                      y_pred=y_predNPP, 
                                                                      y_class=i
                    )
                    NPP_sen_spe[list(L_ALL_CLS_without_NO.keys())[i]] = {'sensitivity': sen, 'specificity': spe}
                pd.DataFrame(NPP_sen_spe).to_csv(os.path.join('bin', 'exp_v', self.now_time, 'NPP_sen_spe.csv'))
                break
            else:
                # 동영상 roi에 맞게 crop & 모델 input size에 맞게 resize & normalization
                frame, croped_frame = self.crop_frame(frame=frame, coord=coord,
                                                      new_width=main_img_shape,
                                                      new_height=main_img_shape,
                )
                croped_frame = croped_frame / 255.0
            # 모든 프레임을 사용하지 않고, 1초에 2~3개의 프레임만 활용
            if count % self.frame_space == 0:
                pred = main_model.predict(croped_frame)
                m0_pred_box.append(pred[0])
                pred_argmax = np.argmax(pred, axis=1)
                if pred_argmax == [0]: # 추론된 값이 0 == 'ES'일 경우
                    pred = sub0_model.predict(croped_frame)
                    # probability 저장
                    frame_num_box.append(count)
                    m1_pred_box.append(pred[0])
                    m2_pred_box.append(0)
                    m3_pred_box.append(0)
                    pred_argmax = np.argmax(pred, axis=1) 
                    result = list(L_01_CLS.keys())[pred_argmax[0]]
                    main_deq.append(result)
                elif pred_argmax == [5]: # 추론된 값이 5 == 'DU'일 경우
                    pred = sub1_model.predict(croped_frame)
                    # probability 저장
                    frame_num_box.append(count)
                    m1_pred_box.append(0)
                    m2_pred_box.append(pred[0])
                    m3_pred_box.append(0)
                    pred_argmax = np.argmax(pred, axis=1) 
                    result = list(L_02_CLS.keys())[pred_argmax[0]]
                    main_deq.append(result)
                elif pred_argmax == [2]: # 추론된 값이 2 == 'BODY'일 경우
                    pred = sub2_model.predict(croped_frame)
                    # probability 저장
                    frame_num_box.append(count)
                    m1_pred_box.append(0)
                    m2_pred_box.append(0)
                    m3_pred_box.append(pred[0])
                    pred_argmax = np.argmax(pred, axis=1) 
                    result = list(L_03_CLS.keys())[pred_argmax[0]]
                    main_deq.append(result)
                else:
                    result = list(L_00_CLS.keys())[pred_argmax[0]]
                    # probability 저장
                    frame_num_box.append(count)
                    m1_pred_box.append(0)
                    m2_pred_box.append(0)
                    m3_pred_box.append(0)
                    main_deq.append(result)
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
            # after post-processing 
            voting_box.append(MAX_CLS)
            non_pp_box.append(result)
            if self.encoder_bool == True: # 프레임 인코딩
                # predict 결과 및 Max_CLS를 프레임에 draw
                frame = self.putText_in_frame_v2(frame=frame,
                                                pred_deq=main_deq,
                                                MAX_CLS=MAX_CLS,
                                                cap_info=cap_info,
                                                pt1=220, pt2=321,
                                                pt3=60, pt4=640,
                )
                frame = self.bit_operation_s0(board=frame,
                                            logo=logo_box[MAX_CLS],
                                            pt1=310, pt2=120
                )
                encode.write(frame)
            pbar.update(1)
            count += 1
            

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_main", default=None, required=True, help='main exp folder', type=str)
    parser.add_argument("--exp_sub0", default=None, required=False, help='ES-GE exp folder', type=str)
    parser.add_argument("--exp_sub1", default=None, required=False, help='SD-BB exp folder', type=str)
    parser.add_argument("--exp_sub2", default=None, required=False, help='BODY exp folder', type=str)
    parser.add_argument("--frame_space", default=20, required=False, help='define space between frames', type=int)
    args = parser.parse_args()
    with open(os.path.join('bin', 'exp', args.exp_main, 'train.yaml'), 'r') as f:
        main_conf = yaml.safe_load(f)
    with open(os.path.join('bin', 'exp', args.exp_sub0, 'train.yaml'), 'r') as f:
        sub0_conf = yaml.safe_load(f)
    with open(os.path.join('bin', 'exp', args.exp_sub1, 'train.yaml'), 'r') as f:
        sub1_conf = yaml.safe_load(f)
    with open(os.path.join('bin', 'exp', args.exp_sub2, 'train.yaml'), 'r') as f:
        sub2_conf = yaml.safe_load(f)
    # video 경로 로드
    video_paths = glob.glob(os.path.join('bin', 'data', 'video', '*.dcm'))
    gpu_num = "0"
    for video_path in video_paths:
        print("video:", video_path)
        os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
        # os.environ["CUDA_VISIBLE_DEVICES"]=str(main_conf['gpu'])
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num)
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
        now_time = f"{gpu_num}-{d.year}-{d.month}-{d.day}-{d.hour}-{d.minute}-{d.second}"
        GPCM = GastricPositionClassificationMain(main_conf, 
                                                 sub0_conf, 
                                                 sub1_conf, 
                                                 sub2_conf,
                                                 args, 
                                                 now_time, 
                                                 video_path
        )
        GPCM.run()

        # 현재 그래프와 상태 초기화
        tf.keras.backend.clear_session()
    
if __name__ == '__main__':
    main()
    sys.exit('save done..!')