from .model.model import CreateGastricModel
from .utils.dataload import DataIOStream
from .utils.video_processing import VideoProcessing
from .utils.roc import AnalyzeROC
from .utils.matrix import PerformanceMeasurement

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
                 args: argparse.ArgumentParser(), learning_stage: bool, now_time: str) -> None:
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
        
        self.video_path = args.video_path
        self.frame_space = args.frame_space
        self.learning_stage = learning_stage
        self.now_time = now_time
    
    def run(self):
        # 코드 실행 시간 확인용
        start = time.time()
        
        # 적용될 동영상 로드, 프레임 정보 추출        
        video_path = os.path.join('bin', 'data', 'video', self.video_path)
        
        cap, cap_info = self.video_load(path=video_path)

        # 인코더 생성
        # exp 내 선택한 best_model 파일에 동영상 저장
        # encode = self.video_save(exp_path=self.exp_path,
                                #  cap_info=cap_info,
                                #  path=video_path)
        
        # 인코더 생성
        # exp_v 파일에 동영상 저장
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
        # 클래스 dict 생성
        main_class_dict, sub0_class_dict, sub1_class_dict = {}, {}, {}
        # 시간 흐름에 따른 MAX CLASS dict 생성
        main_playback_cls_list = []
        
        for i, CLS in enumerate(self.main_data_classes):
            main_status_lv[CLS[0]] = 0
            main_class_dict[CLS[0]] = i
        
        for i, CLS in enumerate(self.sub0_data_classes):
            if CLS[0] not in main_status_lv:
                main_status_lv[CLS[0]] = 0
            sub0_class_dict[CLS[0]] = i
        
        for i, CLS in enumerate(self.sub1_data_classes):
            if CLS[0] not in main_status_lv:
                main_status_lv[CLS[0]] = 0
            sub1_class_dict[CLS[0]] = i
        
        main_class_dict_reverse = {v:k for k,v in main_class_dict.items()}       
        sub0_class_dict_reverse = {v:k for k,v in sub0_class_dict.items()}       
        sub1_class_dict_reverse = {v:k for k,v in sub1_class_dict.items()}       
        
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
        
        #--------------------------------------------------------------------
        
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
                
                # 동영상 GT와 PRED값의 비교값을 엑셀로 저장
                
                # GT데이터 로드
                GT_xlsx_path = glob.glob(os.path.join(video_path, '*.xlsx'))
                GT_xlsx = pd.read_excel(GT_xlsx_path[0], engine='openpyxl')
                
                # GT데이터 분 -> 초 변환
                L_lst, S_lst, E_lst = [], [], []

                for L, S, E in zip(GT_xlsx['Location'], GT_xlsx['Start time'], GT_xlsx['End time']):
                    S_frametime = self.time2frame(S)
                    E_frametime = self.time2frame(E)
                    
                    L_lst.append(L)
                    S_lst.append(S_frametime)
                    E_lst.append(E_frametime)

                # GT데이터의 경우 초 단위로 라벨링이 구성되어 마지막 0.xx초에 대한 라벨링이 없음
                # 하지만 마지막 class와 동일한 라벨링으로 구성되어야 하기때문에 이 부분 수정
                E_lst.pop(-1)
                E_lst.append(len(playback_df))

                GT_info = {'Location': L_lst, 'Start_time': S_lst, 'End_time': E_lst}
                gt_df = pd.DataFrame(GT_info)
                
                new_frame_flow, gt = [], []

                # 영상 특성상 구간이 겹치는 클래스가 존재,
                # 따라서 GT를 중복으로 구성해 df 생성
                for df in gt_df.values:
                    for i in range(df[1], df[2]+1):
                        if not i in new_frame_flow:
                            new_frame_flow.append(i)
                            gt.append([df[0]])
                        else:
                            gt[i].append(df[0])
                
                # 영상 중간에 'NO'가 추론될 경우, 카메라 이동중 노이즈라 판단해 앞의 추론값을 이용함
                # 하지만 영상의 맨 앞,뒤의 경우, 신체외부를 뜻하는 실제 'NO'로 판단되야함
                # 따라서 인덱스 -1 값이 존재할 경우, 그값을 그대로 가져감 **이방법은 영상 뒤의 추론값 정리가능
                # 하지만 영상 앞은 처리가 불가능하기 때문에, df를 뒤집은후 위의 방법을 한번더 실행함
                # 중간의 값과 마지막값은 이미 처리했기때문에 앞의 값만 정리됨
                for i, elem in enumerate(playback_df['main']):
                    if elem == 'NO':
                        try:    
                            playback_df.loc[i, 'main'] = playback_df['main'][i-1]
                        except: 
                            ...
                
                playback_df = playback_df.sort_index(ascending=False).reset_index(drop=True)
                
                for i, elem in enumerate(playback_df['main']):
                    if elem == 'NO':
                        try:
                            playback_df.loc[i, 'main'] = playback_df['main'][i-1]
                        except: 
                            ...

                playback_df = playback_df.sort_index(ascending=False).reset_index(drop=True)
                
                # roc curve등 평가에 활용하기 위해선 중복된 GT값의 정리가 필요함
                # 정답의 경우, PRED값을 활용
                # 오답의 경우, 중복 class중 앞에 존재하는 GT활용
                result = []
                modified_gt = []
                
                for F, G, P in zip(new_frame_flow, gt, playback_df['main']):
                    if P in G:
                        result.append(True)
                        
                        modified_gt.append(P)
                    else:
                        result.append(False)

                        modified_gt.append(G[0])
                
                new_playback_df = pd.DataFrame({'frame_flow':new_frame_flow, 
                                                'gt': gt, 
                                                'modified_gt': modified_gt, 
                                                'pred':playback_df['main'], 
                                                'result': result
                                                })

                # GT상 'NO'는 신체 외부를 뜻함, 이는 평가에 속하지 않음
                # 따라서 이부분 프레임 제거 val[1] == gt
                for i, val in enumerate(new_playback_df.values):
                    if 'NO' in val[1]:
                        new_playback_df.drop(i, inplace=True)

                class_NUM = {'ES': 0, 'GE': 1, 'CR': 2, 'BODY': 3, 'AT': 4, 'AG': 5, 'BB': 6, 'SD': 7}
                
                y_true, y_pred = [], []
                
                for i, (GT, PRED) in enumerate(zip(new_playback_df['modified_gt'], new_playback_df['pred'])):
                    y_true.append(class_NUM[GT])
                    y_pred.append(class_NUM[PRED])
                
                new_y_true = tf.keras.utils.to_categorical(np.array(y_true), num_classes=len(class_NUM))
                new_y_pred = tf.keras.utils.to_categorical(np.array(y_pred), num_classes=len(class_NUM))
                
                sen_spe_dict = {}
                
                # roc curve 생성 
                # sensitivity & specificity 생성
                self.ROC_multi_video(class_dict=class_NUM, y_true=new_y_true, y_pred=new_y_pred, now_time=self.now_time)
                
                for i in range(len(class_NUM)):
                    sen, spe = self.sensitivity_specificity_per_class(y_true=y_true, y_pred=y_pred, y_class=i)
                    sen_spe_dict[list(class_NUM.keys())[i]] = {'sensitivity': sen, 'specificity': spe}
                
                sen_spe_df = pd.DataFrame(sen_spe_dict)
                sen_spe_df.to_csv(os.path.join('bin', 'exp_v', self.now_time, 'sen_spe.csv'))
                
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
                v_time = f"{str(cap_info['playback_time_M']).zfill(2)}:{str(cap_info['playback_time_S']).zfill(2)}"
                
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
                # 동영상 roi에 맞게 crop & 모델 input size에 맞게 resize & normalization
                frame, croped_frame = self.crop_frame(frame=frame, coord=coord,
                                                      new_width=main_img_shape,
                                                      new_height=main_img_shape)
                croped_frame = croped_frame / 255.0

            # 모든 프레임을 사용하지 않고, 1초에 2~3개의 프레임만 활용
            if count % self.frame_space == 0:
                main_pred = np.argmax(main_model.predict(croped_frame), axis=1)
                
                # 추론된 class값이 sub모델을 통해 2차 추론이 필요할경우, sub모델 추론
                # 추론된 값이 0 == 'ES'일 경우
                if main_pred == [0]:
                    sub0_pred = np.argmax(sub0_model.predict(croped_frame), axis=1)
                    main_deq.extend([sub0_class_dict_reverse[sub0_pred[0]]])
                
                # 추론된 값이 5 == 'DU'일 경우
                elif main_pred == [5]:
                    sub1_pred = np.argmax(sub1_model.predict(croped_frame), axis=1)
                    main_deq.extend([sub1_class_dict_reverse[sub1_pred[0]]])
                
                else:
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
            # # 추후 MAX_CLS가 NO로 잡힐 시, main_playback_cls_list에서 NO이전의 CLASS를 사용하도록 코드 수정 필요 
            # # ++만약 main_playback_cls_list의 맨 처음 elem이 NO일 경우 NO 표시
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
    parser.add_argument("--exp_sub0", default=None, required=False, help='ES-GE exp folder', type=str)
    parser.add_argument("--exp_sub1", default=None, required=False, help='SD-BB exp folder', type=str)
    parser.add_argument("--video_path", default=None, required=True, help='select video folder ex)01', type=str)
    parser.add_argument("--frame_space", default=20, required=False, help='define space between frames', type=int)
    parser.add_argument("--learning_stage", default=False, required=False, help='to show the learning stage train(True) or test(False)', type=bool)
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
                                             args, args.learning_stage, now_time)
    GPCM.run()

    
if __name__ == '__main__':
    main()
    sys.exit('save done..!')