from .utils.dataload import DataIOStream
from .utils.matrix import PerformanceMeasurement
from .utils.postprocessing import Postprocessing
from .utils.create_data import DataCreateStream
from .utils.roc import AnalyzeROC
from .model.model import CreatePillModel

import os
import yaml
import time
import datetime
import itertools
import shutil
import glob
import random
import argparse

import time
import datetime

import numpy as np
import pandas as pd

import cv2

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


#----------------------------------------------------------------------------

class PillClassificationMain(DataIOStream, CreatePillModel, PerformanceMeasurement, Postprocessing, AnalyzeROC, DataCreateStream):
    def __init__(self, conf: dict, exp_path: str) -> None:
        DataIOStream.__init__(self)        
        CreatePillModel.__init__(self)
        PerformanceMeasurement.__init__(self)
        Postprocessing.__init__(self)
        AnalyzeROC.__init__(self)
        DataCreateStream.__init__(self)
        
        self.model_name = conf['model_name']
        self.batch = conf['batch']
        self.epoch = conf['epoch']
        self.optimizer = conf['optimizer']
        self.es_patience = conf['es_patience']
        self.reduce_lr_patience = conf['reduce_lr_patience']
        self.reduce_lr_factor = conf['reduce_lr_factor']
        self.pill_count = 0
        self.classify_switch = True
        self.classify_sleep_time = 5
        self.thresh = 150

        self.exp_path = exp_path
        
    def screen_center(self, frame):
        fgbg = cv2.createBackgroundSubtractorMOG2()

        # 배경 제거
        fgmask = fgbg.apply(frame)

        # 이진화
        ret, threshed = cv2.threshold(fgmask, self.thresh, 255, cv2.THRESH_BINARY_INV)

        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel, iterations=2)

        # 물체 탐지
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 100:  # 물체의 크기 조건을 설정
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(opening, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
            if x + w > width * 0.6 :
                print("over center")
                
                return True
        
            else:
                return False
    
    def run(self):
        # 모델명에 맞는 input size를 로드합니다.
        img_shape = self.load_imgShape(model_name=self.model_name)
        
        # Test 혹은 Predict input 파일 Path 로드합니다.
        test_imgs_paths = glob.glob(os.path.join('bin', 'data', 'test', '*.png'))
        
        test_input_lst, test_id_lst = [], []
        
        for i, path in enumerate(test_imgs_paths):
            test_input_lst.append(path)
            # test_id_lst.append(path.split('/')[-1].split('_')[0])
            test_id_lst.append(path.split('\\')[-1].split('_')[0])
        
        # 데이터셋을 로드합니다.
        test_label_lst = tf.keras.utils.to_categorical(np.array(test_id_lst))
            
        test_dict = {'input_path': test_input_lst,
                     'input_label': test_label_lst,
                     'input_id': test_id_lst}
        
        # test_dict = self.resize_n_normalization(object_dict=test_dict, 
                                                    # new_width=img_shape, 
                                                    # new_height=img_shape)
        
        n_classes = len(np.unique(test_dict['input_id']))
        
        # 알약 정보 엑셀 로드
        pill_info = pd.read_excel(os.path.join('bin', 'data', 'pill_list.v1.0.4.xlsx'), engine='openpyxl')
        
        # 모델 경로를 초기화합니다.
        model_path = os.path.join('bin', 'exp', exp_path, 'test.tflite')
        
        # tf lite 모델을 로드합니다.
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']
        
        # 카메라 모듈을 초기화합니다.
        cap = cv2.VideoCapture(0)
        
        # 영상의 프레임 사이즈를 설정합니다.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
        
        if not cap.isOpened():
            print("카메라 연결에 실패했습니다.")
        
        else:
            print("카메라가 정상적으로 연결되었습니다.")
            
        while ret:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # 카메라 화면 출력
            cv2.imshow("Camera Feed", frame)
            
            if self.classify_switch == False:
                # 현재 시간 가져오기
                new_current_time = datetime.datetime.now()
                # 원하는 형식으로 시간을 포맷팅
                new_formatted_time = new_current_time.strftime("%Y-%m-%d %H:%M:%S")
                new_formatted_sec = int(new_current_time.strftime("%S"))
                
                if formatted_sec >= 60-switch_sleep_time:
                    switch_time = formatted_sec + switch_sleep_time
                    
                    if switch_time >= 60:
                        switch_time = switch_time - 60
                    else:
                        pass
                    
                elif formatted_sec < 60-switch_sleep_time:
                    switch_time = formatted_sec + switch_sleep_time
                
                if new_formatted_sec == switch_time:
                    switch = True
                    
            elif self.classify_switch == True:
            
                # 알약이 카메라 중심에 도달했는지 여부를 확인
                center_check = self.screen_center(frame=frame) # True or False
                
                if not center_check:
                    pass
                else:
                    # tf_lite 추론
                    image = np.expand_dims(input_data, axis=0).astype(np.float32)
                    interpreter.set_tensor(input_index, image)
                    
                    # 추론을 진행합니다.
                    interpreter.invoke()
                    
                    # Post-processing: remove batch dimension and find the digit with highest
                    # probability.
                    output = interpreter.tensor(output_index)
                    digit = np.argmax(output()[0]) # 추론 결과
                
                    count += 1
                    
                    # 추론을 마친뒤, 알약이 카메라를 벗어날 시간동안 추론을 멈춘다
                    # 현재 시간 가져오기
                    current_time = datetime.datetime.now()
                    # 원하는 형식으로 시간을 포맷팅
                    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    formatted_sec = int(current_time.strftime("%S"))
                    
                    self.classify_switch = False
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.realese()
        cv2.destroyAllWindows()
            

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default=None, required=True, help='choose exp folder ex)2022-12-22-15-55-46', type=str)
    args = parser.parse_args()
    
    with open(os.path.join('bin', 'exp', 'test', 'train.yaml'), 'r') as f:
        conf = yaml.safe_load(f)

    os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(conf['gpu'])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    print('=='*50)
    for item in conf:
        print(f'{item}: {conf[item]}')
    print('=='*50)

    PCM = PillClassificationMain(conf, 'test')
    PCM.run()

if __name__ == '__main__':
    main()