from .utils.dataload import DataIOStream
from .utils.matrix import PerformanceMeasurement
from .utils.postprocessing import Postprocessing
from .utils.roc import AnalyzeROC
from .model.model import CreateGastricModel


import os
import yaml
import time
import datetime
import itertools
import shutil
import random
import argparse

import numpy as np
import pandas as pd

import cv2

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample


#----------------------------------------------------------------------------

class GastricPositionClassificationMain(DataIOStream, CreateGastricModel, PerformanceMeasurement, Postprocessing, AnalyzeROC):
    def __init__(self, main_conf: dict, sub0_conf: dict, sub1_conf: dict, sub2_conf: dict, args: dict) -> None:
        DataIOStream.__init__(self)        
        CreateGastricModel.__init__(self)
        PerformanceMeasurement.__init__(self)
        Postprocessing.__init__(self)
        AnalyzeROC.__init__(self)
               
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
        
        self.exp_path = args.exp_main
    
    def run(self):
        name_lst = []
        for name in self.main_data_classes:
            name_lst.append(name[0])
        
        dataset_name = '_'.join(name_lst)
        
        category_fpath = os.path.join('bin', 'npy', dataset_name)
        
        img_shape = self.load_imgShape(model_name=self.main_model_name)
        
        # Create a folder based on the shape of the img
        shape_fpath = os.path.join(category_fpath, str(img_shape))
        
        # load dataset
        test = self.dataloader_test(path=shape_fpath)
        
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
        
        all_c_dict = {'ES':0, 'GE':1, 'CR':2, 'UB':3, 'MB':4, 'LB':5, 'AG':6, 'AT':7, 'BB':8, 'SD':9, 'NO':10}
        
        main_c_dict = {'ES':0, 'CR':1, 'BODY':2, 'AG':3, 'AT':4, 'DU':5, 'NO':6}
        main_c_rev_dict = {v:k for k,v in main_c_dict.items()}
        
        sub0_c_dict = {'ES':0, 'GE':1}
        sub0_c_rev_dict = {v:k for k,v in sub0_c_dict.items()}
        
        sub1_c_dict = {'BB':0, 'SD':1}
        sub1_c_rev_dict = {v:k for k,v in sub1_c_dict.items()}
        
        sub2_c_dict = {'UB':0, 'MB':1, 'LB':2}
        sub2_c_rev_dict = {v:k for k,v in sub2_c_dict.items()}
        
        n_classes = list(all_c_dict.keys())
        
        pred_lst = []
        real_label_lst = []
                
        # main_model을 통한 추론합니다.
        # 식도, 위몸통, 십이지장이 추론될 시, sub_model을 추론합니다.
        # 파일 경로를 통해 실제 라벨정보를 가져옵니다.
        for i, (input_img, input_path) in enumerate(zip(test['input_images'], test['input_path'])):
            
            input_img = input_img.reshape(1, input_img.shape[0], input_img.shape[1], input_img.shape[2])
            
            pred = main_model.predict(input_img)
            label = input_path.split('/')[4]
            
            if '-' in label:
                real_label_lst.append(label.split('-')[-1])
            else:
                real_label_lst.append(label)
            
            # ES가 추론될 경우,
            if np.argmax(pred, axis=1)[0] == 0:
                pred = sub0_model.predict(input_img)
                # ES가 추론될 경우,
                if np.argmax(pred, axis=1)[0] == 0:
                    pred_lst.append(sub0_c_rev_dict.get(0))
                # GE가 추론될 경우,
                elif np.argmax(pred, axis=1)[0] == 1:
                    pred_lst.append(sub0_c_rev_dict.get(1))
            # BODY가 추론될 경우,
            elif np.argmax(pred, axis=1)[0] == 2:
                pred = sub2_model.predict(input_img)
                # UB가 추론될 경우,
                if np.argmax(pred, axis=1)[0] == 0:
                    pred_lst.append(sub2_c_rev_dict.get(0))
                # MB가 추론될 경우,
                elif np.argmax(pred, axis=1)[0] == 1:
                    pred_lst.append(sub2_c_rev_dict.get(1))
                # LB가 추론될 경우,
                elif np.argmax(pred, axis=1)[0] == 2:
                    pred_lst.append(sub2_c_rev_dict.get(2))
            # DU가 추론될 경우,
            elif np.argmax(pred, axis=1)[0] == 5:
                pred = sub1_model.predict(input_img)
                # BB가 추론될 경우,
                if np.argmax(pred, axis=1)[0] == 0:
                    pred_lst.append(sub1_c_rev_dict.get(0))
                # SD가 추론될 경우,
                elif np.argmax(pred, axis=1)[0] == 1:
                    pred_lst.append(sub1_c_rev_dict.get(1))
            else:
                pred_lst.append(main_c_rev_dict.get(np.argmax(pred, axis=1)[0]))
        
        for i, pred in enumerate(real_label_lst):
            if pred == 'BODY':
                real_label_lst[i] = 'NO'
        
        pred_all = [all_c_dict[cls] for cls in pred_lst]
        label_all = [all_c_dict[cls] for cls in real_label_lst]
        
        P_ALL = tf.keras.utils.to_categorical(np.array(pred_all))
        L_ALL = tf.keras.utils.to_categorical(np.array(label_all))
        
        # configure bootstrap
        n_iterations = 2000
        n_size = int(len(test['input_images']) * 1)
        
        # run bootstrap
        auc_box = []
        acc_box = []
        f1s_box = []
        pc_box = []
        rc_box = []

        for _ in range(n_iterations):
            boot = {}            
            random_seed = random.randint(0,10000)
            
            boot['pred'] = resample(P_ALL, n_samples=n_size, replace=True, stratify=None, random_state=random_seed)
            boot['true'] = resample(L_ALL, n_samples=n_size, replace=True, stratify=None, random_state=random_seed)
            b_pred = np.argmax(boot['pred'], axis=1)
            b_true = np.argmax(boot['true'], axis=1)
            
            # auc
            auc = self.ROC_multi_ci(n_classes=len(n_classes), 
                                    y_true=boot['true'], y_pred=boot['pred'], 
                                    )
            # classification_reports
            report = classification_report(y_true=b_true, y_pred=b_pred, digits=n_classes, output_dict=True)
            
            auc_box.append(auc)
            acc_box.append(report['accuracy'])
            f1s_box.append(report['weighted avg']['f1-score'])
            pc_box.append(report['weighted avg']['precision'])
            rc_box.append(report['weighted avg']['recall'])
            
        # confidence intervals
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower_auc = max(0.0, np.percentile(auc_box, p))
        p = (alpha+((1.0-alpha)/2.0)) *100
        upper_auc = min(1.0, np.percentile(auc_box, p))
        print('auc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(np.array(auc_box)), np.median(auc_box), lower_auc, upper_auc))

        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower_acc = max(0.0, np.percentile(acc_box, p))
        p = (alpha+((1.0-alpha)/2.0)) *100
        upper_acc = min(1.0, np.percentile(acc_box, p))
        print('acc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(np.array(acc_box)), np.median(acc_box), lower_acc, upper_acc))
        
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower_f1s = max(0.0, np.percentile(f1s_box, p))
        p = (alpha+((1.0-alpha)/2.0)) *100
        upper_f1s = min(1.0, np.percentile(f1s_box, p))
        print('auc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(np.array(f1s_box)), np.median(f1s_box), lower_f1s, upper_f1s))
        
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower_pc = max(0.0, np.percentile(pc_box, p))
        p = (alpha+((1.0-alpha)/2.0)) *100
        upper_pc = min(1.0, np.percentile(pc_box, p))
        print('pc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(np.array(pc_box)), np.median(pc_box), lower_pc, upper_pc))
        
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower_rc = max(0.0, np.percentile(rc_box, p))
        p = (alpha+((1.0-alpha)/2.0)) *100
        upper_rc = min(1.0, np.percentile(rc_box, p))
        print('rc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(np.array(rc_box)), np.median(rc_box), lower_rc, upper_rc))
        
        with open(os.path.join('bin', 'exp', 'total', 'eval_ci.txt'), 'w') as f:
            f.write('auc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(np.array(auc_box)), np.median(auc_box), lower_auc, upper_auc)+'\n')
            f.write('acc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(np.array(acc_box)), np.median(acc_box), lower_acc, upper_acc)+'\n')
            f.write('f1s mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(np.array(f1s_box)), np.median(f1s_box), lower_f1s, upper_f1s)+'\n')
            f.write('pc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(np.array(pc_box)), np.median(pc_box), lower_pc, upper_pc)+'\n')
            f.write('rc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(np.array(rc_box)), np.median(rc_box), lower_rc, upper_rc)+'\n')

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_main", default=None, required=True, help='main exp folder', type=str)
    parser.add_argument("--exp_sub0", default=None, required=False, help='ES-GE exp folder', type=str)
    parser.add_argument("--exp_sub1", default=None, required=False, help='SD-BB exp folder', type=str)
    parser.add_argument("--exp_sub2", default=None, required=False, help='UB-LB exp folder', type=str)
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

    GPCM = GastricPositionClassificationMain(main_conf,
                                             sub0_conf,
                                             sub1_conf,
                                             sub2_conf, 
                                             args)
    GPCM.run()

if __name__ == '__main__':
    main()