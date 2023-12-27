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

from decimal import Decimal


#----------------------------------------------------------------------------

class GastricPositionClassificationMain(DataIOStream, CreateGastricModel, 
                                        PerformanceMeasurement, Postprocessing, 
                                        AnalyzeROC):
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
        
        prob_box = []
        real_label_box = []
                
        # main_model을 통한 추론합니다.
        # 식도, 위몸통, 십이지장이 추론될 시, sub_model을 추론합니다.
        # 파일 경로를 통해 실제 라벨정보를 가져옵니다.
        for i, (input_img, input_path) in enumerate(zip(test['input_images'], test['input_path'])):
            
            input_img = input_img.reshape(1, input_img.shape[0], input_img.shape[1], input_img.shape[2])
            main_pred = main_model.predict(input_img)
            label = input_path.split('/')[4]
            
            if '-' in label:
                real_label_box.append(label.split('-')[-1])
            else:
                real_label_box.append(label)
            
            # 서브 모델의 유무로 인해, probablity의 보정이 항상 요구된다.
            # 서브 모델이 사용 시 
            # 서브 모델의 probablity와 메인 모델의 probablity를 곱해 probablity의 총합이 1이 될 수 있도록 맞춰준다.
            # 서브모델이 사용되지 않을 시
            # probablity * (1 / 서브 모델 클래스 수)를 해당 probablity에 적용해서 probablity의 총합이 1이 될 수 있도록 맞춰준다.
                                   
            # ES가 추론될 경우,
            if np.argmax(main_pred, axis=1)[0] == 0:
                sub_pred = sub0_model.predict(input_img)
                simbol = 'sub0'
                                    
            # BODY가 추론될 경우,
            elif np.argmax(main_pred, axis=1)[0] == 2:
                sub_pred = sub2_model.predict(input_img)
                simbol = 'sub2'
                
            # DU가 추론될 경우,
            elif np.argmax(main_pred, axis=1)[0] == 5:
                sub_pred = sub1_model.predict(input_img)
                simbol = 'sub1'
                
            else:
                simbol = 'main'
            
            # 추론된 클래스에 맞게, probablity를 수정합니다.
            if simbol == 'main':
                es_prob = main_pred[0][main_c_dict['ES']]
                body_prob = main_pred[0][main_c_dict['BODY']]
                du_prob = main_pred[0][main_c_dict['DU']]
                
                main_pred = np.delete(main_pred,[main_c_dict['ES'],
                                                 main_c_dict['BODY'],
                                                 main_c_dict['DU']] ,axis=1)
                
                # ES prob 조정
                main_pred = np.insert(main_pred, all_c_dict['ES'],es_prob/len(sub0_c_dict),axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['GE'],es_prob/len(sub0_c_dict),axis=1)
                # BODY prob 조정
                main_pred = np.insert(main_pred, all_c_dict['UB'],body_prob/len(sub2_c_dict),axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['MB'],body_prob/len(sub2_c_dict),axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['LB'],body_prob/len(sub2_c_dict),axis=1)   
                # DU prob 조정
                main_pred = np.insert(main_pred, all_c_dict['BB'],du_prob/len(sub1_c_dict),axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['SD'],du_prob/len(sub1_c_dict),axis=1)   
                
            elif simbol == 'sub0':
                es_prob = main_pred[0][main_c_dict['ES']] * sub_pred[0][sub0_c_dict['ES']]
                ge_prob = main_pred[0][main_c_dict['ES']] * sub_pred[0][sub0_c_dict['GE']]
                body_prob = main_pred[0][main_c_dict['BODY']]
                du_prob = main_pred[0][main_c_dict['DU']]
                
                main_pred = np.delete(main_pred,[main_c_dict['ES'],
                                                 main_c_dict['BODY'],
                                                 main_c_dict['DU']] ,axis=1)

                # ES prob 조정
                main_pred = np.insert(main_pred, all_c_dict['ES'],es_prob,axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['GE'],ge_prob,axis=1)
                # BODY prob 조정
                main_pred = np.insert(main_pred, all_c_dict['UB'],body_prob/len(sub2_c_dict),axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['MB'],body_prob/len(sub2_c_dict),axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['LB'],body_prob/len(sub2_c_dict),axis=1)   
                # DU prob 조정
                main_pred = np.insert(main_pred, all_c_dict['BB'],du_prob/len(sub1_c_dict),axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['SD'],du_prob/len(sub1_c_dict),axis=1)   
                
            elif simbol == 'sub2':
                es_prob = main_pred[0][main_c_dict['ES']]
                ub_prob = main_pred[0][main_c_dict['BODY']] * sub_pred[0][sub2_c_dict['UB']]
                mb_prob = main_pred[0][main_c_dict['BODY']] * sub_pred[0][sub2_c_dict['MB']]
                lb_prob = main_pred[0][main_c_dict['BODY']] * sub_pred[0][sub2_c_dict['LB']]
                du_prob = main_pred[0][main_c_dict['DU']]
                
                main_pred = np.delete(main_pred,[main_c_dict['ES'],
                                                 main_c_dict['BODY'],
                                                 main_c_dict['DU']] ,axis=1)
                                
                # ES prob 조정
                main_pred = np.insert(main_pred, all_c_dict['ES'],es_prob/len(sub0_c_dict),axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['GE'],es_prob/len(sub0_c_dict),axis=1)
                # BODY prob 조정
                main_pred = np.insert(main_pred, all_c_dict['UB'],ub_prob,axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['MB'],mb_prob,axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['LB'],lb_prob,axis=1)   
                # DU prob 조정
                main_pred = np.insert(main_pred, all_c_dict['BB'],du_prob/len(sub1_c_dict),axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['SD'],du_prob/len(sub1_c_dict),axis=1)   
                
            elif simbol == 'sub1':
                es_prob = main_pred[0][main_c_dict['ES']]
                body_prob = main_pred[0][main_c_dict['BODY']]
                bb_prob = main_pred[0][main_c_dict['DU']] * sub_pred[0][sub1_c_dict['BB']]
                sd_prob = main_pred[0][main_c_dict['DU']] * sub_pred[0][sub1_c_dict['SD']]
                
                main_pred = np.delete(main_pred,[main_c_dict['ES'],
                                                 main_c_dict['BODY'],
                                                 main_c_dict['DU']] ,axis=1)
                                                
                # ES prob 조정
                main_pred = np.insert(main_pred, all_c_dict['ES'],es_prob/len(sub0_c_dict),axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['GE'],es_prob/len(sub0_c_dict),axis=1)
                # BODY prob 조정
                main_pred = np.insert(main_pred, all_c_dict['UB'],body_prob/len(sub2_c_dict),axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['MB'],body_prob/len(sub2_c_dict),axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['LB'],body_prob/len(sub2_c_dict),axis=1)   
                # DU prob 조정
                main_pred = np.insert(main_pred, all_c_dict['BB'],bb_prob,axis=1)   
                main_pred = np.insert(main_pred, all_c_dict['SD'],sd_prob,axis=1)   
            
            prob_box.append(main_pred[0])
            
        for i, label in enumerate(real_label_box):
            if label == 'BODY':
                real_label_box[i] = 'NO'
        
        pred_all = np.argmax(np.array(prob_box), axis=1)
        label_all = [[all_c_dict[cls]] for cls in real_label_box]
        
        P_ALL = np.array(prob_box)
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
        sen_box = {}
        spe_box = {}
        
        # cofusion matrix
        self.plot_confusion_matrix_all(y_true=label_all, y_pred=pred_all, class_lst=list(all_c_dict.keys()), exp_path=self.exp_path)
        
        # roc curve
        self.ROC_multi_all(n_classes=n_classes, 
                           y_true=L_ALL, y_pred=P_ALL, 
                           exp_path=self.exp_path, 
                           name_lst=list(all_c_dict.keys()))
        
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
            
            for i in range(len(all_c_dict.keys())):
                sen, spe = self.sensitivity_specificity_per_class(y_true=b_true, y_pred=b_pred, y_class=i)
                
                try:
                    sen_box[list(all_c_dict.keys())[i]].append(sen)
                    spe_box[list(all_c_dict.keys())[i]].append(spe)
                except:
                    sen_box[list(all_c_dict.keys())[i]] = []
                    sen_box[list(all_c_dict.keys())[i]].append(sen)
                    spe_box[list(all_c_dict.keys())[i]] = []
                    spe_box[list(all_c_dict.keys())[i]].append(spe)
            
        # confidence intervals
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower_auc = max(0.0, np.percentile(auc_box, p))
        p = (alpha+((1.0-alpha)/2.0)) *100
        upper_auc = min(1.0, np.percentile(auc_box, p))
        print('auc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(auc_box)), np.median(auc_box), np.std(auc_box), lower_auc, upper_auc))

        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower_acc = max(0.0, np.percentile(acc_box, p))
        p = (alpha+((1.0-alpha)/2.0)) *100
        upper_acc = min(1.0, np.percentile(acc_box, p))
        print('acc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(acc_box)), np.median(acc_box), np.std(acc_box), lower_acc, upper_acc))
        
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower_f1s = max(0.0, np.percentile(f1s_box, p))
        p = (alpha+((1.0-alpha)/2.0)) *100
        upper_f1s = min(1.0, np.percentile(f1s_box, p))
        print('f1s mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(f1s_box)), np.median(f1s_box), np.std(f1s_box), lower_f1s, upper_f1s))
        
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower_pc = max(0.0, np.percentile(pc_box, p))
        p = (alpha+((1.0-alpha)/2.0)) *100
        upper_pc = min(1.0, np.percentile(pc_box, p))
        print('pc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(pc_box)), np.median(pc_box), np.std(pc_box), lower_pc, upper_pc))
        
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower_rc = max(0.0, np.percentile(rc_box, p))
        p = (alpha+((1.0-alpha)/2.0)) *100
        upper_rc = min(1.0, np.percentile(rc_box, p))
        print('rc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(rc_box)), np.median(rc_box), np.std(rc_box), lower_rc, upper_rc))
        
        with open(os.path.join('bin', 'exp', 'sample', 'eval_ci.txt'), 'w') as f:
            f.write('auc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(auc_box)), np.median(auc_box), np.std(auc_box), lower_auc, upper_auc)+'\n')
            f.write('acc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(acc_box)), np.median(acc_box), np.std(acc_box), lower_acc, upper_acc)+'\n')
            f.write('f1s mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(f1s_box)), np.median(f1s_box), np.std(f1s_box), lower_f1s, upper_f1s)+'\n')
            f.write('pc mean:%.3f, median:%.3f, std:%.3f, CI %.3f - %.3f' % (np.mean(np.array(pc_box)), np.median(pc_box), np.std(pc_box), lower_pc, upper_pc)+'\n')
            f.write('rc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(rc_box)), np.median(rc_box), np.std(rc_box), lower_rc, upper_rc)+'\n')
            f.write('\n')
            
            for i in range(len(all_c_dict.keys())):
                alpha = 0.95
                p = ((1.0-alpha)/2.0) * 100
                lower_sen = max(0.0, np.percentile(sen_box[list(all_c_dict.keys())[i]], p))
                p = (alpha+((1.0-alpha)/2.0)) *100
                upper_sen = min(1.0, np.percentile(sen_box[list(all_c_dict.keys())[i]], p))
                p = ((1.0-alpha)/2.0) * 100
                lower_spe = max(0.0, np.percentile(spe_box[list(all_c_dict.keys())[i]], p))
                p = (alpha+((1.0-alpha)/2.0)) *100
                upper_spe = min(1.0, np.percentile(spe_box[list(all_c_dict.keys())[i]], p))
                print('%s sen mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (list(all_c_dict.keys())[i], 
                                                                            np.mean(np.array(sen_box[list(all_c_dict.keys())[i]])), 
                                                                            np.median(sen_box[list(all_c_dict.keys())[i]]), 
                                                                            np.std(sen_box[list(all_c_dict.keys())[i]]), 
                                                                            lower_sen, upper_sen))
                print('%s spe mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (list(all_c_dict.keys())[i], 
                                                                            np.mean(np.array(spe_box[list(all_c_dict.keys())[i]])), 
                                                                            np.median(spe_box[list(all_c_dict.keys())[i]]), 
                                                                            np.std(spe_box[list(all_c_dict.keys())[i]]), 
                                                                            lower_spe, upper_spe))
                f.write('%s sen mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (list(all_c_dict.keys())[i], 
                                                                            np.mean(np.array(sen_box[list(all_c_dict.keys())[i]])), 
                                                                            np.median(sen_box[list(all_c_dict.keys())[i]]), 
                                                                            np.std(sen_box[list(all_c_dict.keys())[i]]), 
                                                                            lower_sen, upper_sen)+'\n')
                f.write('%s spe mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (list(all_c_dict.keys())[i], 
                                                                            np.mean(np.array(spe_box[list(all_c_dict.keys())[i]])), 
                                                                            np.median(spe_box[list(all_c_dict.keys())[i]]), 
                                                                            np.std(spe_box[list(all_c_dict.keys())[i]]), 
                                                                            lower_spe, upper_spe)+'\n')
            
            # f.write('auc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(np.array(auc_box)), np.median(auc_box), lower_auc, upper_auc)+'\n')
            # f.write('acc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(np.array(acc_box)), np.median(acc_box), lower_acc, upper_acc)+'\n')
            # f.write('f1s mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(np.array(f1s_box)), np.median(f1s_box), lower_f1s, upper_f1s)+'\n')
            # f.write('pc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(np.array(pc_box)), np.median(pc_box), lower_pc, upper_pc)+'\n')
            # f.write('rc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(np.array(rc_box)), np.median(rc_box), lower_rc, upper_rc)+'\n')

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