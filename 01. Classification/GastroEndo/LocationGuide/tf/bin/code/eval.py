from .model.model import CreateGastricModel
from .utils.dataload import DataIOStream
from .utils.video_processing import VideoProcessing
from .utils.roc import AnalyzeROC
from .utils.matrix import PerformanceMeasurement
from .model.model import CreateGastricModel

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from sklearn.utils import resample

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
import random


#----------------------------------------------------------------------------

class GastricPositionClassificationMain(DataIOStream, 
                                        VideoProcessing, 
                                        CreateGastricModel, 
                                        AnalyzeROC, 
                                        PerformanceMeasurement,
):
    def __init__(self, 
                 conf: dict, 
                 args: argparse.ArgumentParser(), 
                 now_time: str,
        ) -> None:
        DataIOStream.__init__(self)    
        VideoProcessing.__init__(self)
        CreateGastricModel.__init__(self)
        AnalyzeROC.__init__(self)
        PerformanceMeasurement.__init__(self)
        
        # exp config 정의
        self.data_classes = conf['data_classes']
        self.model_name = conf['model_name']
        self.optimizer = conf['optimizer']
        self.exp_path = args.exp
        
        self.now_time = now_time
    
    def run(self):
        # 코드 실행 시간 확인용
        start = time.time()
        # Location Guide
        L_ALL_CLS = {'ES':0, 
                     'GE':1, 
                     'CR':2, 
                     'UB':3, 
                     'MB':4, 
                     'LB':5, 
                     'AG':6, 
                     'AT':7, 
                     'BB':8, 
                     'SD':9, 
                     'NO':10
        }
        L_00_CLS = {'ES':0, 
                    'CR':1, 
                    'BODY':2, 
                    'AG':3, 
                    'AT':4, 
                    'DU':5, 
                    'NO':6
        }
        L_01_CLS = {'ES':0, 
                    'GE':1
        }
        L_02_CLS = {'BB':0, 
                    'SD':1
        }
        L_03_CLS = {'UB':0, 
                    'MB':1, 
                    'LB':2
        }      
        
        # dataset load
        class_box = []
        for c in self.data_classes:
            class_box.append(c[0])
        npy_fname = '_'.join(class_box)
        # main_model의 img_shape 로드
        img_shape = self.load_imgShape(model_name=self.model_name)
        npy_fpath = os.path.join('bin', 'npy', npy_fname, str(img_shape))
        # load dataset
        test = self.dataloader_test(path=npy_fpath)
        #--------------------------------------------------------------------
        model = self.load_model(exp_path=self.exp_path)
        # compile & fit
        model.compile(optimizer=self.optimizer, 
                      metrics=['acc'],
                      loss='categorical_crossentropy'
        )
        y_pred = model.predict(test['input_images'])
        y_true = test['input_label']
        # draw ROC curve
        _ = self.ROC_multi(y_true=y_true,
                           y_pred=y_pred,
                           exp_path=self.exp_path,
                           class_box=class_box,
        )
        # draw cm
        self.plot_confusion_matrix(y_true=y_true,
                                   y_pred=y_pred,
                                   class_box=class_box,
                                   exp_path=self.exp_path,
        )
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
        for _ in range(n_iterations):
            boot = {}            
            random_seed = random.randint(0,10000)
            boot['pred'] = resample(y_pred, n_samples=n_size, replace=True, stratify=None, random_state=random_seed)
            boot['true'] = resample(y_true, n_samples=n_size, replace=True, stratify=None, random_state=random_seed)
            b_pred = np.argmax(boot['pred'], axis=1)
            b_true = np.argmax(boot['true'], axis=1)
            # auc
            auc = self.ROC_multi_ci(class_box=class_box, 
                                    y_true=boot['true'], 
                                    y_pred=boot['pred'], 
            )
            # classification_reports
            report = classification_report(y_true=b_true, y_pred=b_pred, digits=class_box, output_dict=True)
            auc_box.append(auc)
            acc_box.append(report['accuracy'])
            f1s_box.append(report['weighted avg']['f1-score'])
            pc_box.append(report['weighted avg']['precision'])
            rc_box.append(report['weighted avg']['recall'])
            for i in range(len(class_box)):
                sen, spe = self.sensitivity_specificity_per_class(y_true=b_true, y_pred=b_pred, y_class=i)
                try:
                    sen_box[class_box[i]].append(sen)
                    spe_box[class_box[i]].append(spe)
                except:
                    sen_box[class_box[i]] = []
                    sen_box[class_box[i]].append(sen)
                    spe_box[class_box[i]] = []
                    spe_box[class_box[i]].append(spe)
        # confidence intervals
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower_auc = max(0.0, np.percentile(auc_box, p))
        p = (alpha+((1.0-alpha)/2.0)) *100
        upper_auc = min(1.0, np.percentile(auc_box, p))
        print('auc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(auc_box)), np.median(auc_box), np.std(auc_box), lower_auc, upper_auc))
        # confidence intervals
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower_acc = max(0.0, np.percentile(acc_box, p))
        p = (alpha+((1.0-alpha)/2.0)) *100
        upper_acc = min(1.0, np.percentile(acc_box, p))
        print('acc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(acc_box)), np.median(acc_box), np.std(acc_box), lower_acc, upper_acc))
        # confidence intervals
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower_f1s = max(0.0, np.percentile(f1s_box, p))
        p = (alpha+((1.0-alpha)/2.0)) *100
        upper_f1s = min(1.0, np.percentile(f1s_box, p))
        print('f1s mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(f1s_box)), np.median(f1s_box), np.std(f1s_box), lower_f1s, upper_f1s))
        # confidence intervals
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower_pc = max(0.0, np.percentile(pc_box, p))
        p = (alpha+((1.0-alpha)/2.0)) *100
        upper_pc = min(1.0, np.percentile(pc_box, p))
        print('pc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(pc_box)), np.median(pc_box), np.std(pc_box), lower_pc, upper_pc))
        # confidence intervals
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
            f.write('pc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(pc_box)), np.median(pc_box), np.std(pc_box), lower_pc, upper_pc)+'\n')
            f.write('rc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(rc_box)), np.median(rc_box), np.std(rc_box), lower_rc, upper_rc)+'\n')
            f.write('\n')
            for i in range(len(class_box)):
                alpha = 0.95
                p = ((1.0-alpha)/2.0) * 100
                lower_sen = max(0.0, np.percentile(sen_box[class_box[i]], p))
                p = (alpha+((1.0-alpha)/2.0)) *100
                upper_sen = min(1.0, np.percentile(sen_box[class_box[i]], p))
                p = ((1.0-alpha)/2.0) * 100
                lower_spe = max(0.0, np.percentile(spe_box[class_box[i]], p))
                p = (alpha+((1.0-alpha)/2.0)) *100
                upper_spe = min(1.0, np.percentile(spe_box[class_box[i]], p))
                print('%s sen mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (class_box[i], 
                                                                            np.mean(np.array(sen_box[class_box[i]])), 
                                                                            np.median(sen_box[class_box[i]]), 
                                                                            np.std(sen_box[class_box[i]]), 
                                                                            lower_sen, upper_sen))
                print('%s spe mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (class_box[i], 
                                                                            np.mean(np.array(spe_box[class_box[i]])), 
                                                                            np.median(spe_box[class_box[i]]), 
                                                                            np.std(spe_box[class_box[i]]), 
                                                                            lower_spe, upper_spe))
                f.write('%s sen mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (class_box[i], 
                                                                            np.mean(np.array(sen_box[class_box[i]])), 
                                                                            np.median(sen_box[class_box[i]]), 
                                                                            np.std(sen_box[class_box[i]]), 
                                                                            lower_sen, upper_sen)+'\n')
                f.write('%s spe mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (class_box[i], 
                                                                            np.mean(np.array(spe_box[class_box[i]])), 
                                                                            np.median(spe_box[class_box[i]]), 
                                                                            np.std(spe_box[class_box[i]]), 
                                                                            lower_spe, upper_spe)+'\n')

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default=None, required=True, help='main exp folder', type=str)
    
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
    # now_time
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day}-{d.hour}-{d.minute}-{d.second}"
    GPCM = GastricPositionClassificationMain(conf,
                                             args, 
                                             now_time
    )
    GPCM.run()
    
if __name__ == '__main__':
    main()
    sys.exit('save done..!')