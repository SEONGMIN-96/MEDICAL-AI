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
    def __init__(self, conf: dict, args: dict) -> None:
        DataIOStream.__init__(self)        
        CreateGastricModel.__init__(self)
        PerformanceMeasurement.__init__(self)
        Postprocessing.__init__(self)
        AnalyzeROC.__init__(self)
        
        self.data_classes = conf['data_classes']
        self.model_name = conf['model_name']
        self.batch = conf['batch']
        self.epoch = conf['epoch']
        self.optimizer = conf['optimizer']
        self.es_patience = conf['es_patience']
        self.reduce_lr_patience = conf['reduce_lr_patience']
        self.reduce_lr_factor = conf['reduce_lr_factor']

        self.exp_path = args.exp
    
    def run(self):
        name_lst = []
        for name in self.data_classes:
            name_lst.append(name[0])
        
        dataset_name = '_'.join(name_lst)
        
        category_fpath = os.path.join('bin', 'npy', dataset_name)
        
        img_shape = self.load_imgShape(model_name=self.model_name)
        
        # Create a folder based on the shape of the img
        shape_fpath = os.path.join(category_fpath, str(img_shape))
        
        # load dataset
        test = self.dataloader_test(path=shape_fpath)
        
        # load best_model.hdf5
        model = self.load_model(exp_path=self.exp_path)
        
        n_classes = len(test['input_class'])
        
        # compile & fit
        model.compile(optimizer=self.optimizer, metrics=['acc'],
                        loss='categorical_crossentropy')
        # # inference
        # eval_result = model.evaluate(test['input_images'], test['input_label'])
        
        y_pred = model.predict(test['input_images'])
        y_true = test['input_label']

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
            auc = self.ROC_multi_ci(n_classes=n_classes, 
                                    y_true=boot['true'], y_pred=boot['pred'], 
                                    )
            # classification_reports
            report = classification_report(y_true=b_true, y_pred=b_pred, digits=n_classes, output_dict=True)
            
            auc_box.append(auc)
            acc_box.append(report['accuracy'])
            f1s_box.append(report['weighted avg']['f1-score'])
            pc_box.append(report['weighted avg']['precision'])
            rc_box.append(report['weighted avg']['recall'])
            
            for i in range(len(name_lst)):
                sen, spe = self.sensitivity_specificity_per_class(y_true=b_true, y_pred=b_pred, y_class=i)
                
                try:
                    sen_box[name_lst[i]].append(sen)
                    spe_box[name_lst[i]].append(spe)
                except:
                    sen_box[name_lst[i]] = []
                    sen_box[name_lst[i]].append(sen)
                    spe_box[name_lst[i]] = []
                    spe_box[name_lst[i]].append(spe)
            
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
        
        with open(os.path.join('bin', 'exp', self.exp_path, 'eval_ci.txt'), 'w') as f:
            f.write('auc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(auc_box)), np.median(auc_box), np.std(auc_box), lower_auc, upper_auc)+'\n')
            f.write('acc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(acc_box)), np.median(acc_box), np.std(acc_box), lower_acc, upper_acc)+'\n')
            f.write('f1s mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(f1s_box)), np.median(f1s_box), np.std(f1s_box), lower_f1s, upper_f1s)+'\n')
            f.write('pc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(pc_box)), np.median(pc_box), np.std(pc_box), lower_pc, upper_pc)+'\n')
            f.write('rc mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (np.mean(np.array(rc_box)), np.median(rc_box), np.std(rc_box), lower_rc, upper_rc)+'\n')
            f.write('\n')
            
            for i in range(len(name_lst)):
                alpha = 0.95
                p = ((1.0-alpha)/2.0) * 100
                lower_sen = max(0.0, np.percentile(sen_box[name_lst[i]], p))
                p = (alpha+((1.0-alpha)/2.0)) *100
                upper_sen = min(1.0, np.percentile(sen_box[name_lst[i]], p))
                p = ((1.0-alpha)/2.0) * 100
                lower_spe = max(0.0, np.percentile(spe_box[name_lst[i]], p))
                p = (alpha+((1.0-alpha)/2.0)) *100
                upper_spe = min(1.0, np.percentile(spe_box[name_lst[i]], p))
                print('%s sen mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (name_lst[i], 
                                                                            np.mean(np.array(sen_box[name_lst[i]])), 
                                                                            np.median(sen_box[name_lst[i]]), 
                                                                            np.std(sen_box[name_lst[i]]), 
                                                                            lower_sen, upper_sen))
                print('%s spe mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (name_lst[i], 
                                                                            np.mean(np.array(spe_box[name_lst[i]])), 
                                                                            np.median(spe_box[name_lst[i]]), 
                                                                            np.std(spe_box[name_lst[i]]), 
                                                                            lower_spe, upper_spe))
                f.write('%s sen mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (name_lst[i], 
                                                                            np.mean(np.array(sen_box[name_lst[i]])), 
                                                                            np.median(sen_box[name_lst[i]]), 
                                                                            np.std(sen_box[name_lst[i]]), 
                                                                            lower_sen, upper_sen)+'\n')
                f.write('%s spe mean:%.4f, median:%.4f, std:%.4f, CI %.4f - %.4f' % (name_lst[i], 
                                                                            np.mean(np.array(spe_box[name_lst[i]])), 
                                                                            np.median(spe_box[name_lst[i]]), 
                                                                            np.std(spe_box[name_lst[i]]), 
                                                                            lower_spe, upper_spe)+'\n')
            


#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default=None, required=True, help='choose exp folder ex)2022-12-22-15-55-46', type=str)
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

    GPCM = GastricPositionClassificationMain(conf, args)
    GPCM.run()

if __name__ == '__main__':
    main()