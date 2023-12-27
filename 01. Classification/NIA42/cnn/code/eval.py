from .utils.dataload import DataIOStream
from .utils.callback import CallBack
from .utils.roc import AnalyzeROC
from .utils.create_data import DataCreateStream
from .utils.matrix import PerformanceMeasurement
from .model.model import CreateOTEModel
from .model.model_v2 import CreateOTEModel_v2
from tqdm import tqdm

import tempfile
import os
import yaml
import time
import datetime
import glob
import random

import numpy as np
import pandas as pd
import pickle
import cv2
import argparse

import matplotlib.pyplot as plt

# from mlflow.tensorflow import autolog, save_model
# from mlflow.models import infer_signature

import tensorflow_model_optimization as tfmot
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#----------------------------------------------------------------------------

class OTEClassificationMain(DataIOStream, CreateOTEModel_v2, CallBack, DataCreateStream, PerformanceMeasurement, AnalyzeROC):
    def __init__(self, conf: dict, args: dict,) -> None:
        DataIOStream.__init__(self)        
        CreateOTEModel_v2.__init__(self)
        CallBack.__init__(self)
        DataCreateStream.__init__(self)
        PerformanceMeasurement.__init__(self)
        AnalyzeROC.__init__(self)
        
        self.conf = conf
        self.train_version = conf['train_version']
        self.model_name = conf['model_name']
        self.class_lst = ['0_No_closure', '1_Lateral_wall', '2_Tongue_base', '3_Epiglottis']
        self.exp_path = args.exp

    def unfreeze_model(self, model):
        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
        for i, layer in enumerate(model.layers):
            if layer.name.startswith('conv2d_180'):
                freeze_from = i
                print('freeze_from:', freeze_from)
                break
                
        for layer in model.layers[freeze_from:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
                
        # for layer in model.layers:
        #     # layer.trainable = True
        #     if not isinstance(layer, tf.keras.layers.BatchNormalization):
        #         layer.trainable = True
        
        return model
    
    def check_trainable(self, model):
        pd.set_option('max_colwidth', -1)
        layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
        df = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
        print(df.tail(30))
        # print(f"Trainable layers: {model.trainable_weights}")

#----------------------------------------------------------------------------

    def run(self):
        img_shape = self.load_imgShape(model_name=self.model_name)
        
        # Create a folder based on the shape of the img
        shape_fpath = os.path.join('bin', 'npy', self.train_version, str(img_shape))
        
        # load dataset
        test, val = self.dataloader_test(path=shape_fpath)
     
        n_classes = len(np.unique([np.argmax(elem) for elem in val['y']]))
        print("n_classes:", n_classes)
        
        
        val_lst = [val["X"][:,0,:,:,:], 
                   val["X"][:,1,:,:,:],
                   val["X"][:,2,:,:,:],
                   val["X"][:,3,:,:,:],
                   val["X"][:,4,:,:,:]
        ]
        test_lst = [test["X"][:,0,:,:,:], 
                    test["X"][:,1,:,:,:],
                    test["X"][:,2,:,:,:],
                    test["X"][:,3,:,:,:],
                    test["X"][:,4,:,:,:]
        ]
        
        # callback set
        # load best_model.hdf5
        best_model = self.load_model(exp_path=self.exp_path)
        
        y_pred = best_model.predict(test_lst)
        y_true = test['y']
        
        self.ROC_multi_all(class_lst=self.class_lst, 
                           y_true=y_true, y_pred=y_pred, 
                           exp_path=self.exp_path)
        
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        
        for (y_p, y_t, path) in zip(y_pred, y_true, list(test['x_path'])):
            if y_p == y_t:
                print(path)
        
        self.plot_confusion_matrix(y_true=y_true, y_pred=y_pred, class_lst=self.class_lst,
                                   exp_path=self.exp_path)
        
        # classification_reports
        report = classification_report(y_true=y_true, y_pred=y_pred, digits=n_classes, output_dict=True)
        
        print(report)

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default=None, required=True, help='main exp folder', type=str)
    args = parser.parse_args()
    
    # load config.yaml
    with open(os.path.join('bin', 'config', 'train_u.yaml')) as f:
        conf = yaml.safe_load(f)
    
    print('=='*50)
    for item in conf:
        print(f'{item}: {conf[item]}')
    print('=='*50)

    os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"]=str(conf['gpu'])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # now_time
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day}-{d.hour}-{d.minute}-{d.second}"
    
    OTECM = OTEClassificationMain(conf=conf, args=args)
    OTECM.run()

if __name__ == '__main__':
    main()