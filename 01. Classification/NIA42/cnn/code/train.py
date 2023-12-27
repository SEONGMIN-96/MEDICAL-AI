from .utils.dataload import DataIOStream
from .utils.callback import CallBack
from .utils.create_data import DataCreateStream
from .utils.matrix import PerformanceMeasurement
from .utils.roc import AnalyzeROC
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

import matplotlib.pyplot as plt

# from mlflow.tensorflow import autolog, save_model
# from mlflow.models import infer_signature

import tensorflow_model_optimization as tfmot
import tensorflow as tf

import imgaug.augmenters as iaa
import imageio

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#----------------------------------------------------------------------------

class OTEClassificationMain(DataIOStream, CreateOTEModel_v2, CallBack, DataCreateStream, PerformanceMeasurement, AnalyzeROC):
    def __init__(self, conf: dict, now_time: str,) -> None:
        DataIOStream.__init__(self)        
        CreateOTEModel_v2.__init__(self)
        CallBack.__init__(self)
        DataCreateStream.__init__(self)
        PerformanceMeasurement.__init__(self)
        AnalyzeROC.__init__(self)
        
        self.conf = conf
        self.dataset = conf['dataset']
        self.train_version = conf['train_version']
        self.model_name = conf['model_name']
        self.batch = conf['batch']
        self.epoch = conf['epoch']
        self.optimizer = conf['optimizer']
        self.es_patience = conf['es_patience']
        self.reduce_lr_factor = conf['reduce_lr_factor']
        self.reduce_lr_patience = conf['reduce_lr_patience']
        self.initial_learning_rate = conf['initial_learning_rate']
        self.learning_rate_scheduler = conf['learning_rate_scheduler']
        self.now_time = now_time
        self.class_lst = ['1_Lateral_wall', '2_Tongue_base', '3_Epiglottis']

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
        
        base_path = os.path.join('bin', 'data', '3285')
        
        if not os.path.exists(shape_fpath):
            os.mkdir(shape_fpath)
            
            folder_00 = glob.glob(os.path.join(base_path, 'classify', '0_No_closure', '*'))
            folder_01 = glob.glob(os.path.join(base_path, 'classify', '1_Lateral_wall', '*'))
            folder_02 = glob.glob(os.path.join(base_path, 'classify', '2_Tongue_base', '*'))
            folder_03 = glob.glob(os.path.join(base_path, 'classify', '3_Epiglottis', '*'))

            folder_00_label = [0 for path in folder_00]
            folder_01_label = [1 for path in folder_01]
            folder_02_label = [2 for path in folder_02]
            folder_03_label = [3 for path in folder_03]
            
            random_seed = 55

            train_X, train_Y = [], []
            test_X, test_Y = [], []
            val_X, val_Y = [], []

            # train_X00, test_X00_temp, train_Y00, test_Y00_temp = train_test_split(folder_00, folder_00_label, test_size=0.1, random_state=random_seed, stratify=folder_00_label)
            # train_X01, test_X01_temp, train_Y01, test_Y01_temp = train_test_split(folder_01, folder_01_label, test_size=0.1, random_state=random_seed, stratify=folder_01_label)
            # train_X02, test_X02_temp, train_Y02, test_Y02_temp = train_test_split(folder_02, folder_02_label, test_size=0.1, random_state=random_seed, stratify=folder_02_label)
            # train_X03, test_X03_temp, train_Y03, test_Y03_temp = train_test_split(folder_03, folder_03_label, test_size=0.1, random_state=random_seed, stratify=folder_03_label)

            # test_X00, val_X00, test_Y00, val_Y00 = train_test_split(test_X00_temp, test_Y00_temp, test_size=0.5, random_state=random_seed, stratify=test_Y00_temp)
            # test_X01, val_X01, test_Y01, val_Y01 = train_test_split(test_X01_temp, test_Y01_temp, test_size=0.5, random_state=random_seed, stratify=test_Y01_temp)
            # test_X02, val_X02, test_Y02, val_Y02 = train_test_split(test_X02_temp, test_Y02_temp, test_size=0.5, random_state=random_seed, stratify=test_Y02_temp)
            # test_X03, val_X03, test_Y03, val_Y03 = train_test_split(test_X03_temp, test_Y03_temp, test_size=0.5, random_state=random_seed, stratify=test_Y03_temp)

            # train_X.extend(train_X00)
            # train_X.extend(train_X01)
            # train_X.extend(train_X02)
            # train_X.extend(train_X03)

            # train_Y.extend(train_Y00)
            # train_Y.extend(train_Y01)
            # train_Y.extend(train_Y02)
            # train_Y.extend(train_Y03)

            # test_X.extend(test_X00)
            # test_X.extend(test_X01)
            # test_X.extend(test_X02)
            # test_X.extend(test_X03)

            # test_Y.extend(test_Y00)
            # test_Y.extend(test_Y01)
            # test_Y.extend(test_Y02)
            # test_Y.extend(test_Y03)

            # val_X.extend(val_X00)
            # val_X.extend(val_X01)
            # val_X.extend(val_X02)
            # val_X.extend(val_X03)

            # val_Y.extend(val_Y00)
            # val_Y.extend(val_Y01)
            # val_Y.extend(val_Y02)
            # val_Y.extend(val_Y03)

            # if len(train_X) == len(train_Y):
            #     print("train:", len(train_X))
            # if len(test_X) == len(test_Y):
            #     print("test:", len(test_X))
            # if len(val_X) == len(val_Y):
            #     print("val:", len(val_X))
            
            new_train_X, new_train_M, new_train_Y = [], [], []
            # new_test_X, new_test_M, new_test_Y = [], [], []
            # new_val_X, new_val_M, new_val_Y = [], [], []
            
            train_X.extend(folder_00)
            train_X.extend(folder_01)
            train_X.extend(folder_02)
            train_X.extend(folder_03)
            
            train_Y.extend(folder_00_label)
            train_Y.extend(folder_01_label)
            train_Y.extend(folder_02_label)
            train_Y.extend(folder_03_label)
            
            for (path, label) in zip(train_X, train_Y):
                imgs = glob.glob(os.path.join(path, '*.png'))
                part = path.split('/')
                masks = glob.glob(os.path.join(base_path, '1_crop_label', 'Total_mask', part[-1], '*.png'))
                new_train_X.append(imgs)
                new_train_M.append(masks)
                new_train_Y.append(label)
            # for (path, label) in zip(test_X, test_Y):
            #     imgs = glob.glob(os.path.join(path, '*.png'))
            #     part = path.split('/')
            #     masks = glob.glob(os.path.join(base_path, '1_crop_label', 'Total_mask', part[-1], '*.png'))
            #     new_test_X.append(imgs)
            #     new_test_M.append(masks)
            #     new_test_Y.append(label)
            # for (path, label) in zip(val_X, val_Y):
            #     imgs = glob.glob(os.path.join(path, '*.png'))
            #     part = path.split('/')
            #     masks = glob.glob(os.path.join(base_path, '1_crop_label', 'Total_mask', part[-1], '*.png'))
            #     new_val_X.append(imgs)
            #     new_val_M.append(masks)
            #     new_val_Y.append(label)
                        
            new_train_Y = tf.keras.utils.to_categorical(np.array(new_train_Y))
            # new_test_Y = tf.keras.utils.to_categorical(np.array(new_test_Y))
            # new_val_Y = tf.keras.utils.to_categorical(np.array(new_val_Y))

            new_train_X = self.resize_n_normalization_v2(object_list=new_train_X, 
                                                    new_width=img_shape, 
                                                    new_height=img_shape)
            # new_test_X = self.resize_n_normalization_v2(object_list=new_test_X, 
            #                                         new_width=img_shape, 
            #                                         new_height=img_shape)
            # new_val_X = self.resize_n_normalization_v2(object_list=new_val_X, 
            #                                     new_width=img_shape, 
            #                                     new_height=img_shape)
            
            train_data = {"X":np.array(new_train_X), "y":np.array(new_train_Y), "X_path": train_X}
            # test_data = {"X":np.array(new_test_X), "y":np.array(new_test_Y), "X_path": test_X}
            # val_data = {"X":np.array(new_val_X), "y":np.array(new_val_Y), "X_path": val_X}
            
            # np.save(file=os.path.join(shape_fpath, 'train'), arr=np.array(train_dict), allow_pickle=True)
            with open(file=os.path.join(shape_fpath, 'train.pickle'), mode='wb') as f:
                pickle.dump(train_data, f, protocol=4)
            # with open(file=os.path.join(shape_fpath, 'test.pickle'), mode='wb') as f:
            #     pickle.dump(test_data, f, protocol=4)    
            # with open(file=os.path.join(shape_fpath, 'val.pickle'), mode='wb') as f:
            #     pickle.dump(val_data, f, protocol=4)
                
            print('resize & normalization done...!')
            print('data save done...!')
        
        # load dataset
        train, test, val = self.dataloader_all(path=shape_fpath)
        # test = self.dataloader_test(path=shape_fpath)
        
        n_classes = len(np.unique([np.argmax(elem) for elem in test['y']]))
        print("n_classes:", n_classes)
        
        # bb = aa[:,2,:,:,:]
        train_lst = [train["X"][:,0,:,:,:], 
                     train["X"][:,1,:,:,:],
                     train["X"][:,2,:,:,:],
                     train["X"][:,3,:,:,:],
                     train["X"][:,4,:,:,:]
        ]
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
        
        # # build_model
        # model = self.build_model(model_name=self.model_name, 
                                # input_shape=train['X'][0][0].shape, 
                                # n_class=n_classes)
        
        # load best_model.hdf5
        model = self.load_model(exp_path='2023-10-25-23-36-33')
        
        # model.summary()
        
        # model = self.unfreeze_model(model=model)
        # self.check_trainable(model=model)

        # callback set
        callback_lst = self.callback_setting(es_patience=self.es_patience,
                                             now_time=self.now_time,
                                             reduce_lr_patience=self.reduce_lr_patience,
                                             reduce_lr_factor=self.reduce_lr_factor)

        # compile & fit
        if self.optimizer == 'Adam':
            if n_classes == 2:
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate), 
                            metrics=['acc'],
                            loss='binary_crossentropy')
            else:
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate), 
                            metrics=['acc'],
                            loss='categorical_crossentropy')
        elif self.optimizer == 'SGD':
            if n_classes == 2:
                model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.initial_learning_rate,
                            momentum=0.9), 
                            metrics=['acc'],
                            loss='binary_crossentropy')
            else:
                model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.initial_learning_rate,
                            momentum=0.9), 
                            metrics=['acc'],
                            loss='categorical_crossentropy')
        
        # normal fit
        fit_hist = model.fit(x=train_lst, y=train['y'],
                    batch_size=self.batch, epochs=self.epoch, verbose=1, 
                    # callbacks=callback_lst)
                    callbacks=callback_lst, validation_data=(val_lst, val['y']))
        
        train_loss_lst = fit_hist.history['loss']
        train_acc_lst = fit_hist.history['acc']
        val_loss_lst = fit_hist.history['val_loss']
        val_acc_lst = fit_hist.history['val_acc']
        
        epoch_lst = [p for p in range(len(train_loss_lst))]

        if not os.path.exists(os.path.join('bin', 'exp', self.now_time)):
            os.mkdir(os.path.join('bin', 'exp', self.now_time))
        
        # save results
        results = pd.DataFrame(data={'epoch' : epoch_lst, 'loss' : train_loss_lst, 'acc' : train_acc_lst, 'val_loss': val_loss_lst, 'val_acc': val_acc_lst})
        results.to_csv(os.path.join('bin', 'exp', self.now_time, 'results.csv'), index=False)
        
        hist_DF = pd.DataFrame(fit_hist.history)
        hist_DF.to_csv(os.path.join('bin', 'exp', self.now_time, 'hist_DF.csv'), index=False)
        
        # save parameter
        with open(os.path.join('bin', 'exp', self.now_time, 'train.yaml'), 'w') as f:
            yaml.safe_dump(self.conf, f)
        
        # load best_model.hdf5
        best_model = self.load_model(exp_path=self.now_time)
        
        # save model to tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_and_pruned_tflite_model = converter.convert()
        
        quantized_and_pruned_tflite_file = os.path.join('bin', 'exp', self.now_time, 'test.tflite')
        
        with open(quantized_and_pruned_tflite_file, 'wb') as f:
            f.write(quantized_and_pruned_tflite_model)
        
        y_pred = model.predict(test_lst)
        y_true = test['y']
        
        self.ROC_multi_all(class_lst=self.class_lst, 
                           y_true=y_true, y_pred=y_pred, 
                           exp_path=self.now_time)
        
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
              
        self.plot_confusion_matrix(y_true=y_true, y_pred=y_pred, class_lst=self.class_lst,
                                   exp_path=self.now_time)
        
        # classification_reports
        report = classification_report(y_true=y_true, y_pred=y_pred, digits=n_classes, output_dict=True)
        
        print(report)

#----------------------------------------------------------------------------

def main():
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
    
    # MkExpDir
    if not os.path.exists(os.path.join('bin', 'exp', now_time)):
        os.mkdir(os.path.join('bin', 'exp', now_time))
    
    OTECM = OTEClassificationMain(conf=conf, now_time=now_time)
    OTECM.run()

if __name__ == '__main__':
    main()