from .utils.dataload import DataIOStream
from .utils.callback import CallBack
from .utils.create_data import DataCreateStream
from .model.model import CreatePillModel

# from .utils.model import GastricModeling


from tqdm import tqdm

import tempfile
import os
import yaml
import time
import datetime
import glob

import numpy as np
import pandas as pd
import pickle

import cv2

import pandas as pd

import matplotlib.pyplot as plt

# from mlflow.tensorflow import autolog, save_model
# from mlflow.models import infer_signature

import tensorflow_model_optimization as tfmot
import tensorflow as tf

#----------------------------------------------------------------------------

class PillClassificationMain(DataIOStream, CreatePillModel, CallBack, DataCreateStream):
    def __init__(self, conf: dict, now_time: str,) -> None:
        DataIOStream.__init__(self)        
        CreatePillModel.__init__(self)
        CallBack.__init__(self)
        DataCreateStream.__init__(self)
        
        self.conf = conf
        self.dataset = conf['dataset']
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
        shape_fpath = os.path.join('bin', 'npy', str(img_shape))
        
        if not os.path.exists(shape_fpath):
            os.mkdir(shape_fpath)
            
            train_id_lst, val_id_lst = [], []
            train_input_lst, val_input_lst = [], []
            
            input_paths = glob.glob(os.path.join('bin', 'data', self.dataset, '*.png'))
            
            for i, path in enumerate(input_paths):
                if i % 20 == 0:
                    val_input_lst.append(path)
                    val_id_lst.append(path.split('/')[-1].split('_')[0])
                else:
                    train_input_lst.append(path)
                    train_id_lst.append(path.split('/')[-1].split('_')[0])
            
            train_label_lst = tf.keras.utils.to_categorical(np.array(train_id_lst))
            val_label_lst = tf.keras.utils.to_categorical(np.array(val_id_lst))
                
            train_dict = {'input_path': train_input_lst,
                          'input_label': train_label_lst,
                          'input_id': train_id_lst}
            
            val_dict = {'input_path': val_input_lst,
                        'input_label': val_label_lst,
                        'input_id': val_id_lst}
            
            train_dict = self.resize_n_normalization(object_dict=train_dict, 
                                                     new_width=img_shape, 
                                                     new_height=img_shape)
            
            val_dict = self.resize_n_normalization(object_dict=val_dict, 
                                                    new_width=img_shape, 
                                                    new_height=img_shape)
            
            # np.save(file=os.path.join(shape_fpath, 'train'), arr=np.array(train_dict), allow_pickle=True)
            with open(file=os.path.join(shape_fpath, 'train.pickle'), mode='wb') as f:
                pickle.dump(train_dict, f, protocol=4)
                
            with open(file=os.path.join(shape_fpath, 'val.pickle'), mode='wb') as f:
                pickle.dump(val_dict, f, protocol=4)
            
            print('resize & normalization done...!')
            print('data save done...!')
        
        # load dataset
        train, val = self.dataloader_all(path=shape_fpath)

        print('data laod done...!')
        
        n_classes = len(np.unique(train['input_id']))
        
        # build_model
        model = self.build_model(model_name=self.model_name, 
                                input_shape=train['input_image'][0].shape, 
                                n_class=n_classes)
        
        # model = self.unfreeze_model(model=model)
        # self.check_trainable(model=model)

        # callback set
        callback_lst = self.callback_setting(es_patience=self.es_patience,
                                             now_time=self.now_time,
                                             reduce_lr_patience=self.reduce_lr_patience,
                                             reduce_lr_factor=self.reduce_lr_factor)

        # compile & fit
        if self.optimizer == 'Adam':
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate), 
                        metrics=['acc'],
                        loss='categorical_crossentropy')
        elif self.optimizer == 'SGD':
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.initial_learning_rate,
                        momentum=0.9), 
                        metrics=['acc'],
                        loss='categorical_crossentropy')
              
        # normal fit
        fit_hist = model.fit(train['input_image'], train['input_label'],
                    batch_size=self.batch, epochs=self.epoch, verbose=1, 
                    callbacks=callback_lst, validation_data=(val['input_image'], val['input_label']))
        
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
    
    PCM = PillClassificationMain(conf=conf, now_time=now_time)
    PCM.run()

if __name__ == '__main__':
    main()