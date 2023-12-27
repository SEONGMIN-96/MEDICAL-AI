from .utils.dataload import DataIOStream
from .utils.callback import CallBack
from .utils.create_data import DataCreateStream
from .model.model import CreateGastricModel

# from .utils.model import GastricModeling


from tqdm import tqdm

import tempfile
import os
import yaml
import time
import datetime

import numpy as np
import pandas as pd
import pickle

import cv2

# from mlflow.tensorflow import autolog, save_model
# from mlflow.models import infer_signature

import tensorflow_model_optimization as tfmot
import tensorflow as tf

#----------------------------------------------------------------------------

class GastricPositionClassificationMain(DataIOStream, CreateGastricModel, CallBack, DataCreateStream):
    def __init__(self, conf: dict, now_time: str,) -> None:
        DataIOStream.__init__(self)        
        CreateGastricModel.__init__(self)
        CallBack.__init__(self)
        DataCreateStream.__init__(self)
        
        self.conf = conf
        self.data_classes = conf['data_classes']
        self.model_name = conf['model_name']
        self.batch = conf['batch']
        self.epoch = conf['epoch']
        self.optimizer = conf['optimizer']
        self.es_patience = conf['es_patience']
        self.reduce_lr_factor = conf['reduce_lr_factor']
        self.reduce_lr_patience = conf['reduce_lr_patience']
        self.learning_rate_scheduler = conf['learning_rate_scheduler']
        self.now_time = now_time
        
        self.startswith_layer = 'activation_28'

    def unfreeze_model(self, model):
        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
        for i, layer in enumerate(model.layers):
            if layer.name.startswith(self.startswith_layer):
                freeze_from = i
                print('freeze_from:', freeze_from)
                break
                
        for layer in model.layers[freeze_from:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
                
        for layer in model.layers:
            # layer.trainable = True
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
        
        return model
    
    def check_trainable(self, model):
        pd.set_option('max_colwidth', -1)
        layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
        df = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
        
        print(df.tail(30))
        # print(f"Trainable layers: {model.trainable_weights}")

        

#----------------------------------------------------------------------------

    def run(self):
        # build_model
        model = self.build_model(model_name=self.model_name, 
                                input_shape=(224,224,3), 
                                n_class=7)
        
        model = self.unfreeze_model(model=model)
        self.check_trainable(model=model)

        
#----------------------------------------------------------------------------

def main():
    # load config.yaml
    with open('./bin/config/train_u.yaml') as f:
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
    
    GPCM = GastricPositionClassificationMain(conf=conf, now_time=now_time)
    GPCM.run()

if __name__ == '__main__':
    main()