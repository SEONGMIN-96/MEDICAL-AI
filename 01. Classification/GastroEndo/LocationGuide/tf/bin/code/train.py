from .utils.dataload import DataIOStream
from .utils.create_data import DataCreateStream
from .model.model import CreateGastricModel

# from .utils.model import GastricModeling


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
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

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

#----------------------------------------------------------------------------

class GastricPositionClassificationMain(DataIOStream, CreateGastricModel, DataCreateStream):
    def __init__(self, conf: dict, now_time: str,) -> None:
        DataIOStream.__init__(self)        
        CreateGastricModel.__init__(self)
        DataCreateStream.__init__(self)
        
        self.conf = conf
        self.data_classes = conf['data_classes']
        self.model_name = conf['model_name']
        self.batch = conf['batch']
        self.epoch = conf['epoch']
        self.optimizer = conf['optimizer']
        self.es_patience = conf['es_patience']
        self.initial_learning_rate = conf['initial_learning_rate']
        self.reduce_lr_factor = conf['reduce_lr_factor']
        self.reduce_lr_patience = conf['reduce_lr_patience']
        self.learning_rate_scheduler = conf['learning_rate_scheduler']
        self.now_time = now_time
        
        self.startswith_layer = 'conv4_block1_1_conv'
        self.trainable = conf['trainable']

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
                
        # for layer in model.layers:
        #     # layer.trainable = True
        #     if not isinstance(layer, tf.keras.layers.BatchNormalization):
        #         layer.trainable = True
        
        return model
    
    def check_trainable(self, model):
        pd.set_option('max_colwidth', -1)
        layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
        df = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
        
        df.to_csv(os.path.join('bin', 'exp', self.now_time, 'check_trainable.csv'), index=False)
        print(df.tail(30))
        # print(f"Trainable layers: {model.trainable_weights}")


#----------------------------------------------------------------------------

    def run(self):
        # Create a folder based on the category of the dataset
        name_lst = []
        for name in self.data_classes:
            name_lst.append(name[0])
        
        dataset_name = '_'.join(name_lst)
        
        category_fpath = os.path.join('bin', 'npy', dataset_name)
        
        if not os.path.exists(category_fpath):
            os.mkdir(category_fpath)
        
        img_shape = self.load_imgShape(model_name=self.model_name)
        
        # Create a folder based on the shape of the img
        shape_fpath = os.path.join(category_fpath, str(img_shape))
        
        if not os.path.exists(shape_fpath):
            os.mkdir(shape_fpath)
            
            train_dict, test_dict, val_dict = self.data_split(data_classes=self.data_classes)
            
            print('data split done...!')
            
            train_dict = self.resize_n_normalization(object_dict=train_dict, 
                                                     new_width=img_shape, 
                                                     new_height=img_shape)
            
            # np.save(file=os.path.join(shape_fpath, 'train'), arr=np.array(train_dict), allow_pickle=True)
            with open(file=os.path.join(shape_fpath, 'train.pickle'), mode='wb') as f:
                pickle.dump(train_dict, f, protocol=4)
            
            test_dict = self.resize_n_normalization(object_dict=test_dict,
                                                    new_width=img_shape,
                                                    new_height=img_shape)
            
            # np.save(file=os.path.join(shape_fpath, 'test'), arr=np.array(test_dict), allow_pickle=True)
            with open(file=os.path.join(shape_fpath, 'test.pickle'), mode='wb') as f:
                pickle.dump(test_dict, f, protocol=4)
            
            val_dict = self.resize_n_normalization(object_dict=val_dict,
                                                   new_width=img_shape,
                                                   new_height=img_shape)
            
            # np.save(file=os.path.join(shape_fpath, 'val'), arr=np.array(val_dict), allow_pickle=True)
            with open(file=os.path.join(shape_fpath, 'val.pickle'), mode='wb') as f:
                pickle.dump(val_dict, f, protocol=4)
            
            print('resize & normalization done...!')
            print('data save done...!')
        
        # load dataset
        train, test, val = self.dataloader_all(path=shape_fpath)

        print('data laod done...!')
        
        n_classes = len(train['input_class'])
        
        # class weight for weight balancing
        class_weight = {0:0.08, 
                        1:0.153, 
                        2:0.153,
                        3:0.153,
                        4:0.153,
                        5:0.153,
                        6:0.153
                        }
        
        # build_model
        model = self.build_model(model_name=self.model_name, 
                                input_shape=train['input_images'][0].shape, 
                                n_class=n_classes)
        
        if self.trainable == True:
            model = self.unfreeze_model(model=model)
            self.check_trainable(model=model)

        # callback set
        es = EarlyStopping(monitor='val_loss', mode='auto', patience=self.es_patience)
        cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                            filepath=os.path.join('bin', 'exp', self.now_time, 'best_model.hdf5'))
        
        # learning_rate_scheduler
        if self.learning_rate_scheduler == 'ReduceLROnPlateau':
            lrs = ReduceLROnPlateau(monitor='val_loss', patience=self.reduce_lr_patience, mode='auto', verbose=1, factor=self.reduce_lr_factor)
        elif self.learning_rate_scheduler == 'CosineDecay':
            cos_decay = tf.keras.experimental.CosineDecay(initial_learning_rate=0.001, decay_steps=50, alpha=0.001)
            lrs = LearningRateScheduler(cos_decay, verbose=1)
        
        # tc = TimeCallback()
        
        # compile & fit
        if self.optimizer == 'Adam':
            if n_classes == 2:
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate), 
                            metrics=['acc'],
                            loss='binary_crossentropy')
            elif n_classes > 2:
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate), 
                            metrics=['acc'],
                            loss='categorical_crossentropy')
        elif self.optimizer == 'SGD':
            if n_classes == 2:
                model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.initial_learning_rate,
                            momentum=0.9), 
                            metrics=['acc'],
                            loss='binary_crossentropy')
            elif n_classes > 2:
                model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.initial_learning_rate,
                            momentum=0.9), 
                            metrics=['acc'],
                            loss='categorical_crossentropy')
              
        # class_weight fit
        # fit_hist = model.fit(train['input_images'], train['input_label'],
                    # batch_size=self.batch, epochs=self.epoch, verbose=1,
                    # validation_data=(val['input_images'], val['input_label']), callbacks=callback_lst
                    # class_weight=class_weight)
                    
        # normal fit
        fit_hist = model.fit(train['input_images'], train['input_label'],
                    batch_size=self.batch, epochs=self.epoch, verbose=1,
                    validation_data=(val['input_images'], val['input_label']), callbacks=[es, cp, lrs])
        
        train_loss_lst = fit_hist.history['loss']
        train_acc_lst = fit_hist.history['acc']
        val_loss_lst = fit_hist.history['val_loss']
        val_acc_lst = fit_hist.history['val_acc']
        
        epoch_lst = [p for p in range(len(train_loss_lst))]

        # test data 결과 evaluation
        y_pred = np.argmax(model.predict(test['input_images']), axis=1) 
        y_true = np.argmax(test['input_label'], axis=1)
        
        # classification_reports
        report = classification_report(y_true=y_true, y_pred=y_pred, digits=n_classes, output_dict=True)

        print('F1-score : {0:.3f}, Precision : {1:.3f}, Recall : {2:.3f}'.format(report['weighted avg']['f1-score'], 
                                                                                 report['weighted avg']['precision'],
                                                                                 report['weighted avg']['recall']))
        
        # if not os.path.exists(os.path.join('bin', 'exp', self.now_time)):
            # os.mkdir(os.path.join('bin', 'exp', self.now_time))
        
        # # save results
        # results = pd.DataFrame(data={'epoch' : epoch_lst, 'loss' : train_loss_lst, 'acc' : train_acc_lst, 'val_loss' : val_loss_lst, 'val_acc' : val_acc_lst})
        # results.to_csv(os.path.join('bin', 'exp', self.now_time, 'results.csv'), index=False)
        
        # # save parameter
        # with open(os.path.join('bin', 'exp', self.now_time, 'train.yaml'), 'w') as f:
        #     yaml.safe_dump(self.conf, f)
        
        # # load best_model.hdf5
        # best_model = self.load_model(exp_path=self.now_time)
        
        # # save model to tflite
        # converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # quantized_and_pruned_tflite_model = converter.convert()
        
        # quantized_and_pruned_tflite_file = os.path.join('bin', 'exp', self.now_time, 'test.tflite')
        
        # with open(quantized_and_pruned_tflite_file, 'wb') as f:
        #     f.write(quantized_and_pruned_tflite_model)

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
    aa = time.time()
    d = datetime.datetime.now()
    now_time = f"{d.year}_{d.month}_{d.day}_{d.hour}_{d.minute}_{d.second}"
    # MkExpDir
    if not os.path.exists(os.path.join('bin', 'exp', now_time)):
        os.mkdir(os.path.join('bin', 'exp', now_time))
    GPCM = GastricPositionClassificationMain(conf=conf, 
                                             now_time=now_time
    )
    GPCM.run()
    print('ETA: {:.3f}s'.format(time.time() - aa))

if __name__ == '__main__':
    main()