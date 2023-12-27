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

import argparse

import numpy as np
import pandas as pd

import cv2

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


#----------------------------------------------------------------------------

class GastricPositionClassificationMain(DataIOStream, CreatePillModel, PerformanceMeasurement, Postprocessing, AnalyzeROC, DataCreateStream):
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

        self.exp_path = exp_path
    
    def run(self):
        # name_lst = []
        # for name in self.data_classes:
            # name_lst.append(name[0])
        
        # dataset_name = '_'.join(name_lst)
        
        # category_fpath = os.path.join('bin', 'npy', dataset_name)
        
        img_shape = self.load_imgShape(model_name=self.model_name)
        
        # Create a folder based on the shape of the img
        # shape_fpath = os.path.join(category_fpath, str(img_shape))
        
        test_imgs_paths = glob.glob(os.path.join('bin', 'data', 'test', '*.png'))
        
        test_input_lst, test_id_lst = [], []
        
        for i, path in enumerate(test_imgs_paths):
            test_input_lst.append(path)
            # test_id_lst.append(path.split('/')[-1].split('_')[0])
            test_id_lst.append(path.split('\\')[-1].split('_')[0])
        
        # load dataset
        test_label_lst = tf.keras.utils.to_categorical(np.array(test_id_lst))
            
        test_dict = {'input_path': test_input_lst,
                     'input_label': test_label_lst,
                     'input_id': test_id_lst}
        
        test_dict = self.resize_n_normalization(object_dict=test_dict, 
                                                    new_width=img_shape, 
                                                    new_height=img_shape)
        
        # load best_model.hdf5
        model = self.load_model(exp_path=self.exp_path)
        
        n_classes = len(np.unique(test_dict['input_id']))
        
        # compile & fit
        model.compile(optimizer=self.optimizer, metrics=['acc'],
                        loss='categorical_crossentropy')
        # inference
        # eval_result = model.evaluate(test_dict['input_image'], test_dict['input_label'])
        
        # 알약 정보 엑셀 로드
        pill_info = pd.read_excel(os.path.join('bin', 'data', 'pill_list.v1.0.4.xlsx'), engine='openpyxl')
        
        import random
        
        # random.seed(1)
        # random.shuffle(test_dict['input_image'])
        # random.seed(1)
        # random.shuffle(test_dict['input_path'])
        # random.seed(1)
        # random.shuffle(test_dict['input_id'])
        # random.seed(1)
        # random.shuffle(test_dict['input_label'])
        
        y_pred = model.predict(test_dict['input_image'])
        y_true = test_dict['input_label']
        
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        
        count = 0
        
        # for i, (y_p, y_t) in enumerate(zip(y_pred, y_true)):
        #     # NUM = np.where(pill_info['접수번호_디렉토리']==int(test_dict['input_path'][y_p*5].split('/')[-1].split('_')[1]))[-1][-1]
        #     NUM = np.where(pill_info['접수번호_디렉토리']==int(test_dict['input_path'][y_p*5].split('\\')[-1].split('_')[1]))[-1][-1]

        #     print('=='*50)
        #     print("Counting pills:", count)
        #     print("Input img's file_name:", test_dict['input_path'][i].split('\\')[-1])
        #     print("Output img's information:", '\n', pill_info.loc[NUM, :])
        #     print('=='*50)
        #     cv2.imshow('Input img', test_dict['input_image'][i])
        #     cv2.imshow('Output img', test_dict['input_image'][y_p*5])
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
            
        #     count += 1
            
        # tflite evaluation
        # model_path = os.path.join('bin', 'exp', self.exp_path, 'test.tflite')
        
        # interpreter = tf.lite.Interpreter(model_path=model_path)
        # interpreter.allocate_tensors()
        
        # input_index = interpreter.get_input_details()[0]['index']
        # output_index = interpreter.get_output_details()[0]['index']
        
        # # Run predictions on ever y image in the "test" dataset.
        # y_pred = []
        
        # for idx, image in enumerate(test['input_image']):
        #     if idx % 10 == 0:
        #         print(f'Evaluated on {idx} results so far.')
        #     image = np.expand_dims(image, axis=0).astype(np.float32)
        #     interpreter.set_tensor(input_index, image)
        #     # Run inference.
        #     interpreter.invoke()
        #     # Post-processing: remove batch dimension and find the digit with highest
        #     # probability.
        #     output = interpreter.tensor(output_index)
        #     digit = np.argmax(output()[0])
        #     y_pred.append(digit)
        
        # roc curve
        # self.ROC_multi(n_classes=n_classes, y_true=y_true, y_pred=predict, exp_path=self.exp_path)
        
        # y_pred = np.argmax(y_pred, axis=1)
        # y_true = np.argmax(y_true, axis=1)  
        # input_path = test_dict['input_path']
        
        self.wrong_pred_save_img(y_pred=y_pred, y_true=y_true, 
                                 input_path=test_dict['input_path'], exp_path=self.exp_path,
                                 class_lst=np.unique(test_dict['input_id']))
        
        # self.plot_confusion_matrix(y_true=y_true, y_pred=y_pred, class_lst=test_dict['input_class'], exp_path=self.exp_path)
        
        # classification_reports
        report = classification_report(y_true=y_true, y_pred=y_pred, digits=n_classes, output_dict=True)
        
        # acc_ci, sens_ci, spec_ci = self.calc_cm(y_pred=y_pred, y_true=y_true)
        # auc, aucCI_lower, aucCI_upper = self.CI_calc(y_true, predict[:,1])
        
        print(report)
        
        # save_results
        # loss = f"loss: {eval_result[0]}"
        accuracy = f"accuracy: {report['accuracy']}"
        # macro_avg = f"macro_avg: {report['macro avg']}"
        # weighted_avg = f"weighted_avg: {report['weighted avg']}"
        
        with open(os.path.join('bin', 'exp', self.exp_path, 'eval.txt'), 'w') as f:
            # for i in range(n_classes):
                # f.write(test_dict['input_class'][i][0]+':'+str(report[str(i)])+'\n')
            # f.write('\n')
            # f.write(loss+'\n')
            f.write(accuracy+'\n')
            # f.write(macro_avg+'\n')
            # f.write(weighted_avg+'\n')
            # f.write('\n')
            # f.write("sensitivity (95% CI) = {0[0]:0.4f} ({0[1]:0.4f} - {0[2]:0.4f})\n".format(self.calc_CL(sens_ci)))   
            # f.write("specificity (95% CI) = {0[0]:0.4f} ({0[1]:0.4f} - {0[2]:0.4f})\n".format(self.calc_CL(spec_ci)))
            # f.write("  accuracy  (95% CI) = {0[0]:0.4f} ({0[1]:0.4f} - {0[2]:0.4f})\n".format(self.calc_CL(acc_ci)))
            # f.write("     AUC    (95% CI) = {:0.4f} ({:0.4f} - {:0.4f})".format(auc, aucCI_lower, aucCI_upper))

#----------------------------------------------------------------------------

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--exp", default=None, required=True, help='choose exp folder ex)2022-12-22-15-55-46', type=str)
    # args = parser.parse_args()
    
    with open(os.path.join('bin', 'exp', 'test', 'train.yaml'), 'r') as f:
        conf = yaml.safe_load(f)

    os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"]=str(conf['gpu'])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    print('=='*50)
    for item in conf:
        print(f'{item}: {conf[item]}')
    print('=='*50)

    GPCM = GastricPositionClassificationMain(conf, 'test')
    GPCM.run()

if __name__ == '__main__':
    main()