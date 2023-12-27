import os
import yaml

import numpy as np
import pandas as pd

import cv2
import glob

import shutil
import random

import datetime

import tensorflow as tf

from tqdm import tqdm


#----------------------------------------------------------------------------

class DataCreateStream():
    def __init__(self) -> None:
        ...

    def text_label_categorical(self, input_label: list, data_classes: list):
        '''
        Args:
            input_label : label 데이터로 구성된 list
        '''
        aaa = {}
        new_input_label = []

        for cls in data_classes:
            aaa[cls[0]] = len(aaa)
        
        for label in input_label:
            new_input_label.append(aaa.get(label))
        
        new_input_label = tf.keras.utils.to_categorical(np.array(new_input_label))
        
        return new_input_label, input_label

    def resize_n_normalization(self, object_dict: dict, new_width: int, new_height: int):
        '''resize and normalization images
        Args:
            object_dict : 데이터셋 관련 정보 dict ex)input_path, input_label, input_id
            
            new_width : resize 시 활용될 new width
            
            new_height : resize 시 활용될 new height
            
        '''
        resized_img_lst = []
        
        for path in object_dict['input_path']:
            image = cv2.imread(path)
            image = cv2.resize(src=image, dsize=(new_width, new_height)) / 255.0
            resized_img_lst.append(image)
            
        resized_imgs = np.array(resized_img_lst)
        
        object_dict['input_image'] = resized_imgs
        
        return object_dict
            