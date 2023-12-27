from .lagacy.preprocess import Preprocess

import tensorflow as tf

import os
import glob
import random

import numpy as np
import pandas as pd
import pickle

import cv2


#----------------------------------------------------------------------------

class DataIOStream(Preprocess):
    def __init__(self) -> None:
        Preprocess.__init__(self)
        
    def dataloader_all(self, path: str):
        """
            
        Args:
            ...

        Return:
            Store incorrectly predicted images in a given path
        """        
        with open(file=os.path.join(path, 'train.pickle'), mode='rb') as f:
            train = pickle.load(f)
            f.close()
            
        # with open(file=os.path.join(path, 'test.pickle'), mode='rb') as f:
        #     test = pickle.load(f)
        #     f.close()
            
        with open(file=os.path.join(path, 'val.pickle'), mode='rb') as f:
            val = pickle.load(f)
            f.close()
            
        # train = np.load(file=os.path.join(path, 'train.npy'))
        # test = np.load(file=os.path.join(path, 'test.npy'))
        # val = np.load(file=os.path.join(path, 'val.npy'))
        
        return train, val
        
    def dataloader_test(self, dataset: str):
        """
            
        Args:
            ...

        Return:
            ...
        """
        f_test = os.path.join('bin', 'data', dataset, 'test')
        
        test = self.npy_csv_load(f_test)

        return test
        
    
    def npy_csv_load(self, path: str):
        """
        
        Args:
            ...

        Return:
            ...
        """
        input_image = np.load(file=os.path.join(path, 'input_image.npy'))
        input_id = np.load(file=os.path.join(path, 'input_id.npy'))
        input_label = np.load(file=os.path.join(path, 'input_label.npy'))
        input_path = pd.read_csv(os.path.join(path, 'input_path.csv'), header=None)
        class_lst = pd.read_csv(os.path.join(path, 'input_class.csv'), header=None)
        
        return {'input_image': input_image, "input_label": input_label, 
                'input_path': input_path, 'input_id': input_id,
                'class_lst': class_lst}
        
    def csv_load(self, path: str):
        """
            
        Args:
            ...

        Return:
            ...
        """
        cls_path = os.path.join('bin', 'data', path, 'test')
        class_lst = pd.read_csv(os.path.join(cls_path, 'input_class.csv'), header=None)
        
        return class_lst