
import os
import glob
import random

import numpy as np
import pandas as pd
import pickle
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

#----------------------------------------------------------------------------

class DataIOStream():
    def __init__(self) -> None:
        ...
        
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
            
        with open(file=os.path.join(path, 'test.pickle'), mode='rb') as f:
            test = pickle.load(f)
            f.close()
            
        with open(file=os.path.join(path, 'val.pickle'), mode='rb') as f:
            val = pickle.load(f)
            f.close()
            
        # train = np.load(file=os.path.join(path, 'train.npy'))
        # test = np.load(file=os.path.join(path, 'test.npy'))
        # val = np.load(file=os.path.join(path, 'val.npy'))
        
        return train, test, val
        
    def dataloader_test(self, path: str):
        """
            
        Args:
            ...

        Return:
            ...
        """
        with open(file=os.path.join(path, 'test.pickle'), mode='rb') as f:
            test = pickle.load(f)
            f.close()
        
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
    
class CustomDataset(Dataset):
    def __init__(self, 
                 x_data_paths: list,
                 y_data : list,
                 img_size : int,
    ):
        super(CustomDataset, self).__init__()
        self.x_data_paths = x_data_paths
        self.y_data = y_data
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225],
            ),
        ])
        
    def __len__(self):
        return len(self.x_data_paths)
    
    def __getitem__(self, idx):
        img_path = self.x_data_paths[idx]
        try:
            # 이미지 파일이 맞는지 확인
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"File {img_path} does not exist.")
            
            x_data = Image.open(img_path)
            x_data = self.transform(x_data)

            y_data = self.y_data[idx]
        except Exception as e:
            # 예외를 처리하고 에러를 로깅
            print(f"Error occurred while processing image at index {idx}: {e}")
            
            # 문제가 있는 경우 더미 데이터로 대체
            x_data = torch.zeros((3, self.img_size, self.img_size))  # 예시로 더미 데이터 생성
            y_data = -1

        return x_data, y_data
    