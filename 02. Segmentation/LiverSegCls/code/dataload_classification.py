import os
import numpy as np
import glob
import torch
import torch.nn as nn
from PIL import Image
import random
import cv2
## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, classes, ex, transform=None, target_transform=None,s= None):
        self.data_dir = data_dir
        self.classes = classes
        self.transform = transform
        self.target_transform= target_transform
        self.s = s


        #lst_data = os.listdir(self.data_dir)

        def get_label(data_list):
            label_list = []
            for path in data_list:
                # 에서 두번째가 class다.
                label_list.append(path.split('/')[-1].split('class_')[-1][0])
            return label_list

        if len(self.data_dir) == 1:
            lst_input_all = glob.glob(os.path.join(self.data_dir[0], '*.tiff'))
        else:
            lst_data1 = glob.glob(os.path.join(self.data_dir[0], '*.tiff'))
            lst_data2 = glob.glob(os.path.join(self.data_dir[1], '*.tiff'))
            lst_data3 = glob.glob(os.path.join(self.data_dir[2], '*.tiff'))
            lst_input_all = lst_data1+lst_data2+lst_data3

        if ex == 1: ## Class 0- Class 1
            lst_input = [i for i in lst_input_all if 'class_0_' in i]+[i for i in lst_input_all if 'class_1_' in i]

        elif ex == 2: ## Class 0- Class 2
            lst_input = [i for i in lst_input_all if 'class_0_' in i]+[i for i in lst_input_all if 'class_2_' in i]

        elif ex ==3: ## Class 1- Class 2
            lst_input = [i for i in lst_input_all if 'class_1_' in i]+[i for i in lst_input_all if 'class_2_' in i]

        elif ex ==4:  ## Class 0-Class 1-Class 2
            lst_input =  lst_input_all 
        lst_input.sort()


        self.label = get_label(lst_input)


        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self, index):


        #label = np.asarray(Image.open( self.lst_label[index]).convert('L'))

        if torch.is_tensor(index):
            index = index.tolist()

        input = np.asarray(Image.open(self.lst_input[index]).convert('L'))
        

        label = self.classes.index(self.label[index])


        data = {'Input_ID': self.lst_input[index],'input': input, 'label': label}
        if self.transform:

            seed = random.randrange(1, 1000)

            #input= cv2.resize(input, (96,96), interpolation=cv2.INTER_CUBIC)  #### wc (raw)
            input= cv2.resize(input, (self.s,self.s), interpolation=cv2.INTER_CUBIC)  #### wc (box,free)

            import matplotlib.pyplot as plt
            torch.manual_seed(seed)
            input = Image.fromarray(input)

            input = self.transform(input)
            

            data_t = {'Input_ID': self.lst_input[index].split('/')[-1],'input' : input,'label' : label }
            data = data_t

        return data
## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        input = data.transpose((2, 0, 1)).astype(np.float32)
        return input

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input = np.asarray(data)
        inputs = (input )/255


        if len(inputs.shape) == 2:
            inputs = inputs[:,:,np.newaxis]
        return inputs
