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
    def __init__(self, data_dir, transform=None,target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        #lst_data = os.listdir(self.data_dir)

        if len(self.data_dir) == 1:
            lst_data = glob.glob(os.path.join(self.data_dir[0], '*.tiff'))
        else:
            lst_data1 = glob.glob(os.path.join(self.data_dir[0], '*.tiff'))
            lst_data2 = glob.glob(os.path.join(self.data_dir[1], '*.tiff'))
            lst_data3 = glob.glob(os.path.join(self.data_dir[2], '*.tiff'))
            lst_data = lst_data1+lst_data2+lst_data3



        lst_label = [f for f in lst_data if 'mask.tiff' in f]
        lst_input = [f for f in lst_data if 'mask.tiff' not in f]

        #lst_label = [f for f in lst_data if f.endswith('IN_unknown_0000001640_HCC_mask.png')]
        #lst_input = [f for f in lst_data if f.endswith('IN_unknown_0000001640_HCC.png')]



        print('---------------------------')

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):


        label_3c = np.asarray(Image.open( self.lst_label[index]))
        label_raw = label_3c[:,:,0]+label_3c[:,:,1]+label_3c[:,:,2]

        input_raw = np.asarray(Image.open(self.lst_input[index]).convert('L'))




        if label_raw.ndim == 2:
            label_raw  = label_raw[:, :,np.newaxis ]
            #print(np.max(label))
        if input_raw.ndim == 2:
            input_raw  = input_raw[:, :,np.newaxis]


        ##### data = {'Input_ID': self.lst_input[index],'Label_ID': self.lst_label[index], 'input': input_raw , 'label': label_raw }
        #if self.transform:

        seed = random.randrange(1, 100)
        #arr = cv2.resize(arr, (256,256), interpolation=cv2.INTER_NEAREST)

        import matplotlib.pyplot as plt
        input = cv2.resize(input_raw, (512,512), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label_raw, (512,512), interpolation=cv2.INTER_NEAREST)
        torch.manual_seed(seed)
        #input_raw = np.array(Image.fromarray(input))
        #label_raw = np.array(Image.fromarray(label))

 



        inputs = self.transform(input)
        torch.manual_seed(seed)
        labels = self.target_transform(label)

        #inputs = inputs[:,np.newaxis]


        #labels = labels[:, np.newaxis]

        #label_conversion = np.abs(labels - 1)

        #labels = np.concatenate((labels, label_conversion), axis=0)

        print('input_raw:', len(input_raw))
        print('label_raw:', len(label_raw))
        print('input:', len(input))
        print('label:', len(label))





        data_t = {'Input_ID': self.lst_input[index].split('/')[-1],'Label_ID': self.lst_label[index].split('/')[-1],'input' : inputs,'label' :labels ,'input_raw' : input_raw,'label_raw' : label_raw }
        data = data_t

        return data
## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        inputss = data.transpose((2, 0, 1)).astype(np.float32)
        return inputss

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
