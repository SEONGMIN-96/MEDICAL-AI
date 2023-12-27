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

from sklearn.model_selection import train_test_split


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
            if image is not None:
                image = cv2.resize(src=image, dsize=(new_width, new_height)) / 255.0
                resized_img_lst.append(image)
            else:
                raise ValueError("이미지 로드 실패")
            
        resized_imgs = np.array(resized_img_lst)
        
        object_dict['input_image'] = resized_imgs
        
        return object_dict
    
    def resize_n_normalization_v2(self, object_list: list, new_width: int, new_height: int):
        '''resize and normalization images
        Args:
            object_list : 데이터셋 관련 정보 
            
            new_width : resize 시 활용될 new width
            
            new_height : resize 시 활용될 new height
            
        '''
        
        result_img_lst = []
        
        for imgs_path in object_list:
            resized_img_lst = []    
            for img_path in imgs_path:
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.resize(src=image, dsize=(new_width, new_height)) / 255.0
                    resized_img_lst.append(image)
                else:
                    raise ValueError("이미지 로드 실패")
            result_img_lst.append(resized_img_lst)
            
        result_img_lst = np.array(result_img_lst)
        
        return result_img_lst

    def resize_n_normalization_v3(self, object_list: list, M_list: list, new_width: int, new_height: int):
        '''resize and normalization images
        Args:
            object_list : 데이터셋 관련 정보 
            
            new_width : resize 시 활용될 new width
            
            new_height : resize 시 활용될 new height
            
        '''
        
        result_img_lst = []
        
        for (imgs_path, m_path) in zip(object_list, M_list):
            resized_img_lst = []    
            for img_path in imgs_path:
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.resize(src=image, dsize=(new_width, new_height)) / 255.0
                    resized_img_lst.append(image)
                else:
                    raise ValueError("이미지 로드 실패")
            result_img_lst.append(resized_img_lst)
            
        result_img_lst = np.array(result_img_lst)
        
        return result_img_lst

    def define_train_test_val_v1(self, excel_file_path:str):
        # 엑셀을 불러옵니다.
        df = pd.read_excel(excel_file_path, engine="openpyxl")
        
        # 이미지 path와 label값들을 리스트화합니다.
        img_paths_lst = list(df["image_paths"])
        labels_lst = [int(elem) for elem in list(df["labels"])]
        
        # train test validation 8:1:1로 나누기위한 기준 값을 정의합니다.
        number_of_train_data = int(len(img_paths_lst) * 0.8)
        number_of_test_data = int(len(img_paths_lst) * 0.1)
        number_of_val_data = int(len(img_paths_lst) * 0.1)
        
        # train, test, validation의 input, label 리스트를 정의합니다.
        train_input_lst, train_label_lst = [], []
        test_input_lst, test_label_lst = [], []
        val_input_lst, val_label_lst = [], []
        
        # 동일인물의 데이터가 섞이면 안되니 한번에 전달하기 위해 담아둘 리스트를 정의합니다.
        same_name_box = []
        same_name_path_box, same_name_label_box = [], []
        
        for i, (path, label) in enumerate(zip(img_paths_lst, labels_lst)):
            # 동일인물 데이터를 확인하기 위한 file name을 정의합니다.
            fname = path.split('/')[-1].split('_')[0][:-1]
                        
            # same_name_box 리스트가 비어있을경우, fname, path, label값을 담아줍니다.
            if len(same_name_box) == 0:
                same_name_box.append(fname)
                same_name_path_box.append(path)
                same_name_label_box.append(label)
                
            # same_name_box의 마지막 요소가 현재 fname과 같을경우,
            # 동일한 환자이기 때문에 same_name_box에 담아줍니다.
            elif same_name_box[-1] == fname:
                same_name_box.append(fname)
                same_name_path_box.append(path)
                same_name_label_box.append(label)
                     
            # same_name_box의 마지막 요소와 현재 fname이 같지 않을경우,
            # 담긴 path, label을 8:1:1기준으로 전달합니다.
            elif same_name_box[-1] != fname:
                # train_input_lst의 크기가 정의해놓은 0.8값보다 작거나 같은경우,
                # 담아놓은 데이터를 train data로 옮깁니다.
                # 그 후 same_name_box를 초기화한뒤, 처음보는 fname을 same_name_box에 담아줍니다.
                if len(train_input_lst) < number_of_train_data: 
                    train_input_lst.extend(same_name_path_box)
                    train_label_lst.extend(same_name_label_box)
                    
                    same_name_box.clear()
                    same_name_path_box.clear()
                    same_name_label_box.clear()
                    
                    same_name_box.append(fname)
                    same_name_path_box.append(path)
                    same_name_label_box.append(label)
                
                # test_input_lst의 크기가 정의해놓은 0.1값보다 작거나 같은경우,
                # 담아놓은 데이터를 test data로 옮깁니다.
                # 그 후 same_name_box를 초기화합니다.
                # 그 후 same_name_box를 초기화한뒤, 처음보는 fname을 same_name_box에 담아줍니다.
                elif len(test_input_lst) < number_of_test_data:
                    test_input_lst.extend(same_name_path_box)
                    test_label_lst.extend(same_name_label_box)
                    
                    same_name_box.clear()
                    same_name_path_box.clear()
                    same_name_label_box.clear()
                    
                    same_name_box.append(fname)
                    same_name_path_box.append(path)
                    same_name_label_box.append(label)
                
                # 나머지 모든 데이터를 val data로 옮깁니다.
                # 그 후 same_name_box를 초기화한뒤, 처음보는 fname을 same_name_box에 담아줍니다.
                else:
                    val_input_lst.extend(same_name_path_box)
                    val_label_lst.extend(same_name_label_box)
                    
                    same_name_box.clear()
                    same_name_path_box.clear()
                    same_name_label_box.clear()
                    
                    same_name_box.append(fname)
                    same_name_path_box.append(path)
                    same_name_label_box.append(label)
            
            # 원본데이터의 마지막 구간은 앞의 정의를 무시하고 버려질수 있으니
            # 모두 val data로 전달하며 데이터 분할을 마무리합니다.
            if i+1 == len(img_paths_lst):
                val_input_lst.extend(same_name_path_box)
                val_label_lst.extend(same_name_label_box)
                
                same_name_box.clear()
                same_name_path_box.clear()
                same_name_label_box.clear()
        
        print(np.unique(np.array(train_label_lst)))
        print(np.unique(np.array(test_label_lst)))
        print(np.unique(np.array(val_label_lst)))
        
        print("total_data_len:", len(img_paths_lst))
        print("train_data_len:", len(train_input_lst))
        print("test_data_len:", len(test_input_lst))
        print("val_data_len:", len(val_input_lst))
        
        train_input_data_paths, train_label_data = self.list_shuffle(list1=train_input_lst,
                                                                     list2=train_label_lst)
        test_input_data_paths, test_label_data = self.list_shuffle(list1=test_input_lst,
                                                                   list2=test_label_lst)
        val_input_data_paths, val_label_data = self.list_shuffle(list1=val_input_lst,
                                                                 list2=val_label_lst)
        
        train_dict = {"input_path": train_input_data_paths,
                      "input_label": train_label_data
        }
        test_dict = {"input_path": test_input_data_paths,
                     "input_label": test_label_data
        }
        val_dict = {"input_path": val_input_data_paths,
                     "input_label": val_label_data
        }
        
        return train_dict, test_dict, val_dict
    
    def define_train_test_val_v2(self, excel_file_path:str):
        # 1. 엑셀을 불러온다.
        # 2. 일련번호(폴더명)가 같은 path의 인덱스끼리 그룹을 생성한다.
        # 3. 해당 그룹을 sklearn train_test_split(stratify=T)를 설정해, 데이터 불균형을 해소한다.
        # 4. Return 값인 train, test 로 나눈뒤, test데이터를 한번더 나눠 test, val 데이터를 구성한다.
        # 5. 이후 일련번호를 통해 데이터셋을 다시 구성하면 train,test,val데이터를 나눌 수 있다.
        
        
        # 1. 엑셀을 불러온다.
        df = pd.read_excel(excel_file_path, engine="openpyxl")
        
        # 이미지 path와 label값들을 리스트화합니다.
        img_paths_lst = list(df["image_paths"])
        labels_lst = [int(elem) for elem in list(df["labels"])]
        
        # 2. 일련번호(폴더명)가 같은 path의 인덱스끼리 그룹을 생성한다.
        data_idx = {}
        data = {}
        
        for i, (path, label) in enumerate(zip(img_paths_lst, labels_lst)):
            fname = path.split('/')[-1].split('_')[0]
            
            try:
                data_idx[fname].append(i)
            except:
                data_idx[fname] = [i]
                data[fname] = [label]
        
        # 3. 해당 그룹을 sklearn train_test_split(stratify=T)를 설정해, 데이터 불균형을 해소한다.        
        X_idx_train, X_temp, y_idx_train, y_temp = train_test_split(list(data.keys()), list(data.values()), 
                                                            test_size=0.2, random_state=42, stratify=list(data.values()))
        
        # 4. Return 값인 train, test 로 나눈뒤, test데이터를 한번더 나눠 test, val 데이터를 구성한다.
        X_idx_val, X_idx_test, y_idx_val, y_idx_test = train_test_split(X_temp, y_temp,
                                                        test_size=0.5, random_state=42, stratify=y_temp)
        
        X_train, y_train = [], []
        X_test, y_test = [], []
        X_val, y_val = [], []
        
        for idx_key in data_idx:
            if idx_key in X_idx_train:
                idx = X_idx_train.index(idx_key)
                
                for idx_key in data_idx[idx_key]:
                    X_train.append(img_paths_lst[idx_key])
                    y_train.append(y_idx_train[idx][0])
                
            elif idx_key in X_idx_test:
                idx = X_idx_test.index(idx_key)
                
                for idx_key in data_idx[idx_key]:
                    X_test.append(img_paths_lst[idx_key])
                    y_test.append(y_idx_test[idx][0])
            elif idx_key in X_idx_val:
                idx = X_idx_val.index(idx_key)
                
                for idx_key in data_idx[idx_key]:
                    X_val.append(img_paths_lst[idx_key])
                    y_val.append(y_idx_val[idx][0])
                            
        data = {"X_train": X_train, "y_train": y_train,
                "X_test": X_test, "y_test": y_test,
                "X_val": X_val, "y_val": y_val}

        return data
    
    def list_shuffle(self, list1:list, list2:list):
        # 입력 데이터와 레이블 데이터의 길이가 같은지 확인
        if len(list1) != len(list2):
            raise ValueError("입력 데이터와 레이블 데이터의 길이가 같아야 합니다.")

        # 입력 데이터와 레이블 데이터를 동일한 순서로 섞기 위해 인덱스 리스트 생성
        indices = list(range(len(list1)))

        # 인덱스 리스트를 섞음
        random.shuffle(indices)

        # 섞인 인덱스 순서에 따라 입력 데이터와 레이블 데이터를 새로운 리스트로 생성
        shuffled_list1 = np.array([list1[i] for i in indices])
        shuffled_list2 = np.array([list2[i] for i in indices])

        # 입력 데이터와 레이블 데이터의 길이가 같은지 확인
        if len(shuffled_list1) != len(shuffled_list2):
            raise ValueError("입력 데이터와 레이블 데이터의 길이가 같아야 합니다.")
        
        return shuffled_list1, shuffled_list2
    
    def add_img_mask(self, img_path, mask_path):
        # 원본 이미지 로드
        original_image = cv2.imread(img_path)
        # 마스크 이미지 로드
        mask_image = cv2.imread(mask_path)

        width = original_image.shape[1]
        height = original_image.shape[0]
        
        mask_image = cv2.resize(mask_image, (width, height))
        
        # 마스크를 원본 이미지에 합성
        composite_image = cv2.addWeighted(original_image, 1, mask_image, 0.3, 0)
        
        # part = mask_path.split('/')
        # new_path = os.path.join('bin', 'data', '1-cycle', 'usable_data', part[-4], 'add_img', part[-2], part[-1])
        # cv2.imwrite(new_path, composite_image)

        return composite_image