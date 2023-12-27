import os
import yaml
import glob
import shutil
import random
import datetime

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

#----------------------------------------------------------------------------

class DataCreateStream():
    def __init__(self) -> None:
        ...
        
    def to_categorical(self, labels, num_classes):
        categorical_labels = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            categorical_labels[i, label] = 1
        return categorical_labels

    def resize_dataset(self, NUM):
        '''
        reisezed images
        '''
        if not os.path.exists(os.path.join('bin', 'data', f'new_pill_list_{NUM}')):
            os.mkdir(os.path.join('bin', 'data', f'new_pill_list_{NUM}'))

        FNAME_LST = glob.glob(os.path.join('bin', 'data', 'old_pill_list', '*'))
        
        for fname in FNAME_LST:
            if not os.path.exists(os.path.join('bin', 'data', f'new_pill_list_{NUM}', fname.split('\\')[-1])):
                os.mkdir(os.path.join('bin', 'data', f'new_pill_list_{NUM}', fname.split('\\')[-1]))
            
            FILES = glob.glob(os.path.join(fname, 'IMG?*.png'))
            
            for f in FILES:
                image = cv2.imread(f)
                resized_image = cv2.resize(src=image, dsize=(0,0), fx=0.05, fy=0.05)
                
                NEW_FNAME = os.path.join('bin', 'data', f'new_pill_list_{NUM}', fname.split('\\')[-1], f.split('\\')[-1])
                
                cv2.imwrite(filename=NEW_FNAME, img=resized_image)

    def ttv_split(self, NUM):
        '''
        train test validation split
        '''
        FNAME_LST = glob.glob(os.path.join('bin', 'data', f'new_pill_list_{NUM}', '*'))

        # if want to make new dataset directory change 'dataset_##'
        DATASET_PATH = os.path.join('bin', 'data', f'dataset_{NUM}')

        if not os.path.exists(DATASET_PATH):
            os.mkdir(DATASET_PATH)
            os.mkdir(os.path.join(DATASET_PATH, 'train'))
            os.mkdir(os.path.join(DATASET_PATH, 'test'))
            os.mkdir(os.path.join(DATASET_PATH, 'val'))
        
        for fname in FNAME_LST:
            TRAIN_PATH = os.path.join(DATASET_PATH, 'train', fname.split('\\')[-1])
            TEST_PATH = os.path.join(DATASET_PATH, 'test', fname.split('\\')[-1])
            VAL_PATH = os.path.join(DATASET_PATH, 'val', fname.split('\\')[-1])
            
            if not os.path.exists(TRAIN_PATH):
                os.mkdir(TRAIN_PATH)
                os.mkdir(TEST_PATH)
                os.mkdir(VAL_PATH)
            
            FILES = glob.glob(os.path.join(fname, 'IMG?*.png'))
            
            count = 0
            
            for f in FILES:
                if count < 14:
                    shutil.copy(src=f, dst=os.path.join(TRAIN_PATH, f.split('\\')[-1]))
                elif 13 < count < 19:
                    shutil.copy(src=f, dst=os.path.join(VAL_PATH, f.split('\\')[-1]))
                elif 18 < count < 20:
                    shutil.copy(src=f, dst=os.path.join(TEST_PATH, f.split('\\')[-1]))
                elif 19 < count < 34:
                    shutil.copy(src=f, dst=os.path.join(TRAIN_PATH, f.split('\\')[-1]))
                elif 33 < count < 39:
                    shutil.copy(src=f, dst=os.path.join(VAL_PATH, f.split('\\')[-1]))
                elif 38 < count < 40:
                    shutil.copy(src=f, dst=os.path.join(TEST_PATH, f.split('\\')[-1]))    
                
                    
                count += 1

    def data_split(self, 
                   class_box: list,
                   data_dir: str,
        ):
        '''load img path & split train test val 
        '''
        all_train_path_lst, all_test_path_lst, all_val_path_lst = [], [], []
        all_train_label_lst, all_test_label_lst, all_val_label_lst = [], [], []

        # count all classes & define max count
        class_num_max = dict()
        
        for class_object in class_box:
            class_name = class_object[0]

            if len(class_object) > 1:
                img_path_lst = []
                for i in range(len(class_object)):
                    img_path_lst_i = glob.glob(os.path.join(data_dir, class_object[i], '*.jpg'))
                    img_path_lst.extend(img_path_lst_i)
            else:
                img_path_lst = glob.glob(os.path.join(data_dir, class_object[0], '*.jpg'))

            print(f'Class Object: {class_object}')
            print(f'Count: {len(img_path_lst)}')
            print('=='*50)

            class_num_max[class_object[0]] = len(img_path_lst)
        
        min_value = min(class_num_max.values())
        
        train_count = int(min_value*0.8)
        test_count = int(min_value*0.085)
        val_count = int(min_value*0.085)
        
        # tqdm
        pbar = tqdm(total=len(class_box))
        
        for class_object in class_box:
            class_name = class_object[0]
            
            if len(class_object) > 1:
                img_path_lst = []
                for i in range(len(class_object)):
                    img_path_lst_i = glob.glob(os.path.join(data_dir, class_object[i], '*.jpg'))
                    img_path_lst.extend(img_path_lst_i)
            else:
                img_path_lst = glob.glob(os.path.join(data_dir, class_object[0], '*.jpg'))
            
            all_p_num_lst, p_num_lst = [], []
            p_num_count = {}
            
            # 환자 번호 리스트 변환, 환자 번호당 이미지 개수 딕셔너리 변환
            for img_path in img_path_lst:
                p_num = img_path.split('/')[-1].split('_')[0]
                
                all_p_num_lst.append(p_num)
                
                if p_num not in p_num_lst:
                    p_num_lst.append(p_num)
            
            for p_num in all_p_num_lst:
                try:
                    p_num_count[p_num] += 1
                except:
                    p_num_count[p_num] = 1
            
            count_01, count_02, count_03 = 0, 0, 0
            train_num_lst, val_num_lst, test_num_lst = [], [], []
            
            # 중복되지 않게 정해진 수만큼 train, val, test 셋 환자번호 랜덤구성
            while count_01 <= train_count:
                r_p_num = random.choice(list(p_num_count.keys()))
        
                if r_p_num not in train_num_lst:
                    train_num_lst.append(r_p_num)
                    count_01 += p_num_count[r_p_num]
            
            while count_02 <= test_count:
                r_p_num = random.choice(list(p_num_count.keys()))
        
                if r_p_num not in train_num_lst and r_p_num not in test_num_lst:
                    test_num_lst.append(r_p_num)
                    count_02 += p_num_count[r_p_num]
                    
            while count_03 <= val_count:
                r_p_num = random.choice(list(p_num_count.keys()))
        
                if r_p_num not in train_num_lst and r_p_num not in test_num_lst and r_p_num not in val_num_lst:
                    val_num_lst.append(r_p_num)
                    count_03 += p_num_count[r_p_num]
            
            # 환자번호를 기준으로 데이터 경로 정리 뒤, 초과되는 데이터 pop
            train_path_lst = self.list_pop(img_path_lst=img_path_lst,
                                        num_lst=train_num_lst,
                                        o_count=train_count)
            test_path_lst = self.list_pop(img_path_lst=img_path_lst,
                                        num_lst=test_num_lst,
                                        o_count=test_count)
            val_path_lst = self.list_pop(img_path_lst=img_path_lst,
                                        num_lst=val_num_lst,
                                        o_count=val_count)
            
            # 클래스당 완성된 train, test, val 경로 파일을 전체 train, test, val 경로 파일에 병합
            all_train_path_lst.extend(train_path_lst)
            all_test_path_lst.extend(test_path_lst)
            all_val_path_lst.extend(val_path_lst)
            
            train_label_lst = [class_name for _ in range(len(train_path_lst))]
            test_label_lst = [class_name for _ in range(len(test_path_lst))]
            val_label_lst = [class_name for _ in range(len(val_path_lst))]
            
            all_train_label_lst.extend(train_label_lst)
            all_test_label_lst.extend(test_label_lst)
            all_val_label_lst.extend(val_label_lst)
            
            pbar.update(1)
        
        all_train_label_lst, train_input_id = self.text_label_categorical(input_label=all_train_label_lst, data_classes=class_box)
        all_test_label_lst, test_input_id = self.text_label_categorical(input_label=all_test_label_lst, data_classes=class_box)
        all_val_label_lst, val_input_id = self.text_label_categorical(input_label=all_val_label_lst, data_classes=class_box)
            
        train_dict = {'input_path': all_train_path_lst, 
                    'input_label': all_train_label_lst, 
                    'input_id': train_input_id, 
                    'input_class' : class_box,
                    }
        test_dict = {'input_path': all_test_path_lst, 
                    'input_label': all_test_label_lst, 
                    'input_id': test_input_id, 
                    'input_class' : class_box,
                    }
        val_dict = {'input_path': all_val_path_lst, 
                    'input_label': all_val_label_lst, 
                    'input_id': val_input_id, 
                    'input_class' : class_box
                    }
            
        return train_dict, test_dict, val_dict

    def list_pop(self, img_path_lst: list, num_lst: list, o_count: int):
        '''
        Args:
            img_path_lst : 전체 이미지 경로 list
            
            num_lst : 정해진 count 만큼 랜덤으로 채워진 환자번호 list
            
            o_count : 데이터셋 수량 지정 count
            
        Note:
            num_lst에서 지정된 환자번호가 o_count를 초과할 경우, 뒤에서 부터 pop 수행 -> o_count에 맞게 데이터셋 정리
        '''
        path_lst = []
                
        for target_num in num_lst:
            for img_path in img_path_lst:
                p_num = img_path.split('/')[-1].split('_')[0]
                
                if target_num == p_num:
                    path_lst.append(img_path)
        
        if len(path_lst) > o_count:
            excess_count = len(path_lst) - o_count
            for _ in range(excess_count):
                path_lst.pop(-1)
                
        return path_lst

    def new_list_pop(self, img_path_lst: list, num_lst: list, o_count: int):
        '''
        Args:
            img_path_lst : 전체 이미지 경로 list
            
            num_lst : 정해진 count 만큼 랜덤으로 채워진 환자번호 list
            
            o_count : 데이터셋 수량 지정 count
            
        Note:
            num_lst에서 지정된 환자번호가 o_count를 초과할 경우, 뒤에서 부터 pop 수행 -> o_count에 맞게 데이터셋 정리
        '''
        path_lst = []
                
        for target_num in num_lst:
            for img_path in img_path_lst:
                p_num = img_path.split('/')[-1].split('_')[0]
                
                if target_num == p_num:
                    path_lst.append(img_path)
        
        # if len(path_lst) > o_count:
        #     excess_count = len(path_lst) - o_count
        #     for _ in range(excess_count):
        #         path_lst.pop(-1)
                
        return path_lst

    def text_label_categorical(self, 
                               input_label: list, 
                               data_classes: list
        ):
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
        
        # new_input_label = tf.keras.utils.to_categorical(np.array(new_input_label))
        new_input_label = self.to_categorical(labels=new_input_label,
                                              num_classes=len(data_classes))
        # encoder = OneHotEncoder(sparse=False)
        # new_input_label = encoder.fit_transform(new_input_label)
        
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
        
        object_dict['input_images'] = resized_imgs
        
        return object_dict

