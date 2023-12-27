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


#----------------------------------------------------------------------------

def resize_dataset(NUM):
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

def ttv_split(NUM):
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

def data_split(TRAIN_COUNT: int, TEST_COUNT: int, VAL_COUNT: int, class_lst: list):
    '''load img path & split train test val 
    '''
    # img path 불러옴
    class_path = os.path.join('bin', 'data', 'src', 'class')

    all_train_path_lst, all_test_path_lst, all_val_path_lst = [], [], []
    all_train_label_lst, all_test_label_lst, all_val_label_lst = [], [], []
        
    for class_object in class_lst:
        class_name = class_object[0]
        
        train_count = int(TRAIN_COUNT)
        test_count = int(TEST_COUNT)
        val_count = int(VAL_COUNT)

        if len(class_object) > 1:
            img_path_lst = []
            for i in range(len(class_object)):
                img_path_lst_i = glob.glob(os.path.join(class_path, class_object[i], '*.jpg'))
            
                img_path_lst.extend(img_path_lst_i)
        else:
            img_path_lst = glob.glob(os.path.join(class_path, class_object[0], '*.jpg'))
        
        print(f'Class Object: {class_object}')
        print(f'Count: {len(img_path_lst)}')
        
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
        train_path_lst = list_pop(img_path_lst=img_path_lst,
                                    num_lst=train_num_lst,
                                    o_count=train_count)
        test_path_lst = list_pop(img_path_lst=img_path_lst,
                                    num_lst=test_num_lst,
                                    o_count=test_count)
        val_path_lst = list_pop(img_path_lst=img_path_lst,
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
    
    # 
    all_train_label_lst, train_input_id = text_label_categorical(input_label=all_train_label_lst, class_lst=class_lst)
    all_test_label_lst, test_input_id = text_label_categorical(input_label=all_test_label_lst, class_lst=class_lst)
    all_val_label_lst, val_input_id = text_label_categorical(input_label=all_val_label_lst, class_lst=class_lst)
        
    train_dict = {'input_path': all_train_path_lst, 
                  'input_label': all_train_label_lst, 
                  'input_id': train_input_id, 
                  'input_class' : class_lst
                  }
    test_dict = {'input_path': all_test_path_lst, 
                 'input_label': all_test_label_lst, 
                 'input_id': test_input_id, 
                 'input_class' : class_lst
                 }
    val_dict = {'input_path': all_val_path_lst, 
                'input_label': all_val_label_lst, 
                'input_id': val_input_id, 
                'input_class' : class_lst
                }
        
    return train_dict, test_dict, val_dict

def data_split_all_count(TRAIN_COUNT: int, TEST_COUNT: int, VAL_COUNT: int, class_lst: list):
    '''load img path & split train test val 
    '''
    # img path 불러옴
    class_path = os.path.join('bin', 'data', 'src', 'class')

    all_train_path_lst, all_test_path_lst, all_val_path_lst = [], [], []
    all_train_label_lst, all_test_label_lst, all_val_label_lst = [], [], []
        
    for class_object in class_lst:
        class_name = class_object[0]
        
        train_count = int(TRAIN_COUNT)
        test_count = int(TEST_COUNT)
        val_count = int(VAL_COUNT)

        if len(class_object) > 1:
            img_path_lst = []
            for i in range(len(class_object)):
                img_path_lst_i = glob.glob(os.path.join(class_path, class_object[i], '*.jpg'))
            
                img_path_lst.extend(img_path_lst_i)
        else:
            img_path_lst = glob.glob(os.path.join(class_path, class_object[0], '*.jpg'))
        
        print(f'Class Object: {class_object}')
        print(f'Count: {len(img_path_lst)}')
        
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
        # test, val 셋 구성후 나머지 데이터 전부 train 셋으로 통합
        # while count_01 <= train_count:
        #     r_p_num = random.choice(list(p_num_count.keys()))
    
        #     if r_p_num not in train_num_lst:
        #         train_num_lst.append(r_p_num)
        #         count_01 += p_num_count[r_p_num]
        
        while count_02 <= test_count:
            r_p_num = random.choice(list(p_num_count.keys()))
    
            if r_p_num not in test_num_lst:
                test_num_lst.append(r_p_num)
                count_02 += p_num_count[r_p_num]
                
        while count_03 <= val_count:
            r_p_num = random.choice(list(p_num_count.keys()))
    
            if r_p_num not in test_num_lst and r_p_num not in val_num_lst:
                val_num_lst.append(r_p_num)
                count_03 += p_num_count[r_p_num]
                
        while True:
            r_p_num = random.choice(list(p_num_count.keys()))
    
            if r_p_num not in train_num_lst and r_p_num not in test_num_lst and r_p_num not in val_num_lst:
                train_num_lst.append(r_p_num)
                count_01 += p_num_count[r_p_num]
                
            if len(train_num_lst) + len(val_num_lst) + len(test_num_lst) == len(list(p_num_count.keys())):
                break
        
        # 환자번호를 기준으로 데이터 경로 정리 뒤, 초과되는 데이터 pop
        train_path_lst = new_list_pop(img_path_lst=img_path_lst,
                                    num_lst=train_num_lst,
                                    o_count=train_count)
        test_path_lst = new_list_pop(img_path_lst=img_path_lst,
                                    num_lst=test_num_lst,
                                    o_count=test_count)
        val_path_lst = new_list_pop(img_path_lst=img_path_lst,
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
    
    # 
    all_train_label_lst, train_input_id = text_label_categorical(input_label=all_train_label_lst, class_lst=class_lst)
    all_test_label_lst, test_input_id = text_label_categorical(input_label=all_test_label_lst, class_lst=class_lst)
    all_val_label_lst, val_input_id = text_label_categorical(input_label=all_val_label_lst, class_lst=class_lst)
        
    train_dict = {'input_path': all_train_path_lst, 
                  'input_label': all_train_label_lst, 
                  'input_id': train_input_id, 
                  'input_class' : class_lst
                  }
    test_dict = {'input_path': all_test_path_lst, 
                 'input_label': all_test_label_lst, 
                 'input_id': test_input_id, 
                 'input_class' : class_lst
                 }
    val_dict = {'input_path': all_val_path_lst, 
                'input_label': all_val_label_lst, 
                'input_id': val_input_id, 
                'input_class' : class_lst
                }
        
    return train_dict, test_dict, val_dict

def list_pop(img_path_lst: list, num_lst: list, o_count: int):
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

def new_list_pop(img_path_lst: list, num_lst: list, o_count: int):
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

def text_label_categorical(input_label: list, class_lst: list):
    '''
    Args:
        input_label : label 데이터로 구성된 list
    '''
    aaa = {}
    new_input_label = []

    for cls in class_lst:
        aaa[cls[0]] = len(aaa)
    
    for label in input_label:
        new_input_label.append(aaa.get(label))
    
    new_input_label = tf.keras.utils.to_categorical(np.array(new_input_label))
    
    return new_input_label, input_label

def resize_n_normalization_(object_dict: dict, new_width: int, new_height: int, now_time: str, folder: str):
    '''resize and normalization images
    Args:
        object_dict : 데이터셋 관련 정보 dict ex)input_path, input_label, input_id
        
        new_width : resize 시 활용될 new width
        
        new_height : resize 시 활용될 new height
        
        now_time : 폴더 생성 및 식별을 위해 생성된 시간 정보
        
        folder : train, val, test
    '''
    resized_img_lst = []
    
    for path in object_dict['input_path']:
        image = cv2.imread(path)
        image = cv2.resize(src=image, dsize=(new_width, new_height)) / 255.0
        resized_img_lst.append(image)
        
    resized_imgs = np.array(resized_img_lst)
    
    save_path = os.path.join('bin', 'data', now_time, folder)
    
    np.save(file=os.path.join(save_path, 'input_image'), arr=resized_imgs)
    np.save(file=os.path.join(save_path, 'input_label'), arr=np.array(object_dict['input_label']))
    np.save(file=os.path.join(save_path, 'input_id'), arr=np.array(object_dict['input_id']))
    
    df_path = pd.DataFrame(object_dict['input_path'])
    df_path.to_csv(os.path.join(save_path, 'input_path.csv'), index=False, header=False)
    df_label = pd.DataFrame(object_dict['input_class'])
    df_label.to_csv(os.path.join(save_path, 'input_class.csv'), index=False, header=False)
            

#----------------------------------------------------------------------------

def main():
    # now_time
    d = datetime.datetime.now()
    now_time = f"{d.year}-{d.month}-{d.day}-{d.hour}-{d.minute}-{d.second}"
    
    # class_lst = [['AG', 'LB-AG'], ['AT', 'LB-AT'], ['CR', 'UB-CR'], ['DU'], ['ES'], ['GE'], ['LB'], ['MB'], ['NO'], ['UB']]                                               # 10 class       
    # class_lst = [['AG', 'LB-AG'], ['AT', 'LB-AT'], ['CR', 'UB-CR'], ['DU'], ['ES'], ['GE'], ['MB', 'UB', 'LB'], ['NO']]                                                   # 8 class 
    # class_lst = [['ES', 'GE'], ['CR', 'UB-CR'], ['MB', 'UB', 'LB'], ['AG', 'LB-AG'], ['AT', 'LB-AT'], ['DU'], ['NO']]                                                     # 7 class
    # class_lst = [['ES', 'GE'], ['CR', 'UB-CR'], ['MB', 'UB', 'LB'], ['AG', 'LB-AG'], ['AT', 'LB-AT'], ['DU'], ['NO', 'B-UB', 'B-MB', 'B-LB']]                             # 7 class
    # class_lst = [['ES', 'GE'], ['CR', 'UB-CR'], ['MB', 'UB', 'LB'], ['AG', 'LB-AG'], ['AT', 'LB-AT'], ['SD', 'BB'], ['NO', 'NO-UB', 'NO-MB', 'NO-LB']]                    # 7 class
    # class_lst = [['ES', 'GE'], ['CR', 'UB-CR'], ['MB', 'UB', 'LB'], ['AG', 'LB-AG'], ['AT', 'LB-AT'], ['SD'], ['BB'], ['NO', 'NO-UB', 'NO-MB', 'NO-LB']]                  # 8 class
    # class_lst = [['ES', 'GE'], ['CR', 'UB-CR'], ['BODY', 'MB', 'UB', 'LB'], ['AG', 'MB-AG', 'LB-AG'], ['AT', 'MB-AT', 'LB-AT'], ['DU', 'SD', 'BB'], ['NO', 'UB-NO', 'MB-NO', 'LB-NO', 'SD-NO', 'BB-NO']]                  # 8 class
    # class_lst = [['UB'], ['MB'], ['LB']]                                                                                                                                     # 3 class
    # class_lst = [['SD'], ['BB']]                                                                                                                                     # 3 class
    class_lst = [['ES'], ['GE']]                                                                                                                                     # 3 class
    
    # all data use
    COUNT_ALL = False
    
    if COUNT_ALL == True:
        train_d, test_d, val_d = data_split_all_count(TRAIN_COUNT=1800, 
                                                      TEST_COUNT=100, 
                                                      VAL_COUNT=100,
                                                      class_lst=class_lst)
    else:
        train_d, test_d, val_d = data_split(TRAIN_COUNT=800, 
                                            TEST_COUNT=100, 
                                            VAL_COUNT=100,
                                            class_lst=class_lst)
    
    folder_name = os.path.join('bin', 'data', now_time)
    
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        os.mkdir(os.path.join(folder_name, 'train'))
        os.mkdir(os.path.join(folder_name, 'test'))
        os.mkdir(os.path.join(folder_name, 'val'))
        
    resize_n_normalization_(object_dict=train_d, 
                            new_height=256, new_width=256, 
                            now_time=now_time, folder='train')
    resize_n_normalization_(object_dict=test_d, 
                            new_height=256, new_width=256, 
                            now_time=now_time, folder='test')
    resize_n_normalization_(object_dict=val_d, 
                            new_height=256, new_width=256, 
                            now_time=now_time, folder='val')
    
    
#----------------------------------------------------------------------------

if __name__ == '__main__':
    main()