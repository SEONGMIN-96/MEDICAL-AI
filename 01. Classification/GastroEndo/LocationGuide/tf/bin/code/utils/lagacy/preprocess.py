import os
import glob
import sys

import cv2
import numpy as np

import shutil


#----------------------------------------------------------------------------

class Preprocess():
    def __init__(self) -> None:
        ...

    def histogram_equalize(self, IMG):
        '''
        이미지 히스토그램 평활화
        '''
        img_ycrcb = cv2.cvtColor(IMG, cv2.COLOR_BGR2YCrCb)
        ycrcb_planes = list(cv2.split(img_ycrcb))

        # 밝기 성분에 대해서만 히스토그램 평활화 수행
        ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
        
        dest_ycrcb = cv2.merge(ycrcb_planes)
        dest = cv2.cvtColor(dest_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        # 이미지 평활화 결과 확인
        cv2.imshow('src', IMG)
        cv2.imshow('dest', dest)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        return dest

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

    def aaa():
        ...
            
            