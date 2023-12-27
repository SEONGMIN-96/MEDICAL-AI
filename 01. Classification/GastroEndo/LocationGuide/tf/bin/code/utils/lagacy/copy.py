from .preprocess import read_roi, extract_roi_box, crop_img

import os
import sys
import glob
import shutil
import cv2

def cp(f_lst, f_n):
    for f in f_lst:
        shutil.copy(src=f, dst=os.path.join('bin', 'data', 'class', f_n, f.split('\\')[-1]) )

def main():
    files = glob.glob(os.path.join('bin', 'data', '02_normal_1000', '01714903', '*.jpg'))
    
    ES, GE, CR, UB, MB, LB, AG, AT, DU = [], [], [], [], [], [], [], [], []
    NO = []
    
    for file in files:
        state = file.split('\\')[-1].split('_')[4]
        
        NAME = state[0] + state[1]
        
        if NAME == 'NO':
            NO.append(file)            
        elif NAME == 'ES':
            ES.append(file)
        elif NAME == 'GE':
            GE.append(file)
        elif NAME == 'CR':
            CR.append(file)
        elif NAME == 'UB':
            UB.append(file)
        elif NAME == 'MB':
            MB.append(file)
        elif NAME == 'LB':
            LB.append(file)
        elif NAME == 'AG':
            AG.append(file)
        elif NAME == 'AT':
            AT.append(file)
        elif NAME == 'DU':
            DU.append(file)
    
    cp(NO, 'NO')
    cp(ES, 'ES')        
    cp(GE, 'GE')
    cp(CR, 'CR')
    cp(UB, 'UB')
    cp(MB, 'MB')
    cp(LB, 'LB')
    cp(AG, 'AG')
    cp(AT, 'AT')
    cp(DU, 'DU')
    

if __name__ == '__main__':
    main()