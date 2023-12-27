import os
import sys
import itertools
import shutil

import matplotlib.pyplot as plt
import numpy as np


class Postprocessing():
    def __init__(self) -> None:
        ...
        
    def wrong_pred_save_img(self, y_pred: int, y_true: int, input_path: str, exp_path: str, class_lst: np.array):
        """Save mispredicted images 
            
        Args:
            ...

        Return:
            Store incorrectly predicted images in a given path
        """
        
        f_path_00 = os.path.join('bin', 'exp', exp_path, 'wrong_img')
        if not os.path.exists(f_path_00):
            os.mkdir(f_path_00)
        
        for i in range(len(y_pred)):
            if y_pred[i] != y_true[i]:
                # f_path_01 = os.path.join(f_path_00, input_path[i].split('/')[-1].split('_')[0])
                f_path_01 = os.path.join(f_path_00, input_path[i].split('\\')[-1].split('_')[0])
                
                if not os.path.exists(f_path_01):
                    os.mkdir(f_path_01)
                    
                PRED = y_pred[i]
                # NAME = input_path[i].split('/')[-1]
                NAME = input_path[i].split('\\')[-1]
                
                shutil.copy(src=input_path[i], dst=os.path.join(f_path_01, f"{class_lst[PRED]}_{NAME}")) 