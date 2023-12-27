import albumentations as A
import numpy as np
import pandas as pd

import os
import sys
import glob
import datetime
import random
import string

import cv2


#----------------------------------------------------------------------------

class TEST(object):
    def __init__(self) -> None:
        ...
        
    def random_state():
        ...
        
    def aug_seq(self):
        transform = A.Compose([
            A.HorizontalFlip(p=1),
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=None, remove_invisible=True, angle_in_degrees=True))
        
        return transform


def main():
    T = TEST()
    
    image_path = os.path.join('data', 'test_img', 'JJI.jpg')
    
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    keypoints = [
            [(10,10)],
            [(50,10)],
    ]
    
    seq = T.aug_seq()
    
    # transformed = seq(image=image)
    transformed = seq(image=image, keypoints=keypoints)
    
    transformed_image = transformed["image"]
    transformed_keypoints = transformed["keypoints"]
    
    print(transformed_keypoints)
    
    cv2.imshow('t_image', transformed_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    
    
if __name__ == "__main__":
    main()