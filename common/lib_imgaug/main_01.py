import os
import sys
import glob
import json

import numpy as np
import pandas as pd 

import cv2
import imgaug as ia
import imgaug.augmenters as iaa


#----------------------------------------------------------------------------

def image_load(PATH):
    return cv2.imread(PATH)

def json_load(PATH):
    with open(PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def img_aug():
    seq = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=0.05*255),
        iaa.Affine(translate_px={})
    ])


#----------------------------------------------------------------------------

def main():
    '''
    샘플 데이터
    img_aug 활용 데이터 augmentation
    '''
    meta_data_path_lst = glob.glob(os.path.join('data', 'label_meta', 'meta', 'Face', '*.json'))
    
    meta_S_path = meta_data_path_lst[0]
    
    meta_data = json_load(meta_S_path)
    
    img_id = meta_data['data_key']
    label_id = meta_data['label_id']
    label_path = meta_data['label_path']
    age_info = meta_data['age']
    frames = meta_data['frames']
    
    label = json_load(os.path.join('data', 'label_meta', label_path[0]))
    img_path_lst = glob.glob(os.path.join('data', 'face_img', age_info, img_id, '*.jpg'))

    # for i in range(len(frames)):
    img = image_load(img_path_lst[0])
    poly_lst = []
    for j in range(len(label['objects'])):
        if int(label['objects'][j]['frames'][0]['num']) == int(0):
            point_lst = []
            if str(label['objects'][j]['frames'][0]['annotation']['coord']['points'][0])[1] == '[':
                for face_area in label['objects'][j]['frames'][0]['annotation']['coord']['points'][0][0]:
                    point_lst.append((face_area['x'], face_area['y']))      
            else:
                for face_area in label['objects'][j]['frames'][0]['annotation']['coord']['points'][0]:
                    point_lst.append((face_area['x'], face_area['y']))      
            
            poly_lst.append(point_lst)
        
    POLYGONS = []
    for i in range(len(poly_lst)):
        POLYGONS.append(
            ia.Polygon(poly_lst[i])
            )
            
    
    seq = iaa.Sequential([
    iaa.Affine(
        scale={"x": (0.8, 0.8), "y": (0.8, 0.8)},
        rotate=(60)
        )
    ])

    img_output, polygon_output = seq(image=img, polygons=POLYGONS)

    print(POLYGONS[1])
    
    print(polygon_output[1])

    # poly_lst1 = []
    # for poly in polygon_output[0]:
        # poly_lst1.append(poly)
        # img = cv2.polylines(img_output, [np.array(poly_lst1, dtype=np.int32)], True, (0,69,255), 3)    

    # cv2.imwrite('aaa.jpg', img_output)
    
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()