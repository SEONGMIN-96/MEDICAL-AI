import os
import sys
import glob
import json

import numpy as np
import pandas as pd 

import cv2
import imgaug


#----------------------------------------------------------------------------

def image_load(PATH):
    return cv2.imread(PATH)

def json_load(PATH):
    with open(PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


#----------------------------------------------------------------------------

def main():
    '''
    어노테이션 정보 해당 이미지 위에 뿌려주기
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

    for i in range(len(frames)):
        img = image_load(img_path_lst[i])
        
        for j in range(len(label['objects'])):
            if int(label['objects'][j]['frames'][0]['num']) == int(i):
                point_lst = []
                if str(label['objects'][j]['frames'][0]['annotation']['coord']['points'][0])[1] == '[':
                    for face_area in label['objects'][j]['frames'][0]['annotation']['coord']['points'][0][0]:
                        point_lst.append([face_area['x'], face_area['y']])      
                else:
                    for face_area in label['objects'][j]['frames'][0]['annotation']['coord']['points'][0]:
                        point_lst.append([face_area['x'], face_area['y']])      

                img = cv2.polylines(img, [np.array(point_lst, dtype=np.int32)], True, (0,69,255), 3)
        
        cv2.imwrite(f'sample_{i}.jpg', img)
        # cv2.imshow('img', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()    

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()