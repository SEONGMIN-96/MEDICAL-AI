import os
import sys
import glob
import json

import numpy as np
import pandas as pd 

import cv2
import imgaug as ia
import imgaug.augmenters as iaa

import string
import random

import datetime
import time

import argparse

import shutil


#----------------------------------------------------------------------------

def image_load(PATH):
    return cv2.imread(PATH)

def json_load(PATH):
    with open(PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def imgaug_seq(height, width):
    seq = iaa.Sequential(
    [
        iaa.Resize({"height": height, "width": width}),
        iaa.Fliplr(0.5),
        iaa.Affine(
            scale={"x": (0.5,1.0), "y": (0.5,1.0)},
            rotate=(-20,20)
        ),
        iaa.OneOf(
            [
                iaa.ElasticTransformation(alpha=(1,4.0), sigma=0.25), # move pixels locally around (with random strengths)
                iaa.PiecewiseAffine(scale=(0.01,0.065)), # sometimes move parts of the image around
                iaa.PerspectiveTransform(scale=(0.01,0.065))
            ]
        )
    ],
)
    
    return seq

def uuid_gen():
    LEN_01 = 8
    LEN_02 = 4
    LEN_03 = 4
    LEN_04 = 4
    LEN_05 = 12

    LEN_ = [LEN_01, LEN_02, LEN_03, LEN_04, LEN_05]

    string_pool = string.ascii_lowercase + string.digits

    exp = ""
    count = 0
    for LEN in LEN_:
        txt = ""
        if count > 0:
            exp += "-"
        for _ in range(LEN):
            txt += random.choice(string_pool)
        count += 1
        exp += txt 
            
    return exp


#----------------------------------------------------------------------------

def main(args):
    '''
    img_aug 활용 데이터 augmentation 
    resize
    label & meta data 생성
    '''
    
    TARGET = args.t
    NUM = 165
    
    count = 0
    
    NEW_height = 1000
    NEW_width = 1500
    
    # call sub info of label
    PORE = "53cf7aa8-bf2c-4323-933c-261615fa1b63" # 모공의 class_id
    ACNE = "c1eeed96-7416-44ff-aaa6-23a241728869" # 여드름의 class_id
    RED_SPOTS = "e15caa3b-a88f-4dc4-8949-a4f19096a5f3" # 홍반의 class_id
    PIGMENTATION = "2cc91760-a687-4126-8787-74325b072217" # 색소침착의 class_id
    WRINKLE = "8d2c0a27-3481-4494-8c82-915e70c7c93c" # 주름의 class_id
    
    # now time
    d = datetime.datetime.now()
    NOW_TIME = f"{d.year}-{d.month}-{d.day}-{d.hour}-{d.minute}-{d.second}"
    
    if not os.path.exists(os.path.join('exp', NOW_TIME)):
        os.mkdir(os.path.join(os.path.join('exp', NOW_TIME)))
    
    # call meta paths
    meta_data_path_lst = glob.glob(os.path.join('data', 'label_meta', 'meta', TARGET, '*.json'))
    
    # age_category_lst
    age_category_lst = ["late_20", "early_30", "mid_30", "late_30", "early_40", "mid_40", "late_40"]
    
    # call meta data from json
    for q in range(len(meta_data_path_lst)):
        meta_S_path = meta_data_path_lst[q]
        meta_data = json_load(meta_S_path)

        # meta info
        DATA_KEY = meta_data['data_key']
        AGE = meta_data['age']
        DATASET = meta_data['dataset']
        IMAGE_INFO = meta_data['image_info']
        LABEL_ID = meta_data['label_id']
        LABEL_PATH = meta_data['label_path']
        LAST_UPDATED_DATE = meta_data['last_updated_date']
        TAGS = meta_data['tags']
        WORK_ASSIGNEE = meta_data['work_assignee']
        STATUS = meta_data['status']
        FRAMES = meta_data['frames']
        
        print("process DATA_KEY :", DATA_KEY)
        
        # call label data from json
        label = json_load(os.path.join('data', 'label_meta', LABEL_PATH[0]))
        img_path_lst = glob.glob(os.path.join('data', 'face_img', AGE, DATA_KEY, '*.jpg'))\
            
        try:
            # call img & stack polygon coord
            for i in range(len(img_path_lst)):
                FORMAT = format(i, '03')
                
                img = image_load(img_path_lst[i])
                poly_lst = []
                class_id_lst, meta_meta_lst = [], []
                for j in range(len(label['objects'])):
                    if int(label['objects'][j]['frames'][0]['num']) == int(i):
                        point_lst = []
                        class_id_lst.append(label['objects'][j]['class_id'])
                        meta_meta_lst.append(label['objects'][j]['frames'][0]['annotation']['meta'])
                        if str(label['objects'][j]['frames'][0]['annotation']['coord']['points'][0])[1] == '[':
                            for face_area in label['objects'][j]['frames'][0]['annotation']['coord']['points'][0][0]:
                                point_lst.append((face_area['x'], face_area['y']))      
                        else:
                            for face_area in label['objects'][j]['frames'][0]['annotation']['coord']['points'][0]:
                                point_lst.append((face_area['x'], face_area['y']))      
                        
                        poly_lst.append(point_lst)
                
                if len(poly_lst) < 3:
                    print(f"{DATA_KEY}_{i} failed")
                    pass                          
                else:
                    # prepare polygon coords according from imgaug format
                    POLYGONS = []
                    for i in range(len(poly_lst)):
                        POLYGONS.append(
                            ia.Polygon(poly_lst[i])
                            )

                    # sequence of imgaug
                    for num in range(NUM):
                        seq = imgaug_seq(height=NEW_height, width=NEW_width)
                        
                        NUM_FORMAT = format(num, '03')

                        # imgaug
                        img_output, polygon_output = seq(image=img, polygons=POLYGONS)

                        # save augmentated imgs
                        if not os.path.exists(os.path.join('exp', NOW_TIME, 'img')):
                            os.mkdir(os.path.join('exp', NOW_TIME, 'img'))
                            for age_c in age_category_lst:
                                os.mkdir(os.path.join('exp', NOW_TIME, 'img', age_c))
                            
                        if not os.path.exists(os.path.join('exp', NOW_TIME, 'img', AGE, DATA_KEY + f"_{FORMAT}")):
                            os.mkdir(os.path.join('exp', NOW_TIME, 'img', AGE, DATA_KEY + f"_{FORMAT}"))
                            
                        cv2.imwrite(os.path.join('exp', NOW_TIME, 'img', AGE, DATA_KEY + f"_{FORMAT}", DATA_KEY + f"_{FORMAT}_{NUM_FORMAT}.jpg"), img_output)
                        
                        print(f"IMG_DATA_FNAME: {DATA_KEY}_{FORMAT}_{NUM_FORMAT}.jpg")
                        
                        NEW_LABEL_ID = uuid_gen()
                        NEW_LAST_UPDATED_DATA = str(datetime.datetime.now())
                        
                        # stack & save meta data
                        exp_meta = {
                            "data_key": DATA_KEY + f"_{FORMAT}",
                            "age": AGE,
                            "dataset": DATASET,
                            "image_info": {
                                "width": NEW_width,
                                "height": NEW_height
                            },
                            "label_id": NEW_LABEL_ID,
                            "label_path": [
                                NEW_LABEL_ID + ".json"
                            ],
                            "last_updated_date": NEW_LAST_UPDATED_DATA,
                            "tags": TAGS,
                            "work_assignee": WORK_ASSIGNEE,
                            "status": STATUS,
                            "frames": [
                                DATA_KEY + f"_{FORMAT}_{NUM_FORMAT}.jpg"
                            ]
                        }
                        
                        EXP_META_FNAME = str(DATA_KEY) + f"_{FORMAT}_{NUM_FORMAT}" + ".json"
                        
                        if not os.path.exists(os.path.join('exp', NOW_TIME, 'label_meta')):
                            os.mkdir(os.path.join('exp', NOW_TIME, 'label_meta'))
                            
                        if not os.path.exists(os.path.join('exp', NOW_TIME, 'label_meta', 'meta')):
                            os.mkdir(os.path.join('exp', NOW_TIME, 'label_meta', 'meta'))
                            for age_c in age_category_lst:
                                os.mkdir(os.path.join('exp', NOW_TIME, 'label_meta', 'meta', age_c))
                        
                        with open(os.path.join('exp', NOW_TIME, 'label_meta', 'meta', AGE, EXP_META_FNAME), 'w') as outfile:
                            json.dump(exp_meta, outfile, indent=4)
                            print(f"META_DATA_FNAME: {EXP_META_FNAME}")
                        
                        # stack & save label data
                        exp_label = {}
                        exp_label["objects"] = []
                        
                        for p in range(len(polygon_output)):
                            poly = polygon_output[p]
                            points = []
                            for po in poly:
                                points.append({
                                    "x": float(po[0]),
                                    "y": float(po[1])
                                })
                            
                            class_id = class_id_lst[p]
                            
                            if class_id == PORE:
                                CLASS_NAME = "모공"
                            elif class_id == ACNE:
                                CLASS_NAME = "여드름"
                            elif class_id == RED_SPOTS:
                                CLASS_NAME = "홍반"
                            elif class_id == PIGMENTATION:
                                CLASS_NAME = "색소침착"
                            elif class_id == WRINKLE:
                                CLASS_NAME = "주름"
                            
                            NEW_ID = uuid_gen()
                            
                            exp_label["objects"].append({
                                "id": NEW_ID,
                                "class_id": class_id,
                                "class_name": CLASS_NAME,
                                "annotation_type": "polygon",
                                "tracking_id": p+1,
                                "frames" : [
                                    {
                                        "num": [],
                                        "properties": [],
                                        "annotation": {
                                            "multiple": True,
                                            "coord": {
                                                "points": points
                                            },
                                            "meta": {
                                                "z_index": meta_meta_lst[p]['z_index'],
                                                "visible": meta_meta_lst[p]['visible'],
                                                "alpha": meta_meta_lst[p]['alpha'],
                                                "color": meta_meta_lst[p]['color']
                                            }
                                        }
                                    }
                                ],
                                "properties": []
                                },)   
                            
                        EXP_LABEL_FNAME = NEW_LABEL_ID + '.json'
                        
                        if not os.path.exists(os.path.join('exp', NOW_TIME, 'label_meta')):
                            os.mkdir(os.path.join('exp', NOW_TIME, 'label_meta'))
                            
                        if not os.path.exists(os.path.join('exp', NOW_TIME, 'label_meta', 'labels')):
                            os.mkdir(os.path.join('exp', NOW_TIME, 'label_meta', 'labels'))
                        
                        with open(os.path.join('exp', NOW_TIME, 'label_meta', 'labels', EXP_LABEL_FNAME), 'w', encoding='utf-8') as outfile:
                            json.dump(exp_label, outfile, indent=4, ensure_ascii=False)
                            print(f"EXP_LABEL_FNAME: {EXP_LABEL_FNAME}")
                            
                            count += 1
                            
                            print(f"Number of transformed images: {count}")
                            
                            print("=="*50)
                            print("=="*50)
        
        except:
            '''
            오류 발생 시, 오류난 DATA_KEY값 Failed_lst 폴더로 복사
            이후 해당 DATA_KEY로 생성된 이미지, 라벨, 메타 데이터 삭제
            '''
            data_key_path = os.path.join('data', 'label_meta', 'meta', TARGET, DATA_KEY + '.json')
            new_data_key_path = os.path.join('data', 'label_meta', 'meta', 'Fail_lst_02', f'{DATA_KEY}.json')
            
            shutil.copy(data_key_path, new_data_key_path)
            
            created_meta_lst = glob.glob(os.path.join('exp', NOW_TIME, 'label_meta', 'meta', AGE, f'{DATA_KEY}_{FORMAT}?*'))
            
            for meta_path in created_meta_lst:
                c_meta_data = json_load(meta_path)
                
                C_DATA_KEY = c_meta_data['data_key']
                C_LABEL_PATH = c_meta_data['label_path'][0]
                C_FRAMES = c_meta_data['frames'][0]
                
                # remove created img
                if os.path.exists(os.path.join('exp', NOW_TIME, 'img', AGE, C_DATA_KEY)):
                    if os.path.exists(os.path.join('exp', NOW_TIME, 'img', AGE, C_DATA_KEY, C_FRAMES)):
                        os.remove(os.path.join('exp', NOW_TIME, 'img', AGE, C_DATA_KEY, C_FRAMES))
                # remove created labels
                if os.path.exists(os.path.join('exp', NOW_TIME, 'label_meta', 'labels', C_LABEL_PATH)):
                    os.remove(os.path.join('exp', NOW_TIME, 'label_meta', 'labels', C_LABEL_PATH))
                # remove created meta
                os.remove(meta_path)
                
                print(f'{C_DATA_KEY} child removed...')
                

#----------------------------------------------------------------------------

if __name__ == "__main__":
    dir_path = "/home/ubuntu/gcubme_ai/Workspace/SM_KANG/workspace/jlk/bin"
    os.chdir(dir_path)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-t", '--t', type=str)
    args = parser.parse_args()
    
    main(args)
    
    print("process done...!")
    sys.exit()