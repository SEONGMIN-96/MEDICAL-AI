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


#----------------------------------------------------------------------------

def image_load(PATH):
    return cv2.imread(PATH)

def json_load(PATH):
    with open(PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def imgaug_seq():
    sometimes_00 = lambda aug: iaa.Sometimes(1.0, aug)
    sometimes_01 = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes_00(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            rotate=(-75, 75), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 3),
            [
                # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                # iaa.OneOf([
                    # iaa.GaussianBlur((0, 0.5)), # blur images with a sigma between 0 and 1.0
                    # iaa.AverageBlur(k=(1, 2)), # blur image using local means with kernel sizes between 0 and 1.0
                    # iaa.MedianBlur(k=(1, 2)), # blur image using local medians with kernel sizes between 0 and 1.0
                # ]),
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                # iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.LinearContrast((0.5, 1.2), per_channel=0.5), # improve or worsen the contrast
                sometimes_01(iaa.ElasticTransformation(alpha=(1, 4.0), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes_01(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.05)))
                iaa.Grayscale(alpha=(0.1, 0.6))
            ],
            random_order=True
        )
    ],
    random_order=True
)
    
    return seq

def random_num_generator():
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

def main():
    '''
    샘플 데이터
    img_aug 활용 데이터 augmentation 
    label & meta data 생성
    '''
    # now time
    d = datetime.datetime.now()
    NOW_TIME = f"{d.year}-{d.month}-{d.day}-{d.hour}-{d.minute}-{d.second}"
    
    if not os.path.exists(os.path.join('exp', NOW_TIME)):
        os.mkdir(os.path.join(os.path.join('exp', NOW_TIME)))
    
    # call meta paths
    meta_data_path_lst = glob.glob(os.path.join('data', 'label_meta', 'meta', 'Face', '*.json'))
    
    # call meta data from json
    meta_S_path = meta_data_path_lst[0]
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
    
    # call label data from json
    label = json_load(os.path.join('data', 'label_meta', LABEL_PATH[0]))
    img_path_lst = glob.glob(os.path.join('data', 'face_img', AGE, DATA_KEY, '*.jpg'))

    NUM = 300
    
    # call img & stack polygon coord
    for i in range(len(img_path_lst)):
        FORMAT = format(i, '03')
        
        img = image_load(img_path_lst[i])
        poly_lst = []
        class_id_lst, meta_meta_lst = [], []
        for j in range(len(label['objects'])):
            if int(label['objects'][j]['frames'][0]['num']) == int(0):
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
        
        # prepare polygon coords according from imgaug format
        POLYGONS = []
        for i in range(len(poly_lst)):
            POLYGONS.append(
                ia.Polygon(poly_lst[i])
                )

        # sequence of imgaug
        for num in range(NUM):
            seq = imgaug_seq()
            
            NUM_FORMAT = format(num, '03')

            # imgaug
            img_output, polygon_output = seq(image=img, polygons=POLYGONS)

            # save augmentated imgs
            if not os.path.exists(os.path.join('exp', NOW_TIME, 'img')):
                os.mkdir(os.path.join('exp', NOW_TIME, 'img'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'img', 'mid_20'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'img', 'late_20'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'img', 'early_30'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'img', 'mid_30'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'img', 'late_30'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'img', 'early_40'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'img', 'mid_40'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'img', 'late_40'))
                
            if not os.path.exists(os.path.join('exp', NOW_TIME, 'img', AGE, DATA_KEY + f"_{FORMAT}")):
                os.mkdir(os.path.join('exp', NOW_TIME, 'img', AGE, DATA_KEY + f"_{FORMAT}"))
                
            cv2.imwrite(os.path.join('exp', NOW_TIME, 'img', AGE, DATA_KEY + f"_{FORMAT}", DATA_KEY + f"_{FORMAT}_{NUM_FORMAT}.jpg"), img_output)
            
            NEW_LABEL_ID = random_num_generator()
            NEW_LAST_UPDATED_DATA = str(datetime.datetime.now())
            
            # stack & save meta data
            exp_meta = {
                "data_key": DATA_KEY + f"_{FORMAT}",
                "age": AGE,
                "dataset": DATASET,
                "image_info": {
                    "width": img_output.shape[0],
                    "height": img_output.shape[1]
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
                    #FRAMES[0].rstrip('.jpg') + "_001" + ".jpg"
                    DATA_KEY + f"_{FORMAT}_{NUM_FORMAT}.jpg"
                ]
            }
            
            EXP_META_FNAME = str(DATA_KEY) + f"_{FORMAT}_{NUM_FORMAT}" + ".json"
            
            if not os.path.exists(os.path.join('exp', NOW_TIME, 'label_meta')):
                os.mkdir(os.path.join('exp', NOW_TIME, 'label_meta'))
                
            if not os.path.exists(os.path.join('exp', NOW_TIME, 'label_meta', 'meta')):
                os.mkdir(os.path.join('exp', NOW_TIME, 'label_meta', 'meta'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'label_meta', 'meta', 'mid_20'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'label_meta', 'meta', 'late_20'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'label_meta', 'meta', 'early_30'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'label_meta', 'meta', 'mid_30'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'label_meta', 'meta', 'late_30'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'label_meta', 'meta', 'early_40'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'label_meta', 'meta', 'mid_40'))
                os.mkdir(os.path.join('exp', NOW_TIME, 'label_meta', 'meta', 'late_40'))
            
            with open(os.path.join('exp', NOW_TIME, 'label_meta', 'meta', AGE, EXP_META_FNAME), 'w') as outfile:
                json.dump(exp_meta, outfile, indent=4)
            
            # call sub info of label
            PORE = "53cf7aa8-bf2c-4323-933c-261615fa1b63" # 모공의 class_id
            ACNE = "c1eeed96-7416-44ff-aaa6-23a241728869" # 여드름의 class_id
            RED_SPOTS = "e15caa3b-a88f-4dc4-8949-a4f19096a5f3" # 홍반의 class_id
            PIGMENTATION = "2cc91760-a687-4126-8787-74325b072217" # 색소침착의 class_id
            WRINKLE = "8d2c0a27-3481-4494-8c82-915e70c7c93c" # 주름의 class_id
                
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
                
                NEW_ID = random_num_generator()
                
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
                    
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()