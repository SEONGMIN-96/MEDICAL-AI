from .utils.opticalflow.orb_homography import OrbHomography
from .utils.opticalflow.opticalflow import OpticalFlow

from PIL import Image
import cv2
import sys
import glob
import os
import time
import yaml
import pandas as pd
import numpy as np


#----------------------------------------------------------------------------

class Video_Play(object):
    def __init__(self, config: dict) -> None:
        self.video_path = os.path.join("data", "video_gastroscopy", config['video_path'])
        self.algorithm = config['algorithm']
        self.save = config['save']
        self.img_save = config['img_save']
        self.cut_frame = config['cut_frame']
        self.playback_time = config['playback_time']
        self.size = config['size'] 
        self.croped_frame = config['croped_frame']
        self.flow_show = config['flow_show']
        self.frame_lst = [None for _ in range(2)]

    def opflow(self):
        cap = cv2.VideoCapture(self.video_path)       

        # 프레임 길이, 너비/높이, 초당 프레임 수 확인
        length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('프레임 길이: %d, 프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d' %(length, width, height, fps))
        print('=='*50)
        
        count = 0
        mean_lst = []
        
        max_count = int(self.playback_time * fps + 1)
        
        if self.save == True:
            # 비디오 저장
            video_name = self.video_path.split('\\')[-1].split('.')[0]
            
            if not os.path.exists(os.path.join('exp', 'colonoscopy', video_name)):
                os.mkdir(os.path.join('exp', 'colonoscopy', video_name))
            
            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            encode = cv2.VideoWriter(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}.avi'), fourcc, fps/4, (self.size, self.size))
            print('saved video...')
    
        a = time.time()

        while cap.isOpened():
            # 1sec = 60frame -> 10초 재생 
            if count == max_count:
                b = time.time()
                print("real time : %ssec" % (str(b - a)))
                print("%d초 재생 완료...!" % (self.playback_time))
                
                df = pd.DataFrame(np.array(mean_lst))
                
                df.to_csv(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'results.csv'))
                
                cap.release()
                # encode.release()
                cv2.destroyAllWindows()
                sys.exit()

            ret, frame = cap.read()
                      
            # 프레임이 읽히면 ret == True
            if not ret:
                print("프레임을 수신할 수 없습니다...")
                sys.exit()
            elif ret:
                if self.img_save == True:
                        # 이미지 저장
                        video_name = self.video_path.split('\\')[-1].split('.')[0]
                        if count % 1 == 0:
                            if not os.path.exists(os.path.join('exp', 'colonoscopy', video_name)):
                                os.mkdir(os.path.join('exp', 'colonoscopy', video_name))
                            if not os.path.exists(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}')):
                                os.mkdir(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}'))
                            if not os.path.exists(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'src')):
                                os.mkdir(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'src'))
                            # cv2.imwrite(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'src', '%06d.jpg' % count), frame) 
                            frame_f = frame[self.croped_frame[0]:self.croped_frame[1], self.croped_frame[2]:self.croped_frame[3]]
                            frame_f = cv2.cvtColor(frame_f, cv2.COLOR_BGR2RGB)
                            PILimg = Image.fromarray(frame_f)
                            PILimg.save(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'src', '%06d.jpg' % count), dpi=(300,300))
                           
                if count % self.cut_frame == 0:
                    frame = frame[self.croped_frame[0]:self.croped_frame[1], self.croped_frame[2]:self.croped_frame[3]]
                    
                    if count == 0:
                        self.frame_lst[1] = frame
                    else:
                        self.frame_lst[0] = self.frame_lst[1] 
                        self.frame_lst[1] = frame
                        
                        # print("%d frame -> %d frame" % (count -2, count))
                        
                        if self.algorithm == 'orb_homography':
                            orb_h = OrbHomography(previous_frame=self.frame_lst[0], current_frame=self.frame_lst[1])
                            res = orb_h.measure_distance()
                        elif self.algorithm == 'lucas_kanade':
                            o_f = OpticalFlow(previous_frame=self.frame_lst[0], current_frame=self.frame_lst[1], size=self.size, flow_show=self.flow_show)
                            res = o_f.lucas_kanade()
                        elif self.algorithm == 'gunner_farneback':
                            o_f = OpticalFlow(previous_frame=self.frame_lst[0], current_frame=self.frame_lst[1], size=self.size, flow_show=self.flow_show)
                            res = o_f.gunner_farneback()
                        elif self.algorithm == 'rlof':
                            o_f = OpticalFlow(previous_frame=self.frame_lst[0], current_frame=self.frame_lst[1], size=self.size, flow_show=self.flow_show)
                            res = o_f.rlof()
                        elif self.algorithm == 'frame_info':
                            o_f = OpticalFlow(previous_frame=self.frame_lst[0], current_frame=self.frame_lst[1], size=self.size, flow_show=self.flow_show)
                            res = o_f.frame_info()
                            
                        if self.save == True:
                            # 비디오 저장
                            encode.write(res[0])
                        
                        mean_lst.append(res[1])
                        
                        if not os.path.exists(os.path.join('exp', 'colonoscopy', video_name)):
                                os.mkdir(os.path.join('exp', 'colonoscopy', video_name))
                        if not os.path.exists(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}')):
                            os.mkdir(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}'))
                        if not os.path.exists(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'flow')):
                            os.mkdir(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'flow'))
                        # cv2.imwrite(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'flow', '%06d.jpg' % count), frame) 
                        frame = cv2.cvtColor(res[0], cv2.COLOR_BGR2RGB)
                        PILimg = Image.fromarray(frame)
                        PILimg.save(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'flow', '%06d.jpg' % count), dpi=(300,300))
                        
                        
                        cv2.imshow('frame', res[0])
                        cv2.waitKey(60)
                        
                count += 1

#----------------------------------------------------------------------------
      
def main():
    with open('./config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for c in config:
        print(f"{c}:{config[c]}")
        
    print('=='*50)
    
    v1 = Video_Play(config)
    v1.opflow()
        
if __name__ == '__main__':
    main()