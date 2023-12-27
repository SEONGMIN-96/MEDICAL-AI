from .utils.opticalflow.orb_homography import OrbHomography
from .utils.opticalflow.opticalflow import OpticalFlow
# from .utils.preprocess import ...

from PIL import Image
import cv2
import sys
import glob
import os
import time
import yaml
import pandas as pd
import numpy as np

from tqdm import tqdm

from collections import deque, Counter


#----------------------------------------------------------------------------

class Video_Play(object):
    def __init__(self, config: dict) -> None:
        self.video_path = os.path.join('bin', 'data', 'video_gastroscopy', config['video_path'])
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
        video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        print('프레임 길이: %d, 프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d' %(video_length, video_width, video_height, video_fps))
        print('=='*50)
        
        count = 0
        mean_lst = []
        
        max_count = int(self.playback_time * video_fps + 1)
        
        if self.save == True:
            # 비디오 저장
            video_name = self.video_path.split('/')[-1].split('.')[0]
            
            if not os.path.exists(os.path.join('bin', 'exp', 'gastroscopy', video_name)):
                os.mkdir(os.path.join('bin', 'exp', 'gastroscopy', video_name))
            
            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            encode = cv2.VideoWriter(os.path.join('bin', 'exp', 'gastroscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}.avi'), fourcc, video_fps, (self.size, self.size))
            print('saved video...')
    
        a = time.time()

        pbar = tqdm(total=max_count)
        
        # 옵티컬 플로우 결과값을 데크에 쌓아
        # 해당 데크의 평균값이 임계값을 넘을 경우
        # 주의 표시를 띄워줍니다.
        main_deq = deque(maxlen=8)
        
        while cap.isOpened():
            # 1sec = 60frame -> 10초 재생 
            if count == max_count:
                b = time.time()
                print("real time : %ssec" % (str(b - a)))
                print("%d초 재생 완료...!" % (self.playback_time))
                
                # df = pd.DataFrame(np.array(mean_lst))
                # df.to_csv('aaa.csv')
                
                cap.release()
                pbar.close()
                
                if self.save == True:
                    encode.release()

            ret, frame = cap.read()
                      
            # 프레임이 읽히면 ret == True
            if not ret:
                print("프레임을 수신할 수 없습니다...")
                cap.release()
                
                if self.save == True:
                    encode.release()
                
                sys.exit()
            elif ret:
                if self.img_save == True:
                        # 이미지 저장
                        video_name = self.video_path.split('\\')[-1].split('.')[0]
                        if count % 1 == 0:
                            if not os.path.exists(os.path.join('bin', 'exp', 'gastroscopy', video_name)):
                                os.mkdir(os.path.join('bin', 'exp', 'gastroscopy', video_name))
                            if not os.path.exists(os.path.join('bin', 'exp', 'gastroscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}')):
                                os.mkdir(os.path.join('bin', 'exp', 'gastroscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}'))
                            if not os.path.exists(os.path.join('bin', 'exp', 'gastroscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'src')):
                                os.mkdir(os.path.join('bin', 'exp', 'gastroscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'src'))
                            # cv2.imwrite(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'src', '%06d.jpg' % count), frame) 
                            # PILimg = Image.fromarray(frame)
                            # PILimg.save(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'src', '%06d.jpg' % count), dpi=(300,300))
                           
                if count % self.cut_frame == 0:
                    count_frame = frame[self.croped_frame[3]:self.croped_frame[1], self.croped_frame[2]:self.croped_frame[0]]
                    
                    if count == 0:
                        self.frame_lst[1] = count_frame
                    else:
                        self.frame_lst[0] = self.frame_lst[1] 
                        self.frame_lst[1] = count_frame
                        
                        # print("%d frame -> %d frame" % (count -2, count))
                        
                        if self.algorithm == 'orb_homography':
                            orb_h = OrbHomography(previous_frame=self.frame_lst[0], current_frame=self.frame_lst[1])
                            res = orb_h.measure_distance()
                        elif self.algorithm == 'lucas_kanade':
                            o_f = OpticalFlow(previous_frame=self.frame_lst[0], current_frame=self.frame_lst[1], size=self.size, flow_show=self.flow_show)
                            res = o_f.lucas_kanade()
                        elif self.algorithm == 'gunner_farneback':
                            o_f = OpticalFlow(previous_frame=self.frame_lst[0], current_frame=self.frame_lst[1], size=self.size, flow_show=self.flow_show)
                            
                            # 옵티컬 플로우의 결과값을 추출합니다.
                            res = o_f.gunner_farneback()
                            
                            # 추출한 결과값을 deq에 쌓아줍니다.
                            main_deq.extend([int(res[-1])])
                            
                            # 쌓인 결과값의 평균을 구합니다.
                            mean_avg = sum(list(main_deq)) // len(list(main_deq))
                            
                            # resize
                            count_frame = cv2.resize(count_frame, (self.size, self.size))
                            
                            # 결과값의 평균에 따른 상태표시를 프레임에 입력합니다.
                            # count_frame = o_f.draw_result(count_frame, mean_avg)
                            count_frame = o_f.draw_result(count_frame, mean_avg, int(res[-1]))
                            
                        elif self.algorithm == 'rlof':
                            o_f = OpticalFlow(previous_frame=self.frame_lst[0], current_frame=self.frame_lst[1], size=self.size, flow_show=self.flow_show)
                            res = o_f.rlof()
                        elif self.algorithm == 'frame_info':
                            o_f = OpticalFlow(previous_frame=self.frame_lst[0], current_frame=self.frame_lst[1], size=self.size, flow_show=self.flow_show)
                            res = o_f.frame_info()
                        
                        if self.save == True:
                            # 프레임을 인코딩합니다.
                            encode.write(count_frame)
                        
                        # mean_lst.append(res[1])
                        
                        # if not os.path.exists(os.path.join('exp', 'gastroscopy', video_name)):
                        #         os.mkdir(os.path.join('exp', 'gastroscopy', video_name))
                        # if not os.path.exists(os.path.join('exp', 'gastroscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}')):
                        #     os.mkdir(os.path.join('exp', 'gastroscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}'))
                        # if not os.path.exists(os.path.join('exp', 'gastroscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'flow')):
                        #     os.mkdir(os.path.join('exp', 'gastroscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'flow'))
                        # # cv2.imwrite(os.path.join('exp', 'colonoscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'flow', '%06d.jpg' % count), frame) 
                        # frame = cv2.cvtColor(res[0], cv2.COLOR_BGR2RGB)
                        # PILimg = Image.fromarray(frame)
                        # PILimg.save(os.path.join('exp', 'gastroscopy', video_name, f'{video_name}_{self.algorithm}_{str(self.size)}', 'flow', '%06d.jpg' % count), dpi=(300,300))
                        
                        # cv2.imshow('frame', res[0])
                        # cv2.waitKey(60)
                
                else:
                    else_frame = frame[self.croped_frame[3]:self.croped_frame[1], self.croped_frame[2]:self.croped_frame[0]]
                    
                    # self.frame_lst[0] = self.frame_lst[1] 
                    # self.frame_lst[1] = frame
                    
                    # print("%d frame -> %d frame" % (count -2, count))
                    
                    if self.algorithm == 'orb_homography':
                        orb_h = OrbHomography(previous_frame=self.frame_lst[0], current_frame=self.frame_lst[1])
                        # res = orb_h.measure_distance()
                    elif self.algorithm == 'lucas_kanade':
                        o_f = OpticalFlow(previous_frame=self.frame_lst[0], current_frame=self.frame_lst[1], size=self.size, flow_show=self.flow_show)
                        # res = o_f.lucas_kanade()
                    elif self.algorithm == 'gunner_farneback':
                        # count % self.cut_frame == 0인 부분에서 얻은 결과값으로만
                        # 해당 프레임에 결과를 그려줍니다.
                        try:
                            # resize
                            else_frame = cv2.resize(else_frame, (self.size, self.size))
                            
                            # else_frame = o_f.draw_result(else_frame, mean_avg)   
                            else_frame = o_f.draw_result(else_frame, mean_avg, int(res[-1]))
                        except:
                            # resize
                            else_frame = cv2.resize(else_frame, (self.size, self.size))
                            
                    elif self.algorithm == 'rlof':
                        o_f = OpticalFlow(previous_frame=self.frame_lst[0], current_frame=self.frame_lst[1], size=self.size, flow_show=self.flow_show)
                        # res = o_f.rlof()
                    elif self.algorithm == 'frame_info':
                        o_f = OpticalFlow(previous_frame=self.frame_lst[0], current_frame=self.frame_lst[1], size=self.size, flow_show=self.flow_show)
                        # res = o_f.frame_info()
                    
                    if self.save == True:
                        # 프레임을 인코딩합니다.
                        encode.write(else_frame)
                        
                count += 1
                pbar.update(1)

#----------------------------------------------------------------------------
      
def main():
    with open(os.path.join('bin', 'config', 'config.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for c in config:
        print(f"{c}:{config[c]}")
        
    print('=='*50)
    
    v1 = Video_Play(config)
    v1.opflow()
        
if __name__ == '__main__':
    main()