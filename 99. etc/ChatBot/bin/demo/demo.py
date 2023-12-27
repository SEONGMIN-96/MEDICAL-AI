from ..utils.stt import transcribe_file
from ..utils.tts import synthesize_text_google, synthesize_text_clova_voice
from ..utils.scenario import dialogflow
from ..utils.sound import wait_sound, wake_up_sound, cov_play, record_audio, ready_sound, check_sound, thanks_sound, mp3_play
from ..utils.text_mining import ans_choice

import os
import argparse
import glob
import concurrent.futures
import cv2


class Chatbot(object):
    def __init__(self) -> None:
        self.key_path = os.path.join(os.getcwd(), 'key', 'stt-chat-220114-aa4fddc8a388.json')
        self.input_path = os.path.join(os.getcwd(), 'bin', 'data', 'input')
        self.output_path = os.path.join(os.getcwd(), 'bin', 'data', 'output')
        
    def chatbot(self, report_path, mode, silence_threshold, service_type, without_mike):
        # Google cloud
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.key_path

        # input/output path
        input_path = self.input_path
        output_path = self.output_path
        
        # Audio path
        audio_name = "response.mp3"
        sample_name = "sample.mp3"

        # dialogflow Sequence
        intent_S = 1
        response = int

        # prepared intents
        max_intents = glob.glob(os.path.join(os.getcwd(), 'bin', 'data', 'dialogflow', 'intents', '*.txt'))

        # api import
        text_S = transcribe_file(os.path.join(output_path, sample_name))
        
        if service_type == 0:
            synthesize_text_google(text_S, output_path)
        elif service_type == 1:
            synthesize_text_clova_voice(text_S, output_path)

        # ready sound
        ready_sound(output_path)
        
        # 
        
        while True:
            if without_mike == 0:           
                # Record audio
                wait_sound(output_path)
                record_audio(os.path.join(output_path, audio_name), silence_threshold, output_path)
            elif without_mike == 1:
                # test code only use typing without recoding
                wait_sound(output_path)
                text = input("명령어 입력 \ninput : ")

            # STT
            if without_mike == 0:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    th1 = executor.submit(check_sound, 
                                        output_path,
                    )
                    th2 = executor.submit(transcribe_file,
                                        os.path.join(output_path, audio_name),
                    )
                text = th2.result()    
            elif without_mike == 1:
                pass

            # replace Dialogflow
            I_res = dialogflow(intent_S, text)

            # TTS
            if I_res == 'ans-exit':
                thanks_sound(output_path)
                break
            if intent_S > 0:
                response = ans_choice(I_res, 
                                    output_path,
                                    input_path,
                                    report_path,
                                    mode,
                                    service_type,
                )
                intent_S += 1
            if response == 999:
                intent_S = 1
            else:
                pass

            # return intent_S to base
            if intent_S == len(max_intents):
                intent_S = 1
                response = int
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_path", default='abdomen_ct/72_30389563.txt', help='choose the path of patient report')
    parser.add_argument("--mode", default='liver', choices=['liver', 'colon'], help='select location of cancer')
    parser.add_argument("--silence_threshold", default=1100, help='silence threshold', type=int)
    parser.add_argument("--service_type", default=0, choices=[0, 1], help='service with google(0) or clova(1)', type=int)
    parser.add_argument("--without_mike", default=0, choices=[0, 1], help='if with mike(0) or without mike(1)', type=int)
    args = parser.parse_args()
    
    report_path = args.report_path
    mode = args.mode
    silence_threshold = args.silence_threshold
    service_type = args.service_type
    without_mike = args.without_mike
    
    print('report_path: %s' % report_path)
    print('mode: %s' % mode)
    print('silence_threshold: %d' % silence_threshold)
    
    if service_type == 0:
        print('service_type: google')
    elif service_type == 1:
        print('service_type: clova')
    
    if without_mike == 0:
        print('without_mike: False')
    elif without_mike == 1:
        print('without_mike: True')
    
    cbot = Chatbot()
    cbot.chatbot(report_path=report_path, mode=mode, silence_threshold=silence_threshold, service_type=service_type, without_mike=without_mike)
    
if __name__ == "__main__":
    main()