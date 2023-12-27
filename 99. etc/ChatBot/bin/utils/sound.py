# record audio
def record_audio(audio_path, silence_threshold, dir_path):
    import pyaudio
    from array import array
    from collections import deque
    from queue import Full

    import wave 
    import time

    # const values for mic streaming
    CHUNK = 1024
    BUFF = CHUNK * 10
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    # WAVE_OUTPUT_FILENAME = "output.wav"
    WAVE_OUTPUT_FILENAME = audio_path

    # const valaues for silence detection
    SILENCE_THREASHOLD = silence_threshold
    SILENCE_SECONDS = 2

     # open stream
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=1,
        frames_per_buffer=CHUNK
    )

    # FIXME: release initial noisy data (1sec)
    for _ in range(0, int(RATE/CHUNK)):
        data = stream.read(CHUNK, exception_on_overflow=False)

    is_started = False
    vol_que = deque(maxlen=SILENCE_SECONDS)

    print('=========================================')
    print("=       Waiting for recording....!      =")
    print('=========================================')
    
    # wait sound
    wait_sound(dir_path)
        
    # frames
    frames = []

    while True:
        try:
            # define temporary variable to store sum of volume for 1 second 
            vol_sum = 0

            # read data for 1 second in chunk
            for _ in range(0, int(RATE/CHUNK)):
                data = stream.read(CHUNK, exception_on_overflow=False)

                # get max volume of chunked data and update sum of volume
                vol = max(array('h', data))
                vol_sum += vol

                # if status is listening, check the volume value
                if not is_started:
                    if vol >= SILENCE_THREASHOLD:
                        print('=========================================')
                        print("=       Recording in progress....!      =")
                        print('=========================================')
                        is_started = True

                # if status is speech started, write data
                if is_started:
                    frames.append(data)

            # if status is speech started, update volume queue and check silence
            if is_started:
                vol_que.append(vol_sum / (RATE/CHUNK) < SILENCE_THREASHOLD)
                if len(vol_que) == SILENCE_SECONDS and all(vol_que):
                    print('=========================================')
                    print('=        End of recording....!          =')
                    print('=========================================')
                    break
                
        except Full:
            pass
           
    # close stream
    stream.stop_stream()
    stream.close()
    p.terminate()
     
    # record franmes(queue) during noise
    wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    sound_norn(WAVE_OUTPUT_FILENAME)
    
def wake_up_sound(dir_path):
    from playsound import playsound
    import winsound
    import os
    
    src = "wakeup_clova.mp3"
    
    # winsound.PlaySound(os.path.join(dir_path, src), winsound.SND_FILENAME)
    playsound(os.path.join(dir_path, src))

def wait_sound(dir_path):
    import winsound
    import os
    import time
    
    time.sleep(0.2)
    src = "wait.wav"
    
    winsound.PlaySound(os.path.join(dir_path, src), winsound.SND_FILENAME)

def ready_sound(dir_path):
    from playsound import playsound
    import winsound
    import os
    import time
    
    time.sleep(0.2)
    src = "ready_clova.mp3"
    
    print(dir_path)
    
    # winsound.PlaySound(os.path.join(dir_path, src), winsound.SND_FILENAME)
    playsound(os.path.join(dir_path, src))
    
def thanks_sound(dir_path):
    from playsound import playsound
    import winsound
    import os
    import time
    
    time.sleep(0.2)
    src = "thanks_clova.mp3"
    
    # winsound.PlaySound(os.path.join(dir_path, src), winsound.SND_FILENAME)
    playsound(os.path.join(dir_path, src))

def check_sound(dir_path):
    from playsound import playsound
    import winsound
    import os
    import time
    
    time.sleep(0.2)
    src = "check_clova.mp3"
    
    # winsound.PlaySound(os.path.join(dir_path, src), winsound.SND_FILENAME)
    playsound(os.path.join(dir_path, src))

def mp3_play(dir_path):
    from playsound import playsound
    import os
    import multiprocessing
    
    src = "response.mp3"
    
    playsound(os.path.join(dir_path, src))

def cov_play(dir_path):
    from pydub import AudioSegment
    import os

    src = "response.mp3"
    dst = "response.wav"

    src_path = os.path.join(dir_path, src)
    dst_path = os.path.join(dir_path, dst)
    
    audSeg = AudioSegment.from_mp3(src_path)
    audSeg.export(dst_path, format="wav")
    
    import winsound

    winsound.PlaySound(dst_path , winsound.SND_FILENAME)
    
def sound_norn(audio_path):
    '''
    As the distance increases, the amplitude of the recorded sound decreases rapidly
    Normalizes the maximum amplitude close to the sampling rate
    '''
    from scipy.io import wavfile

    import numpy as np
    import os

    sample_rate, data = wavfile.read(audio_path) # sr : sampling rate, x : wave data array

    abs1 = abs(np.max(data))
    abs2 = abs(np.min(data))

    MUL = 1

    if abs1 > abs2:
        multiple = (sample_rate/2*MUL) / abs1
    elif abs1 < abs2:
        multiple = (sample_rate/2*MUL) / abs2

    aug_data = (data * multiple).astype(np.int16)

    # print('multiple: %f' % multiple)
    # print("Augaudio data type: {0}".format(type(data[0][0])))

    wavfile.write(filename=audio_path, rate=sample_rate, data=aug_data)