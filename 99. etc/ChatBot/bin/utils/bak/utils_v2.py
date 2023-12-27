# STT
def transcribe_gcs(gcs_url):
    """Asynchronously transcribes the audio file specified by the gcs_uri."""

    import time
    from google.cloud import speech
        
    # start_time = time.time()
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(gcs_url=gcs_url)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code='ko-KR')
    operation = client.long_running_recognize(config = config, audio = audio)
    print('Waiting for operation to complete...')
    response = operation.result(timeout=90)
    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(u'Transcript: {}'.format(result.alternatives[0].transcript))
        print('Confidence: {}'.format(result.alternatives[0].confidence))
    # end_time = time.time() - start_time
    # print(f"소요시간: {end_time} sec")
    print("done")

def transcribe_file(content):
    """Asynchronously transcribes the audio file specified by the local_content."""
    
    import io
    import time
    from google.cloud import speech
    
    text = []

    with io.open(content, 'rb') as audio_file:
        content = audio_file.read()

    # start_time = time.time()
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='ko-KR',
        audio_channel_count=2)
    operation = client.recognize(config = config, audio = audio)
    print('Waiting for operation to complete...')
    response = operation.results
    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response:
        # The first alternative is the most likely one for this portion.
        # print(u'Transcript: {}'.format(result.alternatives[0].transcript))
        # print('Confidence: {}'.format(result.alternatives[0].confidence))
        text = result.alternatives[0].transcript
    # end_time = time.time() - start_time
    # print("time for Speech To Text : {}".format(end_time))
    
    return text

# dialogflow
def detect_intent_texts(project_id, session_id, texts, language_code):
    """Returns the result of detect intent with texts as inputs.

    Using the same `session_id` between requests allows continuation
    of the conversation."""

    from google.cloud import dialogflow_v2beta1 as dialogflow
    import time
    # from google.cloud import dialogflow

    start_time = time.time()
    
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(project_id, session_id)
    # print("Session path: {}\n".format(session))
    
    response = []
    ans = None
    
    print('\n')
    print('Processing...')
    for i in range(2):
        # time.sleep(0.5)
        print('...')
    print('\n')
    
    for text in texts:
        text_input = dialogflow.TextInput(text=text, language_code=language_code)

        query_input = dialogflow.QueryInput(text=text_input)

        response = session_client.detect_intent(
            request={"session": session, "query_input": query_input}
        )

        print("=" * 100)
        # print("Query text: {}".format(response.query_result.query_text))
        # print(
            # "Detected intent: {} (confidence: {})\n".format(
                # response.query_result.intent.display_name,
                # response.query_result.intent_detection_confidence,
            # )
        # )
        # print("Fulfillment text: {}\n".format(response.query_result.fulfillment_text))
    # print(time.time() - start_time)

    if type(response) != list:
        ans = response.query_result.fulfillment_text    
    
    return ans

def create_intent(project_id, display_name, training_phrases_parts, message_texts, input_context_names, output_contexts):
    """Create an intent of the given intent type."""
    from google.cloud import dialogflow_v2beta1 as dialogflow

    intents_client = dialogflow.IntentsClient()
    
    parent = dialogflow.AgentsClient.agent_path(project_id)
    training_phrases = []
    for training_phrases_part in training_phrases_parts:
        part = dialogflow.Intent.TrainingPhrase.Part(text=training_phrases_part)
        # Here we create a new training phrase for each provided part.
        training_phrase = dialogflow.Intent.TrainingPhrase(parts=[part])
        training_phrases.append(training_phrase)

    text = dialogflow.Intent.Message.Text(text=message_texts)
    message = dialogflow.Intent.Message(text=text)
  
    intent = dialogflow.Intent(
        display_name=display_name, training_phrases=training_phrases, messages=[message], input_context_names=input_context_names,
        output_contexts=[output_contexts]
    )
    response = intents_client.create_intent(
        request={"parent": parent, "intent": intent}
    )
    print("Intent created: {}".format(response))
      
def list_intents(project_id):
    from google.cloud import dialogflow_v2beta1 as dialogflow

    intents_client = dialogflow.IntentsClient()

    parent = dialogflow.AgentsClient.agent_path(project_id)

    intents = intents_client.list_intents(request={"parent": parent})

    for intent in intents:
        print("=" * 20)
        print("Intent name: {}".format(intent.name))
        print("Intent display_name: {}".format(intent.display_name))
        print("Action: {}\n".format(intent.action))
        print("Root followup intent: {}".format(intent.root_followup_intent_name))
        print("Parent followup intent: {}\n".format(intent.parent_followup_intent_name))

        print("Input contexts:")
        for input_context_name in intent.input_context_names:
            print("\tName: {}".format(input_context_name))

        print("Output contexts:")
        for output_context in intent.output_contexts:
            print("\tName: {}".format(output_context.name))

def delete_intent(project_id, intent_id):
    """Delete intent with the given intent type and intent value."""
    from google.cloud import dialogflow_v2beta1 as dialogflow

    intents_client = dialogflow.IntentsClient()

    intent_path = intents_client.intent_path(project_id, intent_id)

    intents_client.delete_intent(request={"name": intent_path})
    
# replace dialogflow
def dialogflow_intents(intents_S, intents_path, text, entity_path):
    from nltk.tokenize import word_tokenize, sent_tokenize  
    import os

    words = word_tokenize(text)
    exist = False
    
    for word in words:
        if exist == False:
            if '종료' in word:
                res = 'ans-exit'
                exist = True
            else:
                f = open(os.path.join(intents_path, f'I_{intents_S}.txt'), 'r', encoding='utf-8')
                I_res = f.readline()
                entities = dialogflow_entities(intents_S, entity_path)
                for E in entities:
                    if exist == True:
                        pass
                    elif E in word:
                        res = I_res
                        exist = True
                    elif E not in word:
                        res = 'ans999'
        elif exist == True:
            pass
        
    return res

def dialogflow_entities(intents_S, entities_path):
    import os
    
    f = open(os.path.join(entities_path, f'E_{intents_S}.txt'), 'r', encoding='utf-8')
    entity = f.readline()
    
    return entity


def dialogflow(intent_S, text):
    # 텍스트를 입력받는다
    # 인텐트는 순서에 따라 진행되어야한다.
    # 인텐트는 엔티티 속의 단어 유무에 따라 결정된다.
    import os
    
    dir_path = os.path.join(os.getcwd(), 'cbot_demo', 'dialogflow')
    intents_path = os.path.join(dir_path, 'intents')
    entity_path = os.path.join(dir_path, 'entities')
    
    intent_S = intent_S
    text = text
    res = str
    
    if len(text) > 0:
        # entity = dialogflow_entities(intent_S, entities_path)
        res = dialogflow_intents(intent_S, intents_path, text, entity_path)
    else:
        pass
    
    print("Transcript : {}".format(res))
    
    return res    

# TTS
def synthesize_text(text, dir_path):
    from google.cloud import texttospeech
    import os

    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code = "ko", 
        name = "ko-KR-Wavenet-B", 
        ssml_gender = texttospeech.SsmlVoiceGender.FEMALE
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding = texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        request = {"input" : input_text, "voice" : voice, "audio_config" : audio_config}
    )

    output_path = os.path.join(dir_path, 'response.mp3')
    
    with open(output_path, "wb") as out:
        out.write(response.audio_content)
        # print('Audio content written to file "response.mp3"')

# record audio
def record_audio(audio_path):
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
    SILENCE_THREASHOLD = 6000
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
    for _ in range(0, int(RATE / CHUNK)):
        data = stream.read(CHUNK, exception_on_overflow=False)

    is_started = False
    vol_que = deque(maxlen=SILENCE_SECONDS)

    print('start listening')
    
    # frames
    frames = []

    while True:
        try:
            # define temporary variable to store sum of volume for 1 second 
            vol_sum = 0

            # read data for 1 second in chunk
            for _ in range(0, int(RATE / CHUNK)):
                data = stream.read(CHUNK, exception_on_overflow=False)

                # get max volume of chunked data and update sum of volume
                vol = max(array('h', data))
                vol_sum += vol

                # if status is listening, check the volume value
                if not is_started:
                    if vol >= SILENCE_THREASHOLD:
                        print('start of speech detected')
                        is_started = True

                # if status is speech started, write data
                if is_started:
                    frames.append(data)

            # if status is speech started, update volume queue and check silence
            if is_started:
                vol_que.append(vol_sum / (RATE / CHUNK) < SILENCE_THREASHOLD)
                if len(vol_que) == SILENCE_SECONDS and all(vol_que):
                    print('end of speech detected')
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

def wake_up_sound(dir_path):
    import winsound
    import os
    
    src = "wakeup.wav"
    
    winsound.PlaySound(os.path.join(dir_path, src), winsound.SND_FILENAME)

def wait_sound(dir_path):
    import winsound
    import os
    import time
    
    time.sleep(0.2)
    src = "wait.wav"
    
    winsound.PlaySound(os.path.join(dir_path, src), winsound.SND_FILENAME)

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

def read_txt(txt_path):
    _list = []
    _txt = open(txt_path, 'r', encoding='utf-8')
    while True:
        line = _txt.readline()
        if line == '':
            break
        line = line.rstrip('\n')
        _list.append(line)
    _txt.close

    return _list

def read_paper(txt_path):
    import time

    reports = str
    
    # count = 0
    reports = open(txt_path, 'r', encoding='UTF8').read()
    
    print('Reading Reports...')
    for i in range(2):
        # time.sleep(1)
        print('...')
    print('\n')
    
    return reports

def categorizing_C_keyword(info_path, f_path):
    from nltk.tokenize import word_tokenize, sent_tokenize    
    import os    
    
    txt_path = os.path.join(info_path, f_path)
    cancer_path = os.path.join(info_path,  "01_cancer_list.txt")
    ko_list = os.path.join(info_path, "02_ko_list.txt")
    sub_detail_path = os.path.join(info_path, "04_sub_detail.txt")
    
    reports = read_paper(txt_path=txt_path)
    reports = sent_tokenize(reports)

    cancer_list, cancer_ko_list = [], []
    
    keyword_list = []

    cancer_list = read_txt(cancer_path)
    cancer_ko_list = read_txt(ko_list)
    
    sub_detail_list = read_txt(sub_detail_path)    

    after_conclution = False

    for j in range(len(reports)):
        if 'Conclusion' in reports[j]:
            after_conclution = True
        
        if after_conclution == True:
            for i in range(len(cancer_list)):
                cm_keyword = None
                cancer_keyword = None
                
                if cancer_list[i] in reports[j]:
                    cancer_keyword = cancer_list[i]
                    words = word_tokenize(reports[j])
                    for k in range(len(words)):
                        if 'cm' in words[k]:
                            cm_keyword = words[k]
                else:
                    cancer_keyword = None
                    cm_keyword = None
                    
                if cancer_keyword != None and cm_keyword != None:
                    aa = [cancer_keyword, cm_keyword]
                    keyword_list.append(aa) 
                elif cancer_keyword != None and cm_keyword == None:
                    aa = [cancer_keyword]
                    keyword_list.append(aa)
                elif cancer_keyword == None and cm_keyword == None:
                    pass
        else:
            pass
    
    after_conclution = False
    
    for j in range(len(reports)):
        if 'Conclusion' in reports[j]:
            after_conclution = True
            
        if after_conclution == True:
            if len(keyword_list) == 0:
                if 'colon cancer' in reports[j]:
                    for m in range(len(sub_detail_list)):
                        if sub_detail_list[m] in reports[j]:
                            cancer_keyword = f'{sub_detail_list[m]} cancer'
                            words = word_tokenize(reports[j])
                            for k in range(len(words)):
                                if 'cm' in words[k]:
                                    cm_keyword = words[k]
                else:
                    cancer_keyword = None
                    cm_keyword = None
                    
                if cancer_keyword != None and cm_keyword != None:
                    aa = [cancer_keyword, cm_keyword]
                    keyword_list.append(aa) 
                elif cancer_keyword != None and cm_keyword == None:
                    aa = [cancer_keyword]
                    keyword_list.append(aa)
                elif cancer_keyword == None and cm_keyword == None:
                    pass
        else:
            pass

    for i in range(len(keyword_list)):
        for j in range(len(keyword_list[i])):
                disease = keyword_list[i][j]
                if disease == 'Cecal cancer' or disease == 'Cecal cancer' or disease == 'cecal cancer' or disease == 'cecal ca.':
                    keyword_list[i].append(cancer_ko_list[0])
                elif disease == 'Ascending colon cancer' or disease == 'ascending colon cancer' or disease == 'A-colon cancer' or disease == 'A-colon ca.' or disease == 'A colon ca.' or disease == 'A colon cancer' or disease == 'Ascending colon ca.' or disease == 'ascending colon ca.' or disease == 'Right colon cancer' or disease == 'Right colon ca.' or disease == 'right colon cancer' or disease == 'right colon ca.':
                    keyword_list[i].append(cancer_ko_list[1])
                elif disease == 'Hepatic flexure colon cancer' or disease == 'Hepatic flexure colon ca.' or disease == 'hepatic flexure colon cancer' or disease == 'hepatic flexure colon ca.' or disease == 'HF cancer' or disease == 'HF ca.':
                    keyword_list[i].append(cancer_ko_list[2])
                elif disease == 'T colon cancer' or disease == 'T colon ca.' or disease == 'T-colon cancer' or disease == 'T-colon ca.' or disease == 'Transverse colon cancer' or disease == 'Transverse colon ca.' or disease == 'transverse colon cancer' or disease == 'transverse colon ca.':
                    keyword_list[i].append(cancer_ko_list[3])
                elif disease == 'Splenic flexure colon cancer' or disease == 'Splenic flexure colon ca.' or disease == 'splenic flexure colon cancer' or disease == 'splenic flexure colon ca.':
                    keyword_list[i].append(cancer_ko_list[4])
                elif disease == 'Descending colon cancer' or disease == 'Descending colon ca.' or disease == 'descending colon cancer' or disease == 'descending colon ca.' or disease == 'D-colon cancer' or disease == 'D-colon ca.' or disease == 'D colon cancer' or disease == 'D colon ca.':
                    keyword_list[i].append(cancer_ko_list[5])
                elif disease == 'Sigmoid-descending junction cancer' or disease == 'sigmoid-descending junction cancer' or disease == 'S-D junction cancer' or disease == 'SD junction cancer' or disease == 'SDJ colon cancer' or disease == 'SD colon cancer' or disease == 'SD colon ca.':
                    keyword_list[i].append(cancer_ko_list[6])
                elif disease == 'Sigmoid colon cancer' or disease == 'Sigmoid colon ca.' or disease == 'sigmoid colon cancer' or disease == 'sigmoid colon ca.' or disease == 'S-colon cancer' or disease == 'S-colon ca.' or disease == 'S colon cancer' or disease == 'S colon ca.':
                    keyword_list[i].append(cancer_ko_list[7])
                elif disease == 'Rectosigmoid junction cancer' or disease == 'rectosigmoid junction cancer' or disease == 'Rectosigmoid junction ca.' or disease == 'rectosigmoid colon cancer' or disease == 'Rectosigmoid colon ca.' or disease == 'RS junction cancer' or disease == 'RS colon cancer':
                    keyword_list[i].append(cancer_ko_list[8])
                elif disease == 'Rectal cancer' or disease == 'Rectal ca.' or disease == 'rectal cancer' or disease == 'rectal ca.':
                    keyword_list[i].append(cancer_ko_list[9])
                elif disease == 'Upper rectal cancer' or disease == 'Upper rectal ca.' or disease == 'Upper-rectal cancer' or disease == 'Upper-rectal ca.' or disease == 'upper rectal cancer' or disease == 'upper rectal ca.' or disease == 'upper-rectal cancer' or disease == 'upper-rectal ca,' or disease == 'High rectal cancer' or disease == 'High rectal ca.' or disease == 'High-rectal cancer' or disease == 'High-rectal ca.' or disease == 'high rectal cancer' or disease == 'high rectal ca.' or disease == 'high-rectal cancer' or disease == 'high-rectal ca.':
                    keyword_list[i].append(cancer_ko_list[10])
                elif disease == 'Mid rectal cancer' or disease == 'Mid rectal ca.' or disease == 'Mid-rectal cancer' or disease == 'Mid-rectal ca.' or disease == 'mid rectal cancer' or disease == 'mid rectal ca.' or disease == 'mid-rectal cancer' or disease == 'mid-rectal ca':
                    keyword_list[i].append(cancer_ko_list[11])
                elif disease == 'Low rectal cancer' or disease == 'Low rectal ca.' or disease == 'Lower-rectal cancer' or disease == 'Lower-rectal ca.' or disease == 'low rectal cancer' or disease == 'low rectal ca.' or disease == 'low-rectal cancer' or disease == 'low-rectal ca.' or disease == 'Lower rectal cancer' or disease == 'Lower lectal ca.' or disease == 'Lower-rectal cancer' or disease == 'Lower-rectal ca.'  or disease == 'lower rectal cancer' or disease == 'lower rectal ca.' or disease == 'lower-rectal cancer' or disease == 'lower-rectal ca.':
                    keyword_list[i].append(cancer_ko_list[12])
                elif disease == 'DS junction cancer':
                    keyword_list[i].append(cancer_ko_list[13])
                else:
                    pass
                
    return keyword_list

def keyword_count(keyword_list):
    response_ = str
    response_list = []
        
    if len(keyword_list) == 0:
        response_ = None
    else:
        sen = keyword_list
        
        if len(sen) == 1:
            if len(sen[-1]) == 3:
                response_ = f"{sen[-1][-2]}의 {sen[-1][-1]}"
                response_list.append(response_)
            elif len(sen[-1]) == 2:
                response_ = f"{sen[-1][-1]}"
                response_list.append(response_)
        elif len(sen) >= 2:

            unique, unique_list = [], []
            
            sen.reverse()
            
            for value in sen:
                if value[-1] not in unique:
                    unique.append(value[-1])
                    unique_list.append(value)
            
            for i in range(len(unique_list)):
                if len(unique_list[i]) == 3:
                    response_ = f"{unique_list[i][-2]}의 {unique_list[i][-1]}"
                    response_list.append(response_)
                elif len(unique_list[i]) == 2:
                    response_ = f"{unique_list[i][-1]}"
                    response_list.append(response_)
        else:
            pass
        
        response_list.reverse()
        
        # 에스 결장암, 직결장암
        words0 = ['에스 결장암', '직결장암']        
        response_list = find_word(response_list, words0)
        
        words1 = ['일반 직장암', '상부 직장암']
        response_list = find_word(response_list, words1)
        
        words2 = ['직장암', '중부 직장암']
        response_list = find_word(response_list, words2)
        
        words3 = ['직장암', '하부 직장암']
        response_list = find_word(response_list, words3)
                
    return response_, response_list

def find_word(response_list, words):
    matching_list0, matching_list1 = [], []

    for i in range(len(response_list)):
        if words[0] == response_list[i]:
            matching_list0.append(0)
        else:
            matching_list0.append(1)
    
    for i in range(len(response_list)):
        if words[1] == response_list[i]:
            matching_list1.append(0)
        else:
            matching_list1.append(1)

    for i in range(len(matching_list0)):
        try:
            if matching_list0[i] == 0 and matching_list1[i+1] == 0:
                response_list.pop(i)
        except:
            pass
    
    return response_list

def ans_choice(I_res, output_path, input_path, txt_path):
    response = int
    
    if I_res == 'ans01':
        keyword_list = categorizing_C_keyword(input_path, txt_path)
        response_, response_list = keyword_count(keyword_list)
        if response_ != None:
            response = f"환자의 판독지 내 발견된 증상은 {response_list}입니다."
            synthesize_text(response, output_path)
            cov_play(output_path)
            response = 1
        elif response_ == None:
            ans = "환자의 판독지 내 증상이 존재하지 않습니다."
            synthesize_text(ans, output_path)
            cov_play(output_path)
            response = 1
    elif I_res == 'ans999' or I_res == None:
        response = "방금 하신 말씀을 잘 못 알아들었어요. \n다시 말씀해 주세요."
        synthesize_text(response, output_path)
        cov_play(output_path)
        response = 2
    else:
        response = None
    
    return response

def test(input_path, txt_path):
    keyword_list = categorizing_C_keyword(input_path, txt_path)
    
    ans, ans_list = keyword_count(keyword_list)
    
    if ans == None:
        ans = "환자의 판독지 내 증상이 존재하지 않습니다."
        print(ans)
    else:
        print(f"환자의 판독지 내 발견된 증상은 {ans_list} 입니다.")
        