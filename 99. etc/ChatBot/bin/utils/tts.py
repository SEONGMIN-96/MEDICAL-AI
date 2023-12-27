# google TTS
def synthesize_text_google(text, dir_path):
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
        
# clova
def synthesize_text_clova_voice(text, dir_path):
    from pydub import AudioSegment

    import os
    import sys
    import urllib.request

    client_id = "c417gymxgc"
    client_secret = "7C29vkJ1kQXzcKg9n8jU2NeOV4Pa9DEum8kOVRS1"
    encText = urllib.parse.quote(text)
    data = "speaker=nara_call&volume=0&speed=0&pitch=0&format=mp3&text=" + encText;
    url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
    request.add_header("X-NCP-APIGW-API-KEY",client_secret)
    response = urllib.request.urlopen(request, data=data.encode('utf-8'))
    rescode = response.getcode()
    
    audio_path = os.path.join(dir_path, 'response.mp3')
    
    print('audio_path:', audio_path)
    
    if(rescode==200):
        response_body = response.read()
        with open(audio_path, 'wb') as f:
            f.write(response_body)
    else:
        print("Error Code:" + rescode)