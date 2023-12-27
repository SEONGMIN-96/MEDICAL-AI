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