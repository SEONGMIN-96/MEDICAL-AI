from .tts import synthesize_text_google, synthesize_text_clova_voice
from .sound import cov_play, mp3_play


def read_txt(txt_path):
    '''
    .txt 형태의 키워드 모음 읽기
    '''
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
    '''
    .txt 형태의 판독문 읽기
    '''
    reports = str
    
    reports = open(txt_path, 'r', encoding='UTF8').read()
    
    print('Reading Reports...')
    for i in range(2):
        print('...')
    print('\n')
    
    return reports

def categorizing_colon_keyword(info_path, f_path):
    '''
    판독문 속 존재하는 대장암 키워드 및 세부정보 추출
    '''
    from nltk.tokenize import word_tokenize, sent_tokenize    
    import os    
    
    txt_path = os.path.join(info_path, f_path)
    cancer_path = os.path.join(info_path,  "01_colon_cancer_list.txt")
    ko_list = os.path.join(info_path, "02_colon_ko_list.txt")
    sub_detail_path = os.path.join(info_path, "03_colon_sub_detail.txt")
    
    reports = read_paper(txt_path=txt_path)
    
    return_reports = reports
    
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
            print("No Conclusion in the report...")
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
                if disease == 'Cecal cancer' or \
                        disease == 'Cecal cancer' or \
                        disease == 'cecal cancer' or \
                        disease == 'cecal ca.':
                    keyword_list[i].append(cancer_ko_list[0])
                elif disease == 'Ascending colon cancer' or \
                        disease == 'ascending colon cancer' or \
                        disease == 'A-colon cancer' or \
                        disease == 'A-colon ca.' or \
                        disease == 'A colon ca.' or \
                        disease == 'A colon cancer' or \
                        disease == 'Ascending colon ca.' or \
                        disease == 'ascending colon ca.' or \
                        disease == 'Right colon cancer' or \
                        disease == 'Right colon ca.' or \
                        disease == 'right colon cancer' or \
                        disease == 'right colon ca.':
                    keyword_list[i].append(cancer_ko_list[1])
                elif disease == 'Hepatic flexure colon cancer' or \
                        disease == 'Hepatic flexure colon ca.' or \
                        disease == 'hepatic flexure colon cancer' or \
                        disease == 'hepatic flexure colon ca.' or \
                        disease == 'HF cancer' or \
                        disease == 'HF ca.':
                    keyword_list[i].append(cancer_ko_list[2])
                elif disease == 'T colon cancer' or \
                        disease == 'T colon ca.' or \
                        disease == 'T-colon cancer' or \
                        disease == 'T-colon ca.' or \
                        disease == 'Transverse colon cancer' or \
                        disease == 'Transverse colon ca.' or \
                        disease == 'transverse colon cancer' or \
                        disease == 'transverse colon ca.':
                    keyword_list[i].append(cancer_ko_list[3])
                elif disease == 'Splenic flexure colon cancer' or \
                        disease == 'Splenic flexure colon ca.' or \
                        disease == 'splenic flexure colon cancer' or \
                        disease == 'splenic flexure colon ca.':
                    keyword_list[i].append(cancer_ko_list[4])
                elif disease == 'Descending colon cancer' or \
                        disease == 'Descending colon ca.' or \
                        disease == 'descending colon cancer' or \
                        disease == 'descending colon ca.' or \
                        disease == 'D-colon cancer' or \
                        disease == 'D-colon ca.' or \
                        disease == 'D colon cancer' or \
                        disease == 'D colon ca.':
                    keyword_list[i].append(cancer_ko_list[5])
                elif disease == 'Sigmoid-descending junction cancer' or \
                        disease == 'sigmoid-descending junction cancer' or \
                        disease == 'S-D junction cancer' or \
                        disease == 'SD junction cancer' or \
                        disease == 'SDJ colon cancer' or \
                        disease == 'SD colon cancer' or \
                        disease == 'SD colon ca.':
                    keyword_list[i].append(cancer_ko_list[6])
                elif disease == 'Sigmoid colon cancer' or \
                        disease == 'Sigmoid colon ca.' or \
                        disease == 'sigmoid colon cancer' or \
                        disease == 'sigmoid colon ca.' or \
                        disease == 'S-colon cancer' or \
                        disease == 'S-colon ca.' or \
                        disease == 'S colon cancer' or \
                        disease == 'S colon ca.':
                    keyword_list[i].append(cancer_ko_list[7])
                elif disease == 'Rectosigmoid junction cancer' or \
                        disease == 'rectosigmoid junction cancer' or \
                        disease == 'Rectosigmoid junction ca.' or \
                        disease == 'rectosigmoid colon cancer' or \
                        disease == 'Rectosigmoid colon ca.' or \
                        disease == 'RS junction cancer' or \
                        disease == 'RS colon cancer':
                    keyword_list[i].append(cancer_ko_list[8])
                elif disease == 'Rectal cancer' or \
                        disease == 'Rectal ca.' or \
                        disease == 'rectal cancer' or \
                        disease == 'rectal ca.':
                    keyword_list[i].append(cancer_ko_list[9])
                elif disease == 'Upper rectal cancer' or \
                        disease == 'Upper rectal ca.' or \
                        disease == 'Upper-rectal cancer' or \
                        disease == 'Upper-rectal ca.' or \
                        disease == 'upper rectal cancer' or \
                        disease == 'upper rectal ca.' or \
                        disease == 'upper-rectal cancer' or \
                        disease == 'upper-rectal ca,' or \
                        disease == 'High rectal cancer' or \
                        disease == 'High rectal ca.' or \
                        disease == 'High-rectal cancer' or \
                        disease == 'High-rectal ca.' or \
                        disease == 'high rectal cancer' or \
                        disease == 'high rectal ca.' or \
                        disease == 'high-rectal cancer' or \
                        disease == 'high-rectal ca.':
                    keyword_list[i].append(cancer_ko_list[10])
                elif disease == 'Mid rectal cancer' or \
                        disease == 'Mid rectal ca.' or \
                        disease == 'Mid-rectal cancer' or \
                        disease == 'Mid-rectal ca.' or \
                        disease == 'mid rectal cancer' or \
                        disease == 'mid rectal ca.' or \
                        disease == 'mid-rectal cancer' or \
                        disease == 'mid-rectal ca':
                    keyword_list[i].append(cancer_ko_list[11])
                elif disease == 'Low rectal cancer' or \
                        disease == 'Low rectal ca.' or \
                        disease == 'Lower-rectal cancer' or \
                        disease == 'Lower-rectal ca.' or \
                        disease == 'low rectal cancer' or \
                        disease == 'low rectal ca.' or \
                        disease == 'low-rectal cancer' or \
                        disease == 'low-rectal ca.' or \
                        disease == 'Lower rectal cancer' or \
                        disease == 'Lower lectal ca.' or \
                        disease == 'Lower-rectal cancer' or \
                        disease == 'Lower-rectal ca.' or \
                        disease == 'lower rectal cancer' or \
                        disease == 'lower rectal ca.' or \
                        disease == 'lower-rectal cancer' or \
                        disease == 'lower-rectal ca.':
                    keyword_list[i].append(cancer_ko_list[12])
                elif disease == 'DS junction cancer':
                    keyword_list[i].append(cancer_ko_list[13])
                else:
                    pass
                
    return keyword_list, return_reports

def colon_keyword_count(keyword_list):
    '''
    대장의 "결장암" 등 단어가 중첩되는 키워드 카운트
    '''
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
        response_list = find_similarity(response_list, words0)
        
        words1 = ['직장암', '상부 직장암']
        response_list = find_similarity(response_list, words1)
        
        words2 = ['직장암', '중부 직장암']
        response_list = find_similarity(response_list, words2)
        
        words3 = ['직장암', '하부 직장암']
        response_list = find_similarity(response_list, words3)
                
    return response_, response_list

def categorizing_liver_keyword(info_path, f_path):
    '''
    판독문 속 존재하는 간암 키워드 및 세부정보 추출
    '''
    from nltk.tokenize import word_tokenize, sent_tokenize    
    import os    
    
    txt_path = os.path.join(info_path, f_path)
    cancer_path = os.path.join(info_path,  "04_liver_cancer_list.txt")
    location_path = os.path.join(info_path, "05_liver_location_list.txt")
    ko_list_path = os.path.join(info_path, "06_liver_ko_list.txt")
    
    reports = read_paper(txt_path=txt_path)
    
    return_reports = reports
    
    reports = sent_tokenize(reports)

    cancer_list, cancer_ko_list = [], []
    
    keyword_list = []

    cancer_list = read_txt(cancer_path)
    location_list = read_txt(location_path)
    
    cancer_ko_list = read_txt(ko_list_path)    

    after_conclution = False

    for j in range(len(reports)):
        if 'Conclusion' in reports[j]:
            after_conclution = True
        
        if after_conclution == True:
            for i in range(len(cancer_list)):
                cancer_keyword = None
                location_keyword = None
                
                if cancer_list[i] in reports[j]:
                    cancer_keyword = cancer_list[i]
                    words = word_tokenize(reports[j])
                    for word in words:
                        for location in location_list:
                            if location in word:
                                location_keyword = location
                            
                else:
                    cancer_keyword = None
                    location_keyword = None
                    
                if cancer_keyword != None and location_keyword != None:
                    aa = [cancer_keyword, location_keyword]
                    keyword_list.append(aa) 
                elif cancer_keyword != None and location_keyword == None:
                    aa = [cancer_keyword]
                    keyword_list.append(aa)
                elif cancer_keyword == None and location_keyword == None:
                    pass
        else:
            print("No Conclusion in the report...")
            pass
    
    for i in range(len(keyword_list)):
        for j in range(len(keyword_list[i])):
                disease = keyword_list[i][j]
                for c in cancer_list:
                    if disease == c:
                        keyword_list[i].append(cancer_ko_list[0])
    
    return keyword_list, return_reports

def liver_keyword_count(keyword_list):
    '''
    간의 암 단어가 중첩되는 키워드 카운트
    '''
    response_ = str
    response_list = []
    
    if len(keyword_list) == 0:
        response_ = None
    else:
        sen = keyword_list
        
        if len(sen) == 1:
            if len(sen[-1]) > 2:
                response_ = f"{sen[-1][-2]}의 {sen[-1][-1]}"
                response_list.append(response_)
            elif len(sen[-1]) == 2:
                response_ = f"{sen[-1][1]}"
                response_list.append(response_)
        elif len(sen) >= 2:

            unique, unique_list = [], []
            
            sen.reverse()
            
            for value in sen:
                if value[-1] not in unique:
                    unique.append(value[-1])
                    unique_list.append(value)
            
            for i in range(len(unique_list)):
                if len(unique_list[i]) == 2:
                    response_ = f"{unique_list[i][-1]}의 {unique_list[i][-2]}"
                    response_list.append(response_)
                elif len(unique_list[i]) == 1:
                    response_ = f"{unique_list[i][-1]}"
                    response_list.append(response_)
        else:
            pass
        
        response_list.reverse()
        
    return response_, response_list

def find_similarity(response_list, words):
    '''
    "직장암" 등 중복포함된 단어로인해 잘못 마이닝된 키워드 제거
    '''
    matching_list0, matching_list1 = [], []

    for i in range(len(response_list)):
        if words[0] in response_list[i]:
            matching_list0.append(0)
        else:
            matching_list0.append(1)
    
    for i in range(len(response_list)):
        if words[1] in response_list[i]:
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

def ans_choice(I_res, output_path, input_path, report_path, mode, service_type):
    '''
    텍스트 마이닝을 통한 키워드 추출
    '''
    response = int

    if I_res == 'ans01':
        if mode == 'colon':
            keyword_list, return_reports = categorizing_colon_keyword(input_path, report_path)
            print('keyword_list: %s' % keyword_list)
            response_, response_list = colon_keyword_count(keyword_list)
        elif mode == 'liver':
            keyword_list, return_reports = categorizing_liver_keyword(input_path, report_path)
            print('keyword_list: %s' % keyword_list)
            response_, response_list = liver_keyword_count(keyword_list)

        print(return_reports)
        for _ in range(2):
            print('\n')
        print('=========================================')
        print("Is the patient's symptom correct???\n", response_list)
        print('=========================================')
        for _ in range(3):
            print('...')

        feedback = False
        
        while not feedback:        
            ox = input("True or False?? : ")
            if ox == 'True':
                feedback = True
                if response_ != None:
                    response = f"{response_list}입니다."
                    if service_type == 0:            
                        synthesize_text_google(response, output_path)
                        cov_play(output_path)
                    elif service_type == 1:
                        synthesize_text_clova_voice(response, output_path)
                        mp3_play(output_path)
                    response = 1
                elif response_ == None:
                    response = "증상이 존재하지 않습니다."
                    if service_type == 0:            
                        synthesize_text_google(response, output_path)
                        cov_play(output_path)
                    elif service_type == 1:
                        synthesize_text_clova_voice(response, output_path)
                        mp3_play(output_path)
                    response = 1
            elif ox == 'False':
                feedback = True
            else:
                print("choose True or False")        
    elif I_res == 'ans999' or I_res == str:
        response = "방금 하신 말씀을 잘 못 알아들었어요. \n다시 말씀해 주세요."
        if service_type == 0:            
            synthesize_text_google(response, output_path)
            cov_play(output_path)
        elif service_type == 1:
            synthesize_text_clova_voice(response, output_path)
            mp3_play(output_path)
        response = 999
    else:
        response = None
    
    return response

