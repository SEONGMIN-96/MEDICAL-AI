# replace dialogflow
def dialogflow_intents(intents_S, intents_path, entity_path, text):
    from nltk.tokenize import word_tokenize, sent_tokenize  
    import os

    words = word_tokenize(text)
    exist = False
    
    for word in words:        
        if exist == False:
            if '종료' in word or '그만' in word or 'exit' in word: 
                res = 'ans-exit'
                exist = True
            else:
                I_res, entities = dialogflow_intents_entities(intents_S, 
                                                              intents_path, 
                                                              entity_path,
                                )
                for entity in entities:
                    if exist == False:
                        if entity in word:
                            res = I_res
                            exist = True
                        elif entity not in word:
                            res = 'ans999'
                    elif exist == True:
                        pass
        elif exist == True:
            pass
    return res

def dialogflow_intents_entities(intents_S, intents_path, entities_path):
    import os
    
    entities = []
    
    # read intents
    f = open(os.path.join(intents_path, f'I_{intents_S}.txt'), 'r', encoding='utf-8')
    I_res = f.readline()
    f.close
    # read entities
    f = open(os.path.join(entities_path, f'E_{intents_S}.txt'), 'r', encoding='utf-8')
    while True:
        line = f.readline()
        if line == '':
            break
        entity = line.rstrip('\n')
        entities.append(entity)
    f.close
    
    return I_res, entities


def dialogflow(intent_S, text):
    # 텍스트를 입력받는다
    # 인텐트는 순서에 따라 진행되어야한다.
    # 인텐트는 엔티티 속의 단어 유무에 따라 결정된다.
    import os
    
    dir_path = os.path.join(os.getcwd(), 'bin', 'data', 'dialogflow')
    intents_path = os.path.join(dir_path, 'intents')
    entity_path = os.path.join(dir_path, 'entities')
    
    intent_S = intent_S
    text = text
    res = str
    
    if len(text) > 0:
        res = dialogflow_intents(intent_S, intents_path, entity_path, text)
    else:
        pass
    
    print("Response of Dialogflow : {}".format(res))
    
    return res    