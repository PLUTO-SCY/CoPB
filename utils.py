'''
2023/11/11 begin
2023/11/19 last modification
'''

import logging
import os
import pickle
import sys

import openai

MODEL = "gpt-3.5-turbo"


def printSeq(seq):
    for item in seq:
        print(item)


def setup_logger(agentid):
    # os.remove("Logs/record.log")
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler("Logs/record_{}.log".format(agentid))

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger,"Logs/record_{}.log".format(agentid)


def askChatGPT(messages, model="gpt-3.5-turbo", temperature = 1):
    response = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            temperature = temperature,
        )
    addtoken(response.usage.total_tokens)
    answer = response.choices[0].message["content"]
    return answer.strip()

def askChatGPT_saveDialogue(messages, model="gpt-3.5-turbo", temperature = 1):
    response = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            temperature = temperature,
        )
    addtoken(response.usage.total_tokens)
    answer = response.choices[0].message["content"]
    return answer.strip()

def saveDialogue(conversation, path, ifexist):
    if ifexist:
        file = open(path,'rb') 
        data = pickle.load(file)
        data.append(conversation) 
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    else:
        data = [conversation]
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    

def askLocalGPT(messages, model_name, temperature = 1):
    
    # response = openai.chat.completions.create(
    #     model = model_name,
    #     messages = messages,
    #     temperature = temperature,
    #     max_tokens = 2048,)  # which control the randomness of the results, between 0 and 2
    # addtoken(response.usage.total_tokens)
    # answer = response.choices[0].message.content
    # return answer.strip()

    response = openai.ChatCompletion.create(
            model = model_name,
            messages = messages,
            temperature = temperature,
            max_tokens = 2048,
        )
    # addtoken(response.usage.total_tokens)
    answer = response.choices[0].message["content"]
    return answer.strip()


def setOpenAi(keyid = 0):
    '''
    Set the Internet access port and key.
    '''

    if keyid == 0:
        openai.api_key = ""
    elif keyid == 1:
        openai.api_key = ""
    elif keyid == 2:
        openai.api_key = ""
    elif keyid == 3:
        openai.api_key = ""
    elif keyid == 4:
        openai.api_key = ""
        
    addtoken(-1)
    return 0

def setLocalOpenAi(choice = 0):
    openai.api_base = "http://101.6.69.60:5000/v1"
    openai.api_key = "token-fiblab-20240513"
    model_name = "Meta-Llama-3-8B-Instruct"
    return model_name

def getTime(timestr):
    timelist = timestr[1:-1].split(',')
    return [int(timelist[0]), int(timelist[1])]

def printQA(Q, A, logger, additional_info = ''):
    logger.info(additional_info + 'Question: {}'.format(Q))
    logger.info(additional_info + 'Answer: {}'.format(A+'\n'))
    
def addtoken(num):
    try:
        with open("tokens.txt", "r") as f:  
            data = f.read()  
            nownum = int(data)        
            
        if num == -1:
            nownum = 0
        else:
            nownum = nownum + num
        
        with open("tokens.txt","w+") as f:
            f.write(str(nownum))
    except:
        pass

def timeEval(dis):
    return int(dis/70) 

def calTime(start, duration):
    h1, m1  = start.split(':')
    h1 = int(h1)
    m1 = int(m1)
    m1 = m1 + duration
    if m1 > 59:
        h1 += 1
        m1 -= 60
        if h1 > 23:
            h1 = h1-24
    h1 = str(h1)
    m1 = str(m1)
    if len(m1) == 1:
        m1 = '0' + m1
    return str(h1) + ':' + str(m1)
        
def turntime2list(time):
    time = time[1:-1]
    time = time.split(',')
    time = [i.strip() for i in time]
    return (time[0], time[1])

def getDirectEventID(event):
    if event in ['have breakfast', 'have lunch', 'have dinner', 'eat']:
        return "10"
    elif event == 'do shopping':
        return "13"
    elif event == 'do sports':
        return "18"
    elif event == 'excursion': 
        return "22"
    elif event == 'leisure or entertainment':
        return "16"
    elif event == 'medical treatment':
        return "20"
    elif event == 'handle the trivialities of life':
        return "14"
    elif event == 'banking and financial services':
        return "25"
    elif event == 'government and political services':
        return "12"
    elif event == 'cultural institutions and events':
        return "23"
    else:
        print('\nIn function event2cate: The selected choice: {} is not in the range!\n'.format(event))
        sys.exit(0)
    
    
def printDict(d):
    for key, value in d.items():
        print(key)
        print(value)
    

def count_files(folder_path):
    file_count = 0    
    for root, dirs, files in os.walk(folder_path):
        file_count += len(files)
    return file_count


