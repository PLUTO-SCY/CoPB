import argparse
import json
import logging
import os
import pickle
import random
import sys
from datetime import datetime, timedelta

import numpy as np
import openai
from tqdm import tqdm


def add_time(start_time, minutes):
    
    start_datetime = datetime.strptime('2000-01-01 ' + start_time, '%Y-%m-%d %H:%M')

    end_datetime = start_datetime + timedelta(minutes=minutes)

    if end_datetime.day != start_datetime.day:
        cross_day = True
    else:
        cross_day = False

    end_time = end_datetime.strftime('%H:%M')

    return end_time, cross_day

def getTimeFromZone(timeZone0):
    time0, time1 = timeZone0.split('-')
    time0 = float(time0)/2  
    time1 = float(time1)/2
    
    sampleResult = random.uniform(time0, time1)  
    minutes = int(sampleResult*60)
    return minutes

def sampleTime(event):    
    timeDistribution_Event = getTimeDistribution() 
    timeDis = timeDistribution_Event[event]
    timeZones = list(timeDis.keys())
    length = len(list(timeDis.keys()))
    weightList = list(timeDis.values())
    indexx = list(range(length))
    sample = np.random.choice(indexx, size=1, p=weightList)
    timeZone = timeZones[sample[0]] 
    minutes = getTimeFromZone(timeZone)
    return minutes

def sampleGapTime():
    minutes = getTimeFromZone('0-2') 
    return minutes

def getEventNum():
    value = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    prob = [0.2266657982153325, 0.42213248225102584, 0.14472741483749105, 0.10851299420308734, 0.04604963199374715, 0.024946264573698952, 0.011789226861199766, 0.006839054256497101, 0.004559369504331401, 0.0017586139516706832, 0.0011072754510519118, 0.0003256692503093858, 0.00013026770012375432, 0.00026053540024750864, 0.00019540155018563148]
    index = list(range(len(value)))
    sample = np.random.choice(index, size=1, p=prob)  
    return value[sample[0]]

def getDay():
    value = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    prob = [0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432]
    index = list(range(len(value)))
    sample = np.random.choice(index, size=1, p=prob) 
    return value[sample[0]]

def get_Ordinal_Numeral(n):
    ordinal_dict = {
        "0": "zeroth",
        "1": "first",
        "2": "second",
        "3": "third",
        "4": "fourth",
        "5": "fifth",
        "6": "sixth",
        "7": "seventh",
        "8": "eighth",
        "9": "ninth",
        "10": "tenth",
        "11": "eleventh",
        "12": "twelfth",
        "13": "thirteenth",
        "14": "fourteenth",
        "15": "fifteenth",
        "16": "sixteenth",
        "17": "seventeenth",
        "18": "eighteenth",
        "19": "nineteenth",
        "20": "twentieth",
        "21": "twenty-first",
        "22": "twenty-second",
        "23": "twenty-third",
        "24": "twenty-fourth",
        "25": "twenty-fifth",
        "26": "twenty-sixth",
        "27": "twenty-seventh",
        "28": "twenty-eighth",
        "29": "twenty-ninth",
        "30": "thirtieth"
    }
    return ordinal_dict[str(n)]

def getTimeDistribution():
    file = open('Data/timeDistribution_Event.pkl','rb') 
    timeDistribution_Event = pickle.load(file)
    timeDistribution_Event['banking and financial services'] = timeDistribution_Event['handle the trivialities of life']
    timeDistribution_Event['cultural institutions and events'] = timeDistribution_Event['handle the trivialities of life']
    return timeDistribution_Event
    
def setIO(modeChoice, keyid, bias=0):
    if modeChoice == 'labeld':
        file = open('Data/Labeled_profile.pkl','rb') 
        profileDict = pickle.load(file) 
        profileIds = list(profileDict.keys())
        profiles = list(profileDict.values())
        personNum = len(profiles)
        genIdList = [keyid+bias] 
        loopStart = 0
        loopEnd = personNum
        return profileIds, profiles, personNum, genIdList, loopStart, loopEnd
    
    elif modeChoice == 'realRatio':
        file = open('Data/profileWordDict.pkl','rb') 
        profileDict = pickle.load(file) 
        profileIds = list(profileDict.keys())
        profilesAndNum = list(profileDict.values())
        profiles = [item[0] for item in profilesAndNum]
        personNum = len(profiles)
        genIdList = [0]
        loopStart = keyid*60  
        loopEnd = keyid*60+60
        return profileIds, profiles, personNum, genIdList, loopStart, loopEnd
    
    elif modeChoice == 'shy':
        file = open('Data/profileWordDict.pkl','rb') 
        profileDict = pickle.load(file) 
        profileIds = list(profileDict.keys())
        profilesAndNum = list(profileDict.values())
        profiles = [item[0] for item in profilesAndNum]
        personNum = len(profiles)
        genIdList = [0]
        loopStart = keyid*200  
        loopEnd = keyid*200+200
        return profileIds, profiles, personNum, genIdList, loopStart, loopEnd
    
    else:
        print('error!')
        sys.exit(0)

def sampleNoiseTime():
    noise = random.randint(-10, 10)
    return noise




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


def processRealTraces(data, map):
    traces = []
    for key, value in data.items():
        trace = []
        for point in value:
            try:
                trace.append([(point[0], point[1]), point[3], map.lnglat2xy(point[4][0], point[4][1])])
            except:
                print(point)
        
        traces.append(trace)
    return traces


def timeInday(time):
    try:
        h,m = time.split(':')
    except:
        h,m,_ = time.split(':')
    h = int(h)
    m = int(m)   
    minutes = h*60+m
    return minutes/(24*60)

def timeSplit(time):
    time = time[1:-1]
    start, end = time.split(',')
    start = start.strip()
    end = end.strip()
    return (timeInday(start), timeInday(end))

def genDataProcess(trace, map):
    res = []
    for item in trace:
        poiid = item[2][1]
        poi = map.get_poi(poiid)
        xy = poi['position']
        position = (xy['x'], xy['y'])
        SEtime = timeSplit(item[1])
        
        res.append([SEtime, poiid, position])
    return res

def readGenTraces(map, folderName):
    traces = []
    filePath = 'Results/{}/Res0'.format(folderName)
    allfiles = os.listdir(filePath)
    
    success = 0
    for filename in tqdm(allfiles):
        try:
            f = open("Results/{}/Res0/".format(folderName) + filename, 'r', encoding='utf-8')
            content = f.read()
            oneTrace = json.loads(content)
            trace = genDataProcess(oneTrace, map)
            traces.append(trace)
            success += 1
        except:
            pass
        
    print("read all num: {}".format(success))
    print("actually all num: {}".format(len(allfiles)))
    return traces

