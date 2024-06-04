import argparse
import json
import pickle
import random
# -*- coding: utf-8 -*-
import sys
from datetime import datetime, timedelta

import numpy as np
import openai


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

