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
    """
    计算结束后的时间，给出起始时间和增量时间（分钟）
    :param start_time: 起始时间，格式为 '%H:%M'
    :param minutes: 增量时间（分钟）
    :return: 结束后的时间，格式为 '%H:%M'；是否跨越了一天的标志
    """
    # 将字符串转换为 datetime 对象，日期部分设为一个固定的日期
    start_datetime = datetime.strptime('2000-01-01 ' + start_time, '%Y-%m-%d %H:%M')

    # 增加指定的分钟数
    end_datetime = start_datetime + timedelta(minutes=minutes)

    # 判断是否跨越了一天
    if end_datetime.day != start_datetime.day:
        cross_day = True
    else:
        cross_day = False

    # 将结果格式化为字符串，只包含时间部分
    end_time = end_datetime.strftime('%H:%M')

    return end_time, cross_day

def getTimeFromZone(timeZone0):
    time0, time1 = timeZone0.split('-')
    time0 = float(time0)/2  # 这里已经化成小时了
    time1 = float(time1)/2
    # print(time0)
    # print(time1)
    
    sampleResult = random.uniform(time0, time1)  # 采样一个具体的时间值出来,单位是小时
    # print(sampleResult)  
    minutes = int(sampleResult*60)
    return minutes

def sampleTime(event):
    '''
    根据事件的类型,在真实数据统计出的分布中进行采样,获取时间
    '''
    # genEs = ["go to work", "go home", "eat", "do shopping", "do sports", "excursion", "leisure or entertainment", "go to sleep", "medical treatment", "handle the trivialities of life", "banking and financial services", "cultural institutions and events"]
    # realEs = ['excursion', 'leisure or entertainment', 'eat', 'go home', 'do sports', 'handle the trivialities of life', 'do shopping', 'go to work', 'medical treatment', 'go to sleep']
    # for item in genEs:
    #     if item not in realEs:
    #         print(item)
    # print(timeDistribution_Event.keys())
    
    timeDistribution_Event = getTimeDistribution() 
    timeDis = timeDistribution_Event[event]
    timeZones = list(timeDis.keys())
    length = len(list(timeDis.keys()))
    weightList = list(timeDis.values())
    indexx = list(range(length))
    sample = np.random.choice(indexx, size=1, p=weightList)  # 根据计算出来的概率值进行采样
    timeZone = timeZones[sample[0]]  # 选好了以半小时为度量的区间
    minutes = getTimeFromZone(timeZone)
    return minutes

def sampleGapTime():
    '''
    事件之间的间隔时间
    '''
    minutes = getTimeFromZone('0-2')  # 将事件的间隔时间设置为0到1个小时
    return minutes

def getEventNum():
    '''
    通过真实概率值采样,得到事件总数 
    实际上如果太多的话,会自动截断的.也生成不了这么多.
    事件的数量是从真实数据中采样而来
    '''
    value = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    prob = [0.2266657982153325, 0.42213248225102584, 0.14472741483749105, 0.10851299420308734, 0.04604963199374715, 0.024946264573698952, 0.011789226861199766, 0.006839054256497101, 0.004559369504331401, 0.0017586139516706832, 0.0011072754510519118, 0.0003256692503093858, 0.00013026770012375432, 0.00026053540024750864, 0.00019540155018563148]
    index = list(range(len(value)))
    sample = np.random.choice(index, size=1, p=prob)  # 根据计算出来的概率值进行采样
    return value[sample[0]]

def getDay():
    '''
    随机采样今天是星期几
    '''
    value = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    prob = [0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432]
    index = list(range(len(value)))
    sample = np.random.choice(index, size=1, p=prob)  # 根据计算出来的概率值进行采样
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
    file = open('F:/Coding/Datasets/事件的时间分布/timeDistribution_Event.pkl','rb') 
    timeDistribution_Event = pickle.load(file)
    timeDistribution_Event['banking and financial services'] = timeDistribution_Event['handle the trivialities of life']
    timeDistribution_Event['cultural institutions and events'] = timeDistribution_Event['handle the trivialities of life']
    return timeDistribution_Event
    
def setIO(modeChoice, keyid, bias=0):
    if modeChoice == 'labeld':
        file = open('../Datasets/人物profile统计无年龄/标注数据人物profile.pkl','rb') 
        profileDict = pickle.load(file)  # 标注数据是需要id的
        profileIds = list(profileDict.keys())
        profiles = list(profileDict.values())
        personNum = len(profiles)
        genIdList = [keyid+bias]  # 对一类人生成5个模板,将key分配到id上进行加速
        loopStart = 0
        loopEnd = personNum
        return profileIds, profiles, personNum, genIdList, loopStart, loopEnd
    
    elif modeChoice == 'realRatio':
        file = open('../Datasets/人物profile统计无年龄/profileWordDict.pkl','rb') 
        profileDict = pickle.load(file)  # 其中的keys已经做了匿名化
        profileIds = list(profileDict.keys())
        profilesAndNum = list(profileDict.values())
        profiles = [item[0] for item in profilesAndNum]
        personNum = len(profiles)
        genIdList = [0] # list(range(10))  # 现在生成10次
        loopStart = keyid*60  # 将多key放到id上加速,top300其实也是可以的
        loopEnd = keyid*60+60
        return profileIds, profiles, personNum, genIdList, loopStart, loopEnd
    
    elif modeChoice == 'shy':
        file = open('../Datasets/人物profile统计无年龄/profileWordDict.pkl','rb') 
        profileDict = pickle.load(file)  # 其中的keys已经做了匿名化
        profileIds = list(profileDict.keys())
        profilesAndNum = list(profileDict.values())
        profiles = [item[0] for item in profilesAndNum]
        personNum = len(profiles)
        genIdList = [0] # list(range(10))  # 现在生成10次
        loopStart = keyid*200  # 将多key放到id上加速,top300其实也是可以的
        loopEnd = keyid*200+200
        return profileIds, profiles, personNum, genIdList, loopStart, loopEnd
    
    else:
        print('error!')
        sys.exit(0)

def sampleNoiseTime():
    noise = random.randint(-10, 10)
    return noise

