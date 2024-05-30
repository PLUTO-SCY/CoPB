'''
LLM来生成持续时间
并且用TPB理论构建prompts
'''
# -*- coding: utf-8 -*-
import argparse
import json
import sys
import time

import numpy as np

from All_In_One_Utils import *
from setBasicData import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

GPT_MODEL = "gpt-4-turbo-preview"  # 速度快,完成问题的成功率高,价格略高
# GPT_MODEL = "gpt-3.5-turbo-1106"  # 在快速迭代阶段就用3.5 turbo吧，不然确实是有点昂贵  3.5虽然响应快,但是成功率太低了

parser = argparse.ArgumentParser(description='gen DayIntention multi-step')
parser.add_argument('--keyid', default = 0, type=int)  # 从0开始的,还是得用5个
parser.add_argument('--TemplateRes_PATH', default = 'TemplateRes_0521_只是为了测试生成的时间', type=str)
parser.add_argument('--mode_choice', default = 'labeld', type=str)  # realRatio
args = parser.parse_args()

# 只生成50个就够了

if __name__ == '__main__':

    timeDistribution_Event = getTimeDistribution()
    setOpenAi(keyid = args.keyid)  # keyid 从0到5进行加速
    profileIds, profiles, personNum, genIdList, loopStart, loopEnd = setIO(modeChoice=args.mode_choice, keyid = args.keyid)
    
    # loopEnd = 5
    # print(len(profiles))
    # 这个loopstart可以根据文件夹中的文件数量进行读取
    
    
    for genid in genIdList:  # 提前把所有文件夹建好
        if not os.path.exists('Results/{}/Res{}'.format(args.TemplateRes_PATH, genid)):
            os.makedirs('Results/{}/Res{}'.format(args.TemplateRes_PATH, genid))                


    for genid in genIdList:
        
        loopStart = count_files('Results/{}/Res{}'.format(args.TemplateRes_PATH, genid))
        for index in range(loopStart, loopEnd): # top500 profiles生成,计划生成top_250的profile的模板.
            personBasicInfo = profiles[index]
        
            nowTime = '00:00'
            N = getEventNum()  # 随机选一个事件数目
            N = 3
            day = getDay()  # 随机选一天
            globalInfo = getBasicData_0509(personBasicInfo, day, N, GPT_MODEL)
            history = []
            
            # if os.path.exists('Results/{}/Res{}/GenResult_{}.json'.format(args.TemplateRes_PATH, genid, profileIds[index])):  # 生成的路径是唯一且确定的,如果已经生成过了就跳过
            #     break
            
            for i in range(N):  # 这个N要用来控制总事件数量啊
                
                # 在生成这一步的intent之前多了一个行为完成度的置信度打分机制。
                if i == 0:
                    question = """Now is 00:00. Based on current time, your profile and lifestyle habits, rank your level of confidence of next intention in completion.
Specifically, for the [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events] candidate intents, you need to give a ranking from most to least completion confidence.
Answer format: [intent1, intent2, intent3...]"""
                else:
                    question = """Now is {}. Based on current time, your profile and lifestyle habits, rank your level of confidence of next intention in completion.
Specifically, for the [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events] candidate intents, you need to give a ranking from most to least completion confidence.
Answer format: [intent1, intent2, intent3...]""".format(nowTime)  # 只考虑当前的时间说不定会有用呢

    
                PBC_Q = [{"role": "user", "content": question}]  # 看看这个是否有效果
                nowQ = globalInfo + PBC_Q
                rank_response = askChatGPT(nowQ, model=GPT_MODEL, temperature = 1)
                # print(rank_response)
                nowQ = nowQ + [{"role": "assistant", "content": rank_response}]
                 
                
                ordinal_Numeral = get_Ordinal_Numeral(i+1)
                if i == N-1:  # 表示是最后一件事情
                    historyPrompts = "The arrangement of the previous {} things is as follows: {}. ".format(i, history)
                    # question = "{}What's the last arrangement today? Please just output the event name. No quotation marks required. No explanation is required.".format(historyPrompts)
                    question = "{}What's the last arrangement today? Please output the event name and explain your thought process or reasons for your choice. Answer format:[\"event name\", \"reasons\"]".format(historyPrompts)
                elif i == 0:
                    historyPrompts = ""
                    # question = "{}What's the {} arrangement today? Please just output the event name. No quotation marks required. No explanation is required.".format(historyPrompts, ordinal_Numeral)
                    question = "{}What's the {} arrangement today? Please output the event name and explain your thought process or reasons for your choice. Answer format:[\"event name\", \"reasons\"]".format(historyPrompts, ordinal_Numeral)
                else:
                    historyPrompts = "The arrangement of the previous {} things is as follows: {}. ".format(i, history)
                    # question = "{}What's the next arrangement today? Please just output the event name. No quotation marks required. No explanation is required.".format(historyPrompts)
                    question = "{}What's the next arrangement today? Please output the event name and explain your thought process or reasons for your choice. Answer format:[\"event name\", \"reasons\"]".format(historyPrompts)
                
                # print(question+'\n')
                # 注意让回答不要加引号,不然会出错
                # print(question)

                Q = [{"role": "user", "content": question}]
                nowQ = nowQ + Q
                
                genOK = False
                while not genOK:
                    try:
                        # 生成合法
                        response = askChatGPT(nowQ, model=GPT_MODEL, temperature = 1)
                        response = eval(response)
                        reason = response[1]
                        answer = response[0]

                        # answer = answer.strip().lower()
                        # if answer in ["go to work", "go home", "have breakfast", "have lunch", "have dinner", "do shopping", "do sports", "excursion", "leisure or entertainment", "go to sleep", "medical treatment", "handle the trivialities of life", "banking and financial services", "cultural institutions and events"]:
                        if answer in ["go to work", "go home", "eat", "do shopping", "do sports", "excursion", "leisure or entertainment", "go to sleep", "medical treatment", "handle the trivialities of life", "banking and financial services", "cultural institutions and events"]: 
                            genOK = True
                        else:
                            # 死循环直至生成成功
                            pass
                    except:
                        pass
                
                # if i == N-1 and N!=2:  # 最后一件事,还要不要控制
                #     answer = 'go home'
                
                nowQ = nowQ + [{"role": "assistant", "content": answer}]
                Q2 = [{"role": "user", "content": """How long will you spend on this arrangement?
You must consider some fragmented time, such as 3 hours plus 47 minute, and 7 hours and 13 minutes.
Please answer as a list: [x,y]. Which means x hours and y minutes."""}]
                nowQ = nowQ + Q2
                
                    
                # increment_minutes = sampleTime(answer)  # 需要根据事件类型来进行采样。
                # 之前是从数据中进行采样,但是现在不直接采样了,而是问LLM直接生成。
                genOK = False
                while not genOK:
                    try:
                        increment_minutes = askChatGPT(nowQ, model=GPT_MODEL, temperature = 1.5)
                        increment_minutes = eval(increment_minutes)
                        genOK = True
                    except:
                        pass
                increment_minutes = 60*increment_minutes[0]+increment_minutes[1]
                noiseTime = sampleNoiseTime()
                if increment_minutes + noiseTime > 0:  # 防止变成负数
                    increment_minutes = increment_minutes + noiseTime  # 转化成分钟数量
                
                end_time, cross_day = add_time(nowTime, increment_minutes)
                
                # 一旦时间跨天就提前终止生成过程,说明生成满1天
                if cross_day or end_time == "23:59":
                    seTime = "("+ nowTime+", 23:59)"
                    history.append([answer, seTime])
                    break
                else:
                    seTime = "("+ nowTime+", "+end_time+")"
                    history.append([answer, seTime])
                    
                    # print(history)
                    gapTime = sampleGapTime()
                    tmpnowTime, cross_day = add_time(end_time, gapTime)  
                    if cross_day:
                        nowTime = end_time
                    else:
                        nowTime = tmpnowTime
                            
            print("GEN OK!!!  ProfileIndex: {}  |  personId: {}  |  genIndex: {}  |  eventNum: {}".format(index, profileIds[index], genid, N))                     
                
            with open('Results/{}/Res{}/GenResult_{}.json'.format(args.TemplateRes_PATH, genid, profileIds[index]),'w') as file_obj:  # 这里的index就是模板的序号了,而genid是生成结果的序号
                json.dump(history, file_obj)


