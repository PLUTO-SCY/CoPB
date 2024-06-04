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

GPT_MODEL = "gpt-4-turbo-preview" 
# GPT_MODEL = "gpt-3.5-turbo-1106"  

parser = argparse.ArgumentParser(description='gen DayIntention multi-step')
parser.add_argument('--keyid', default = 0, type=int)  
parser.add_argument('--TemplateRes_PATH', default = 'TemplateRes', type=str)
parser.add_argument('--mode_choice', default = 'labeld', type=str)  # realRatio
args = parser.parse_args()

if __name__ == '__main__':

    timeDistribution_Event = getTimeDistribution()
    setOpenAi(keyid = args.keyid)  
    profileIds, profiles, personNum, genIdList, loopStart, loopEnd = setIO(modeChoice=args.mode_choice, keyid = args.keyid)
    
    
    for genid in genIdList:  
        if not os.path.exists('Results/{}/Res{}'.format(args.TemplateRes_PATH, genid)):
            os.makedirs('Results/{}/Res{}'.format(args.TemplateRes_PATH, genid))                


    for genid in genIdList:
        
        loopStart = count_files('Results/{}/Res{}'.format(args.TemplateRes_PATH, genid))
        for index in range(loopStart, loopEnd): 
            personBasicInfo = profiles[index]
        
            nowTime = '00:00'
            N = getEventNum() 
            N = 3
            day = getDay()
            globalInfo = getBasicData_0509(personBasicInfo, day, N, GPT_MODEL)
            history = []
            
            for i in range(N):  
                
                if i == 0:
                    question = """Now is 00:00. Based on current time, your profile and lifestyle habits, rank your level of confidence of next intention in completion.
Specifically, for the [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events] candidate intents, you need to give a ranking from most to least completion confidence.
Answer format: [intent1, intent2, intent3...]"""
                else:
                    question = """Now is {}. Based on current time, your profile and lifestyle habits, rank your level of confidence of next intention in completion.
Specifically, for the [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events] candidate intents, you need to give a ranking from most to least completion confidence.
Answer format: [intent1, intent2, intent3...]""".format(nowTime) 

    
                PBC_Q = [{"role": "user", "content": question}] 
                nowQ = globalInfo + PBC_Q
                rank_response = askChatGPT(nowQ, model=GPT_MODEL, temperature = 1)
                # print(rank_response)
                nowQ = nowQ + [{"role": "assistant", "content": rank_response}]
                 
                
                ordinal_Numeral = get_Ordinal_Numeral(i+1)
                if i == N-1: 
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


                Q = [{"role": "user", "content": question}]
                nowQ = nowQ + Q
                
                genOK = False
                while not genOK:
                    try:
                        response = askChatGPT(nowQ, model=GPT_MODEL, temperature = 1)
                        response = eval(response)
                        reason = response[1]
                        answer = response[0]

                        if answer in ["go to work", "go home", "eat", "do shopping", "do sports", "excursion", "leisure or entertainment", "go to sleep", "medical treatment", "handle the trivialities of life", "banking and financial services", "cultural institutions and events"]: 
                            genOK = True
                        else:
                            pass
                    except:
                        pass
                
                
                nowQ = nowQ + [{"role": "assistant", "content": answer}]
                Q2 = [{"role": "user", "content": """How long will you spend on this arrangement?
You must consider some fragmented time, such as 3 hours plus 47 minute, and 7 hours and 13 minutes.
Please answer as a list: [x,y]. Which means x hours and y minutes."""}]
                nowQ = nowQ + Q2
                

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
                if increment_minutes + noiseTime > 0: 
                    increment_minutes = increment_minutes + noiseTime  
                
                end_time, cross_day = add_time(nowTime, increment_minutes)
                
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
                
            with open('Results/{}/Res{}/GenResult_{}.json'.format(args.TemplateRes_PATH, genid, profileIds[index]),'w') as file_obj: 
                json.dump(history, file_obj)


