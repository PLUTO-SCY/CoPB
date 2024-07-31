import argparse
import json

from setBasicData import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

GPT_MODEL = "gpt-4-turbo-preview" 

parser = argparse.ArgumentParser(description='gen DayIntention multi-step')
parser.add_argument('--keyid', default = 0, type=int)
parser.add_argument('--TemplateRes_PATH', default = 'TemplateRes_COT', type=str) 
parser.add_argument('--mode_choice', default = 'realRatio', type=str)
args = parser.parse_args()


if __name__ == '__main__':
    
    timeDistribution_Event = getTimeDistribution()
    setOpenAi(keyid = args.keyid)
    profileIds, profiles, personNum, genIdList, loopStart, loopEnd = setIO(modeChoice=args.mode_choice, keyid = args.keyid)
    
    for index in range(loopStart, loopEnd):
        personBasicInfo = profiles[index]
        
        for genid in genIdList: 
            
            nowTime = '00:00'
            N = getEventNum()
            print('N: {}'.format(N))
            day = getDay()
            globalInfo = basicGlobalInfo(personBasicInfo, day, N)
            history = []
            
            if not os.path.exists('Results/{}/Res{}'.format(args.TemplateRes_PATH, genid)): 
                os.makedirs('Results/{}/Res{}'.format(args.TemplateRes_PATH, genid))
                
            if os.path.exists('Results/{}/Res{}/GenResult_{}.json'.format(args.TemplateRes_PATH, genid, profileIds[index])):
                continue
            
            for i in range(N):
                ordinal_Numeral = get_Ordinal_Numeral(i+1)
                if i == N-1:
                    historyPrompts = "The arrangement of the previous {} things is as follows: {}. ".format(i, history)
                    question = "{}What's the last arrangement today? Please output the event name and explain your thought process or reasons for your choice. Answer format:[\"event name\", \"reasons\"]".format(historyPrompts)
                elif i == 0:
                    historyPrompts = ""
                    question = "{}What's the {} arrangement today? Please output the event name and explain your thought process or reasons for your choice. Answer format:[\"event name\", \"reasons\"]".format(historyPrompts, ordinal_Numeral)
                else:
                    historyPrompts = "The arrangement of the previous {} things is as follows: {}. ".format(i, history)
                    question = "{}What's the next arrangement today? Please output the event name and explain your thought process or reasons for your choice. Answer format:[\"event name\", \"reasons\"]".format(historyPrompts)

                Q = [{"role": "user", "content": question}]
                nowQ = globalInfo + Q
                
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
                        print(response)
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
                        print(increment_minutes)
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
                 
                    gapTime = sampleGapTime()
                    tmpnowTime, cross_day = add_time(end_time, gapTime)  
                    if cross_day:
                        nowTime = end_time
                    else:
                        nowTime = tmpnowTime
                            
            print("GEN OK!!!  ProfileIndex: {}  |  personId: {}  |  genIndex: {}  |  eventNum: {}".format(index, profileIds[index], genid, N))
        
            with open('Results/{}/Res{}/GenResult_{}.json'.format(args.TemplateRes_PATH, genid, profileIds[index]),'w') as file_obj:
                json.dump(history, file_obj)


        
    