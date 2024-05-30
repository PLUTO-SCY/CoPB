import argparse
import json
import pickle
import sys
from random import choice

import openai

from modules import *
from setBasicData import *
from utils import *

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

parser = argparse.ArgumentParser(description='gen DayIntention just Once')
parser.add_argument('--genid', default = 1, type=int) 
parser.add_argument('--Template_PATH', default = 'TemplateRes', type=str)  # where the generated templates are
parser.add_argument('--Trajectory_PATH', default = 'TrajectoryRes', type=str)  
args = parser.parse_args()

startEnd = [(0,4), (4,14), (14,35), (35,70), (70,140), (140,250)]

(startIndex, endIndex) = startEnd[args.genid-1]


if __name__ == '__main__':

    file = open('Data/profileWordDict.pkl','rb') 
    profileDict = pickle.load(file)
    file = open('Data/hws.pkl','rb') 
    HWs = pickle.load(file)
    
    with open('Data/data.pkl', 'rb') as f:
        cate1, cate2, cate3, cate12, cate23, cate3id = pickle.load(f) 
    database = DataBase(cate1, cate2, cate3, cate12, cate23, cate3id)  
    print('DataBase load done!')
    
    filepath = 'Results/{}'.format(args.Trajectory_PATH)
    if not os.path.isdir(filepath):
        os.mkdir(filepath)

    for profileindex in range(startIndex, endIndex): 
        print('begin')
        personBasicInfo, num = profileDict[str(profileindex)]
        f = open('Results/{}/Res0/GenResult_{}.json'.format(args.Template_PATH, profileindex), 'r')
        content = f.read()
        plan = json.loads(content)
            
        num = num // 25
        if num < 1: 
            num = 1
            
        for genIndex in range(num):
            (home, work) = choice(HWs) 
            Homeplace = (home[0], home[1])
            Workplace = (work[0], work[1])
            nowPlace = Homeplace
            
            hisTraj = []
            
            numIntentions = len(plan)
            for i in range(numIntentions):       
                nextIntention = plan[i][0]
                nextIntention = nextIntention.lower()
                
                if nextIntention == 'go to work': 
                    nextPlace = Workplace
                    
                elif nextIntention == 'go home':
                    nextPlace = Homeplace
                    
                elif nextIntention == 'go to sleep':
                    nextPlace = Homeplace  
                
                elif nextIntention in ['eat', 'have breakfast', 'have lunch', 'have dinner', 'do shopping', 'do sports', 'excursion', 'leisure or entertainment', 'medical treatment', 'handle the trivialities of life', 'banking and financial services', 'government and political services', 'cultural institutions and events']: 
                    eventId = getDirectEventID(nextIntention)
                  
                    POIs = database.event2poi_gravity(eventId, nowPlace)                   
                    
                    options = list(range(len(POIs)))
                    probabilities = [item[2] for item in POIs]
                    sample = np.random.choice(options, size=1, p=probabilities) 
                    nextPlace = POIs[sample[0]] 
                    
                else: 
                    eventId = ""
                    print('error {}'.format(nextIntention))
                    POIs = database.event2poi_gravity(eventId, nowPlace) 
                    options = list(range(len(POIs)))
                    probabilities = [item[2] for item in POIs]
                    sample = np.random.choice(options, size=1, p=probabilities) 
                    nextPlace = POIs[sample[0]] 
                    
                thisThing = [plan[i][0], plan[i][1], nextPlace]
                hisTraj.append(thisThing)
                nowPlace = nextPlace

            with open('Results/{}/trace_{}_{}.json'.format(args.Trajectory_PATH, profileindex, genIndex),'w',encoding='utf-8') as file_obj:  
                json.dump(hisTraj, file_obj, ensure_ascii=False)

            print("OK profile_Index: {}, gen_Index: {}".format(profileindex, genIndex))
        
            
            