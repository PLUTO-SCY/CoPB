import asyncio
import math
import os
import pickle
import random
import sys
import time
from itertools import accumulate
from operator import itemgetter
from typing import cast

import numpy as np
# from pycitysim.map import Map
# from pycitysim.routing import RoutingClient
# from pycitysim.utils import wrap_feature_collection
from pywolong.map import Map
from pywolong.routing import RoutingClient
from pywolong.utils import wrap_feature_collection
from tqdm import tqdm

from utils import *

XMIN = 435300
XMAX = 457300
YMIN = 4405900
YMAX = 4427900
D = 250 
N = int((XMAX-XMIN)/D)

def Grid_ID2ij(id):
    i = id // N  
    j = id - i*N
    return (i,j)

def mesureDis(h, w):
    ih, jh = Grid_ID2ij(int(h))
    iw, jw = Grid_ID2ij(int(w))
    dis = math.sqrt((ih-iw)**2+(jh-jw)**2)*D
    return dis

def calTuple(PPS, alpha, beta):
    (Pi, Pj, Sij)  = PPS
    if Pj == 0:
        return 0  
    else:
        Pi = Pi if Pi>0 else 1
        Sij = Sij if Sij>0 else 1
        value = (Pi+alpha*Sij)*Pj / ((Pi+(alpha+beta)*Sij)*(Pi+Pj+(alpha+beta)*Sij))
    return value     
        
class DataBase:  
    def __init__(self, cate1, cate2, cate3, cate12, cate23, cate3id):
        self.cate1 = cate1
        self.cate2 = cate2
        self.cate3 = cate3
        self.cate12 = cate12
        self.cate23 = cate23
        self.cate3id = cate3id
        
        self.map = Map(
            mongo_uri="mongodb://sim:FiblabSim1001@mgo.db.fiblab.tech:8635/",
            mongo_db="llmsim",
            mongo_coll="map_beijing5ring_withpoi_0424",
            cache_dir="Cache",
        )
    
    
    def event2cate(self, event):
        if event in ['have breakfast', 'have lunch', 'have dinner', 'eat']:
            return self.cate12['美食']
        elif event == 'do shopping':
            return str(self.cate12['购物'])
        elif event == 'do sports':
            return self.cate12['运动健身']
        elif event == 'excursion':  
            return self.cate12['旅游景点']
        elif event == 'leisure or entertainment':
            return self.cate12['娱乐休闲']
        elif event == 'medical treatment':
            return self.cate12['医疗保健']
        elif event == 'handle the trivialities of life':
            return self.cate12['生活服务']
        elif event == 'banking and financial services':
            return self.cate12['银行金融']
        elif event == 'Government and Political Services':
            return self.cate12['机构团体']
        elif event == 'Cultural Institutions and Events':
            return self.cate12['文化场馆']
        else:
            print('\nIn function event2cate: The selected choice is not in the range!\n')
            sys.exit(0)
            

    def cate2label(self, cate):
        try:
            return self.cate23[cate]
        except:
            print('\nIn function cate2label: The selected choice is not in the range!\n')
            sys.exit(0)
        
    
    def event2label(self, event):
        if event == 'shopping':
            return str(self.cate12['购物'])
        elif event == 'do sports':
            return self.cate12['运动健身']
        else:
            print('\nIn function event2label: The selected event is not in the range!\n')
            sys.exit(0)
            
    def label2poi(self, label, nowPlace):
        nowPlace = self.map.get_poi(nowPlace[1])
        labelQueryId = str(self.cate3id[label])
        pois10k = self.map.query_pois(
                center = (nowPlace['position']['x'], nowPlace['position']['y']), 
                radius = 10000, 
                category_prefix= labelQueryId, 
                limit = 500
            )  
        
        if len(pois10k) < 1:  
            pois10k = self.map.query_pois(
                center = (nowPlace['position']['x'], nowPlace['position']['y']), 
                radius = 10000,
                category_prefix= labelQueryId[:-1], 
                limit = 500
            )  
            if len(pois10k) < 1:
                pois10k = self.map.query_pois(
                    center = (nowPlace['position']['x'], nowPlace['position']['y']), 
                    radius = 10000,
                    category_prefix= labelQueryId[:-2], 
                    limit = 500
                )  
                if len(pois10k) < 1:
                    pois10k = self.map.query_pois(
                        center = (nowPlace['position']['x'], nowPlace['position']['y']), 
                        radius = 10000,
                        category_prefix= labelQueryId[:-3], 
                        limit = 500
                    )  
                    if len(pois10k) < 1:
                        pois10k = self.map.query_pois(
                            center = (nowPlace['position']['x'], nowPlace['position']['y']), 
                            radius = 10000,
                            category_prefix= labelQueryId[:-4], 
                            limit = 500
                        )  

        pois_1k = []
        pois_2k = []
        pois_6k = []
        pois_6km = []  
        for poi in pois10k:
            if poi[1]<1000:
                pois_1k.append(poi)
            elif poi[1]<2000:
                pois_2k.append(poi)
            elif poi[1]<6000:
                pois_6k.append(poi)
            else:
                pois_6km.append(poi)
        
        poiForChoose = []
        if len(pois_1k)>10:
            poiForChoose = poiForChoose + random.sample(pois_1k, 10)  
        else:
            poiForChoose = poiForChoose + pois_1k
            
        if len(pois_2k)>3:
            poiForChoose = poiForChoose + random.sample(pois_2k, 3)
        else:
            poiForChoose = poiForChoose + pois_2k
        
        if len(pois_6k)>2:
            poiForChoose = poiForChoose + random.sample(pois_6k, 2)  
        else:
            poiForChoose = poiForChoose + pois_6k
        
        if len(pois_6km)>4:
            poiForChoose = poiForChoose + random.sample(pois_6km, 4)  
        else:
            poiForChoose = poiForChoose + pois_6km
        
    
        res = []
        for poi in poiForChoose:
            res.append((poi[0]['name'], poi[0]['id'], poi[1]))  
            
        return res

    def event2poi_gravity(self, label, nowPlace):
        
        nowPlace = self.map.get_poi(nowPlace[1]) 
        labelQueryId = label
        
        pois10k = self.map.query_pois(
                center = (nowPlace['position']['x'], nowPlace['position']['y']), 
                radius = 10000,  
                category_prefix= labelQueryId, 
                limit = 10000  
            ) 
        if pois10k[-1][1] < 5000:
            pois10k = self.map.query_pois(
                center = (nowPlace['position']['x'], nowPlace['position']['y']), 
                radius = 10000,
                category_prefix= labelQueryId, 
                limit = 20000 
            ) 
            if pois10k[-1][1] < 5000:
                    pois10k = self.map.query_pois(
                    center = (nowPlace['position']['x'], nowPlace['position']['y']), 
                    radius = 10000,  
                    category_prefix= labelQueryId, 
                    limit = 30000  
                ) 
        
        N = len(pois10k)
        pois_Dis = {"1k":[], "2k":[], "3k":[], "4k":[], "5k":[], "6k":[], "7k":[], "8k":[], "9k":[], "10k":[], "more":[]}
        for poi in pois10k:
            iflt10k = True
            for d in range(1,11):
                if (d-1)*1000 <= poi[1] < d*1000:
                    pois_Dis["{}k".format(d)].append(poi)
                    iflt10k = False
                    break
            if iflt10k:
                pois_Dis["more"].append(poi)
    
        res = []
        distanceProb = []
        for poi in pois10k:
            iflt10k = True
            for d in range(1,11):
                if (d-1)*1000 <= poi[1] < d*1000:
                    n = len(pois_Dis["{}k".format(d)])
                    S = math.pi*((d*1000)**2 - ((d-1)*1000)**2)
                    density = n/S
                    distance = poi[1]
                    distance = distance if distance>1 else 1
                    
                  
                    weight = density / (distance**2)                 
                    res.append((poi[0]['name'], poi[0]['id'], weight, distance))
                    
                  
                    distanceProb.append(1/(math.sqrt(distance)))  
                    
                    iflt10k = False
                    break
        distanceProb = np.array(distanceProb)
        distanceProb = distanceProb/np.sum(distanceProb)
        distanceProb = list(distanceProb)
        
        options = list(range(len(res)))
        sample = list(np.random.choice(options, size=50, p=distanceProb)) 
        
        get_elements = itemgetter(*sample)
        random_elements = get_elements(res)
        weightSum = sum(item[2] for item in random_elements)
        final = [(item[0], item[1], item[2]/weightSum, item[3]) for item in random_elements]
        return final
    
    
    def samplePOI_Oppo_Work(self, nowloc, cate, params, N=700000):
        (alpha, beta) = params
        pois11km = self.map.query_pois(
                center = nowloc, 
                radius = 11000, 
                category_prefix= cate, 
                limit = N,  
            )  
        
        radiuses = list(range(11))
        radiuses = [item*1000 for item in radiuses]
        poiList = {}
        for i in range(11): 
            poiList['{}-{}'.format(i,i+1)] = []
        
        for poi in pois11km:
            for i in range(11):
                if i*1000<poi[1]<(i+1)*1000:
                    poiList['{}-{}'.format(i,i+1)].append((poi[0]['name'], poi[0]['id'], (poi[0]['position']['x'], poi[0]['position']['y'])))
            
        ringList = []
        for i in range(11): 
            ringList.append(len(poiList['{}-{}'.format(i,i+1)]))
        
        Pi = len(poiList['0-1'])
        Pj = ringList
        Sij = list(accumulate(ringList))
    
        Pi = Pi if Pi>0 else 1
        Pj = [pj if pj>0 else 1 for pj in Pj]
        Sij = [sij if sij>0 else 1 for sij in Sij]
        weightList = []

        for i in range(11): 
            value = (Pi+alpha*Sij[i])*Pj[i] / ((Pi+(alpha+beta)*Sij[i])*(Pi+Pj[i]+(alpha+beta)*Sij[i]))
            weightList.append(value)
        
        summ = sum(weightList)
        weightList = [item/summ for item in weightList]
        indexx = list(range(11))
        OK = False
        while not OK:
            try:
                sample = np.random.choice(indexx, size=1, p=weightList) 
                choiceList = poiList['{}-{}'.format(sample[0], sample[0]+1)]
                choosePlace = random.choice(choiceList) 
                OK = True
            except:
                pass
        return choosePlace  
    
    def getRoadTime(self, nowP, tP):
        place1 = self.map.get_poi(nowP[1])
        place2 = self.map.get_poi(tP[1])

        xy1 = (place1['position']['x'], place1['position']['y'])
        xy2 = (place2['position']['x'], place2['position']['y'])
        dis = np.linalg.norm(np.array(xy1) - np.array(xy2))
        return timeEval(dis)

    async def queryPoiTime(self, aoi_id1, aoi_id2):
        client = RoutingClient("localhost:52101")
        req = {
            "type": 1,
            "start": {"aoi_position": {"aoi_id": aoi_id1}},
            "end": {"aoi_position": {"aoi_id": aoi_id2}},
        }  

        res = await client.GetRoute(req)
        res = cast(dict, res)
        feature = self.map.export_route_as_geojson(req, res, {"PathKey1": "something"})
        eta = self.map.estimate_route_time(req, res)
        return int(eta/60)+5  
    
    
    def poiNameId2AoiId(self, poiNameId):
        poi = self.map.get_poi(poiNameId[1])
        return poi['aoi_id']


class DataBaseSimple: 
    def __init__(self, cate1, cate2, cate3, cate12, cate23, cate3id):
        self.cate1 = cate1
        self.cate2 = cate2
        self.cate3 = cate3
        self.cate12 = cate12
        self.cate23 = cate23
        self.cate3id = cate3id
    
    
    def event2cate(self, event):
        if event in ['have breakfast', 'have lunch', 'have dinner']:
            return self.cate12['美食']
        elif event == 'do shopping':
            return str(self.cate12['购物'])
        elif event == 'do sports':
            return self.cate12['运动健身']
        elif event == 'excursion':
            return self.cate12['旅游景点']
        elif event == 'leisure or entertainment':
            return self.cate12['娱乐休闲']
        elif event == 'medical treatment':
            return self.cate12['医疗保健']
        elif event == 'handle the trivialities of life':
            return self.cate12['生活服务']
        elif event == 'banking and financial services':
            return self.cate12['银行金融']
        elif event == 'Government and Political Services':
            return self.cate12['机构团体']
        elif event == 'Cultural Institutions and Events':
            return self.cate12['文化场馆']
        else:
            print('\nIn function event2cate: The selected choice is not in the range!\n')
            sys.exit(0)
            

    def cate2label(self, cate):
        try:
            return self.cate23[cate]
        except:
            print('\nIn function cate2label: The selected choice is not in the range!\n')
            sys.exit(0)
        
    
    