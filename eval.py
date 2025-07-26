# encoding: utf-8

import argparse
import json
import os
import pickle
import shutil
from collections import Counter
from datetime import date, datetime, time
from math import asin, cos, radians, sin, sqrt

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import setproctitle
from pycitysim.map import Map

from utils import *


def drawTwoBars(values1, values2, afterName="", path="."):
    
    values1 = values1 / np.sum(values1)
    values2 = values2 / np.sum(values2)
    
    bar_width = 0.3
    x = np.arange(len(values1))

    plt.bar(x, values1, width=bar_width, label='Bar 1', color='blue')
    plt.bar(x + bar_width, values2, width=bar_width, label='Bar 2', color='orange')

    plt.title(afterName)
    plt.xlabel('Categories (unnamed)')
    plt.ylabel('Values')

    plt.legend()
    plt.savefig(path + '/compare_{}.png'.format(afterName))



def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) 
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 
    distance=round(distance/1000,3)
    return distance


class EvalUtils(object):
    """
    some commonly-used evaluation tools and functions
    """

    @staticmethod
    def filter_zero(arr):
        """
        remove zero values from an array
        :param arr: np.array, input array
        :return: np.array, output array
        """
        arr = np.array(arr)
        filtered_arr = np.array(list(filter(lambda x: x != 0., arr)))
        return filtered_arr

    @staticmethod
    def arr_to_distribution(arr, min, max, bins):
        """
        convert an array to a probability distribution
        :param arr: np.array, input array
        :param min: float, minimum of converted value
        :param max: float, maximum of converted value
        :param bins: int, number of bins between min and max
        :return: np.array, output distribution array
        """
        distribution, base = np.histogram(
            arr, np.arange(
                min, max, float(
                    max - min) / bins))
        return distribution, base[:-1]

    @staticmethod
    def norm_arr_to_distribution(arr, bins=100):
        """
        normalize an array and convert it to distribution
        :param arr: np.array, input array
        :param bins: int, number of bins in [0, 1]
        :return: np.array, np.array
        """
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        distribution, base = np.histogram(arr, np.arange(0, 1, 1. / bins))
        return distribution, base[:-1]

    @staticmethod
    def log_arr_to_distribution(arr, min=-30., bins=100):
        """
        calculate the logarithmic value of an array and convert it to a distribution
        :param arr: np.array, input array
        :param bins: int, number of bins between min and max
        :return: np.array,
        """
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        arr = np.log(arr)
        distribution, base = np.histogram(arr, np.arange(min, 0., 1./bins))
        ret_dist, ret_base = [], []
        for i in range(bins):
            if int(distribution[i]) == 0:
                continue
            else:
                ret_dist.append(distribution[i])
                ret_base.append(base[i])
        return np.array(ret_dist), np.array(ret_base)

    @staticmethod
    def get_js_divergence(p1, p2):
        """
        calculate the Jensen-Shanon Divergence of two probability distributions
        :param p1:
        :param p2:
        :return:
        """
        # normalize
        p1 = p1 / (p1.sum()+1e-14)
        p2 = p2 / (p2.sum()+1e-14)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + \
            0.5 * scipy.stats.entropy(p2, m)
        return js
    


class IndividualEval(object):

    def __init__(self):
        print('IndividualEval init!')
        self.max_distance = 10000  
   

    def get_topk_visits(self,trajs, k):
        topk_visits_loc = []
        topk_visits_freq = []
        for traj in trajs:
            topk = Counter(traj).most_common(k)
            for i in range(len(topk), k):
                topk += [(-1, 0)]
            loc = [l for l, _ in topk]
            freq = [f for _, f in topk]
            loc = np.array(loc, dtype=int)
            freq = np.array(freq, dtype=float) / trajs.shape[1]
            topk_visits_loc.append(loc)
            topk_visits_freq.append(freq)
        topk_visits_loc = np.array(topk_visits_loc, dtype=int)
        topk_visits_freq = np.array(topk_visits_freq, dtype=float)
        return topk_visits_loc, topk_visits_freq

    
    def get_overall_topk_visits_freq(self, trajs, k):
        _, topk_visits_freq = self.get_topk_visits(trajs, k)
        mn = np.mean(topk_visits_freq, axis=0)
        return mn / np.sum(mn)


    def get_overall_topk_visits_loc_freq_arr(self, trajs, k=1):
        topk_visits_loc, _ = self.get_topk_visits(trajs, k)
        k_top = np.zeros(self.max_locs, dtype=float)
        for i in range(k):
            cur_k_visits = topk_visits_loc[:, i]
            for ckv in cur_k_visits:
                index = int(ckv)
                if index == -1:
                    continue
                k_top[index] += 1
        k_top = k_top / np.sum(k_top)
        return k_top

    
    def get_overall_topk_visits_loc_freq_dict(self, trajs, k):
        topk_visits_loc, _ = self.get_topk_visits(trajs, k)
        k_top = {}
        for i in range(k):
            cur_k_visits = topk_visits_loc[:, i]
            for ckv in cur_k_visits:
                index = int(ckv)
                if index in k_top:
                    k_top[int(ckv)] += 1
                else:
                    k_top[int(ckv)] = 1
        return k_top

    def get_overall_topk_visits_loc_freq_sorted(self, trajs, k):
        k_top = self.get_overall_topk_visits_loc_freq_dict(trajs, k)
        k_top_list = list(k_top.items())
        k_top_list.sort(reverse=True, key=lambda k: k[1])
        return np.array(k_top_list)


    def get_geodistances(self, trajs):
        distances = []
        seq_len = 48
        for traj in trajs:
            for i in range(seq_len - 1):
                lng1 = self.X[traj[i]]
                lat1 = self.Y[traj[i]]
                lng2 = self.X[traj[i + 1]]
                lat2 = self.Y[traj[i + 1]]
                distances.append(geodistance(lng1,lat1,lng2,lat2))
        distances = np.array(distances, dtype=float)
        return distances
    
    def get_distances(self, trajs):
      
        distances = []
        for traj in trajs:
            seq_len = len(traj)
            for i in range(seq_len - 1):
                dx = traj[i][2][0] - traj[i + 1][2][0]
                dy = traj[i][2][1] - traj[i + 1][2][1]
                distances.append(sqrt(dx**2 + dy**2))
        distances = np.array(distances, dtype=float)
        return distances
    
    def get_durations(self, trajs):
        d = []
        for traj in trajs:
            for item in traj:
                d.append(item[0][1] - item[0][0])
        
        return np.array(d) 
    
    def get_gradius(self, trajs):
        gradius = []
        for traj in trajs:
            seq_len = len(traj)
            xs = np.array([t[2][0] for t in traj])
            ys = np.array([t[2][1] for t in traj])
            xcenter, ycenter = np.mean(xs), np.mean(ys)
            dxs = xs - xcenter
            dys = ys - ycenter
            rad = [sqrt(dxs[i]**2 + dys[i]**2) for i in range(seq_len)]
            rad = np.mean(np.array(rad, dtype=float))
            gradius.append(rad)
        gradius = np.array(gradius, dtype=float)
        return gradius
    
    def get_periodicity(self, trajs):
        reps = []
        for traj in trajs:
            locs = [item[1] for item in traj]
            reps.append(float(len(set(locs)))/len(locs))
        reps = np.array(reps, dtype=float)
        return reps

    def get_timewise_periodicity(self, trajs):
        pass


    def get_geogradius(self, trajs):
        """
        get the std of the distances of all points away from center as `gyration radius`
        :param trajs:
        :return:
        """
        gradius = []
        for traj in trajs:
            xs = np.array([self.X[t] for t in traj])
            ys = np.array([self.Y[t] for t in traj])
            lng1, lat1 = np.mean(xs), np.mean(ys)
            rad = []
            for i in range(len(xs)):                   
                lng2 = xs[i]
                lat2 = ys[i]
                distance = geodistance(lng1,lat1,lng2,lat2)
                rad.append(distance)
            rad = np.mean(np.array(rad, dtype=float))
            gradius.append(rad)
        gradius = np.array(gradius, dtype=float)
        return gradius

    def get_individual_jsds(self, t1, t2):
        """
        get jsd scores of individual evaluation metrics
        :param t1: real_data
        :param t2: gene_data
        :return:
        """
        print('get individual jsds here!')
        d1 = self.get_distances(t1)
        d2 = self.get_distances(t2)
        d12 = np.array(list(d1)+list(d2))
        bins = int(3.5 * np.std(d12) / (len(d12) ** (1/3))) 
        d1_dist, _ = EvalUtils.arr_to_distribution(
            d1, 0, self.max_distance, bins)
        d2_dist, _ = EvalUtils.arr_to_distribution(
            d2, 0, self.max_distance, bins) 
        d_jsd = EvalUtils.get_js_divergence(d1_dist, d2_dist)
        g1 = self.get_gradius(t1)
        g2 = self.get_gradius(t2)
        g12 = np.array(list(g1)+list(g2))
        bins = int(3.5 * np.std(g12) / (len(g12) ** (1/3)))  
        g1_dist, _ = EvalUtils.arr_to_distribution(
            g1, 0, self.max_distance, bins)
        g2_dist, _ = EvalUtils.arr_to_distribution(
            g2, 0, self.max_distance, bins)
        g_jsd = EvalUtils.get_js_divergence(g1_dist, g2_dist)
        du1 = self.get_durations(t1)
        du2 = self.get_durations(t2)          
        du1_dist, _ = EvalUtils.arr_to_distribution(du1, 0, 1, 48)
        du2_dist, _ = EvalUtils.arr_to_distribution(du2, 0, 1, 48)
        du_jsd = EvalUtils.get_js_divergence(du1_dist, du2_dist)        
        p1 = self.get_periodicity(t1)
        p2 = self.get_periodicity(t2)
        p1_dist, _ = EvalUtils.arr_to_distribution(p1, 0, 1, 48)
        p2_dist, _ = EvalUtils.arr_to_distribution(p2, 0, 1, 48)
        p_jsd = EvalUtils.get_js_divergence(p1_dist, p2_dist)
        return round(d_jsd, 5), round(g_jsd, 5), round(du_jsd, 5), round(p_jsd, 5)
        
        
    def draw_jsds(self, t1, t2):
        """
        get jsd scores of individual evaluation metrics
        :param t1: real_data
        :param t2: gene_data
        :return:
        """
        current_date = date.today()
        current_time = datetime.now().time()
        
        folderName = str(current_date)+'_'+str(current_time.hour)+'-'+str(current_time.minute)
        path = 'ComparePics/'+folderName
        os.mkdir(path)
        
        d1 = self.get_distances(t1)
        d2 = self.get_distances(t2)
        d12 = np.array(list(d1)+list(d2))
        # bins = int(3.5 * np.std(d12) / (len(d12) ** (1/3))) 
        bins = 20
        d1_dist, _ = EvalUtils.arr_to_distribution(
            d1, 0, self.max_distance, bins)
        d2_dist, _ = EvalUtils.arr_to_distribution(
            d2, 0, self.max_distance, bins) 
        drawTwoBars(d1_dist, d2_dist, "distance", path)
        plt.clf()


        g1 = self.get_gradius(t1)
        g2 = self.get_gradius(t2)
        g12 = np.array(list(g1)+list(g2))
        # bins = int(3.5 * np.std(g12) / (len(g12) ** (1/3))) 
        bins = 20
        g1_dist, _ = EvalUtils.arr_to_distribution(
            g1, 0, self.max_distance, bins)
        g2_dist, _ = EvalUtils.arr_to_distribution(
            g2, 0, self.max_distance, bins)
        drawTwoBars(g1_dist, g2_dist, "gradius", path)
        plt.clf()
        
        
        du1 = self.get_durations(t1)
        du2 = self.get_durations(t2)          
        du1_dist, _ = EvalUtils.arr_to_distribution(du1, 0, 1, 48)
        du2_dist, _ = EvalUtils.arr_to_distribution(du2, 0, 1, 48)
        du_jsd = EvalUtils.get_js_divergence(du1_dist, du2_dist)
        drawTwoBars(du1_dist, du2_dist, "durations", path)
        plt.clf()
        
        
        p1 = self.get_periodicity(t1)
        p2 = self.get_periodicity(t2)
        p1_dist, _ = EvalUtils.arr_to_distribution(p1, 0, 1, 48)
        p2_dist, _ = EvalUtils.arr_to_distribution(p2, 0, 1, 48)
        p_jsd = EvalUtils.get_js_divergence(p1_dist, p2_dist)
        drawTwoBars(p1_dist, p2_dist, "periodicity", path)
        plt.clf()

        return 0


parser = argparse.ArgumentParser(description='eval')
parser.add_argument('--trajPath', default = "TrajectoryRes", type=str)
args = parser.parse_args()

if __name__ == "__main__":    
    profileindex = 0
    genIndex = 0
    map = Map(
            mongo_uri="mongodb://sim:FiblabSim1001@mgo.db.fiblab.tech:8635/",
            mongo_db="llmsim",
            mongo_coll="map_beijing5ring_withpoi_0424",
            cache_dir="Cache",
        )
    
    file = open('Data/allRes9115.pkl','rb')  
    real_data = pickle.load(file)
    real_data = processRealTraces(real_data, map)
    print(len(real_data))
    path = args.trajPath
    gen_data = readGenTraces(map, path) 
    
    print("gen num: {}".format(len(gen_data)))
    print("real num: {}".format(len(real_data)))        
    print("data process load done!")
    
    individualEval = IndividualEval()
    print(individualEval.get_individual_jsds(real_data, gen_data))


