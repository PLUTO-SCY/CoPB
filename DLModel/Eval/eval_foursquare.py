# encoding: utf-8

import json
import os
import pickle
import random
import shutil
from collections import Counter
from datetime import date, datetime, time
from math import asin, ceil, cos, radians, sin, sqrt
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from pycitysim.map import Map
from utils import *


def printMetrics(metrics): 
    d_jsd,  g_jsd,  du_jsd,  p_jsd,  e_jsd,  f_jsd, z_jsd = metrics
    print('%.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f' % (d_jsd,  g_jsd,  du_jsd,  p_jsd,  e_jsd,  f_jsd, z_jsd))  
    
def printMetrics2(metrics): 
    d_jsd,  g_jsd,  du_jsd,  p_jsd,  e_jsd,  f_jsd, z_jsd = metrics
    print('%.4f  %.4f  %.4f  %.4f  %.4f  %.4f' % (d_jsd, g_jsd, p_jsd, e_jsd, f_jsd, z_jsd))  

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


def getId(xy):
    y_min, x_min = 11539961, -9408732
    y_max, x_max = 11588801, -9376545
    y_interval = (y_max - y_min) / 50
    x_interval = (x_max - x_min) / 50
    x, y = xy
    i = (x - x_min) // x_interval
    j = (y - y_min) // y_interval
    i = np.clip(i, 0, 49)
    j = np.clip(j, 0, 49)
    id = (i * 50 + j).astype(int)
    return id


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
        self.max_distance = 30_000  

    def get_topk_visits(self, trajs, k):
        topk_visits_loc = []
        topk_visits_freq = []
        
        for traj in trajs:
            topk = Counter(traj).most_common(k)
            
            topk += [(-1, 0)] * (k - len(topk)) 
            loc = np.array([l for l, _ in topk], dtype=int)
            freq = np.array([f for _, f in topk], dtype=float) / 144
            
            topk_visits_loc.append(loc)
            topk_visits_freq.append(freq)

        return np.array(topk_visits_loc, dtype=int), np.array(topk_visits_freq, dtype=float)
    
    
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
        seq_len = 144
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
                dx = traj[i][0] - traj[i + 1][0]
                dy = traj[i][1] - traj[i + 1][1]
                distances.append(sqrt(dx**2 + dy**2))
        distances = np.array(distances, dtype=float)
        return distances

    def get_durations(self, trajs):
        durations = []
        for traj in trajs:
            diff = np.diff(traj, axis=0)
            change_indices = np.where(~np.all(diff == 0, axis=1))[0]
            segment_lengths = np.diff(np.concatenate(([-1], change_indices, [len(traj) - 1])))
            durations.extend(segment_lengths)
        return np.array(durations) / 144
    
    
    def get_gradius(self, trajs):
        gradius = []
        for traj in trajs:
            seq_len = len(traj)
            xs = np.array([t[0] for t in traj])
            ys = np.array([t[1] for t in traj])
            xcenter, ycenter = np.mean(xs), np.mean(ys)
            dxs = xs - xcenter
            dys = ys - ycenter
            rad = [(dxs[i]**2 + dys[i]**2) for i in range(seq_len)]
            rad = np.mean(np.array(rad, dtype=float))
            gradius.append(sqrt(rad))
        gradius = np.array(gradius, dtype=float)
        return gradius
    
    def get_periodicity(self, trajs):
        """
        stat how many repetitions within a single trajectory
        :param trajs:
        :return:
        """
        reps = []
        for traj in trajs:
            tuple_traj = [(item[0], item[1]) for item in traj]
            num = len(set(tuple_traj))
            reps.append(float(num) / 144)
        reps = np.array(reps, dtype=float)
        return reps
    

    def get_dailyEvents(self, trajs):
        reps = []
        for traj in trajs:
            diff = np.diff(traj, axis=0) 
            change_indices = np.where(~np.all(diff == 0, axis=1))[0]
            events = len(change_indices)
            reps.append(events / 144) 
        
        return np.array(reps, dtype=float)

    
    
    def get_dailyLocNum(self, trajs):
        from collections import Counter

        def calculate_frequencies(my_list):
            total_items = len(my_list)
            frequency_counter = Counter(my_list)
            frequencies = {value: count / total_items for value, count in frequency_counter.items()}
            return frequencies
        
        reps = []
        for traj in trajs:
            locs = [item[1] for item in traj]
            reps.append(len(locs))
        result = calculate_frequencies(reps)

        v = []
        f = []
        for value, frequency in result.items():
            v.append(value)
            f.append(frequency)
        return reps

    def get_timewise_periodicity(self, trajs):
        """
        stat how many repetitions of different times
        :param trajs:
        :return:
        """
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
    
    

    def get_gridsFreq(self, trajs):
        ids = [point for traj in trajs for point in traj]
        counter = Counter(ids)
        distribution = np.zeros(2500)
        for element, frequency in counter.items():
            distribution[int(element)] = frequency
        distribution = distribution / np.sum(distribution)
        return distribution

    def get_individual_jsds(self, t1, t2):

        d1 = self.get_distances(t1)  
        d2 = self.get_distances(t2)
        bins = 50
        d1_dist, d1_bjs = EvalUtils.arr_to_distribution(
            d1, 0, self.max_distance, bins)
        d2_dist, _ = EvalUtils.arr_to_distribution(
            d2, 0, self.max_distance, bins)
        d_jsd = EvalUtils.get_js_divergence(d1_dist, d2_dist)
        print('get_distances done!')

        g1 = self.get_gradius(t1)
        g2 = self.get_gradius(t2)
        g12 = np.array(list(g1)+list(g2))
        bins = 50
        g1_dist, _ = EvalUtils.arr_to_distribution(
            g1, 0, self.max_distance, bins)
        g2_dist, _ = EvalUtils.arr_to_distribution(
            g2, 0, self.max_distance, bins)
        g_jsd = EvalUtils.get_js_divergence(g1_dist, g2_dist)
        print('get_gradius done!')
        
        
        du1 = self.get_durations(t1)
        du2 = self.get_durations(t2)          
        du1_dist, _ = EvalUtils.arr_to_distribution(du1, 0, 1, 144)
        du2_dist, _ = EvalUtils.arr_to_distribution(du2, 0, 1, 144)
        du_jsd = EvalUtils.get_js_divergence(du1_dist, du2_dist) 
        
        p1 = self.get_periodicity(t1) 
        p2 = self.get_periodicity(t2)
        p1_dist, _ = EvalUtils.arr_to_distribution(p1, 0, 1.1, 144) 
        p2_dist, _ = EvalUtils.arr_to_distribution(p2, 0, 1.1, 144)
        p_jsd = EvalUtils.get_js_divergence(p1_dist, p2_dist)

        e1 = self.get_dailyEvents(t1)
        e2 = self.get_dailyEvents(t2)
        e1_dist, _ = EvalUtils.arr_to_distribution(e1, 0, 1.1, 144)
        e2_dist, _ = EvalUtils.arr_to_distribution(e2, 0, 1.1, 144)
        e_jsd = EvalUtils.get_js_divergence(e1_dist, e2_dist)

        t1 = np.apply_along_axis(getId, axis=2, arr=t1)
        t2 = np.apply_along_axis(getId, axis=2, arr=t2)

        f1 = self.get_overall_topk_visits_freq(t1, 100)
        f2 = self.get_overall_topk_visits_freq(t2, 100)
        f1_dist, _ = EvalUtils.arr_to_distribution(f1, 0, 1, 100)
        f2_dist, _ = EvalUtils.arr_to_distribution(f2, 0, 1, 100)
        f_jsd = EvalUtils.get_js_divergence(f1_dist, f2_dist)  

        z1 = self.get_gridsFreq(t1) 
        z2 = self.get_gridsFreq(t2)
        z_jsd = EvalUtils.get_js_divergence(z1, z2)

        print(d_jsd,  g_jsd,  du_jsd,  p_jsd,  e_jsd,  f_jsd, z_jsd)

        return d_jsd,  g_jsd,  du_jsd,  p_jsd,  e_jsd,  f_jsd, z_jsd
        
  

lat0 = 35.6895 
lon0 = 139.6917 
R = 6371000  

def deg_to_rad(deg):
    return deg * np.pi / 180

def latlon_to_xy(lat, lon):
    lat_rad = deg_to_rad(lat)
    lon_rad = deg_to_rad(lon)
    lat0_rad = deg_to_rad(lat0)
    lon0_rad = deg_to_rad(lon0)
    x = R * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    y = R * (lat_rad - lat0_rad)
    return x, y

def latlon_array_to_xy_vectorized(data):
    lat0_rad = deg_to_rad(lat0)
    lon0_rad = deg_to_rad(lon0)
    lats_rad = deg_to_rad(data[:, :, 1])
    lons_rad = deg_to_rad(data[:, :, 0])
    x = R * (lons_rad - lon0_rad) * np.cos(lat0_rad)
    y = R * (lats_rad - lat0_rad)
    xy_data = np.stack([x, y], axis=-1)
    return xy_data


def process_sequences(seq1, seq2):
    seq2 = np.array(seq2)
    for i in range(1, len(seq1)):
        if seq1[i] == seq1[i-1]:
            seq2[i] = seq2[i-1]
    return seq2


if __name__ == "__main__":    

    expIndex = 100
    sampleIndex = 5
    gen_data = readGenTraces(expIndex, sampleIndex)

    print(np.mean(gen_data)) 
    print(np.mean(gen_data[:,:,1]))  

    with open('../data/foursquare_scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)

    gen_data = gen_data.reshape(-1, 2)

    gen_data = loaded_scaler.inverse_transform(gen_data)

    gen_data = gen_data.reshape(-1, 144, 2) 
    print('gen_data.shape: ', gen_data.shape)  

    path = '/data2/shaochenyang/scywork/Mobility_Revision/DLModel/data/foursquare_condition.pt'
    tensor = torch.load(path)
    conditions = tensor.cpu().numpy()
    print('conditions.shape: ', conditions.shape)

    gen_data = latlon_array_to_xy_vectorized(gen_data)

    for i in range(gen_data.shape[0]):
        gen_data[i] = process_sequences(conditions[i], gen_data[i])
    real_data = np.load('../data/foursquare_realTrajs.npy')

    real_data = latlon_array_to_xy_vectorized(real_data)
    print('real trajectories.shape: ', real_data.shape)

    individualEval = IndividualEval()
    printMetrics2(individualEval.get_individual_jsds(gen_data, real_data))


