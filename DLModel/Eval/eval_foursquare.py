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


def printMetrics(metrics):  # 仅仅加入了对小数位数的控制
    d_jsd,  g_jsd,  du_jsd,  p_jsd,  e_jsd,  f_jsd, z_jsd = metrics
    print('%.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f' % (d_jsd,  g_jsd,  du_jsd,  p_jsd,  e_jsd,  f_jsd, z_jsd))  
    
def printMetrics2(metrics):  # 仅仅加入了对小数位数的控制
    d_jsd,  g_jsd,  du_jsd,  p_jsd,  e_jsd,  f_jsd, z_jsd = metrics
    print('%.4f  %.4f  %.4f  %.4f  %.4f  %.4f' % (d_jsd, g_jsd, p_jsd, e_jsd, f_jsd, z_jsd))  

def drawTwoBars(values1, values2, afterName="", path="."):
    
    # print(values1.shape)
    # print(values2.shape)
    values1 = values1 / np.sum(values1)
    values2 = values2 / np.sum(values2)
    
    bar_width = 0.3
    x = np.arange(len(values1))

    # 创建柱状图
    plt.bar(x, values1, width=bar_width, label='Bar 1', color='blue')
    plt.bar(x + bar_width, values2, width=bar_width, label='Bar 2', color='orange')

    # 添加标题和标签
    plt.title(afterName)
    plt.xlabel('Categories (unnamed)')
    plt.ylabel('Values')

    # 添加图例
    plt.legend()
    plt.savefig(path + '/compare_{}.png'.format(afterName))



def geodistance(lng1,lat1,lng2,lat2):  # long是精度,lat是纬度
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
    
    # 使用 numpy 的 clip 函数来限制 i 和 j 的范围
    i = np.clip(i, 0, 49)
    j = np.clip(j, 0, 49)
    
    id = (i * 50 + j).astype(int)  # 分成50乘50的格
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
        self.max_distance = 30_000  # 作为分布的上限，还是需要设置的.  10KM理应作为上限,可以统一
   

    # def get_topk_visits(self,trajs, k):
    #     topk_visits_loc = []
    #     topk_visits_freq = []
    #     for traj in trajs:
    #         topk = Counter(traj).most_common(k)
    #         for i in range(len(topk), k):
    #             # supplement with (loc=-1, freq=0)
    #             topk += [(-1, 0)]
    #         loc = [l for l, _ in topk]
    #         freq = [f for _, f in topk]
    #         loc = np.array(loc, dtype=int)
    #         freq = np.array(freq, dtype=float) / 144
    #         topk_visits_loc.append(loc)
    #         topk_visits_freq.append(freq)
    #     topk_visits_loc = np.array(topk_visits_loc, dtype=int)
    #     topk_visits_freq = np.array(topk_visits_freq, dtype=float)
    #     return topk_visits_loc, topk_visits_freq

    def get_topk_visits(self, trajs, k):
        topk_visits_loc = []
        topk_visits_freq = []
        
        for traj in trajs:
            # 获取频率最高的 k 个位置
            topk = Counter(traj).most_common(k)
            
            # 补充不足 k 个位置的部分
            topk += [(-1, 0)] * (k - len(topk))  # 更简洁的补充方式
            
            # 提取位置和频率
            loc = np.array([l for l, _ in topk], dtype=int)
            freq = np.array([f for _, f in topk], dtype=float) / 144
            
            topk_visits_loc.append(loc)
            topk_visits_freq.append(freq)
        
        # 转换成 NumPy 数组
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
        """
        已改,适配
        """
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
            # 计算相邻点是否相同
            diff = np.diff(traj, axis=0)
            # 找到变化点的索引
            change_indices = np.where(~np.all(diff == 0, axis=1))[0]
            # 计算每个连续段的长度
            segment_lengths = np.diff(np.concatenate(([-1], change_indices, [len(traj) - 1])))
            durations.extend(segment_lengths)
        return np.array(durations) / 144
    
    
    def get_gradius(self, trajs):
        """
        已改,已适配
        TODO:这里涉及到一个是否需要对时间进行加权的问题        
        """
        gradius = []
        for traj in trajs:
            seq_len = len(traj)
            xs = np.array([t[0] for t in traj])
            ys = np.array([t[1] for t in traj])
            xcenter, ycenter = np.mean(xs), np.mean(ys)
            dxs = xs - xcenter
            dys = ys - ycenter
            rad = [(dxs[i]**2 + dys[i]**2) for i in range(seq_len)]  # 先进行距离的平方
            rad = np.mean(np.array(rad, dtype=float))
            gradius.append(sqrt(rad))  # 最后再开根号
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
        """
        计算地点转换的频率，即重复性，越大代表地点转换越多。
        计算每个轨迹中地点的变化次数，并归一化到[0, 1]范围。
        """
        reps = []
        for traj in trajs:
            # 计算相邻地点是否相同
            diff = np.diff(traj, axis=0)  # 计算相邻位置之间的差异
            # 找到变化点的索引（即地点发生变化的地方）
            change_indices = np.where(~np.all(diff == 0, axis=1))[0]
            # 计算每个轨迹中变化的次数
            events = len(change_indices)
            reps.append(events / 144)  # 归一化到 [0, 1]，假设每条轨迹有 144 个位置
        
        return np.array(reps, dtype=float)

    
    
    def get_dailyLocNum(self, trajs):
        """
        已改,已适配
        这个值越大,说明重复性越低,现阶段的重复性可能主要体现在家和工作地上
        """
        from collections import Counter

        def calculate_frequencies(my_list):
            total_items = len(my_list)
            frequency_counter = Counter(my_list)

            # 计算频率值
            frequencies = {value: count / total_items for value, count in frequency_counter.items()}
            
            return frequencies
        
        reps = []
        for traj in trajs:
            locs = [item[1] for item in traj]
            # reps.append(len(set(locs)))
            reps.append(len(locs))
        # reps = np.array(reps, dtype=float)
        
        result = calculate_frequencies(reps)

        v = []
        f = []
        # 打印结果
        for value, frequency in result.items():
            v.append(value)
            f.append(frequency)
            # print(f"{value}: {frequency:.2}")
        print(v)
        print(f)
            
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
    
    
    # def get_gridsFreq(self, trajs):

    #     ids = []
    #     for traj in trajs:
    #         for point in traj:
    #             ids.append(point)                
     
    #     counter = Counter(ids)  # 统计
        
    #     gridTimesDict = {}
    #     for element, frequency in counter.items():
    #         gridTimesDict[str(element)] = frequency
        
    #     distribution = []
    #     for i in range(2500):
    #         if str(i) in gridTimesDict.keys():
    #             distribution.append(gridTimesDict[str(i)])
    #         else:
    #             distribution.append(0)
                
    #     distribution = np.array(distribution)
    #     distribution = distribution / np.sum(distribution)
                
    #     return distribution

    def get_gridsFreq(self, trajs):
        # 将所有点展开为一个列表
        ids = [point for traj in trajs for point in traj]
        
        # 统计各个点的频率
        counter = Counter(ids)
        
        # 创建一个大小为2500的零数组，直接将频率映射到对应位置
        distribution = np.zeros(2500)
        
        for element, frequency in counter.items():
            distribution[int(element)] = frequency
        
        # 归一化
        distribution = distribution / np.sum(distribution)
        
        return distribution

    def get_individual_jsds(self, t1, t2):

        d1 = self.get_distances(t1)  
        d2 = self.get_distances(t2)
        bins = 50
        d1_dist, d1_bjs = EvalUtils.arr_to_distribution(
            d1, 0, self.max_distance, bins)
        d2_dist, _ = EvalUtils.arr_to_distribution(
            d2, 0, self.max_distance, bins)  # 这里分多少桶也需要再斟酌一下
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
        du_jsd = EvalUtils.get_js_divergence(du1_dist, du2_dist)  # 这个可能有点问题啊
        print('get_durations done!')
        
        p1 = self.get_periodicity(t1)  # 现在这个指标不太理想,事件数量少了以后,就很难有重复的地点出现
        p2 = self.get_periodicity(t2)
        p1_dist, _ = EvalUtils.arr_to_distribution(p1, 0, 1.1, 144)  # 为什么设置成1.1就是因为会有很多1出现,感觉是生成模板的时候对事件数量进行采样的问题,不应该给1件事情那么高的采样频率
        p2_dist, _ = EvalUtils.arr_to_distribution(p2, 0, 1.1, 144)
        p_jsd = EvalUtils.get_js_divergence(p1_dist, p2_dist)
        # 这好像也写的没啥问题
        print('get_periodicity done!')

        e1 = self.get_dailyEvents(t1)
        e2 = self.get_dailyEvents(t2)
        e1_dist, _ = EvalUtils.arr_to_distribution(e1, 0, 1.1, 144)  # 为什么设置成1.1就是因为会有很多1出现,感觉是生成模板的时候对事件数量进行采样的问题,不应该给1件事情那么高的采样频率
        e2_dist, _ = EvalUtils.arr_to_distribution(e2, 0, 1.1, 144)
        e_jsd = EvalUtils.get_js_divergence(e1_dist, e2_dist)
        # 这两个都需要格点坐标
        # 所有人top100的地点的访问频率,的分布
        # 需要进行时间的加权求和嘛？其实我觉得不需要
        # 一共是2500个网格
        print('get_dailyEvents done!')

        t1 = np.apply_along_axis(getId, axis=2, arr=t1)
        t2 = np.apply_along_axis(getId, axis=2, arr=t2)

        # print(type(t2))

        # data_flat = t2.flatten()
        # # 使用 Counter 统计每个数字的出现次数
        # counter = Counter(data_flat)
        # # 获取出现次数最多的前 10 个数字
        # most_common_10 = counter.most_common(10)
        # print("出现次数最多的前 10 个数字及其次数：")
        # for number, count in most_common_10:
        #     print(f"数字: {number}, 出现次数: {count}")
        # sys.exit(0)

        f1 = self.get_overall_topk_visits_freq(t1, 100)
        f2 = self.get_overall_topk_visits_freq(t2, 100)
        f1_dist, _ = EvalUtils.arr_to_distribution(f1, 0, 1, 100)  # 原来是500桶
        f2_dist, _ = EvalUtils.arr_to_distribution(f2, 0, 1, 100)
        f_jsd = EvalUtils.get_js_divergence(f1_dist, f2_dist)  # 这个就是top100的freq的指标
        print('get_topk_visits done!')
        
        z1 = self.get_gridsFreq(t1)  # 原来是get_overall_topk_visits_freq
        z2 = self.get_gridsFreq(t2)
        z_jsd = EvalUtils.get_js_divergence(z1, z2)
        print('get_gridsFreq done!')

       

        
        print(d_jsd,  g_jsd,  du_jsd,  p_jsd,  e_jsd,  f_jsd, z_jsd)

        return d_jsd,  g_jsd,  du_jsd,  p_jsd,  e_jsd,  f_jsd, z_jsd
        # l_jsd, f_jsd
        
  

# 东京的参考经纬度（单位：度）
lat0 = 35.6895  # 参考纬度
lon0 = 139.6917  # 参考经度

# 地球半径（单位：米）
R = 6371000  # 地球平均半径

# 将经纬度从度转换为弧度
def deg_to_rad(deg):
    return deg * np.pi / 180

# 将经纬度转换为平面坐标（单位：米）
def latlon_to_xy(lat, lon):
    # 将经纬度从度转换为弧度
    lat_rad = deg_to_rad(lat)
    lon_rad = deg_to_rad(lon)
    lat0_rad = deg_to_rad(lat0)
    lon0_rad = deg_to_rad(lon0)
    
    # 计算xy坐标
    x = R * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    y = R * (lat_rad - lat0_rad)
    return x, y


# 将整个数组的经纬度转换为xy坐标
def latlon_array_to_xy_vectorized(data):
    lat0_rad = deg_to_rad(lat0)
    lon0_rad = deg_to_rad(lon0)
    
    # 将经纬度从度转换为弧度
    lats_rad = deg_to_rad(data[:, :, 1])
    lons_rad = deg_to_rad(data[:, :, 0])
    
    # 计算xy坐标
    x = R * (lons_rad - lon0_rad) * np.cos(lat0_rad)
    y = R * (lats_rad - lat0_rad)
    
    # 组合成新的数组
    xy_data = np.stack([x, y], axis=-1)
    return xy_data



def process_sequences(seq1, seq2):
    # 确保 seq2 是一个 NumPy 数组，便于处理
    seq2 = np.array(seq2)
    
    for i in range(1, len(seq1)):
        # 如果 seq1[i] == seq1[i-1]，则 seq2[i] 更新为 seq2[i-1]
        if seq1[i] == seq1[i-1]:
            seq2[i] = seq2[i-1]
    
    return seq2


if __name__ == "__main__":    

    expIndex = 100
    sampleIndex = 5
    gen_data = readGenTraces(expIndex, sampleIndex)

    # print('numpy_array.shape: ', gen_data.shape)   # (7920, 144, 2)
    # # 衡量数据分布是否符合要求
    print(np.mean(gen_data)) 
    print(np.mean(gen_data[:,:,1]))  
  
    # 加载 scaler0.
    with open('../data/foursquare_scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)

    gen_data = gen_data.reshape(-1, 2)  # 或者 arr.reshape(7920*144, 2)

    gen_data = loaded_scaler.inverse_transform(gen_data)

    gen_data = gen_data.reshape(-1, 144, 2)  # (7920, 144, 2)
    print('gen_data.shape: ', gen_data.shape)  # gen_data.shape:  （14829， 144）

    path = '/data2/shaochenyang/scywork/Mobility_Revision/DLModel/data/foursquare_condition.pt'
    tensor = torch.load(path)
    conditions = tensor.cpu().numpy()
    print('conditions.shape: ', conditions.shape)
    # xy转换
    gen_data = latlon_array_to_xy_vectorized(gen_data)
    # print(gen_data.shape)
    # print(gen_data[80])
    # sys.exit(0)


    # 对生成的轨迹做修正
    for i in range(gen_data.shape[0]):
        gen_data[i] = process_sequences(conditions[i], gen_data[i])

    # print(conditions[5000])
    # print(gen_data[5000])
    # sys.exit(0)
    

    # # 输出结果
    # print("原始数据（经纬度）：")
    # print(gen_data[0, 0])  # 打印第一个batch的第一个点的经纬度
    # print("转换后的数据（xy坐标，单位：米）：")
    # print(xy_data[0, 0])  # 打印第一个batch的第一个点的xy坐标
    # print(xy_data[1, 1]) 
    # sys.exit(0)

    # with open('../data/processed_trajectories_dataset.pkl', "rb") as f:
    #     processed_trajectories = pickle.load(f)
    # longlats = []
    # for df in processed_trajectories:
    #     lat_lon = df[["Latitude", "Longitude"]].values
    #     longlats.append(lat_lon)
    # longlats = np.array(longlats)
    # np.save('../data/realTrajs.npy', longlats)
    # sys.exit(0)

    real_data = np.load('../data/foursquare_realTrajs.npy')

    real_data = latlon_array_to_xy_vectorized(real_data)
    print('real trajectories.shape: ', real_data.shape)

    # print(np.max(real_data[:,:,0]))
    # print(np.min(real_data[:,:,0]))
    # print(np.max(real_data[:,:,1]))
    # print(np.min(real_data[:,:,1]))
    # sys.exit(0)
    
    individualEval = IndividualEval()
    printMetrics2(individualEval.get_individual_jsds(gen_data, real_data))
    # radius\Dailyloc\IntentDist\G-rank\LocFreq


