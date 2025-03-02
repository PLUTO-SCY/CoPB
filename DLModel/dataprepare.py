import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import random
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import pickle
import copy
import json

# 读取已处理的数据
def load_trajectories(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# 类别编码和经纬度归一化
def initialize_encoders(user_trajectories):
    category_list = sorted(set(
        category for trajectories in user_trajectories.values() 
        for df in trajectories for category in df["General Category"].unique()
    ))

    category_encoder = LabelEncoder()
    category_encoder.fit(category_list)
    # 查看每个标签对应的类别
    category_to_label = dict(zip(category_encoder.classes_, category_encoder.transform(category_encoder.classes_)))
    numCategory = len(category_encoder.classes_)
    
    return category_encoder, numCategory

# 类别编码和经纬度归一化
def initialize_encoders_tencent(user_trajectories):

    # 汇总所有唯一的category，并去重后排序
    category_list = sorted(set(category for df in user_trajectories for category in df["General Category"].unique()))

    category_encoder = LabelEncoder()
    category_encoder.fit(category_list)
    # 查看每个标签对应的类别
    category_to_label = dict(zip(category_encoder.classes_, category_encoder.transform(category_encoder.classes_)))
    # print('category_to_label: ', category_to_label)
    numCategory = len(category_encoder.classes_)
    
    return category_encoder, numCategory 

# 数据插值处理
def interpolate_trajectory(df):
    df["Local time"] = pd.to_datetime(df["Local time"])
    df["Local time"] = df["Local time"].dt.tz_localize(None)  # 去掉时区信息
    
    date_str = df["Local time"].dt.date.iloc[0]
    full_time_range = pd.date_range(start=f"{date_str} 00:00:00", end=f"{date_str} 23:55:00", freq="10T")
    full_df = pd.DataFrame({"Local time": full_time_range})
    df = pd.merge_asof(full_df, df.sort_values("Local time"), on="Local time", direction="nearest")

    # 线性插值填充经纬度数据
    df["Latitude"] = df["Latitude"].interpolate(method="linear")
    df["Longitude"] = df["Longitude"].interpolate(method="linear")

    # 前向填充类别数据
    df["Venue category name"] = df["Venue category name"].fillna(method="ffill")
    df["General Category"] = df["General Category"].fillna(method="ffill")

    # 填充边界缺失值
    df["Latitude"] = df["Latitude"].fillna(method="bfill").fillna(method="ffill")
    df["Longitude"] = df["Longitude"].fillna(method="bfill").fillna(method="ffill")

    return df

# 处理所有用户数据
def process_trajectories(user_trajectories):
    processed_trajectories = []
    for user_id, daily_trajectories in tqdm(user_trajectories.items()):
        for df in daily_trajectories:
            df = interpolate_trajectory(df)
            processed_trajectories.append(df)
    return processed_trajectories


# Prepare Dataset
class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, category_encoder, lat_lon_scaler, repeat=1):
        self.trajectories = trajectories
        self.category_encoder = category_encoder
        self.lat_lon_scaler = lat_lon_scaler

        self.processed_lat_lon = []
        self.processed_categories = []
        print(f'repeat: {repeat}')

        # 直接加载处理好的数据，就不需要再进行处理了
        for df in self.trajectories:
            lat_lon = self.lat_lon_scaler.transform(df[["Latitude", "Longitude"]].values)
            self.processed_lat_lon.append(lat_lon)
            category_indices = self.category_encoder.transform(df["General Category"].values)
            self.processed_categories.append(category_indices)
        
        self.processed_categories_init = copy.deepcopy(self.processed_categories)

        if repeat > 1:
            self.processed_lat_lon = self.processed_lat_lon * repeat
            self.processed_categories = self.processed_categories * repeat
        combined = list(zip(self.processed_lat_lon, self.processed_categories))
        random.shuffle(combined)
        self.processed_lat_lon, self.processed_categories = zip(*combined)
        self.processed_lat_lon = list(self.processed_lat_lon)
        self.processed_categories = list(self.processed_categories)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        lat_lon = self.processed_lat_lon[idx]
        category_indices = self.processed_categories[idx]

        return torch.tensor(lat_lon, dtype=torch.float32).transpose(0, 1), torch.tensor(category_indices, dtype=torch.long)


def get_dataset(repeat = 1, batch_size=8, dataset = 'foursquare'):
    
    if dataset == 'foursquare':
        user_trajectories = load_trajectories("../Datasets/filtered_trajectories_with_newcategories.pkl")
        category_encoder, numCategory = initialize_encoders(user_trajectories)
        with open('data/processed_trajectories_dataset.pkl', "rb") as f:
            processed_trajectories = pickle.load(f)
    
    elif dataset == 'tencent':
        processed_trajectories = load_trajectories("../Datasets/tencent/tencent_cate_trajs_44_sample14.pkl")
        category_encoder, numCategory = initialize_encoders_tencent(processed_trajectories)
        repeat = 1

    elif dataset == 'mobile':
        processed_trajectories = load_trajectories("../Datasets/mobile/chinamobile_cate_trajs_144.pkl")
        category_encoder, numCategory = initialize_encoders_tencent(processed_trajectories)

    lat_lon_scaler = StandardScaler()
    lat_lon_data = np.vstack([df[["Latitude", "Longitude"]].values for df in processed_trajectories])
    lat_lon_scaler.fit(lat_lon_data)
    print('lat_lon_scaler.mean_: ', lat_lon_scaler.mean_)

    with open(f'data/{dataset}_scaler.pkl', 'wb') as f:
        pickle.dump(lat_lon_scaler, f)
     
    dataset = TrajectoryDataset(processed_trajectories, category_encoder, lat_lon_scaler, repeat)
   
    return dataset, numCategory
