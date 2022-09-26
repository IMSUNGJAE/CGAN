import numpy as np
import glob
import os
import pandas as pd


def all_Data():
    input_path = r'C:/data/data_setting/all_data/'
    file_list = glob.glob(os.path.join(input_path, '*.csv'))
    all_data = []

    for i, file in enumerate(file_list):
        df = pd.read_csv(file)
        all_data.append(df)

    return np.array(all_data)

# def origin_data(data_path):
#     input_path = r'C:/data/data_setting/' + data_path
#     file_list = glob.glob(os.path.join(input_path, '*.csv'))
#     origin = []
#     for i, file in enumerate(file_list):
#         df = pd.read_csv(file)
#         origin.append(df)
#
#     return np.array(origin) # shape 3차

# def normalized_data(data_path):
#
#     """train_data기준 정규분포 데이터 표준화화"""
#     origin = origin_data(data_path)
#     all_data = all_Data()
#
#     norm_data = (origin - np.min(all_data)) / (np.max(all_data) - np.min(all_data))
#
#     data = np.array([])
#
#     for i in range(origin.shape[0]):
#         data = np.append(data, norm_data[i])
#         data = np.append(data, [i + 1])
#
#     return data.reshape(origin.shape[0],1,17)

def origin_data(data_path):
    input_path = r'C:/data/data_setting/'+data_path
    file_list = glob.glob(os.path.join(input_path, '*.csv'))
    origin = []
    for i, file in enumerate(file_list):
        df = pd.read_csv(file)
        # df[16] = i+1
        origin.append(df)

    return np.array(origin) # shape 3차

def normalized_data(data_path):

    """train_data기준 정규분포 데이터 표준화화"""

    origin = origin_data(data_path)
    all_data = all_Data()

    norm = (origin - np.min(all_data)) / (np.max(all_data) - np.min(all_data))

    norm_data = np.array([])

    for num, i in enumerate(norm):
        i = np.append(i, num + 1)
        norm_data = np.append(norm_data, i)

    norm_data = norm_data.reshape(len(norm),1,17)
    return norm_data

a= normalized_data('test/')
b =origin_data('test/')

# print(a)
print(b)