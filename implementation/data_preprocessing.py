import numpy as np
import glob
import os
import pandas as pd
from argsparse_file import args
import itertools

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
    input_path = r'C:/data/data_setting/' + data_path
    file_list = glob.glob(os.path.join(input_path, '*.csv'))
    origin = []
    for i, file in enumerate(file_list):
        df = pd.read_csv(file)
        # df[16] = i+1
        origin.append(df)

    return np.array(origin) # shape 3차

# def normalized_data(data_path):
#
#     """train_data기준 정규분포 데이터 표준화화"""
#
#     origin = origin_data(data_path)
#     all_data = all_Data()
#
#     norm_data = (origin - np.min(all_data)) / (np.max(all_data) - np.min(all_data))
#
#     return norm_data

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

def de_normalized_data(input_data):
    all_data = all_Data()
    de_normal = input_data * (np.max(all_data) - np.min(all_data)) + np.min(all_data)
    return de_normal

def nan_data(): #원하는 결측치 개수 입력해서 총 경우의 수중에 결측치개수만큼 가지고 있는 리스트 반환

    L_vector = []
    lst = np.array(list(itertools.product([0, 1], repeat=16))).tolist()
    for i in range(len(lst)):
        lst[i].append(1)
        zero = lst[i].count(0)
        if zero == args.num:
            L_vector.append(lst[i])

    pos = [np.where(np.array(L_vector[i]) == 0)[0] for i in range(len(L_vector))]

    L_vector = np.array(L_vector)
    pos = np.array(pos)

    return L_vector, pos # shape 2차원
