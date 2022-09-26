import numpy as np
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools

def all_Data():
    input_path = r'C:/data/data_setting/all_data/'
    file_list = glob.glob(os.path.join(input_path, '*.csv'))
    all_data = []

    for i, file in enumerate(file_list):
        df = pd.read_csv(file)
        all_data.append(df)

    return np.array(all_data)

def origin_data(data_path):
    input_path = r'C:/data/data_setting/' + data_path
    file_list = glob.glob(os.path.join(input_path, '*.csv'))
    origin = []
    for i, file in enumerate(file_list):
        df = pd.read_csv(file)
        origin.append(df)

    return np.array(origin) # shape 3차

def normalized_data(data_path):

    """train_data기준 정규분포 데이터 표준화화"""
    origin = origin_data(data_path)
    all_data = all_Data()

    norm_data = (origin - np.min(all_data)) / (np.max(all_data) - np.min(all_data))

    data = np.array([])

    for i in range(origin.shape[0]):
        data = np.append(data, norm_data[i])
        data = np.append(data, [i + 1])

    return data.reshape(origin.shape[0],1,17)


def nan_data(): #원하는 결측치 개수 입력해서 총 경우의 수중에 결측치개수만큼 가지고 있는 리스트 반환

    L_vector = []
    lst = np.array(list(itertools.product([0, 1], repeat=16))).tolist()
    for i in range(len(lst)):
        lst[i].append(1)
        zero = lst[i].count(0)
        if zero == 5:
            L_vector.append(lst[i])

    pos = [np.where(np.array(L_vector[i]) == 0)[0] for i in range(len(L_vector))]

    L_vector = np.array(L_vector)
    pos = np.array(pos)

    return L_vector, pos # shape 2차원





a = all_Data()
n = a[:,:,:16].reshape(-1,16)
x = np.array(np.arange(0,16))
print(n.shape)

plt.rc('font', family='Times New Roman')
plt.figure()
plt.title("data_distribution")
# plt.plot(n,
#         color='red',
#          marker="o",
#         linewidth=1, label='original_data')

for i, array in enumerate(n):
    plt.plot(x, array, color = np.random.rand(3, ), marker = "o", label = f"Array #{i}")
# plt.scatter(k,n1,
#         color='blue',
#         linestyle='--',
#         linewidth=1, label='original_data')
#
# plt.scatter(k,n2,
#         color='black',
#         linestyle='--',
#         linewidth=1, label='original_data')
#
# plt.scatter(k,n3,
#         color='green',
#         linestyle='--',
#         linewidth=1, label='original_data')


plt.legend()
plt.grid()
plt.xlabel('')
plt.ylabel('')
plt.xticks(np.arange(1,17))
plt.show()