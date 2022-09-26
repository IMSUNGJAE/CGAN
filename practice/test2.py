import numpy as np
import glob
import os
import pandas as pd
import itertools

def nan_data(): #원하는 결측치 개수 입력해서 총 경우의 수중에 결측치개수만큼 가지고 있는 리스트 반환

    L_vector = []
    lst = np.array(list(itertools.product([0, 1], repeat=16))).tolist()
    for i in range(len(lst)):
        lst[i].append(1)
        zero = lst[i].count(0)
        if zero == 2:
            L_vector.append(lst[i])

    pos = [np.where(np.array(L_vector[i]) == 0)[0] for i in range(len(L_vector))]

    L_vector = np.array(L_vector)
    pos = np.array(pos)

    return L_vector, pos # shape 2차원

a,b = nan_data()

g = np.array([])

for i in range(a.shape[0]):
    g = np.append(g, np.delete(a[i][:16],b[i]))

print(g.reshape(120,-1).shape)


