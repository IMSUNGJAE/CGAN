import numpy as np
import pandas as pd
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt

""" csv 파일 읽어와서 열 평균 내고 전치시켜서 저장"""
# path = 'C:/data/Displacement Control/'
# # path = 'C:/data/strain Control/'
# # path = 'C:/data/Validation data/'
#
# file_list = os.listdir(path)
# file_list_py = [file for file in file_list if file.endswith('.csv')]
#
# for i , name in enumerate(file_list_py):
#     data = pd.read_csv(path + name)
#     D = data.mean()
#     Data = pd.DataFrame(D)
#     D = data.T
#     Data.to_csv('C:/data/'+ name ,index=None)
#
#
# print('finish')
#

# """최대, 최소 값 구하기 위해 가져온 코드, csv 파일 읽어와서 answer image로 변환"""
# input_path = 'C:/data/data_setting/train/'
#
# file_list = glob.glob(os.path.join(input_path, '*.csv'))
# all_data = []
#
# for i, file in enumerate(file_list):
#     df = pd.read_csv(file)
#     all_data.append(df)
#
# all_data = np.array(all_data)
# print(np.max(all_data),np.min(all_data))
#
#
#
# #path = 'C:/data/data_setting/train/'
# # path = 'C:/data/data_setting/test/'
# path = 'C:/data/data_setting/Validation/'
#
# file_list = os.listdir(path)
# file_list_py = [file for file in file_list if file.endswith('.csv')]
#
# for i , name in enumerate(file_list_py):
#     data = pd.read_csv(path + name)
#     data = np.array(data)
#
#
#     norm_data = (data - np.min(all_data)) / (np.max(all_data) - np.min(all_data))
#     # norm_data = np.reshape(norm_data,(1,16))
#     # im = Image.fromarray(norm_data)
#     # im.save('C:/data/data_setting/image/validation_image/'+ name.split('.')[0] + '.jpg','JPEG')
#     plt.axis('off')
#     plt.imshow(norm_data,cmap='gray')
#     plt.savefig(fname='C:/data/data_setting/image/validation_image/'+ name.split('.')[0] + '.jpg' , bbox_inches='tight', pad_inches=0)
#
#
# print('finish')
#
# '''============================='''

path = 'C:/data/data_setting/validation/'
# path = 'C:/data/strain Control/'
# path = 'C:/data/Validation data/'

file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.csv')]


list=[]
for i , name in enumerate(file_list_py):
    list.append(name)


list = pd.DataFrame(list)
list.to_csv('C:/data/list.csv')
"""image mse loss 적용했을 때 변환 과정 확인"""

