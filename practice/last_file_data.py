import pandas as pd
import numpy as np
import os

path = 'C:/data/data_setting/lastdata/'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.csv')]

for i , name in enumerate(file_list_py):
    data = pd.read_csv(path + name)
    last_data = np.array(data)[1:,1:]
    Data = pd.DataFrame(last_data[-1])
    Data.T.to_csv('C:/data/data_setting/last_data/'+ name +'.csv',  index=None)

print('finish')


