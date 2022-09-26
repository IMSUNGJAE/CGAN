import numpy as np
import glob
import os
import pandas as pd


def all_Data():
    input_path = r'C:/data/data_setting/all_data/'
    file_list = glob.glob(os.path.join(input_path, '*.csv'))


    return file_list

a= np.array(all_Data())

print(a)