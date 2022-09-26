import numpy as np
from torch.utils.data import Dataset

class custom_dataset(Dataset):

    def __init__(self,data,L_vector,L_index):

        self.data = data # shape = [데이터 개수 ,1,17]
        self.L_vector = L_vector # shape = [l_vector 개수, 17]
        self.L_index = L_index # shape = [총 l_vector 데이터 개수 중에 결측치가 num개인 데이터 개수 ,num]

    def __len__(self):
        return self.data.shape[0] * self.L_vector.shape[0]

    def __getitem__(self, index):

        nan_data = self.data * self.L_vector # shape = [데이터개수, l_vector 개수, 17]
        nan_data = nan_data.reshape((self.data.shape[0] * self.L_vector.shape[0]),17)# shape = [데이터개수*l_vector 개수, 17]
        nan_value = nan_data[index][:16].reshape(1, 16)  # 결측치 포함 데이터
        nan_data_index = self.L_index[index % self.L_vector.shape[0]]

        label = int(nan_data[index][-1])
        ground_truth = self.data[label - 1][:, :16]

        return ground_truth, nan_value, nan_data_index, label-1 # 2차원

