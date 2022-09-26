import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np


k = nn.Embedding(91,91)

a= torch.LongTensor(np.random.randint(0,91,128))


n= k(a)

print(n)

