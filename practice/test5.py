import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random

n_classes = 91
latent_dim = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        """수정부분"""
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            # *block(128, 64),
            # *block(128, 64),
            nn.Linear(128, 16),
            nn.Tanh(),

        )

    def forward(self, z, gen_labels):
        gen_input = torch.cat((self.label_emb(gen_labels), z), -1)
        z = self.model(gen_input)
        return z


z = torch.FloatTensor(np.random.normal(0, 1, (1, 16)))
val_label = torch.LongTensor(np.arange(12,13)).cuda()

label = torch.randint(0,91,(1,))

model = Generator()

print(val_label)


