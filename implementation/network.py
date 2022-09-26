import torch.nn as nn
import torch
import torch.nn.functional as F
from argsparse_file import args

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        """수정부분"""
        self.label_emb = nn.Embedding(args.n_classes, args.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.latent_dim + args.n_classes, 128, normalize=False),
            # *block(128, 64),
            # *block(128, 64),
            nn.Linear(128, 16),

        )

    def forward(self, z, gen_labels):
        gen_input = torch.cat((self.label_emb(gen_labels), z), -1)
        z = self.model(gen_input)
        return z


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        """수정부분"""
        self.label_embedding = nn.Embedding(args.n_classes, args.n_classes)

        self.model = nn.Sequential(

            nn.Linear(args.n_classes + args.latent_dim, 256),
            nn.LeakyReLU(0.2,inplace=True),

            # nn.Linear(256, 128),  # 기존 512
            # nn.LeakyReLU(0.2,inplace=True),

            # nn.Linear(128, 64),  # 기존 512
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x,labels):
        d_in = torch.cat((x,self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity
