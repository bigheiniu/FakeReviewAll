import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, hidden_size, z_size, z_repre_layer):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # normalize
        # activate function

        self.gen = nn.Sequential(
            *block(z_size, z_size),
            *block(z_size, z_size, normalize=False),
            nn.Linear(z_size, hidden_size),
            nn.Tanh()
        )
    def forward(self, z):
        hidden = self.gen(z)
        return hidden
