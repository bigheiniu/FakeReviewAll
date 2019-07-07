import torch
import torch.nn as nn

from .baseRNN import BaseRNN

class C2S(nn.Module):
    def __init__(self):
        super(C2S, self).__init__()

