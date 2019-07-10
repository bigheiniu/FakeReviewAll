import torch
import torch.nn as nn
import numpy as np

class FakeClassify(nn.Module):
    '''
    The discriminator in GAN to classify the deceptive reviews and real reviews.
    '''
    def __init__(self):
        super(FakeClassify, self).__init__()
        #feature extraction -> RNN based model

        #classifier -> MLP binary classifier

    def forward(self, deceptive_review, real_review):
        pass