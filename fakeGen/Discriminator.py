import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, clf_layers):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.represent_layer_list = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(clf_layers)
        ])
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def forward(self, hidden):
        # classify the simulating hidden representation and real hidden representation
        for layer in self.represent_layer_list:
            hidden = layer(hidden)
        out = self.hidden2out(hidden)
        return out

    def batchBCELoss(self, in_hidden, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.
         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        out = self.forward(in_hidden)
        return loss_fn(out, target)
