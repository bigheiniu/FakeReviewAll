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
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden):
        # classify the simulating hidden representation and real hidden representation
        for layer in self.represent_layer_list:
            hidden = layer(hidden)
        out = self.hidden2out(hidden)
        logit = self.sigmoid(out)
        return logit

    def batchBCELoss(self, in_hidden, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.
         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        out = self.forward(in_hidden)
        return loss_fn(out, target), {'y_pre': torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out)),
                                      'y_true': target}

class RNNclaissfier(nn.Module):
    def __init__(self, encoderRNN, discriminator):
        super(RNNclaissfier, self).__init__()
        self.encoder = encoderRNN
        self.clf = discriminator

    def forward(self, onehot_seq, onehot_label, con_seq=None,
                con_label=None):
        _, hidden = self.encoder(onehot_seq, is_onehot=True)
        onehot_state = hidden[0].view(hidden.shape[0], -1)
        if con_seq is not None:
            _, hidden = self.encoder(con_seq, is_onehot=False)
            con_state = hidden[0].view(hidden.shape[0], -1)
            all_state = torch.cat((onehot_state, con_state), dim=0)
            all_label = torch.cat((onehot_label, con_label), dim=0)
        else:
            all_state = onehot_state
            all_label = onehot_label

        #shuffle data
        shuffle_index = torch.randperm(all_state.shape[0])
        all_state = all_state[shuffle_index]
        all_label = all_label[shuffle_index]
        loss, out = self.clf.batchBCELoss(all_state, all_label)
        return loss, out




