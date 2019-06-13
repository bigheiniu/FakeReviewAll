import torch.nn as nn
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

from .baseRNN import BaseRNN
import numpy as np
from .ItemEncoder import ItemEncoder
from .EncoderRNN import EncoderRNN

class NeuralEditorEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size, latent_size, context_size, input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False,
                 embedding=None, update_embedding=True):
        super(NeuralEditorEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_factor = (2 * n_layers) if bidirectional else n_layers
        self.comment_encoder = EncoderRNN(vocab_size, max_len, hidden_size, input_dropout_p, dropout_p,
                 n_layers, bidirectional, rnn_cell, variable_lengths, embedding, update_embedding)
        self.item_encoder = ItemEncoder(context_size, self.hidden_size * self.hidden_factor)
        self.rnn_cell = rnn_cell
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.gate = nn.Linear(2 * self.hidden_size * self.hidden_factor, self.hidden_factor * self.hidden_size)
        self.hidden2mean = nn.Linear(self.hidden_factor * self.hidden_size, self.latent_size)
        self.hidden2var = nn.Linear(self.hidden_factor * self.hidden_size, self.latent_size)
        self.hidden2latent = nn.Linear(self.latent_size, self.hidden_size * self.hidden_factor)

    def forward(self, input_itemId, input_rating, output_comment, comment_length=None):
        comment_output, comment_hidden = self.comment_encoder(output_comment, comment_length)
        batch_size = comment_output.shape[0]
        if self.rnn_cell.lower() == 'gru':
            comment_hidden = comment_hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        elif self.rnn_cell.lower() == 'lstm':
            comment_hidden = comment_hidden[0].view(batch_size, self.hidden_size * self.hidden_factor)
        _, item_hidden = self.item_encoder(input_rating, input_itemId)

        gate = torch.sigmoid(self.gate(torch.cat((item_hidden, comment_hidden), 1)))
        # #
        final_hidden = (1 - gate) * comment_hidden + gate * item_hidden
        z_mean = self.hidden2mean(final_hidden)
        z_var = self.hidden2latent(final_hidden)

        std = torch.exp(0.5 * z_var)

        z = torch.randn([batch_size, self.latent_size]).to(device)
        z = z * std + z_mean
        z_dis = {'var':z_var, 'mean':z_mean}
        # z = torch.randn(th.shape[0], self.latent_size).to(device)
        # z_dis = {'var':1.1, 'mean': 0.9}

        # return z, z_dis, item_hidden.unsqueeze_(0)
        return z, z_dis, item_hidden.unsqueeze(0)

    #
    # def sample_vmf(self, mu, kappa):
    #     """vMF sampler in pytorch.
    #            http://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python
    #            Args:
    #                mu (Tensor): of shape (batch_size, 2*word_dim)
    #                kappa (Float): controls dispersion. kappa of zero is no dispersion.
    #            """
    #     batch_size, id_dim = mu.size()
    #     result_list = []
    #     for i in range(batch_size):
    #         munorm = mu[i].norm().expand(id_dim)
    #         munoise = self.add_norm_noise(munorm, self.norm_eps)
    #         if float(mu[i].norm().data.cpu().numpy()) > 1e-10:
    #             # sample offset from center (on sphere) with spread kappa
    #             w = self._sample_weight(kappa, id_dim)
    #             wtorch = w * torch.ones(id_dim).to(device)
    #
    #             # sample a point v on the unit sphere that's orthogonal to mu
    #             v = self._sample_orthonormal_to(mu[i] / munorm, id_dim)
    #
    #             # compute new point
    #             scale_factr = torch.sqrt(torch.ones(id_dim).to(device) - torch.pow(wtorch, 2))
    #             orth_term = v * scale_factr
    #             muscale = mu[i] * wtorch / munorm
    #             sampled_vec = (orth_term + muscale) * munoise
    #         else:
    #             rand_draw = torch.randn(id_dim).to(device)
    #             rand_draw = rand_draw / torch.norm(rand_draw, p=2).expand(id_dim)
    #             rand_norms = (torch.rand(1) * self.norm_eps).expand(id_dim)
    #             sampled_vec = rand_draw * rand_norms.to(device)  # mu[i]
    #         result_list.append(sampled_vec)
    #
    #     return torch.stack(result_list, 0)
    #
    # def _sample_weight(self, kappa, dim):
    #     """Rejection sampling scheme for sampling distance from center on
    #     surface of the sphere.
    #     """
    #     dim = dim - 1  # since S^{n-1}
    #     b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)  # b= 1/(sqrt(4.* kdiv**2 + 1) + 2 * kdiv)
    #     x = (1. - b) / (1. + b)
    #     c = kappa * x + dim * np.log(1 - x ** 2)  # dim * (kdiv *x + np.log(1-x**2))
    #
    #     while True:
    #         z = np.random.beta(dim / 2., dim / 2.)  # concentrates towards 0.5 as d-> inf
    #         w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
    #         u = np.random.uniform(low=0, high=1)
    #         if kappa * w + dim * np.log(1. - x * w) - c >= np.log(
    #                 u):  # thresh is dim *(kdiv * (w-x) + log(1-x*w) -log(1-x**2))
    #             return w