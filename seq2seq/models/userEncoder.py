import torch.nn as nn
import torch

class UserEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, input_dropout_p=0, mlp_layer=2, n_layers=1, bidirectional=False):
        super(UserEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_factor = (2 * n_layers) if bidirectional else n_layers
        self.input_embed = nn.Embedding(input_size, hidden_size)

        self.mlp_list = nn.ModuleList([
           nn.Linear(3 * hidden_size, 3 * hidden_size) for _ in range(mlp_layer)
        ])
        self.mlp_list.append(nn.Linear(3 * hidden_size, hidden_size))

        self.user2mean = nn.Linear(self.hidden_size, self.hidden_size)
        self.user2var = nn.Linear(self.hidden_size, self.hidden_size)
        self.latent2decode_init = nn.Linear(self.hidden_size, self.hidden_size * self.hidden_factor)

    def forward(self, input_itemId, input_rating, input_userId,output_comment, comment_length=None):
        device = torch.device('gpu') if input_userId.is_cuda else torch.device('cpu')
        user_embed = self.input_embed(input_userId)
        item_embed = self.input_embed(input_itemId)
        rate_embed = self.input_embed(input_rating)
        user_mean = self.hidden2mean(user_embed)
        user_var = self.hidden2latent(user_embed)
        std = torch.exp(0.5 * user_var)

        batch_size = user_embed.shape[0]
        feature_size = user_embed.shape[-1]
        user_resample = torch.randn([batch_size, feature_size]).to(device)
        user_resample = user_resample * std + user_mean

        latent = torch.cat([user_resample, item_embed, rate_embed], dim=-1)
        for layer in self.mlp_list:
            latent = layer(latent)
        decode_init = self.latent2decode_init(latent)
        user_dis = {'var':user_var, 'mean':user_mean}
        return decode_init, user_dis


