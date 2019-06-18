import torch
import torch.nn as nn

class ItemEncoder(nn.Module):
    def __init__(self, embedding_count, hidden_size, lstm_layer=1):
        super(ItemEncoder, self).__init__()
        self.embedding = nn.Embedding(embedding_count, hidden_size)
        self.linear = nn.Linear(3 * hidden_size, lstm_layer * hidden_size)

    def forward(self, userId, itemId, rate):
        rate_embed = self.embedding(rate)
        item_embed = self.embedding(itemId)
        user_embed = self.embedding(userId)
        context = torch.cat((rate_embed, item_embed, user_embed), dim=1)
        hidden = torch.tanh(self.linear(context))
        output = torch.cat((rate_embed, item_embed, user_embed), 0)
        output_list = output.unsqueeze(1)
        output_list = output_list.contiguous().view(rate_embed.shape[0], 3, -1)
        return output_list, hidden