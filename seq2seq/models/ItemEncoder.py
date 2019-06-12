import torch
import torch.nn as nn

class ItemEncoder(nn.Module):
    def __init__(self, embedding_count, hidden_size):
        super(ItemEncoder, self).__init__()
        self.embedding = nn.Embedding(embedding_count, hidden_size)

        self.linear = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, ratings, items):

        rate_embed = self.embedding(ratings)
        item_embed = self.embedding(items)
        context = torch.cat((rate_embed, item_embed), 1)
        hidden = torch.tanh(self.linear(context))
        output = None
        hidden.unsqueeze_(0)
        return output, hidden