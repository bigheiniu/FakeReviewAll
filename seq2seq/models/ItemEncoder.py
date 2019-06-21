import torch
import torch.nn as nn

class ItemEncoder(nn.Module):
    def __init__(self, embedding_count, hidden_size, decode_layer=1, predic_rate=False, predic_count=5):
        super(ItemEncoder, self).__init__()
        self.embedding = nn.Embedding(embedding_count, hidden_size)
        self.linear = nn.Linear(3 * hidden_size, decode_layer * hidden_size)
        self.predic_rate = predic_rate
        #the rate prediction is a regression task
        self.rate_linear = nn.Linear(2 * hidden_size, predic_count)

    def forward(self, userId, itemId, rate):
        if self.predic_rate is False:
            rate_embed = self.embedding(rate)
            item_embed = self.embedding(itemId)
            user_embed = self.embedding(userId)
            context = torch.cat((rate_embed, item_embed, user_embed), dim=1)
            hidden = torch.tanh(self.linear(context))
            output = torch.cat((rate_embed, item_embed, user_embed), 0)
            output_list = output.unsqueeze(1)
            output_list = output_list.contiguous().view(rate_embed.shape[0], 3, -1)
            #In atn, the hidden should be split into the layer count
            return output_list, hidden, None
        else:
            item_embed = self.embedding(itemId)
            user_embed = self.embedding(userId)
            context = torch.cat((item_embed, user_embed), dim=1)
            rate_pred = self.rate_linear(context)
            # add a subtask for rate prediction
            hidden_init = self.linear(context)
            return hidden_init, None, rate_pred