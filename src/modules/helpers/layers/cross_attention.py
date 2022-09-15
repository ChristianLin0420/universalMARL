
import torch.nn as nn
import torch.nn.functional as F
import torch

class CrossAttention(nn.Module):
    def __init__(   self, 
                    value_in_channel, 
                    value_out_channel, 
                    key_in_channel, 
                    key_out_channel, 
                    query_in_channel, 
                    query_out_channel, 
                    heads   ):

        super().__init__()

        self.value_in_channel = value_in_channel
        self.value_out_channel = value_out_channel
        self.key_in_channel = key_in_channel
        self.key_out_channel = key_out_channel
        self.query_in_channel = query_in_channel
        self.query_out_channel = query_out_channel
        self.heads = heads

        self.values = nn.Linear(value_in_channel, value_out_channel * heads, bias = False)
        self.keys = nn.Linear(key_in_channel, key_out_channel * heads, bias = False)
        self.querys = nn.Linear(query_in_channel, query_out_channel * heads, bias = False)
        self.unifyheads = nn.Linear(value_out_channel * heads, query_out_channel)

    def forward(self, x, y):

        x_b, x_t, x_e = x.size()
        y_b, y_t, y_e = y.size()
        
        # compute scaled dot-product self-attention
        # - fold heads into the batch dimension
        values = self.values(x).view(x_b, x_t, self.heads, self.value_out_channel)
        keys = self.keys(x).view(x_b, x_t, self.heads, self.key_out_channel)
        queries = self.querys(y).view(y_b, y_t, self.heads, self.query_out_channel)

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(x_b * self.heads, x_t, self.key_out_channel)
        queries = queries.transpose(1, 2).contiguous().view(y_b * self.heads, y_t, self.query_out_channel)
        values = values.transpose(1, 2).contiguous().view(x_b * self.heads, x_t, self.value_out_channel)

        queries = queries / (y_e ** (1 / 4))
        keys = keys / (x_e ** (1 / 4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (y_b * self.heads, y_t, x_t)

        dot = F.softmax(dot, dim = 2)

        # - dot now has row-wise self-attention probabilities
        # apply the self attention to the values
        out = torch.bmm(dot, values).view(y_b, self.heads, y_t, self.value_out_channel)
        out = out.transpose(1, 2).contiguous().view(y_b, y_t, self.heads * self.value_out_channel)

        return self.unifyheads(out)