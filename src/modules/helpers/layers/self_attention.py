
import torch.nn as nn
import torch.nn.functional as F
import torch

class SelfAttention(nn.Module):
    def __init__(self, args, emb, heads=8, mask=False, cross=False):

        super().__init__()

        self.args = args
        self.emb = emb
        self.heads = heads
        self.mask = mask

        key_in  = emb // 2 if args.agent == "trackformer" and cross else emb
        query_in = emb 
        value_in = key_in

        self.tokeys = nn.Linear(key_in, emb * heads, bias=False)
        self.toqueries = nn.Linear(query_in, emb * heads, bias=False)
        self.tovalues = nn.Linear(value_in, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, mask, encoder_output = None):

        b, t, e = x.size()
        h = self.heads

        if self.args.agent == "trackformer":
            e = self.emb

        if encoder_output is not None:
            bq, tq, eq = encoder_output.size()
            keys = self.tokeys(x).view(b, t, h, e)
            queries = self.toqueries(encoder_output).view(b, tq, h, e)
            values = self.tovalues(x).view(b, t, h, e)
        else:
            tq = t
            keys = self.tokeys(x).view(b, t, h, e)
            queries = self.toqueries(x).view(b, tq, h, e)
            values = self.tovalues(x).view(b, t, h, e)

        # print("values: {}".format(values))
        # print("keys: {}".format(keys))
        # print("queries: {}".format(queries))

        # compute scaled dot-product self-attention
        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, tq, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # print("=" * 50)

        # print("values: {}".format(values))
        # print("keys: {}".format(keys))
        # print("queries: {}".format(queries))

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # print("dot: {}".format(dot))
        # print("-" * 50)

        assert dot.size() == (b * h, tq, t)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities
        # print("dot: {}".format(dot))

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, tq, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, tq, h * e)
        # print("out: {}".format(out))

        return self.unifyheads(out)

def mask_(matrices, maskval=0.0, mask_diagonal=True):

    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval