
import torch.nn as nn
import torch

from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):

    def __init__(self, args, input_dim, output_dim, dummy = False):
        super().__init__()

        self.args = args
        self.dummy = dummy

        self.token_embedding = nn.Linear(input_dim, args.emb)
        self.query_embedding = nn.Linear(input_dim, args.emb)
        self.encoder = Encoder(args, False, 0.0)
        self.decoder = Decoder(args, False, 0.0, self.dummy)
        self.toprobs = nn.Linear(args.emb, output_dim)

        self.output_dim = output_dim
        self.d_tokens = None

    def forward(self, x, h, mask, query = None):
        tokens = self.token_embedding(x)
        tokens = torch.cat((tokens, h), 1)

        b, t, e,  = tokens.size()

        x = self.encoder(tokens, mask)

        # reward token/hidden token
        final_size = self.args.max_agents_len + 1

        if self.dummy:
            x = self.decoder(tokens, x, mask, mask, self.args.max_agents_len)
        else:
            x = self.decoder(tokens, x, mask, mask, t)
            final_size = t

        x = self.toprobs(x.view(b * final_size, e)).view(b, final_size, self.output_dim)

        return x, tokens