from ast import arg
from lib2to3.pgen2 import token
from numpy import size
import torch.nn as nn
import torch.nn.functional as F
import torch

from .encoder import Encoder
from .decoder import Decoder

from modules.helpers.embedding.random_layer import RandomLayer

class Transformer(nn.Module):

    def __init__(self, args, input_dim, output_dim, dummy = False):
        super().__init__()

        self.args = args
        self.dummy = dummy

        self.token_embedding = nn.Linear(input_dim, args.emb)
        self.encoder = Encoder(args, False, 0.0)
        self.decoder = Decoder(args, False, 0.0)
        self.toprobs = nn.Linear(args.emb, output_dim)

        self.output_dim = output_dim
        self.d_tokens = None

    def forward(self, x, h, mask):
        tokens = self.token_embedding(x)
        tokens = torch.cat((tokens, h), 1)

        b, t, e,  = tokens.size()

        x = self.encoder(tokens, mask)

        # reward token/hidden token
        latent_size = 1
        final_size = self.args.max_agents_len + 1

        if self.d_tokens is None:
            self.d_tokens = RandomLayer(self.args).get_random_vector(512, 50, 64)

        d = self.d_tokens[:b, :latent_size, :e]

        if self.dummy:
            x = self.decoder(d, x, mask, mask, self.args.max_agents_len)
        else:
            x = self.decoder(d, x, mask, mask, t)
            final_size = t + 1

        x = self.toprobs(x.view(b * final_size, e)).view(b, final_size, self.output_dim)

        return x, tokens