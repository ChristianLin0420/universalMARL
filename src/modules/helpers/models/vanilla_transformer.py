from ast import arg
from lib2to3.pgen2 import token
import torch.nn as nn
import torch.nn.functional as F
import torch

from .encoder import Encoder
from .decoder import Decoder

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

    def forward(self, x, h, mask):
        tokens = self.token_embedding(x)
        tokens = torch.cat((tokens, h), 1)

        b, t, e,  = tokens.size()

        x = self.encoder(tokens, mask)

        # reward token/hidden token
        latent_size = 1
        final_size = self.args.max_agents_len + 1
        # if self.dummy:
        #     latent_size = self.args.max_agents_len + 1
        # else:
        #     latent_size = 2

        if self.args.use_cuda:
            d_tokens = torch.rand(b, latent_size, e).cuda()
        else:
            d_tokens = torch.rand(b, latent_size, e)

        x = self.decoder(d_tokens, x, mask, mask)

        x = self.toprobs(x.view(b * final_size, e)).view(b, final_size, self.output_dim)

        return x, tokens