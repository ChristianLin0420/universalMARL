from lib2to3.pgen2 import token
import torch.nn as nn
import torch.nn.functional as F
import torch

from encoder import Encoder
from decoder import Decoder

class VanillaTransformer(nn.Module):

    def __init__(self, args, input_dim, output_dim):
        super().__init__()

        self.token_embedding = nn.Linear(input_dim, args.emb)
        self.encoder = Encoder(args, False, 0.0)
        self.decoder = Decoder(args, False, 0.0)
        self.toprobs = nn.Linear(args.emb, output_dim)

        self.output_dim = output_dim

    def forward(self, x, h, mask):
        tokens = self.token_embedding(x)
        tokens = torch.cat((tokens, h), 1)

        b, t, e,  = tokens.size()

        x = self.encoder(x, mask)

        # reward token/hidden token
        d_tokens = torch.rand(b, 2, e)

        x = self.decoder(d_tokens, x, False, False)

        x = self.toprobs(x.view(b * 2, e)).view(b, 2, self.output_dim)

        return x, tokens