import torch.nn as nn
import torch

from .encoder import Encoder
from .decoder import Decoder

class TransferableTransformer(nn.Module):

    def __init__(self, args, input_dim, output_dim, dummy = False):
        super().__init__()

        self.args = args

        self.token_embedding = nn.Linear(input_dim, args.emb)
        self.query_embedding = nn.Linear(input_dim, args.emb)
        self.encoder = Encoder(args, False, 0.0)
        self.decoder = Decoder(args, False, 0.0)
        self.toprobs = nn.Linear(args.emb, output_dim)

        self.output_dim = output_dim
        self.d_tokens = None

    def forward(self, x, d, h, mask):
        tokens = self.token_embedding(x)
        tokens = torch.cat((tokens, h), 1)

        x = self.encoder(tokens, mask)

        d = self.query_embedding(d)
        d = self.decoder(d, x, mask, mask, self.args.max_agents_len, False)

        return d, tokens