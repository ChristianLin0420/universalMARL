import torch.nn as nn
import torch

from .encoder import Encoder
from .transfermer_decoder import TransfermerDecoder

class TransferableTransformerPlus(nn.Module):

    def __init__(self, args, input_dim, output_dim):
        super().__init__()

        self.args = args

        self.token_embedding = nn.Linear(input_dim, args.emb)
        self.encoder = Encoder(args, False, 0.0)
        self.decoder = TransfermerDecoder(args)

        self.output_dim = output_dim
        self.d_tokens = None

    def forward(self, x, d, h, mask):
        tokens = self.token_embedding(x)
        x = self.encoder(tokens, mask)

        tokens = self.token_embedding(d)
        tokens = torch.cat((tokens, h), 1)
        d = self.decoder(tokens, x)

        return d