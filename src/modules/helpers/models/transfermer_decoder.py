import torch
import torch.nn as nn

from modules.helpers.layers.transfermer_decoder_layer import TransfermerDecoderLayer
from modules.helpers.embedding.positional_embedding import PositionalEncoding

class TransfermerDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.posit_emb = PositionalEncoding(args.emb, args.max_len, args.device)

        self.layers = nn.ModuleList([TransfermerDecoderLayer(args=args, emb=args.emb) for _ in range(args.depth)])

    def forward(self, d, enc_src, position_emb = True):

        if position_emb:
            p_emb = self.posit_emb(d)
            d = d + p_emb

        for layer in self.layers:
            d = layer(d, enc_src)

        return d
