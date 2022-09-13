import torch
import torch.nn as nn

from modules.helpers.layers.transformer_decoder_layer import DecoderLayer
from modules.helpers.embedding.positional_embedding import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, args, mask, drop_prob, dummy = False):
        super().__init__()
        self.args = args
        self.dummy = dummy

        self.posit_emb = PositionalEncoding(args.emb, args.max_len, args.device)

        self.layers = nn.ModuleList([DecoderLayer(args=args,
                                                  emb=args.emb,
                                                  heads=args.heads,
                                                  mask=mask,
                                                  dropout=drop_prob)
                                     for _ in range(1)])


    def forward(self, d, enc_src, trg_mask, src_mask, len, position_emb = True):

        if position_emb:
            p_emb = self.posit_emb(d)
            d = d + p_emb

        for layer in self.layers:
            d = layer(d, enc_src, trg_mask, src_mask)

        return d
