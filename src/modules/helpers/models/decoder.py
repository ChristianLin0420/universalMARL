import torch
from torch import nn

from modules.helpers.layers.transformer_decoder_layer import DecoderLayer
from modules.helpers.embedding.positional_embedding import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, args, mask, drop_prob):
        super().__init__()
        self.posit_emb = PositionalEncoding(args.emb, args.max_len, args.device)

        self.layers = nn.ModuleList([DecoderLayer(emb=args.emb,
                                                  heads=args.heads,
                                                  mask=mask,
                                                  drop_prob=drop_prob)
                                     for _ in range(args.depth)])


    def forward(self, trg, enc_src, trg_mask, src_mask):

        p_emb = self.posit_emb(trg)
        trg = trg + p_emb

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        trg = torch.cat((trg, trg), 1)

        print("Decoder output dimension: {}".format(trg.size()))
        
        return trg
