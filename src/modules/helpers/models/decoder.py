import torch
import torch.nn as nn

from modules.helpers.layers.transformer_decoder_layer import DecoderLayer
from modules.helpers.embedding.positional_embedding import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, args, mask, drop_prob):
        super().__init__()
        self.args = args

        self.posit_emb = PositionalEncoding(args.emb, args.max_len, args.device)

        self.layers = nn.ModuleList([DecoderLayer(emb=args.emb,
                                                  heads=args.heads,
                                                  mask=mask,
                                                  dropout=drop_prob)
                                     for _ in range(args.depth)])


    def forward(self, d, enc_src, trg_mask, src_mask):

        trg = d

        for i in range(self.args.max_agents_len):
            p_emb = self.posit_emb(trg)
            tmp = trg[:, -1:, :] + p_emb[-1:, :]

            for layer in self.layers:
                tmp = layer(tmp[:, -1:, :], enc_src, trg_mask, src_mask)

            trg = torch.concat([trg, tmp[:, -1:, :]], 1)

        # print("Decoder output dimension: {}".format(trg.size()))
        
        return trg
