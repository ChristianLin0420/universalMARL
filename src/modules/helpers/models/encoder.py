import torch.nn as nn

from modules.helpers.layers.transformer_encoder_layer import EncoderLayer
from modules.helpers.embedding.positional_embedding import PositionalEncoding


class Encoder(nn.Module):

    def __init__(self, args, mask, drop_prob):
        super().__init__()
        self.posit_emb = PositionalEncoding(args.emb, args.max_len, args.device)

        self.layers = nn.ModuleList([EncoderLayer(args=args,
                                                  emb=args.emb,
                                                  heads=args.heads,
                                                  mask=mask,
                                                  dropout=drop_prob)
                                     for _ in range(args.depth)])

    def forward(self, x, s_mask):
        p_emb = self.posit_emb(x)
        _x = x + p_emb
        
        for layer in self.layers:
            _x = layer(_x, s_mask)

        # print("Encoder output dimension: {}".format(_x.size()))

        return _x