from torch import nn

from modules.helpers.layers.transformer_encoder_layer import EncoderLayer
from modules.helpers.embedding.positional_embedding import PositionalEncoding


class Encoder(nn.Module):

    def __init__(self, args, mask, drop_prob):
        super().__init__()
        self.posit_emb = PositionalEncoding(args.emb, args.max_len, args.device)

        self.layers = nn.ModuleList([EncoderLayer(emb=args.emb,
                                                  heads=args.heads,
                                                  mask=mask,
                                                  drop_prob=drop_prob)
                                     for _ in range(args.depth)])

    def forward(self, x, s_mask):
        x = self.posit_emb(x)

        for layer in self.layers:
            x = layer(x, s_mask)

        print("Encoder output dimension: {}".format(x.size()))

        return x