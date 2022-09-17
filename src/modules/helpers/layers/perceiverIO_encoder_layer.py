import torch.nn as nn

from .cross_attention import CrossAttention

class PerceiverIOEncoderLayer(nn.Module):

    def __init__(self, args, ff_hidden_mult=2, dropout=0.0):
        super(PerceiverIOEncoderLayer, self).__init__()

        self.args = args

        self.attention = CrossAttention(  args.emb, 
                                                args.encode_out, 
                                                args.emb, 
                                                args.encode_out * args.mapping_scalar, 
                                                args.process_out, 
                                                args.encode_out * args.mapping_scalar, 
                                                args.heads  )
        
        # self.norm1 = nn.LayerNorm(args.encode_out)
        # self.drop1 = nn.Dropout(dropout)

    def forward(self, dec, enc):
        # _x = enc
        # x = self.attention(dec, enc)

        # x = self.norm1(x + _x)
        # x = self.drop1(x)

        return self.attention(dec, enc)