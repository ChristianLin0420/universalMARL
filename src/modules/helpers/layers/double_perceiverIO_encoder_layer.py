import torch.nn as nn

from .cross_attention import CrossAttention

class DoublePerceiverIOEncoderLayer(nn.Module):

    def __init__(self, args, dropout=0.0):
        super(DoublePerceiverIOEncoderLayer, self).__init__()

        self.args = args

        self.attention = CrossAttention(    args.encode_out, 
                                            args.encode_out, 
                                            args.encode_out, 
                                            args.encode_out * args.mapping_scalar, 
                                            args.encode_out, 
                                            args.encode_out * args.mapping_scalar, 
                                            args.heads  )
        
        self.norm1 = nn.LayerNorm(args.encode_out)
        self.drop1 = nn.Dropout(dropout)

    def forward(self, dec, enc):
        _x = enc
        x = self.attention(dec, enc)

        x = self.norm1(x + _x)
        x = self.drop1(x)

        return x