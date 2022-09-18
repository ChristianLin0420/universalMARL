import torch.nn as nn

from .cross_attention import CrossAttention

class PerceiverIODecoderLayer(nn.Module):

    def __init__(self, args, emb, ff_hidden_mult=2, dropout=0.0):
        super(PerceiverIODecoderLayer, self).__init__()

        self.args = args

        self.attention = CrossAttention(    args.process_out, 
                                            args.decode_out, 
                                            args.process_out, 
                                            args.decode_out, 
                                            emb, 
                                            args.decode_out, 
                                            args.heads  )

        self.norm1 = nn.LayerNorm(args.decode_out)
        self.drop1 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(args.decode_out, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.norm2 = nn.LayerNorm(emb)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, dec, enc):
        _x = enc
        x = self.attention(dec, enc)

        x = self.norm1(x + _x)
        x = self.drop1(x)

        _x = x
        x = self.ffn(x)
        
        x = self.norm2(x + _x)
        x = self.drop2(x)

        return x