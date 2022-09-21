import torch.nn as nn

from .cross_attention import CrossAttention
from .self_attention import SelfAttention

class TransfermerDecoderLayer(nn.Module):

    def __init__(self, args, emb, ff_hidden_mult=2, dropout=0.0):
        super(TransfermerDecoderLayer, self).__init__()

        self.args = args

        self.self_attention = SelfAttention(args, emb, heads = args.heads, mask = None)
        self.norm1 = nn.LayerNorm(emb)
        self.drop1 = nn.Dropout(dropout)

        self.cross_attention = CrossAttention(  emb, 
                                                args.decode_out, 
                                                emb, 
                                                args.decode_out, 
                                                emb, 
                                                args.decode_out, 
                                                args.heads  )

        self.norm2 = nn.LayerNorm(args.decode_out)
        self.drop2 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(args.decode_out, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.norm3 = nn.LayerNorm(emb)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, dec, enc):

        _x = dec
        x = self.self_attention(dec, None)
        
        x = self.norm1(x + _x)
        x = self.drop1(x)

        _x = dec
        x = self.cross_attention(enc, x)

        x = self.norm1(x + _x)
        x = self.drop1(x)

        _x = x
        x = self.ffn(x)
        
        x = self.norm2(x + _x)
        x = self.drop2(x)

        return x