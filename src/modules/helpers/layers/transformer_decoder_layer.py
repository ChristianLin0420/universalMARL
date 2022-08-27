import torch.nn as nn

from .self_attention import SelfAttention

class DecoderLayer(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=2, dropout=0.0):
        super(DecoderLayer, self).__init__()
        
        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.norm1 = nn.LayerNorm(emb)
        self.drop1 = nn.Dropout(dropout)

        self.enc_dec_attention = SelfAttention(emb, heads=heads, mask=mask)
        self.norm2 = nn.LayerNorm(emb)
        self.drop2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.GELU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.norm3 = nn.LayerNorm(emb)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, dec, enc, t_mask = False, s_mask = False):
        _x = dec
        x = self.attention(dec, t_mask)
        
        x = self.norm1(x + _x)
        x = self.drop1(x)

        if enc is not None:
            _x = x
            b, t, e = x.size()
            x = self.enc_dec_attention(x, s_mask, enc)
            x = self.norm2(x[:, :t, :] + _x)
            x = self.drop2(x)

        _x = x
        x = self.ffn(x)
        
        x = self.norm3(x + _x)
        x = self.drop3(x)

        return x
        