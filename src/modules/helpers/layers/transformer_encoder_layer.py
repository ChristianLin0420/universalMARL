from torch.nn import nn

from self_attention import SelfAttention

class EncoderLayer(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super(EncoderLayer, self).__init__()
        
        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.norm1 = nn.LayerNorm(emb)
        self.drop1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.norm2 = nn.LayerNorm(emb)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, _x, mask):
        x = _x
        attended = self.attention(x, mask)

        x = self.norm1(attended + x)
        x = self.drop1(x)

        fedforward = self.ffn(x)

        x = self.norm2(fedforward + x)
        x = self.drop2(x)

        return x, mask