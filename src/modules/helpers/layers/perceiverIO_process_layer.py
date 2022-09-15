import torch.nn as nn

from .self_attention import SelfAttention

class PerceiverIOProcessLayer(nn.Module):

    def __init__(self, args, emb, ff_hidden_mult=2, dropout=0.0):
        super(PerceiverIOProcessLayer, self).__init__()

        self.args = args

        self.attention = SelfAttention(args, emb, heads=args.heads, mask=None)
        self.norm1 = nn.LayerNorm(emb)
        self.drop1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.norm2 = nn.LayerNorm(emb)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x, None)

        x = self.norm1(attended + x)
        x = self.drop1(x)

        fedforward = self.ffn(x)

        x = self.norm2(fedforward + x)
        x = self.drop2(x)

        return x