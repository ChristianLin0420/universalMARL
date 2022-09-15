import torch.nn as nn

from .self_attention import SelfAttention

class PerceiverIOProcessLayer(nn.Module):

    def __init__(self, args, emb, ff_hidden_mult=2, dropout=0.0):
        super(PerceiverIOProcessLayer, self).__init__()

        self.args = args

        self.attention = SelfAttention(args, args.latent_embedding_size, heads=args.heads, mask=None)
        self.norm1 = nn.LayerNorm(args.latent_embedding_size)
        self.drop1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(args.latent_embedding_size, ff_hidden_mult * args.latent_embedding_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * args.latent_embedding_size, args.latent_embedding_size)
        )

        self.norm2 = nn.LayerNorm(args.latent_embedding_size)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x, None)

        x = self.norm1(attended + x)
        x = self.drop1(x)

        fedforward = self.ffn(x)

        x = self.norm2(fedforward + x)
        x = self.drop2(x)

        return x