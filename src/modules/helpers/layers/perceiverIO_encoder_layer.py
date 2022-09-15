import torch.nn as nn

from .cross_attention import CrossAttention

class PerceiverIOEncoderLayer(nn.Module):

    def __init__(self, args, emb, ff_hidden_mult=2, dropout=0.0):
        super(PerceiverIOEncoderLayer, self).__init__()

        self.args = args

        self.attention = CrossAttention(    args.emb, 
                                            args.value_out_channel, 
                                            args.emb, 
                                            args.query_out_channel, 
                                            args.latent_embedding_size, 
                                            args.query_out_channel, 
                                            args.heads  )
        
        self.norm1 = nn.LayerNorm(args.latent_embedding_size)
        self.drop1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(args.latent_embedding_size, ff_hidden_mult * args.latent_embedding_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * args.latent_embedding_size, args.latent_embedding_size)
        )

        self.norm2 = nn.LayerNorm(args.latent_embedding_size)
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