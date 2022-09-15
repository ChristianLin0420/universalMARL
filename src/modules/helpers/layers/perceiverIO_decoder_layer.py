import torch.nn as nn

from .cross_attention import CrossAttention

class PerceiverIODecoderLayer(nn.Module):

    def __init__(self, args, emb, ff_hidden_mult=2, dropout=0.0):
        super(PerceiverIODecoderLayer, self).__init__()

        self.args = args

        self.attention = CrossAttention(    args.emb, 
                                            args.value_out_channel, 
                                            args.emb, 
                                            args.query_out_channel, 
                                            args.latent_embedding_size, 
                                            args.query_out_channel, 
                                            args.heads  )
        
        self.ffn = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.norm1 = nn.LayerNorm(emb)
        self.drop1 = nn.Dropout(dropout)

    def forward(self, dec, enc):
        _x = dec
        x = self.attention(dec, enc)

        _x = x
        x = self.ffn(x)
        
        x = self.norm1(x + _x)
        x = self.drop1(x)

        return x