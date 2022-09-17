import torch.nn as nn

from .cross_attention import CrossAttention

class PerceiverIOProcessLayer(nn.Module):

    def __init__(self, args, emb, ff_hidden_mult=2, dropout=0.0):
        super(PerceiverIOProcessLayer, self).__init__()

        self.args = args

        self.attention = CrossAttention(    args.encode_out, 
                                            args.process_out, 
                                            args.encode_out, 
                                            args.process_out, 
                                            args.encode_out, 
                                            args.process_out, 
                                            args.heads  )

        self.norm1 = nn.LayerNorm(args.process_out)
        self.drop1 = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x, x)

        x = self.norm1(attended + x)
        x = self.drop1(x)

        return x