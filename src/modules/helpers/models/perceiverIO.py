from re import A
import torch.nn as nn
import torch
from einops import repeat
from modules.helpers.layers.cross_attention import CrossAttention

class PerceiverIO(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.latent_length = args.latent_length
        self.latent_embedding_size = args.latent_embedding_size

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.rand(args.latent_length, args.latent_embedding_size))
        # self._init_parameters(0.02)

        # Embedding
        self.token_embedding = nn.Linear(args.token_dim, args.emb)

        # Encoder
        self.encode_cross_attention = CrossAttention(  
                                                        args.emb, 
                                                        args.value_out_channel, 
                                                        args.emb, 
                                                        args.query_out_channel, 
                                                        args.latent_embedding_size, 
                                                        args.query_out_channel, 
                                                        args.heads)

        # Process
        self.process = nn.ModuleList([CrossAttention(
                                                        args.query_out_channel, 
                                                        args.query_out_channel, 
                                                        args.query_out_channel, 
                                                        args.query_out_channel, 
                                                        args.query_out_channel, 
                                                        args.query_out_channel, 
                                                        args.heads)
                                        for _ in range(args.depth)])

        # Decoder 
        self.decode_cross_attention = CrossAttention(
                                                        args.query_out_channel,
                                                        args.value_out_channel, 
                                                        args.query_out_channel, 
                                                        args.key_out_channel, 
                                                        args.emb,
                                                        args.key_out_channel,
                                                        args.heads)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self.latent.normal_(0.0, init_scale)
            _init_parameters(self, init_scale)
    
    def forward(self, x, h, query):
        tokens = self.token_embedding(x)
        tokens = torch.cat((tokens, h), 1)

        b, _, _,  = tokens.size()

        latent = torch.repeat_interleave(torch.unsqueeze(self.latent, dim = 0), b, dim = 0) #repeat(self.latent, "... -> b ...", b = b)
        x = self.encode_cross_attention(tokens, latent)
        print("-" * 50)
        print("x: {}".format(x))
        print("1" * 50)

        for layer in self.process:
            print("x: {}".format(x))
            x = layer(x, x)
        print("2" * 50)
        query = self.token_embedding(query)
        query = torch.cat((query, h), 1)
        x = self.decode_cross_attention(x, query).view(b, self.args.action_space_size + self.args.enemy_num + 1, self.args.key_out_channel)

        return x

def _init_parameters(module, init_scale):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=init_scale)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=init_scale)