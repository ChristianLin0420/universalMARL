from re import A
import torch.nn as nn
import torch
from modules.helpers.layers.perceiverIO_encoder_layer import PerceiverIOEncoderLayer
from modules.helpers.layers.double_perceiverIO_encoder_layer import DoublePerceiverIOEncoderLayer
from modules.helpers.layers.perceiverIO_decoder_layer import PerceiverIODecoderLayer

class DoublePerceiver(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

        # learnable initial latent vectors
        self.ally_latent = nn.Parameter(torch.empty(args.latent_length, args.encode_out))
        self.enemy_latent = nn.Parameter(torch.empty(args.latent_length, args.encode_out))
        self._init_parameters(0.02)

        # Embedding
        self.token_embedding = nn.Linear(args.token_dim, args.emb)

        # Ally Encoder
        self.ally_encoder = PerceiverIOEncoderLayer(args, 0.2)

        # Enemy Encoder
        self.enemy_encoder = PerceiverIOEncoderLayer(args, 0.2)

        # Hidden Embedding
        self.hidden_embedding = nn.Linear(args.latent_length * args.encode_out * 2, args.emb)

        # Process
        self.process = nn.ModuleList([DoublePerceiverIOEncoderLayer(args, 0.2) for _ in range(args.depth)] )

        # Decoder 
        self.decoder = PerceiverIODecoderLayer(args, args.emb, dropout = 0.2)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self.ally_latent.normal_(0.0, init_scale)
            self.enemy_latent.normal_(0.0, init_scale)
            _init_parameters(self, init_scale)
    
    def forward(self, ally, enemy, h, query):
        ally_tokens = self.token_embedding(ally)
        enemy_tokens = self.token_embedding(enemy)
        # ally_tokens = torch.cat((ally_tokens, h), 1)
        # enemy_tokens = torch.cat((enemy_tokens, h), 1)

        b = ally_tokens.size(0)
        ally_latent = torch.repeat_interleave(torch.unsqueeze(self.ally_latent, dim = 0), b, dim = 0)
        enemy_latent = torch.repeat_interleave(torch.unsqueeze(self.enemy_latent, dim = 0), b, dim = 0)

        ally = self.ally_encoder(ally_tokens, ally_latent)
        enemy = self.enemy_encoder(enemy_tokens, enemy_latent)
        # hidden = torch.cat([ally, enemy], -1)
        # hidden = self.hidden_embedding(hidden.view(-1, self.args.latent_length * self.args.encode_out * 2)).view(b, 1, self.args.emb)

        for layer in self.process:
            x = layer(enemy, ally)
        
        x = self.decoder(x, query).view(b, 1, self.args.emb)

        return x, None

def _init_parameters(module, init_scale):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=init_scale)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=init_scale)