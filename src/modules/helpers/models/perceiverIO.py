from re import A
import torch.nn as nn
import torch
from modules.helpers.layers.perceiverIO_encoder_layer import PerceiverIOEncoderLayer
from modules.helpers.layers.perceiverIO_process_layer import PerceiverIOProcessLayer
from modules.helpers.layers.perceiverIO_decoder_layer import PerceiverIODecoderLayer

class PerceiverIO(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(args.latent_length, args.encode_out))
        self._init_parameters(0.02)

        # Embedding
        self.token_embedding = nn.Linear(args.token_dim, args.emb)

        # Encoder
        self.encoder = PerceiverIOEncoderLayer(args)
        self.hidden_embedding = nn.Linear(args.encode_out, args.emb)

        # Process
        self.process = nn.ModuleList([PerceiverIOProcessLayer(args, 0.5) for _ in range(args.depth)] )

        # Decoder 
        self.decoder = PerceiverIODecoderLayer(args, args.emb)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self.latent.normal_(0.0, init_scale)
            _init_parameters(self, init_scale)
    
    def forward(self, x, h, query):
        tokens = self.token_embedding(x)
        tokens = torch.cat((tokens, h), 1)

        b = tokens.size(0)
        latent = torch.repeat_interleave(torch.unsqueeze(self.latent, dim = 0), b, dim = 0)

        x = self.encoder(tokens, latent)
        hidden = self.hidden_embedding(x)

        for layer in self.process:
            x = layer(x)

        query = self.token_embedding(query)
        
        if self.args.agent == "perceiver_io":
            x = self.decoder(x, query).view(b, self.args.action_space_size + self.args.enemy_num, self.args.emb)
        else:
            x = self.decoder(x, query).view(b, 1, self.args.emb)

        return x, hidden

def _init_parameters(module, init_scale):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=init_scale)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=init_scale)