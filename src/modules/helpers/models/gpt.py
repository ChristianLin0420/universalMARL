import torch.nn as nn
import torch

from .encoder import Encoder
from .decoder import Decoder

class GPT(nn.Module):

    def __init__(self, args, input_dim, output_dim):
        super().__init__()

        self.args = args

        self.observation_embedding = nn.Linear(input_dim * 2, args.emb)

        # encoder part
        self.previous_encoder_output = None
        self.encoder = Encoder(args, False, 0.0)
        self.encoder_embedding = nn.Linear(args.emb, args.emb)

        # decoder part
        self.previous_decoder_output = None
        self.decoder = Decoder(args, False, 0.0)
        self.decoder_embedding = nn.Linear(args.emb, args.emb)

    def forward(self, ally, enemy, hidden, mask):

        if self.previous_encoder_output is not None and self.previous_encoder_output.size(0) == ally.size(0):
            encoder_input = torch.cat((self.previous_encoder_output.cuda(), ally), -1)
        else:
            zero_padding = torch.zeros(ally.size()) if not self.args.use_cuda else torch.zeros(ally.size()).cuda()
            encoder_input = torch.cat((zero_padding, ally), -1)

        if self.previous_decoder_output is not None and self.previous_decoder_output.size(0) == enemy.size(0):
            decoder_input = torch.cat((self.previous_decoder_output.cuda(), enemy), -1)
        else:
            zero_padding = torch.zeros(enemy.size()) if not self.args.use_cuda else torch.zeros(enemy.size()).cuda()
            decoder_input = torch.cat((zero_padding, enemy), -1)

        self.previous_encoder_output = ally.detach().cpu()
        self.previous_decoder_output = enemy.detach().cpu()

        decode_input = self.observation_embedding(encoder_input)
        cross_input = self.observation_embedding(decoder_input)
        cross_input = torch.cat((cross_input, hidden), 1)

        decode_out = self.decoder(cross_input, decode_input, mask, mask, self.args.max_agents_len, False)
        actions_out = self.decoder_embedding(decode_out)

        return actions_out, decode_out[:, -1:, :]


