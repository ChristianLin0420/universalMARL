import torch.nn as nn
import torch

from .encoder import Encoder
from .decoder import Decoder

class TrackFormer(nn.Module):

    def __init__(self, args, input_dim, output_dim):
        super().__init__()

        self.args = args
        self.track_emb = args.emb // 2

        self.observation_embedding = nn.Linear(input_dim, self.track_emb)

        # encoder part
        self.previous_encoder_output = None
        self.encoder = Encoder(args, False, 0.0)
        self.encoder_embedding = nn.Linear(args.emb, self.track_emb)

        # decoder part
        self.previous_decoder_output = None
        self.decoder = Decoder(args, False, 0.0)
        self.decoder_embedding = nn.Linear(args.emb, self.track_emb)

    def forward(self, ally, enemy, hidden, mask):

        # encoder part
        tokens = self.observation_embedding(ally)
        tokens = torch.cat((tokens, hidden), 1)
        b, l, e = tokens.size()

        if self.previous_encoder_output is not None and self.previous_encoder_output.size(0) == tokens.size(0):
            encoder_input = torch.cat((self.previous_encoder_output.cuda(), tokens), -1)
        else:
            zero_padding = torch.zeros(b, l, e) if not self.args.use_cuda else torch.zeros(b, l, e).cuda()
            encoder_input = torch.cat((zero_padding, tokens), -1)

        self.previous_encoder_output = tokens.detach().cpu()

        encode_out = self.encoder(encoder_input, mask)
        encode_out = self.encoder_embedding(encode_out)

        # decoder part
        tokens = self.observation_embedding(enemy)
        b, l, e = tokens.size()

        if self.previous_decoder_output is not None and self.previous_decoder_output.size(0) == tokens.size(0):
            decoder_input = torch.cat((self.previous_decoder_output.cuda(), tokens), -1)
        else:
            zero_padding = torch.zeros(b, l, e) if not self.args.use_cuda else torch.zeros(b, l, e).cuda()
            decoder_input = torch.cat((zero_padding, tokens), -1)

        decode_out = self.decoder(decoder_input, encode_out, mask, mask, self.args.max_agents_len, False)
        decode_out = self.decoder_embedding(decode_out)

        self.previous_decoder_output = decode_out.detach().cpu()

        return decode_out, self.previous_encoder_output.cuda()


