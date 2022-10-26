from json import encoder
import torch.nn as nn
import torch

from modules.helpers.embedding.twod_positional_embedding import TwoDPositionalEncoding
from .encoder import Encoder
from .transfermer_decoder import TransfermerDecoder

class FouseformerPlus(nn.Module):

    def __init__(self, args, input_dim, output_dim):
        super().__init__()

        self.args = args

        self.agent_token_embedding = nn.Linear(input_dim, args.emb)
        self.entity_token_embedding = nn.Linear(input_dim, args.emb)
        self.agent_action_token_embedding = nn.Linear(input_dim, args.emb)
        self.enemy_action_token_embedding = nn.Linear(input_dim, args.emb)

        if not args.agent_positional_embedding:
            self.position_embedding = nn.Linear(2, args.emb)
        else:
            self.position_embedding = TwoDPositionalEncoding(args, args.emb, args.max_len, args.device)

        self.encoder = Encoder(args, False, 0.0)
        self.decoder = TransfermerDecoder(args)

        self.output_dim = output_dim
        self.d_tokens = None

    def forward(self, a, e, h, mask):

        agent_token = self.agent_token_embedding(a[:, :1, :])
        entity_token = self.entity_token_embedding(torch.cat((a[:, 1:, :], e), 1))

        pos_input = torch.cat((a, e), 1)
        pos_emb = self.position_embedding(pos_input[:, :, 2:4])

        tokens = torch.cat((agent_token, entity_token), 1)
        tokens = tokens + pos_emb
        encoder_tokens = torch.cat((tokens, h), 1)

        encoder_tokens = self.encoder(encoder_tokens, mask)

        enemy_action_token = self.enemy_action_token_embedding(e[:, :self.args.enemy_num])
        q = self.agent_action_token_embedding(a[:, :1, :])
        decoder_tokens = torch.cat((q, enemy_action_token), 1)
        d = self.decoder(decoder_tokens, encoder_tokens)

        return d, encoder_tokens[:, -1:, :]