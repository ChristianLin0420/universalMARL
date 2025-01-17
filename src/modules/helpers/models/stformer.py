from json import encoder
import torch.nn as nn
import torch

from modules.helpers.embedding.twod_positional_embedding import TwoDPositionalEncoding
from .encoder import Encoder
from .transfermer_decoder import TransfermerDecoder

class STformer(nn.Module):

    def __init__(self, args, input_dim, output_dim):
        super().__init__()

        self.args = args

        self.agent_token_embedding = nn.Linear(input_dim, args.emb // 2)
        self.entity_token_embedding = nn.Linear(input_dim, args.emb // 2)

        if not args.agent_positional_embedding:
            self.position_embedding = nn.Linear(2, args.emb)
        else:
            self.position_embedding = TwoDPositionalEncoding(args, args.emb // 2, args.max_len, args.device)

        self.global_pos_emb = nn.Parameter(torch.zeros(1, args.max_memory_decoder * 2, args.emb))

        self.encoder = Encoder(args, False, 0.0)
        self.decoder = TransfermerDecoder(args)

        self.output_dim = output_dim

    def forward(self, a, e, h, m, mask):

        agent_token = self.agent_token_embedding(a[:, :1, :])
        entity_token = self.entity_token_embedding(torch.cat((a[:, 1:, :], e), 1))

        pos_input = torch.cat((a, e), 1)
        pos_emb = self.position_embedding(pos_input[:, :, 2:4], True)

        tokens = torch.cat((agent_token, entity_token), 1)
        tokens = torch.cat((tokens, pos_emb), -1)
        encoder_tokens = torch.cat((tokens, h), 1)

        encoder_tokens = self.encoder(encoder_tokens, mask)

        b = a.size(0)
        global_pos_emb = torch.repeat_interleave(self.global_pos_emb, b, dim = 0)
        m = m + global_pos_emb
        m = self.decoder(m, encoder_tokens[:, :1, :])

        return encoder_tokens[:, :1, :], m, encoder_tokens[:, -1:, :]

    def fixed_models_weight(self):
        self.agent_token_embedding.requires_grad = False
        self.entity_token_embedding.requires_grad = False
        self.position_embedding.requires_grad = False
        self.encoder.requires_grad = False
        self.global_pos_emb.requires_grad = False