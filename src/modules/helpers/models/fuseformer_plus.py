from json import encoder
import torch.nn as nn
import torch

from modules.helpers.embedding.twod_positional_embedding import TwoDPositionalEncoding
from modules.helpers.embedding.positional_embedding import PositionalEncoding
from .encoder import Encoder
from .transfermer_decoder import TransfermerDecoder

class FouseformerPlus(nn.Module):

    def __init__(self, args, input_dim, output_dim):
        super().__init__()

        self.args = args

        self.agent_token_embedding = nn.Linear(input_dim, args.emb // 2)
        self.entity_token_embedding = nn.Linear(input_dim, args.emb // 2)

        if not args.agent_positional_embedding:
            self.position_embedding = nn.Linear(2, args.emb)
        else:
            self.position_embedding = TwoDPositionalEncoding(args, args.emb // 2, args.max_len, args.device)

        self.memory_embedding = PositionalEncoding(args.emb, args.max_memory_decoder, args.device)
        self.memory_pos_emb = self.memory_embedding.generate()

        self.encoder = Encoder(args, False, 0.0)
        self.decoder = TransfermerDecoder(args)

        self.output_dim = output_dim

        self.d_tokens = None

    def forward(self, a, e, h, mask):

        agent_token = self.agent_token_embedding(a[:, :1, :])
        entity_token = self.entity_token_embedding(torch.cat((a[:, 1:, :], e), 1))

        pos_input = torch.cat((a, e), 1)
        pos_emb = self.position_embedding(pos_input[:, :, 2:4], True)

        tokens = torch.cat((agent_token, entity_token), 1)
        tokens = torch.cat((tokens, pos_emb), -1)
        encoder_tokens = torch.cat((tokens, h), 1)

        encoder_tokens = self.encoder(encoder_tokens, mask)

        agent_encdoer_tokens = encoder_tokens[:, :1, :]

        if self.d_tokens is None:
            self.d_tokens = torch.zeros(agent_encdoer_tokens.size(0), self.args.max_memory_decoder, agent_encdoer_tokens.size(-1)).to(self.args.device)
        else:
            if self.d_tokens.size(0) <= agent_encdoer_tokens.size(0):
                repeat = agent_encdoer_tokens.size(0) // self.d_tokens.size(0) - 1
                tmp = self.d_tokens

                for _ in range(repeat):
                    self.d_tokens = torch.cat((self.d_tokens, tmp), 0)

                tmp = None
            else:
                self.d_tokens = None
                self.d_tokens = torch.zeros(agent_encdoer_tokens.size(0), self.args.max_memory_decoder, agent_encdoer_tokens.size(-1)).to(self.args.device)
                
        assert self.d_tokens.size(0) == a.size(0)
        
        decoder_tokens = self.d_tokens + self.memory_pos_emb
        d = self.decoder(decoder_tokens, encoder_tokens)

        self.d_tokens = torch.cat((d[:, :1, :], self.d_tokens[:, :-1, :]), 1)

        return d[:, :1, :], encoder_tokens[:, -1:, :]