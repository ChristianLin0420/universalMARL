from json import encoder
import torch.nn as nn
import torch
import os
from modules.helpers.embedding.twod_positional_embedding import TwoDPositionalEncoding
from modules.helpers.embedding.positional_embedding import PositionalEncoding
from modules.helpers.embedding.identity_embedding import IdentityEmbedding
from .encoder import Encoder
from .transfermer_decoder import TransfermerDecoder

class FouseformerExtra(nn.Module):

    def __init__(self, args, input_dim, output_dim):
        super().__init__()

        self.args = args

        self.agent_token_embedding = nn.Linear(input_dim, args.emb // 2)
        self.entity_token_embedding = nn.Linear(input_dim, args.emb // 2)

        if not args.agent_positional_embedding:
            self.position_embedding = nn.Linear(2, args.emb // 2)
        else:
            if args.use_identity:
                self.position_embedding = TwoDPositionalEncoding(args, args.emb // 4, args.max_len, args.device)
                self.identity_embedding = IdentityEmbedding(args, args.batch_size, args.emb // 4, args.map_name, args.dummy_entity, args.device).get_identity_embedding()
            else:
                self.position_embedding = TwoDPositionalEncoding(args, args.emb // 2, args.max_len, args.device)

        self.memory_encoder = nn.Linear(args.emb * 2, args.emb)
        self.memory_embedding = PositionalEncoding(args.emb, args.max_memory_decoder * 2, args.device)
        self.memory_pos_emb = self.memory_embedding.generate()

        self.encoder = Encoder(args, False, 0.0)
        self.decoder = TransfermerDecoder(args)

        self.output_dim = output_dim

        if args.save_attention_maps:
            self.frame_idx = 0
            self.save_attention_maps = True
            self.save_path = args.save_attention_maps_path
            self.save_path = os.path.join(self.save_path, args.map_name, args.mixer, args.agent)
            os.makedirs(self.save_path, exist_ok=True)

    def forward(self, a, e, h, m, mask):

        agent_token = self.agent_token_embedding(a[:, :1, :])
        entity_token = self.entity_token_embedding(torch.cat((a[:, 1:, :], e), 1))

        pos_input = torch.cat((a, e), 1)

        if not self.args.agent_positional_embedding:
            pos_emb = self.position_embedding(pos_input[:, :, 2:4])
        else:
            pos_emb = self.position_embedding(pos_input[:, :, 2:4], True)

        tokens = torch.cat((agent_token, entity_token), 1)
        b = tokens.size(0)

        if self.args.use_identity:
            tokens = torch.cat((tokens, self.identity_embedding[:b, :, :]), -1)

        tokens = torch.cat((tokens, pos_emb), -1)
        encoder_tokens = torch.cat((tokens, h), 1)

        encoder_tokens = self.encoder(encoder_tokens, mask, save_attention_maps=self.save_attention_maps, save_path=self.save_path, frame_idx=self.frame_idx)
        
        m = self.memory_encoder(m)
        m = m + self.memory_pos_emb
        m = self.decoder(m, encoder_tokens[:, :1, :])
        self.frame_idx += 1

        return encoder_tokens[:, :1, :], m, encoder_tokens[:, -1:, :]

    def fixed_models_weight(self):
        self.agent_token_embedding.requires_grad = False
        self.entity_token_embedding.requires_grad = False
        self.position_embedding.requires_grad = False
        self.memory_encoder.requires_grad = False
        self.memory_embedding.requires_grad = False
        self.encoder.requires_grad = False