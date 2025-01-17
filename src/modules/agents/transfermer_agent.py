import torch.nn as nn
import torch
from random import shuffle
from modules.helpers.models.transfer_transformer import TransferableTransformer
from envs.smac_config import get_entity_extra_information


class Transfermer(nn.Module):
    def __init__(self, input_shape, args):
        super(Transfermer, self).__init__()
        self.args = args
        
        self.transformer = TransferableTransformer(args, args.token_dim, args.emb)
        
        self.max_ally_num = args.max_ally_num
        self.max_enemy_num = args.max_enemy_num
        self.max_entity_num = max(self.max_ally_num, self.max_enemy_num)

        if not args.random_encoder_inputs_zero:
            self.encoder_query = torch.unsqueeze(torch.rand(args.action_space_size + self.max_entity_num, args.token_dim), 0)
            self.decoder_query = torch.unsqueeze(torch.rand(args.action_space_size + self.max_enemy_num, args.token_dim), 0)
        else:
            self.encoder_query = torch.unsqueeze(torch.zeros(args.action_space_size + self.max_entity_num, args.token_dim), 0)
            self.decoder_query = torch.unsqueeze(torch.zeros(args.action_space_size + self.max_enemy_num, args.token_dim), 0)

        # Output optimal action
        self.action_embedding = nn.Linear(args.emb, 1)

        # helper
        self.encoder_indices = [i for i in range(1, args.action_space_size + self.max_entity_num + 1)]
        self.decoder_indices = [i for i in range(self.args.action_space_size + 1, args.action_space_size + self.max_enemy_num + 1)]

        # Mapping to target action spaces
        if args.checkpoint_path == "":
            self.map_action_embedding = nn.Linear(args.action_space_size + args.max_enemy_num, args.action_space_size + args.enemy_num)
        else:
            self.map_action_embedding = nn.Linear(args.action_space_size + args.max_enemy_num, args.action_space_size + args.min_enemy_num)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.emb).to(self.args.device)
    
    def append_extra_infos(self, inputs, encoder_input = True):
        b, _, _ = inputs.size()

        query = self.encoder_query if encoder_input else self.decoder_query
        indices = self.encoder_indices if encoder_input else self.decoder_indices
        query = torch.repeat_interleave(query, b, dim = 0)

        if self.args.random_inputs:
            shuffle(indices)

            if encoder_input:
                query[:, :1, :] = inputs[:, :1, :]  # agent fix at first position

            agent_count = self.args.ally_num - 1 if encoder_input else self.args.enemy_num

            for i in range(agent_count):
                query[:, indices[i]-1:indices[i], :] = inputs[:, i:i+1, :] # ally
        else:
            if encoder_input:
                query[:, :self.args.ally_num, :] = inputs
            else:
                query[:, self.args.action_space_size:self.args.action_space_size + self.args.enemy_num, :] = inputs

        return query

    def forward(self, inputs, hidden_state, task_enemy_num = None, task_ally_num = None, env = "sc2"):
        
        b, t, e = inputs.size()

        encoder_inputs = inputs[:, :task_ally_num, :]
        decoder_inputs = inputs[:, task_ally_num:, :]

        encoder_inputs = self.append_extra_infos(encoder_inputs).to(self.args.device)
        decoder_inputs = self.append_extra_infos(decoder_inputs, False).to(self.args.device)
        
        outputs, tokens = self.transformer.forward(encoder_inputs, decoder_inputs, hidden_state, None)
        # first output for 6 action (no_op stop up down left right)
        q = self.action_embedding(outputs.view(-1, self.args.emb)).view(b, -1, 1)
        q = self.map_action_embedding(q.reshape(-1, self.args.action_space_size + self.args.max_enemy_num))
        # last dim for hidden state
        h = tokens[:, -1:, :]

        return q, h

    def save_query(self, path):
        torch.save(self.encoder_query, "{}/encoder_query.pt".format(path))
        torch.save(self.decoder_query, "{}/decoder_query.pt".format(path))

    def load_query(self, path):
        self.encoder_query = torch.load("{}/encoder_query.pt".format(path)).to(self.args.device)
        self.decoder_query = torch.load("{}/decoder_query.pt".format(path)).to(self.args.device)
        self.encoder_query.requires_grad =  False
        self.decoder_query.requires_grad =  False

    def fixed_models_weight(self):
        self.map_action_embedding = nn.Linear(self.args.action_space_size + self.args.max_enemy_num, self.args.action_space_size + self.args.enemy_num).to(self.args.device)
        self.transformer.requires_grad = False
        self.action_embedding.requires_grad = False
        self.map_action_embedding.requires_grad = True