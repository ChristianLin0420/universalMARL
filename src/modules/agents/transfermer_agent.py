import torch.nn as nn
import torch
from random import shuffle
from modules.helpers.models.vanilla_transformer import Transformer
from envs.smac_config import get_entity_extra_information


class Transfermer(nn.Module):
    def __init__(self, input_shape, args):
        super(Transfermer, self).__init__()
        self.args = args
        
        self.transformer = Transformer(args, args.token_dim, args.emb)

        if not args.random_encoder_inputs_zero:
            self.encoder_query = torch.unsqueeze(torch.rand(args.action_space_size + args.max_mixing_size, args.token_dim), 0)
        else:
            self.encoder_query = torch.unsqueeze(torch.zeros(args.action_space_size + args.max_mixing_size, args.token_dim), 0)

        self.decoder_query = torch.unsqueeze(torch.rand(args.action_space_size, args.token_dim), 0)

        # Output optimal action
        self.action_embedding = nn.Linear(args.emb, 1)

        # helper
        self.encoder_indices = [i for i in range(1, args.action_space_size + args.max_mixing_size)]

        # create entity queries
        self.init_entity_infos()

    def init_entity_infos(self):
        infos = []

        for i in range(self.args.ally_num + self.args.enemy_num):
            if i == 0:
                infos.append(get_entity_extra_information("self", "marine"))
            elif i < self.args.ally_num:
                infos.append(get_entity_extra_information("ally", "marine"))
            else:
                infos.append(get_entity_extra_information("enemy", "marine"))

        infos = torch.tensor(infos)
        infos = torch.unsqueeze(infos, 0)
        self.entity_infos = infos

    def init_hidden(self):
        # make hidden states on same device as model
        if self.args.use_cuda:
            return torch.zeros(1, self.args.emb).cuda()
        else:
            return torch.zeros(1, self.args.emb)
    
    def append_extra_infos(self, inputs, encoder_input = True):
        b, t, e = inputs.size()

        query = self.encoder_query if encoder_input else self.decoder_query
        query = torch.repeat_interleave(query, b, dim = 0)

        if encoder_input:
            if self.args.random_inputs:
                shuffle(self.encoder_indices)

                query[:, :1, :] = inputs[:, :1, :]  # agent fix at first position

                for i in range(self.args.ally_num - 1):
                    query[:, self.encoder_indices[i]-1:self.encoder_indices[i], :] = inputs[:, i:i+1, :] # ally
            else:
                query[:, :self.args.ally_num, :] = inputs
        else:
            query = torch.cat([query, inputs], dim = 1)

        return query

    def forward(self, inputs, hidden_state, task_enemy_num = None, task_ally_num = None, env = "sc2"):
        
        b, t, e = inputs.size()

        if env == "sc2":
            inputs = torch.reshape(inputs, (b, t * e))
            new = torch.zeros(b, t * e)

            f_size = self.args.token_dim - 2
            own_feature = 1
            move_feature = 4

            new[:, :move_feature] = inputs[:, :move_feature]                                # agent movement features
            new[:, move_feature:(move_feature + own_feature)] = inputs[:, -own_feature:]    # agent own feature
            new[:, f_size:(f_size + (task_ally_num - 1) * f_size)] = inputs[:, (move_feature + task_enemy_num * f_size):(t * e - own_feature)]  # ally features
            new[:, task_ally_num * f_size:] = inputs[:, move_feature:(move_feature + task_enemy_num * f_size)]                                  # enemy features
            new = torch.reshape(new, (b, t, e))
            infos = torch.repeat_interleave(self.entity_infos, b, dim = 0)
            new = torch.cat([new, infos], axis = 2)
            encoder_inputs = new[:, :task_ally_num, :]
            decoder_inputs = new[:, task_ally_num:, :]

            encoder_inputs = self.append_extra_infos(encoder_inputs)
            decoder_inputs = self.append_extra_infos(decoder_inputs, False)

            if self.args.use_cuda:
                encoder_inputs = encoder_inputs.cuda()
                decoder_inputs = decoder_inputs.cuda()
        
        outputs, tokens = self.transformer.forward(encoder_inputs, hidden_state, None, decoder_inputs)

        # first output for 6 action (no_op stop up down left right)
        q = self.action_embedding(outputs.view(-1, self.args.emb)).view(b, -1, 1)

        # last dim for hidden state
        h = tokens[:, -1:, :]

        return q, h

    def save_query(self, path):
        torch.save(self.encoder_query, "{}/encoder_query.pt".format(path))
        torch.save(self.decoder_query, "{}/decoder_query.pt".format(path))

    def load_query(self, path):
        self.encoder_query = torch.load("{}/encoder_query.pt".format(path))
        self.decoder_query = torch.load("{}/decoder_query.pt".format(path))