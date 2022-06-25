import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse

from modules.helpers.models.vanilla_transformer import Transformer


class DummyTransformer(nn.Module):
    def __init__(self, input_shape, args):
        super(DummyTransformer, self).__init__()
        self.args = args
        self.max_agents_len = args.max_agents_len
        self.transformer = Transformer(args, args.token_dim, args.emb, True)
        self.q_basic = nn.Linear(args.emb, args.action_space_size)
        
    def init_hidden(self):
        # make hidden states on same device as model
        if self.args.use_cuda:
            return torch.zeros(1, self.args.emb).cuda()
        else:
            return torch.zeros(1, self.args.emb)

    def forward(self, inputs, hidden_state, task_enemy_num = None, task_ally_num = None, env = "sc2"):

        b, t, e = inputs.size()

        if self.args.use_cuda:
            tmp_inputs = torch.zeros(b, self.max_agents_len, e).cuda()
        else:
            tmp_inputs = torch.zeros(b, self.max_agents_len, e)

        if env == "sc2":
            inputs = torch.reshape(inputs, (b, t * e))
            new = torch.zeros(b, self.max_agents_len * e)
            f_size = self.args.token_dim
            new[:, :4] = inputs[:, :4] # agent movement features
            new[:, 4] = inputs[:, 4]   # agent own feature
            new[:, 5:(5 + (task_ally_num - 1) * f_size)] = inputs[:, (4 + task_enemy_num * f_size):(t * e - 1)]    # ally features
            new[:, int(self.max_agents_len * f_size / 2):int(self.max_agents_len * f_size / 2 + task_enemy_num * f_size)] = inputs[:, 4:(4 + task_enemy_num * f_size)]       # enemy features
            inputs = torch.reshape(new, (b, self.max_agents_len, e))
        elif env == "simple_spread":
            tmp_inputs[:, :t, :] = inputs
            inputs = tmp_inputs
        elif env == "simple_tag":
            tmp_inputs[:, :self.args.n_agents, :] = inputs[:, :self.args.n_agents, :]
            tmp_inputs[:, self.args.n_agents:(self.args.n_agents + self.args.n_adverary), :] = inputs[:, self.args.n_agents:, :]
            inputs = tmp_inputs

        outputs, _ = self.transformer.forward(inputs, hidden_state, None)

        q_basic_actions = self.q_basic(outputs[:, 0, :])

        # last dim for hidden state
        h = outputs[:, -1:, :]

        if env == "sc2":
            q_enemies_list = []

            # each enemy has an output Q
            for i in range(task_enemy_num):
                q_enemy = self.q_basic(outputs[:, int(self.max_agents_len / 2) + i, :])
                q_enemy_mean = torch.mean(q_enemy, 1, True)
                q_enemies_list.append(q_enemy_mean)

            # concat enemy Q over all enemies
            q_enemies = torch.stack(q_enemies_list, dim=1).squeeze()

            # concat basic action Q with enemy attack Q
            q = torch.cat((q_basic_actions, q_enemies), 1)

            return q, h
        
        elif env in ["simple_spread", "simple_tag"]:
            return q_basic_actions, h