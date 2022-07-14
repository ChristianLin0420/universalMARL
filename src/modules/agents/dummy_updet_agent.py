import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse

from modules.helpers.models.simple_transformer import Transformer


class DummyUPDeT(nn.Module):
    def __init__(self, input_shape, args):
        super(DummyUPDeT, self).__init__()
        self.args = args
        self.max_agents_len = args.max_agents_len
        self.transformer = Transformer(args.token_dim, args.emb, args.heads, args.depth, args.emb)
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
            enemy_feature = self.args.enemy_feature
            own_feature = self.args.own_feature
            move_feature = self.args.token_dim - self.args.own_feature
            new[:, :move_feature] = inputs[:, :move_feature]                                # agent movement features
            new[:, move_feature:(move_feature + own_feature)] = inputs[:, -own_feature:]    # agent own feature
            new[:, f_size:(f_size + (task_ally_num - 1) * f_size)] = inputs[:, (move_feature + task_enemy_num * enemy_feature):(t * e - own_feature)]    # ally features
            new[:, int(self.max_agents_len * f_size / 2):int(self.max_agents_len * f_size / 2 + task_enemy_num * enemy_feature)] = inputs[:, move_feature:(move_feature + task_enemy_num * enemy_feature)]       # enemy features
            inputs = torch.reshape(new, (b, self.max_agents_len, e))

            if self.args.use_cuda:
                inputs = inputs.cuda()
        elif env == "simple_spread":
            tmp_inputs[:, :t, :] = inputs
            inputs = tmp_inputs

        outputs, _ = self.transformer.forward(inputs, hidden_state, None)
        # first output for 6 action (no_op stop up down left right)
        q_basic_actions = self.q_basic(outputs[:, 0, :])

        # last dim for hidden state
        h = outputs[:, -1:, :]

        if env == "sc2":
            q_enemies_list = []

            # each enemy has an output Q
            for i in range(task_enemy_num):
                q_enemy = self.q_basic(outputs[:, 1 + i, :])
                q_enemy_mean = torch.mean(q_enemy, 1, True)
                q_enemies_list.append(q_enemy_mean)

            # concat enemy Q over all enemies
            q_enemies = torch.stack(q_enemies_list, dim=1).squeeze()

            # concat basic action Q with enemy attack Q
            q = torch.cat((q_basic_actions, q_enemies), 1)

            return q, h
        
        elif env in ["simple_spread"]:
            return q_basic_actions, h


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unit Testing')
    parser.add_argument('--token_dim', default='5', type=int)
    parser.add_argument('--emb', default='32', type=int)
    parser.add_argument('--heads', default='3', type=int)
    parser.add_argument('--depth', default='2', type=int)
    parser.add_argument('--ally_num', default='5', type=int)
    parser.add_argument('--enemy_num', default='5', type=int)
    parser.add_argument('--episode', default='20', type=int)
    args = parser.parse_args()


    # testing the agent
    agent = DummyUPDeT(None, args).cuda()
    hidden_state = agent.init_hidden().cuda().expand(args.ally_num, 1, -1)
    tensor = torch.rand(args.ally_num, args.ally_num+args.enemy_num, args.token_dim).cuda()
    q_list = []
    for _ in range(args.episode):
        q, hidden_state = agent.forward(tensor, hidden_state, args.ally_num, args.enemy_num)
        q_list.append(q)
