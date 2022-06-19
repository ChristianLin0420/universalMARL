import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse

from modules.helpers.models.simple_transformer import Transformer


class UPDeT(nn.Module):
    def __init__(self, input_shape, args):
        super(UPDeT, self).__init__()
        self.args = args
        self.transformer = Transformer(args.token_dim, args.emb, args.heads, args.depth, args.emb)
        self.q_basic = nn.Linear(args.emb, args.action_space_size)

    def init_hidden(self):
        # make hidden states on same device as model
        if self.args.use_cuda:
            return torch.zeros(1, self.args.emb).cuda()
        else:
            return torch.zeros(1, self.args.emb)

    def forward(self, inputs, hidden_state, task_enemy_num = None, task_ally_num = None, env = "sc2"):
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
        
        elif env == "particle":
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
    agent = UPDeT(None, args).cuda()
    hidden_state = agent.init_hidden().cuda().expand(args.ally_num, 1, -1)
    tensor = torch.rand(args.ally_num, args.ally_num+args.enemy_num, args.token_dim).cuda()
    q_list = []
    for _ in range(args.episode):
        q, hidden_state = agent.forward(tensor, hidden_state, args.ally_num, args.enemy_num)
        q_list.append(q)
