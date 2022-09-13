import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse

from modules.helpers.models.simple_transformer import Transformer


class TransformerAggregationAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TransformerAggregationAgent, self).__init__()
        self.args = args
        self.transformer = Transformer(args, args.token_dim, args.emb, args.heads, args.depth, args.emb)
        self.q_linear = nn.Linear(args.emb, 6 + args.enemy_num)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.emb).cuda()

    def forward(self, inputs, hidden_state, task_enemy_num, task_ally_num):
        outputs, _ = self.transformer.forward(inputs, hidden_state, None)

        # last output for hidden state
        h = outputs[:,-1:,:]
        q_agg = torch.mean(outputs, 1)
        q = self.q_linear(q_agg)

        return q, h


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
    agent = TransformerAggregationAgent(None, args).cuda()
    hidden_state = agent.init_hidden().cuda().expand(args.ally_num, 1, -1)
    tensor = torch.rand(args.ally_num, args.ally_num+args.enemy_num, args.token_dim).cuda()
    q_list = []
    for _ in range(args.episode):
        q, hidden_state = agent.forward(tensor, hidden_state, args.ally_num, args.enemy_num)
        q_list.append(q)