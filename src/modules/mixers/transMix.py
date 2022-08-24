import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransMixer(nn.Module):
    def __init__(self, args):
        super(TransMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.input_size = 6

        self.token_embedding = nn.Linear(self.input_size, args.emb)
        self.attention = nn.Linear(args.emb, args.emb)
        self.q_basic = nn.Linear(args.emb, 1)
        self.total_q = nn.Linear(args.max_mixing_size, 1)

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)

        if self.args.use_cuda:
            agent_qs = agent_qs.cuda()
            states = states.cuda()
        
        agent_qs = agent_qs.view(-1, self.input_size)
        token = self.token_embedding(agent_qs).view(-1, self.args.max_mixing_size, self.args.emb)
        atten = self.attention(token).view(-1, self.args.max_mixing_size, self.args.emb)
        q_basic = self.q_basic(atten).view(-1, self.args.max_mixing_size)
        total_q = self.total_q(q_basic).view(bs, -1, 1)

        return total_q

