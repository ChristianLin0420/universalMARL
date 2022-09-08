import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.helpers.models.decoder import Decoder

class AdaptiveMixer(nn.Module):
    def __init__(self, args):
        super(AdaptiveMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape)) - args.ally_num * (args.action_space_size + args.enemy_num)
        self.action_dim = args.action_space_size + args.enemy_num

        self.observation_embedding = nn.Linear(args.token_dim, args.emb)
        self.action_embedding = nn.Linear(args.action_context_length, args.emb)
        self.hyper_net = Decoder(args, False, 0.0)
        self.hyper_b = nn.Linear(self.state_dim, args.emb)

        self.adaptive_net = nn.Sequential(nn.Linear(self.state_dim,  args.emb),
                                          nn.ReLU(),
                                          nn.Linear(args.emb, self.args.min_ally_num))

        self.adaptive_final_net = nn.Sequential(nn.Linear(self.state_dim, args.emb), 
                                                nn.ReLU(),
                                                nn.Linear(args.emb, 1))
                               

    def forward(self, agent_qs, states, observations):
        bs = agent_qs.size(0)
        current_states = states.reshape(-1, int(np.prod(self.args.state_shape)))[:, :self.state_dim]
        current_observations = observations.reshape(-1, self.n_agents, self.args.token_dim)
        previous_actions = states.reshape(-1, int(np.prod(self.args.state_shape)))[:, self.state_dim:].view(-1, self.n_agents, self.action_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # action padding
        padding_size = self.args.action_context_length - self.action_dim
        padding = th.zeros(previous_actions.size(0), self.n_agents, padding_size)
        previous_actions = th.concat([previous_actions, padding], axis = -1)

        if self.args.use_cuda:
            current_states.cuda()
            current_observations.cuda()
            previous_actions.cuda()

        a_emb = self.action_embedding(previous_actions)
        o_emb = self.observation_embedding(current_observations)
        h_emb = self.hyper_net(o_emb, a_emb, None, None, 0, False)
        h_b = self.hyper_b(current_states).view(-1, self.args.emb, 1)
        hidden = th.bmm(h_emb, h_b).view(-1, 1, self.n_agents)

        adaptive = self.adaptive_net(current_states).view(-1, self.n_agents, 1)
        adaptive_final = self.adaptive_final_net(current_states).view(-1, 1, 1)

        # Compute final output
        y = th.bmm(hidden, adaptive) + adaptive_final
        # Reshape and return
        q_tot = y.view(bs, -1, 1)

        return q_tot

    def new_adaptive_layer(self):
        self.observation_embedding.requires_grad = False
        self.action_embedding.requires_grad = False
        self.hyper_net.requires_grad = False
        self.hyper_b.requires_grad = False
        self.adaptive_net = nn.Sequential(nn.Linear(self.state_dim,  self.emb),
                                          nn.ReLU(),
                                          nn.Linear(self.emb, self.n_agents))
        self.adaptive_final_net.requires_grad = True


