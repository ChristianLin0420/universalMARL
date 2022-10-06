import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RMixer(nn.Module):
    def __init__(self, args):
        super(RMixer, self).__init__()

        self.args = args
        self.n_agents = args.max_ally_num
        self.input_dim = args.max_states_dim // 4
        self.embed_dim = args.mixing_embed_dim

        self.state_map_network = nn.Linear(args.max_states_dim, self.input_dim)

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.input_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.input_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.input_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.input_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.input_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def state_padding(self, states):

        padding_size = self.args.max_states_dim - states.size(2)
        padding = th.zeros(states.size(0), states.size(1), padding_size).to(self.args.device)
        states = th.cat([states, padding], axis = -1)
        
        return states.view(-1, self.args.max_states_dim)

    def agent_qs_padding(self, agent_qs):
        
        padding_size = self.args.max_ally_num - agent_qs.size(2)
        padding = th.zeros(agent_qs.size(0), agent_qs.size(1), padding_size).to(self.args.device)
        agent_qs = th.cat([agent_qs, padding], axis = -1)

        # shuffle the last dimension
        p = th.randperm(self.args.max_ally_num)
        agent_qs = agent_qs[:, :, p]

        return agent_qs.view(-1, 1, self.args.max_ally_num)

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)

        if self.args.random_mixing_inputs:
            permutation = th.randperm(self.args.ally_num).to(self.args.device)
            agent_qs = agent_qs[:, :, permutation]
        
        states = self.state_padding(states)
        agent_qs = self.agent_qs_padding(agent_qs)

        # map states to lower dimension
        states = self.state_map_network(states)
        
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        
        return q_tot
