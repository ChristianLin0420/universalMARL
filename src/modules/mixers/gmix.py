import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GMixer(nn.Module):
    def __init__(self, args):
        super(GMixer, self).__init__()

        self.args = args
        self.n_agents = args.max_ally_num
        self.input_dim = ...
        self.emb_dim = args.mixing_embed_dim


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