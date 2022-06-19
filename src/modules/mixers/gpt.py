import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GPTMixer(nn.Module):
    def __init__(self, args):
        
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        

    def forward(self,agent_qs, state):
        pass