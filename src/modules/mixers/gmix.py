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


    