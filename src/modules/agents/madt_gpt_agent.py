import torch.nn as nn
import torch

from modules.helpers.models.madt_gpt import madtGPT


class madtGPT(nn.Module):
    def __init__(self, args):
        self.args = args
        self.madt = madtGPT()

    def init_hidden(self):
        if self.args.use_cuda:
            return torch.zeros(1, self.args.emb).cuda()
        else:
            return torch.zeros(1, self.args.emb)

    def forward(self, states, pre_actions, rtgs = None, timesteps = None):
        logits = self.madt.forward(states, pre_actions, rtgs, timesteps)
        return logits
