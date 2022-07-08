import imp
import torch.nn as nn
import torch.nn.functional as F
import torch

from modules.helpers.models.simple_transformer import Transformer


class HyperTransformer(nn.Module):
    def __init__(self, args):
        super(HyperTransformer, self).__init__()
        self.args = args

    