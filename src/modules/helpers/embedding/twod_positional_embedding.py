import math
import torch
from torch import nn

class TwoDPositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, args, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(TwoDPositionalEncoding, self).__init__()

        self.delta = max_len // 2
        self.device = device
        self.args = args

        # same size with input matrix (for adding with input matrix)
        self.x_encoding = torch.zeros(max_len, d_model, device=device)
        self.x_encoding.requires_grad = False  # we don't need to compute gradient
        self.y_encoding = torch.zeros(max_len, d_model, device=device)
        self.y_encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.x_encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.x_encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.y_encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.y_encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, tokens):

        visible_range = 9
        pos_emb = torch.zeros(tokens.size(0), tokens.size(1), self.args.emb)

        for b in range(tokens.size(0)):
            for idx in range(tokens.size(1)):
                x = int(tokens[b, idx, 0] * visible_range)
                y = int(tokens[b, idx, 1] * visible_range)
                pos_emb[b, idx:idx+1, :] = torch.add(self.x_encoding[self.delta + x:self.delta + x + 1, :], self.y_encoding[self.delta + y:self.delta + y + 1, :])

        return pos_emb.to(self.device)