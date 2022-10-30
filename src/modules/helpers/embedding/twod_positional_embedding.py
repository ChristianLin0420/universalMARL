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

        d_model = d_model // 2

        # same size with input matrix (for adding with input matrix)
        self.x_encoding = torch.zeros(max_len, d_model, device=device)
        self.y_encoding = torch.zeros(max_len, d_model, device=device)
        self.x_encoding.requires_grad = False
        self.y_encoding.requires_grad = False  

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.x_encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.x_encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.y_encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.y_encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        # two dimensional positional matrix
        self.encoding = torch.zeros(max_len, max_len, d_model * 2, device=device)

        for x in range(max_len):
            for y in range(max_len):
                self.encoding[x, y, :] = torch.cat((self.x_encoding[x, :], self.y_encoding[y, :]), -1)

    def forward(self, tokens, concat = False):

        visible_range = 9

        if concat:
            pos_emb = torch.zeros(tokens.size(0), tokens.size(1), self.args.emb // 2).to(self.device)
        else:
            pos_emb = torch.zeros(tokens.size(0), tokens.size(1), self.args.emb).to(self.device)

        tokens = torch.mul(tokens, visible_range)
        tokens = torch.round(tokens).type(torch.LongTensor)

        # for b in range(tokens.size(0)):
        #     for idx in range(tokens.size(1)):
        #         x = int(tokens[b, idx, 0])
        #         y = int(tokens[b, idx, 1])
        #         pos_emb[b, idx:idx+1, :] = self.encoding[self.delta + x:self.delta + x + 1, 
        #                                                  self.delta + y:self.delta + y + 1, :]

        x = tokens[:, :, 0]
        y = tokens[:, :, 1]
        pos_emb[:, :, :] = self.encoding[x, y, :]

        return pos_emb
