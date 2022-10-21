
import torch
from modules.helpers.embedding.twod_positional_embedding import TwoDPositionalEncoding

encoding = TwoDPositionalEncoding(32, 512, "cpu")

tmp = torch.rand(1, 32, 32)
pos = encoding(tmp)

print(pos)