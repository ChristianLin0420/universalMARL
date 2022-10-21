
import torch
import matplotlib.pyplot as plt

from modules.helpers.embedding.twod_positional_embedding import TwoDPositionalEncoding

size = 32

encoding = TwoDPositionalEncoding(None, size, 512, "cpu")

tmp = torch.rand(1, 18, size)
pos = encoding(tmp).squeeze(0)

plt.matshow(pos)
plt.show()