import torch
from numpy import random

# a = torch.unsqueeze(torch.rand(8, 7), 0)
# print(a.size())
# b = torch.repeat_interleave(a, 2, dim = 0)
# print(b.size())

# print(a)
# print(b)

k = [0, 1, 2, 3, 4, 5, 6, 7]
print(random.choice(k, 3))