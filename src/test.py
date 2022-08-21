import torch
from numpy import random

a = torch.unsqueeze(torch.rand(8, 7), 0)
print(a.size())
b = torch.repeat_interleave(a, 2, dim = 0)
print(b.size())

print(a)
print(b)