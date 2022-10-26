
from re import T
import torch


random = torch.rand(3, 2, 2)

print(random)

avg = torch.mean(random, 1, True)
print(avg.size())
print(avg)