
import torch


random = torch.rand(3, 2, 2)

print(random)

random = torch.mul(random, 0.0)

print(random)