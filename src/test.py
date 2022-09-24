import torch
from random import shuffle
from numpy import random

x = torch.rand(3, 3, 3)
p = torch.randperm(3)
print(x)
print(p)
print(x[:, :, p])