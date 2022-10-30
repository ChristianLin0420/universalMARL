
from re import T
import torch


random = torch.rand(3, 2, 2)

print(random)
print(random.size())

for _ in range(1):
    random = torch.cat((random, random), 0)

print(random)

print(random[:3, :, :])
print(random[3:, :, :])

print(random.size())