import torch

x = torch.rand(3, 3, 3)
y = x[:, 1:, :]
p = torch.randperm(2)
print(x)
print(p)
print(y)
y = y[:, p, :]
print(y)
x = torch.cat((x[:, :1, :], y), 1)
print(x)