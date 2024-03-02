import torch
import sys
import os

a = torch.tensor([[1.], [1.]])
b = torch.tensor([[2.], [2.]])
print(torch.norm(b - a))
print(a.reshape(-1,2))
print(a)