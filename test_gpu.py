"""
    test gpu,cuda,cudnn
"""
import torch

print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())