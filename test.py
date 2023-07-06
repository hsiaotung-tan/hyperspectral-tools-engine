import torch

x = torch.arange(0, 9).reshape(1, 3, 3)
print(x.mean(dtype=torch.float32))

# print(x-amax)



