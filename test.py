import torch

x = torch.arange(0, 9).reshape(1, 3, 3)
print(x)
shape = x.shape
print(shape)
amin = x.min()
amax = x.max()
print(amin)
print(amax)

print(x-amax)



