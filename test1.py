from data.transforms.noises import AddGaussanNoiseStd
import torch
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure, spectral_angle_mapper
from pytorch_lightning import seed_everything
seed_everything(2000)
a = AddGaussanNoiseStd(sigma=30)
x = torch.randn((3,31,31,31))
y = a(x)
print(peak_signal_noise_ratio(x,y))
y = x + torch.randn_like(x) * 0.117
print(peak_signal_noise_ratio(x,y))


