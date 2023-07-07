from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, SpectralAngleMapper
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure, spectral_angle_mapper
import torch
from pytorch_lightning import seed_everything

seed_everything(3000)
x = torch.randn((3,31,31,31))
y = x + torch.randn_like(x) * 0.5
p = torch.tensor(0.)

for i,j in zip(x,y):
    
    t = peak_signal_noise_ratio(i.unsqueeze(dim=0), j.unsqueeze(dim=0))
    print(t)
    p+=t

print(p/len(x))

print(peak_signal_noise_ratio(x,y))


