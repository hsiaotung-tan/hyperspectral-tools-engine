from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, SpectralAngleMapper
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure, spectral_angle_mapper
from pytorch_lightning import seed_everything
import torch

seed_everything(2000)

x = torch.randn((2, 4))
print(x)
print(torch.argmax(x,dim=1))



