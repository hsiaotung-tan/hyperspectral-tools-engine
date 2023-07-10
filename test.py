from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, SpectralAngleMapper
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure, spectral_angle_mapper
from pytorch_lightning import seed_everything
import torch
from pathlib import Path
import h5py
import os
seed_everything(2000)
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--a', action='store_true') # hsdt
print(parser.parse_args())


