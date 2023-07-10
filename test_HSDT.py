import torchvision.transforms as transforms
from torch import nn, optim, utils
from typing import Any
import torch

import os
import lightning.pytorch as pl
from lightning.pytorch import seed_everything


from model.HSDT import HSDT
from data.ICVLDataset import get_gaussian_icvl_loader_test

seed = 2000
seed_everything(seed)
train_crop_size = (64, 64)
test_val_crop_size = (512, 512)
batch_size = 16
test_loader = get_gaussian_icvl_loader_test(num_workers=0, crop_size=test_val_crop_size)

# config_dict = {
#     'in_channels':1,
#     'channels':16, 
#     'num_half_layer':7, 
#     'sample_idx':[1, 3, 5],
#     'seed':seed, 
#     'train_crop_size':train_crop_size, 
#     'test_val_crop_size':test_val_crop_size, 
#     'batch_size':batch_size,
#     'Fusion':None
# }

config_dict = {
    'in_channels':1,
    'channels':16, 
    'num_half_layer':5, 
    'sample_idx':[1, 3],
    'seed':seed, 
    'train_crop_size':train_crop_size, 
    'test_val_crop_size':test_val_crop_size, 
    'batch_size':batch_size,
    'Fusion':None
}
model = HSDT(**config_dict)
t = torch.load('/home/tanxiaodong/Desktop/Project/HSIR/checkpoints/hsir.model.hsdt.hsdt/model_best.pth')
# print(t.keys())
model.net.load_state_dict(t['net'])
trainer = pl.Trainer(devices=1, default_root_dir=os.getcwd()+'/'+'logs/HSDT/test')
trainer.test(model, dataloaders=test_loader,)
