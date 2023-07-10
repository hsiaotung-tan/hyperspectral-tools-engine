import torchvision.transforms as transforms
from torch import nn, optim, utils
from typing import Any
import torch

import os
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from model.HSDT import HSDT
from data.ICVLDataset import get_gaussian_icvl_loader_s1, get_gaussian_icvl_loader_s2, get_gaussian_icvl_loader_val, get_gaussian_icvl_loader_test

seed = 2023
seed_everything(seed)
train_crop_size = (64, 64)
test_val_crop_size = (512, 512)
batch_size = 16
loader1 = get_gaussian_icvl_loader_s1(num_workers=0, pin_memory=False, crop_size=train_crop_size, batch_size=batch_size)
loader2 = get_gaussian_icvl_loader_s2(num_workers=0, pin_memory=False, crop_size=train_crop_size, batch_size=batch_size)
val_loader = get_gaussian_icvl_loader_val(num_workers=0, crop_size=test_val_crop_size)
test_loader = get_gaussian_icvl_loader_test(num_workers=0, crop_size=test_val_crop_size)

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





class SwitchDatasetCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == 29:
            # 切换到另一个数据集
            trainer.replace_dataloader(train_dataloader=loader2)

switchDatasetCallback = SwitchDatasetCallback()

# saves top-K checkpoints based on "val_psnr" metric
# checkpoint_callback = ModelCheckpoint(
#     save_top_k=2,
#     monitor="val_psnr",
#     mode="max",
#     filename="hsdt-{epoch:02d}-{val_psnr:.2f}",
# )
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# accumulate_grad_batches=4
trainer = pl.Trainer(max_epochs=80, devices=1, log_every_n_steps=1, callbacks=[lr_monitor,switchDatasetCallback], default_root_dir=os.getcwd()+'/'+'logs/HSDT')
# trainer = pl.Trainer(max_epochs=80, devices=1, log_every_n_steps=1, callbacks=[checkpoint_callback,lr_monitor], default_root_dir=os.getcwd()+'/'+'logs/HSDT')
trainer.fit(model, train_dataloaders=loader1, val_dataloaders=val_loader, )
trainer.test(model, dataloaders=test_loader,)
