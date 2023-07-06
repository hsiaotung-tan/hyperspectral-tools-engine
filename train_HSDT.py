import torchvision.transforms as transforms
from torch import nn, optim, utils
from typing import Any
import torch


import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from model.HSDT import HSDT
from data.ICVLDataset import get_gaussian_icvl_loader_s1, get_gaussian_icvl_loader_s2, get_gaussian_icvl_loader_val, get_gaussian_icvl_loader_test


seed_everything(2000)
# 1, 16, 5, [1, 3]  version_4
# 1, 16, 7, [1, 3, 5]  version_3
model = HSDT(in_channels=1, channels=16, num_half_layer=5, sample_idx=[1, 3])



loader1 = get_gaussian_icvl_loader_s1()
loader2 = get_gaussian_icvl_loader_s2()
val_loader = get_gaussian_icvl_loader_val()
test_loader = get_gaussian_icvl_loader_test()

class SwitchDatasetCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == 29:
            # 切换到另一个数据集
            trainer.replace_dataloader(train_dataloader=loader2)

switchDatasetCallback = SwitchDatasetCallback()

# saves top-K checkpoints based on "val_psnr" metric
checkpoint_callback = ModelCheckpoint(
    save_top_k=2,
    monitor="val_psnr",
    mode="max",
    filename="hsdt-{epoch:02d}-{val_psnr:.2f}",
)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# accumulate_grad_batches=4
trainer = pl.Trainer(max_epochs=80, devices=1, log_every_n_steps=1, callbacks=[checkpoint_callback,lr_monitor,switchDatasetCallback], )
trainer.fit(model, train_dataloaders=loader1, val_dataloaders=val_loader)
trainer.test(model, test_loader=test_loader, ckpt_path='best')
