from typing import Any
import pytorch_lightning as pl
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure, spectral_angle_mapper
import torch
from torch import nn, optim

class Base(pl.LightningModule):
    def __init__(self, use_2d, base_lr, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.net = None
        self.use_2d = use_2d
        self.base_lr = base_lr
        self.psnr = peak_signal_noise_ratio
        self.ssim = structural_similarity_index_measure
        self.sam = spectral_angle_mapper

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward    
        x, y = batch
        y_hat = self.net(x)
        loss = nn.functional.mse_loss(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        # log默认是一次记录一个batch的
        if not self.use_2d:
            y_hat_d, y_d = y_hat.detach().squeeze(1), y.detach().squeeze(1)
        else:
             y_hat_d, y_d = y_hat.detach(), y.detach()
        psnr_v = self.psnr(y_hat_d, y_d)
        ssim_v = self.ssim(y_hat_d, y_d)
        sam_v = self.sam(y_hat_d, y_d)
        self.log("train_loss", loss)
        self.log('train_psnr', psnr_v)
        self.log('train_ssim', ssim_v)
        self.log('train_sam', sam_v)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y_hat = self.net(x)
        val_loss = nn.functional.mse_loss(y_hat, y)
        # log默认是一次记录一个batch的，但在test_step和validation_step会自动的给你accumulates并且average
        if not self.use_2d:
            y_hat_d, y_d = y_hat.detach().squeeze(1), y.detach().squeeze(1)
        else:
             y_hat_d, y_d = y_hat.detach(), y.detach()
        psnr_v = self.psnr(y_hat_d, y_d)
        ssim_v = self.ssim(y_hat_d, y_d)
        sam_v = self.sam(y_hat_d, y_d)
        self.log("val_loss", val_loss)
        self.log('val_psnr', psnr_v, True)
        self.log('val_ssim', ssim_v)
        self.log('val_sam', sam_v)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y_hat = self.net(x)
        test_loss = nn.functional.mse_loss(y_hat, y)
        # log默认是一次记录一个batch的，但在test_step和validation_step会自动的给你accumulates并且average
        if not self.use_2d:
            y_hat_d, y_d = y_hat.detach().squeeze(1), y.detach().squeeze(1)
        else:
             y_hat_d, y_d = y_hat.detach(), y.detach()
        psnr_v = self.psnr(y_hat_d, y_d)
        ssim_v = self.ssim(y_hat_d, y_d)
        sam_v = self.sam(y_hat_d, y_d)
        self.log("test_loss", test_loss)
        self.log('test_psnr', psnr_v)
        self.log('test_ssim', ssim_v)
        self.log('test_sam', sam_v)

    def forward(self, x):
        """
            So forward() defines your prediction/inference actions.
        """
        return self.net(x)
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.base_lr)
        return optimizer
        