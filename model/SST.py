from torch import nn, optim
import torch
import lightning.pytorch as pl
from .sst import sst
from .sst.SST import SST as aSST
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure, spectral_angle_mapper

class SST(pl.LightningModule):
    def __init__(self, inp_channels, dim, window_size, depths, num_heads, mlp_ratio, seed:int, train_crop_size:tuple, test_val_crop_size:tuple, batch_size:int):
        super().__init__()
        self.example_input_array=torch.Tensor(1, 1, 31, *(train_crop_size))
        self.save_hyperparameters()

        # self.automatic_optimization = False

        self.net = aSST(inp_channels, dim, window_size, depths, num_heads, mlp_ratio)
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
        
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # multiStepLr = MultiStepLR(optimizer=optimizer, milestones=[30, 45, 55, 60, 65, 75, 80], gamma=0.5)
        # warmupScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=1.0, total_epoch=2,after_scheduler=multiStepLr)
        # return [optimizer], [warmupScheduler]
        return optimizer

if __name__ == '__main__':
    x = torch.randn((1, 31, 64, 64))
    model = SST(inp_channels=31,dim = 90,
        window_size=8,
        depths=[ 6,6,6,6,6,6],
        num_heads=[ 6,6,6,6,6,6],mlp_ratio=2,seed=1,train_crop_size=(64,64), test_val_crop_size=(512, 512), batch_size=16)
    print(model(x).shape)