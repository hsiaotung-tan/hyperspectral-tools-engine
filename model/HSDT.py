from torch import nn, optim
import torch
import lightning.pytorch as pl
from .hsdt import arch
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure, spectral_angle_mapper

class HSDT(pl.LightningModule):
    def __init__(self, in_channels:int, channels:int, num_half_layer:int, sample_idx:list, Fusion=None):
        super().__init__()
        self.example_input_array=torch.Tensor(1, 1, 31, 64, 64)
        self.save_hyperparameters()

        # self.automatic_optimization = False

        self.net = arch.HSDT(in_channels=in_channels, channels=channels, num_half_layer=num_half_layer, sample_idx=sample_idx, Fusion=Fusion)
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
        y_hat_d, y_d = y_hat.detach().squeeze(1), y.detach().squeeze(1)
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
        y_hat_d, y_d = y_hat.detach().squeeze(1), y.detach().squeeze(1)
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
        y_hat_d, y_d = y_hat.detach().squeeze(1), y.detach().squeeze(1)
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
        # warmupScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=1.0, total_epoch=10,after_scheduler=multiStepLr)
        return optimizer
    

if __name__ == '__main__':
    x = torch.randn((1,1, 64, 64))
    model = HSDT(in_channels=1, channels=16, num_half_layer=5, sample_idx=[1, 3])
    print(model(x).shape)