from torch import nn, optim
import torch
import lightning.pytorch as pl
from .hsdt import arch

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import MultiStepLR
from criteria.utils import peak_snr,sam

class HSDT(pl.LightningModule):
    def __init__(self, in_channels:int, channels:int, num_half_layer:int, sample_idx:list, Fusion=None):
        super().__init__()
        self.example_input_array=torch.Tensor(1, 1, 31, 128, 128)
        self.save_hyperparameters()

        # self.automatic_optimization = False

        self.net = arch.HSDT(in_channels=in_channels, channels=channels, num_half_layer=num_half_layer, sample_idx=sample_idx, Fusion=Fusion)
        self.net.use_2dconv = False
        self.net.bandwise = False
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        
        # it is independent of forward
        # opt = self.optimizers()
        # opt.zero_grad()
        x, y = batch
        length = len(x)
        y_hat = self.net(x)

        loss = nn.functional.mse_loss(y_hat, y)
        # self.manual_backward(loss)
        # opt.step()

        # Logging to TensorBoard (if installed) by default
        # log默认是一次记录一个batch的
        psnr = 0
        ssim = 0
        sam_v = 0
        for y,y_hat in zip(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy()):
            y, y_hat = y.squeeze(0), y_hat.squeeze(0)
            psnr += peak_signal_noise_ratio(y, y_hat, data_range=1)
            ssim += structural_similarity(y, y_hat, data_range=1, channel_axis=-3)
            sam_v += sam(torch.tensor(y_hat), torch.tensor(y)).mean()
        psnr/=length
        ssim/=length
        sam_v/=length
        # psnr = peak_signal_noise_ratio(y.detach().cpu().permute(0,1,4,2,3).numpy(), y_hat.detach().cpu().permute(0,1,4,2,3).numpy(), data_range=1)
        # ssim = structural_similarity(y.detach().cpu().squeeze(1).numpy(), y_hat.detach().cpu().squeeze(1).numpy(),channel_axis=-3, data_range=1)
        self.log_dict({"train_loss":loss, 'train_psnr':psnr, 'train_ssim': ssim, 'train_sam': sam_v})
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        length = len(x)
        y_hat = self.net(x)
        val_loss = nn.functional.mse_loss(y_hat, y)
        # log默认是一次记录一个batch的，但在test_step和validation_step会自动的给你accumulates并且average
        psnr = 0
        ssim = 0
        sam_v = 0
        for y,y_hat in zip(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy()):
            y, y_hat = y.squeeze(0), y_hat.squeeze(0)
            psnr += peak_signal_noise_ratio(y, y_hat,data_range=1)
            ssim += structural_similarity(y, y_hat, data_range=1, channel_axis=-3)
            sam_v += sam(torch.tensor(y_hat), torch.tensor(y)).mean()
        psnr/=length
        ssim/=length
        sam_v/=length
        # psnr = peak_signal_noise_ratio(y.detach().cpu().permute(0,1,4,2,3).numpy(), y_hat.detach().cpu().permute(0,1,4,2,3).numpy(), data_range=1)
        # ssim = structural_similarity(y.detach().cpu().squeeze(1).numpy(), y_hat.detach().cpu().squeeze(1).numpy(),channel_axis=-3, data_range=1)
        self.log_dict({"val_loss":val_loss, 'val_psnr':psnr, 'val_ssim': ssim, 'val_sam': sam_v})
        # self.log_dict({"train_loss":loss, 'train_psnr':psnr, 'train_ssim': ssim, 'train_sam': sam_v})
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        length = len(x)
        y_hat = self.net(x)
        test_loss = nn.functional.mse_loss(y_hat, y)
        # log默认是一次记录一个batch的，但在test_step和validation_step会自动的给你accumulates并且average
        psnr = 0
        ssim = 0
        sam_v = 0
        for y,y_hat in zip(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy()):
            y, y_hat = y.squeeze(0), y_hat.squeeze(0)
            psnr += peak_signal_noise_ratio(y, y_hat,data_range=1)
            ssim += structural_similarity(y, y_hat, data_range=1, channel_axis=-3)
            sam_v += sam(torch.tensor(y_hat), torch.tensor(y)).mean()
        psnr/=length
        ssim/=length
        sam_v/=length
        # psnr = peak_signal_noise_ratio(y.detach().cpu().permute(0,1,4,2,3).numpy(), y_hat.detach().cpu().permute(0,1,4,2,3).numpy(), data_range=1)
        # ssim = structural_similarity(y.detach().cpu().squeeze(1).numpy(), y_hat.detach().cpu().squeeze(1).numpy(),channel_axis=-3, data_range=1)
        self.log_dict({"test_loss":test_loss, 'test_psnr':psnr, 'test_ssim': ssim, 'test_sam': sam_v})
    

    def backward(self, loss):
        return loss.backward()
    
    def forward(self, x):
        """
            So forward() defines your prediction/inference actions.
        """
        return self.net(x)

    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        multiStepLr = MultiStepLR(optimizer=optimizer, milestones=[30, 45, 55, 60, 65, 75, 80], gamma=0.5)
        warmupScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=1.0, total_epoch=10,after_scheduler=multiStepLr)
        return {
            'optimizer' : optimizer,
            # 'lr_scheduler' : {
            #     'scheduler' : warmupScheduler,
            #     'interval' : 'epoch',
            #     'frequency' : 1, 
            #     'name' : 'learning rate variety'
            # }
        }