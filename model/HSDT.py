from torch import nn, optim
import torch
import lightning.pytorch as pl
from .hsdt import arch
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import MultiStepLR
from criteria.utils import peak_snr,sam,ssim

class HSDT(pl.LightningModule):
    def __init__(self, in_channels:int, channels:int, num_half_layer:int, sample_idx:list, Fusion=None):
        super().__init__()
        self.example_input_array=torch.Tensor(1, 1, 31, 64, 64)
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.net = arch.HSDT(in_channels=in_channels, channels=channels, num_half_layer=num_half_layer, sample_idx=sample_idx, Fusion=Fusion)
        self.net.use_2dconv = False
        self.net.bandwise = False
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        
        # it is independent of forward
        opt = self.optimizers()
        opt.zero_grad()
        x, y = batch
        length = len(x)
        y_hat = self.net(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.manual_backward(loss)
        opt.step()

        # Logging to TensorBoard (if installed) by default
        # log默认是一次记录一个batch的
        
        psnr_v = 0
        ssim_v = 0
        sam_v = 0
        for y,y_hat in zip(y.detach(), y_hat.detach()):
            y, y_hat = y.squeeze(0), y_hat.squeeze(0)
            psnr_v += peak_snr(y, y_hat)
            ssim_v += ssim(y, y_hat)
            sam_v += sam(y_hat, y)
        psnr_v/=length
        ssim_v/=length
        sam_v/=length
        
        self.log_dict({"train_loss":loss, 'train_psnr':psnr_v, 'train_ssim': ssim_v, 'train_sam': sam_v})
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        length = len(x)
        y_hat = self.net(x)
        val_loss = nn.functional.mse_loss(y_hat, y)
        # log默认是一次记录一个batch的，但在test_step和validation_step会自动的给你accumulates并且average
        psnr_v = 0
        ssim_v = 0
        sam_v = 0
        for y,y_hat in zip(y.detach(), y_hat.detach()):
            y, y_hat = y.squeeze(0), y_hat.squeeze(0)
            psnr_v += peak_snr(y, y_hat)
            ssim_v += ssim(y, y_hat)
            sam_v += sam(y_hat, y)
        psnr_v/=length
        ssim_v/=length
        sam_v/=length
        self.log_dict({'val_loss':val_loss, 'val_psnr':psnr_v, 'val_ssim': ssim_v, 'val_sam': sam_v})
        # self.log_dict({"train_loss":loss, 'train_psnr':psnr, 'train_ssim': ssim, 'train_sam': sam_v})
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        length = len(x)
        y_hat = self.net(x)
        test_loss = nn.functional.mse_loss(y_hat, y)
        # log默认是一次记录一个batch的，但在test_step和validation_step会自动的给你accumulates并且average
        psnr_v = 0
        ssim_v = 0
        sam_v = 0
        for y,y_hat in zip(y.detach(), y_hat.detach()):
            y, y_hat = y.squeeze(0), y_hat.squeeze(0)
            psnr_v += peak_snr(y, y_hat)
            ssim_v += ssim(y, y_hat)
            sam_v += sam(y_hat, y)
        psnr_v/=length
        ssim_v/=length
        sam_v/=length
        self.log_dict({'test_loss':test_loss, 'test_psnr':psnr_v, 'test_ssim': ssim_v, 'test_sam': sam_v})
    

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
    

if __name__ == '__main__':
    x = torch.randn((1,1, 64, 64))
    model = HSDT(in_channels=1, channels=16, num_half_layer=5, sample_idx=[1, 3])
    print(model(x).shape)