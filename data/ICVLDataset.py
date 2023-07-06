import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import numpy as np
from torchvision import transforms
from transforms import utils, noises

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class ICVLDataset(Dataset):
    """
    The torch Dataset to use for training a NN on the ICVL dataset.

    NOTE: this will automatically do random rotations and vertical flips on both the network
    input and label

    Parameters
    ----------
    datadir: str
        location of the data to be loaded
    crop_size: tuple, optional
        the size to crop the image to. Many datasets have non-uniform data
        default: (128, 128)
    input_transform: optional
        transforms to apply to the network input
    target_transform: optional
        transforms to perform on the label
    common_transforms: optional
        transforms to apply to both the network input and the label
    use2d: bool, optional
        flag indicating if use conv2d
    """

    def __init__(
        self,
        datadir,
        crop_size=(128, 128),
        input_transform=None,
        target_transform=None,
        common_transforms=None,
        use2d=True,
        repeat=1
    ):
        super(ICVLDataset, self).__init__()
        datadir = Path(datadir)
        self.files = [datadir / f for f in os.listdir(datadir) if f.endswith(".npy")]
        self.base_transforms = transforms.RandomCrop(crop_size)  # RandomCrop(crop_size)
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.common_transforms = common_transforms
        self.repeat = repeat
        self.length = len(self.files) * self.repeat
        self.use2d=use2d

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        # crop HSI to designated size
        img = self.base_transforms(torch.tensor(np.load(self.files[idx]), dtype=torch.float32))
        
        img = utils.minmax_normalize(img)
        if not self.use2d:
            img = img.unsqueeze(0)
        
        # transforms of input and target 
        if self.common_transforms is not None:
            img = self.common_transforms(img)
        
        target = img.clone().detach()

        # exert input data transform
        if self.input_transform:
            img = self.input_transform(img)

        # exert target data transform
        if self.target_transform is not None:
            target = self.target_transform(target)

        # # clip after noise
        # img = utils.clip_outliers(img, min=0, max=1)
        # target = utils.clip_outliers(target, min=0, max=1)

        return img, target
    
def get_gaussian_icvl_loader_s1(use_conv2d=False, crop_size=(64,64), batch_size=16, shuffle=True, num_workers=8, pin_memory=False):
    data_dir = '/HDD/Datasets/HSI_denoising/ICVL/Origin/train'
    input_transform = noises.AddGaussanNoiseStd(50)
    dataset = ICVLDataset(datadir=data_dir, crop_size=crop_size, input_transform=input_transform, use2d=use_conv2d)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn)
    return loader

def get_gaussian_icvl_loader_s2(use_conv2d=False, crop_size=(64,64), batch_size=16, shuffle=True, num_workers=8, pin_memory=False):
    data_dir = '/HDD/Datasets/HSI_denoising/ICVL/Origin/train'
    input_transform = noises.AddGaussanBlindNoiseStd(10, 30, 50, 70)
    dataset = ICVLDataset(datadir=data_dir, crop_size=crop_size, input_transform=input_transform, use2d=use_conv2d)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn)
    return loader

def get_gaussian_icvl_loader_val(use_conv2d=False, crop_size=(512, 512), batch_size=1, shuffle=True, num_workers=8, pin_memory=False):
    data_dir = '/HDD/Datasets/HSI_denoising/ICVL/Origin/val'
    input_transform = noises.AddGaussanNoiseStd(50)
    dataset = ICVLDataset(datadir=data_dir, crop_size=crop_size, input_transform=input_transform, use2d=use_conv2d)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn)
    return loader


if __name__ == '__main__':
    data_dir = '/HDD/Datasets/HSI_denoising/ICVL_HyDe/test'
    input_transform = noises.AddGaussanNoiseStd(70)
    sets = ICVLDataset(datadir=data_dir,input_transform=input_transform, use2d=False)
    # print(len(sets))
    print(sets[1][0])
    # print(sets[1][0]- sets[1][1])
