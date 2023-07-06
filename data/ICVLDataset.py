import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import numpy as np
from torchvision import transforms
from transforms import utils, noises
# from .noise_tools import transforms as hyde_transforms


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
        
        img = utils.minmax_normalize(torch.tensor(np.load(self.files[idx]), dtype=torch.float32))
        if not self.use2d:
            img = img.unsqueeze(0)
        

        # crop HSI to designated size
        img = self.base_transforms(img)
        
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
    

if __name__ == '__main__':
    data_dir = '/HDD/Datasets/HSI_denoising/ICVL_HyDe/test'
    input_transform = noises.AddGaussanNoiseStd(30)
    sets = ICVLDataset(datadir=data_dir,input_transform=input_transform, use2d=False)
    # print(sets[1][0].min())
    print(sets[1][0])
