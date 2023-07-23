import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Any, List, Union
import torch
import numpy as np
import os
from pathlib import Path
import hst_engine.data.tools.general as G
import hst_engine.data.tools.noise as N
import torchvision.transforms as T


class ICVLOriginDataset(Dataset):
    def __init__(self, 
                 data_dir:str, 
                 input_transform=None, 
                 target_transform=None, 
                 common_transform=None, 
                 repeat:int=1) -> None:
        super().__init__()
        datadir = Path(data_dir)
        self.files = [datadir / f for f in os.listdir(datadir) if f.endswith(".npy")]
        self.length = len(self.files)
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.common_transform = common_transform
        self.repeat = repeat

    def __getitem__(self, index) -> Any:
        # Origin Dataset is organized by numpy.ndarray (np.float32), 
        # we first transform them to torch.Tensor then use torchvision.transforms to transform.
        index = index % self.length
        data:torch.Tensor = torch.from_numpy(np.load(self.files[index]))
        print(data.max())

        if self.common_transform:
            data = self.common_transform(data)
        
        if isinstance(data, np.ndarray):
            target = data.copy()
        elif isinstance(data, torch.Tensor):
            target = data.clone()
        else:
            raise NotImplementedError()
        
        if self.input_transform:
            data = self.input_transform(data)

        if self.target_transform:
            target = self.target_transform(target)
        # return {'input':data, 'target':target}
        return data, target
    
    def __len__(self):
        return self.length * self.repeat

if __name__ == '__main__':
    t = ICVLOriginDataset('/HDD/Datasets/HSI_denoising/ICVL/Origin/train', repeat=530)
    print(len(t))
    t[0]