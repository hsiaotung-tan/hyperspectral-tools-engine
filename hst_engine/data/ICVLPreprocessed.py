from torch.utils.data import DataLoader, Dataset
from typing import Any
import torch
import numpy as np
import os
from pathlib import Path
import hst_engine.data.tools.general as G
import hst_engine.data.tools.noise as N
import torchvision.transforms as T
import scipy



class ICVLPreprocessDataset(Dataset):
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
        index = index % self.length
        data:np.ndarray = np.load(self.files[index])
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

def get_train_dataset(data_dir='/HDD/Datasets/HSI_denoising/ICVL/Processed/data', input_trans=None, target_trans=None):
    return ICVLPreprocessDataset(data_dir=data_dir, input_transform=input_trans, target_transform=target_trans)

def get_train_loader_s1(
        data_dir='/HDD/Datasets/HSI_denoising/ICVL/Processed/data', 
        batch_size=16, 
        use_2d = True,
        num_workers=8, 
        pin_memory=True, 
        *args, **kwargs):
    input_trans = T.Compose([N.AddNoise(50), G.NdArray2Tensor(use_2d)])
    target_trans = T.Compose([G.NdArray2Tensor(use_2d)])
    ds = get_train_dataset(data_dir=data_dir, input_trans=input_trans, target_trans=target_trans)
    return DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

def get_train_loader_s2(
        data_dir='/HDD/Datasets/HSI_denoising/ICVL/Processed/data', 
        batch_size=16, 
        use_2d = True,
        num_workers=8, 
        pin_memory=True, 
        *args, **kwargs):
    input_trans = T.Compose([N.AddNoiseBlind([10, 30, 50, 70]), G.NdArray2Tensor(use_2d)])
    target_trans = T.Compose([G.NdArray2Tensor(use_2d)])
    ds = get_train_dataset(data_dir=data_dir, input_trans=input_trans, target_trans=target_trans)
    return DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

class ICVLPreprocessedTestDataset(Dataset):
    def __init__(self, 
                 data_dir:str = '/HDD/Datasets/HSI_All/Dataset/Denoise/ICVL_Test/icvl_512_50',
                 size=None,
                 input_transform=None,
                 target_transform=None,  ) -> None:
        super().__init__()
        datadir = Path(data_dir)
        self.files = [datadir / f for f in os.listdir(datadir) if f.endswith(".mat")]
        if size and len(self.files) > size:
            self.files = self.files[:size]
        self.length = len(self.files)
        self.input_transform = input_transform
        self.target_transform = target_transform
            
    def __getitem__(self, index) -> Any:
        data = scipy.io.loadmat(self.files[index])
        data, target = data['input'].transpose(2, 0, 1), data['gt'].transpose(2, 0, 1)
        if self.input_transform:
            data = self.input_transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target
    
    def __len__(self):
        return self.length
    
def get_val_dataset(data_dir='/HDD/Datasets/HSI_All/Dataset/Denoise/ICVL_Test/icvl_512_50', size=5, input_trans=None, target_trans=None):
    return ICVLPreprocessedTestDataset(data_dir=data_dir, size=size, input_transform=input_trans, target_transform=target_trans)
def get_val_loader(data_dir='/HDD/Datasets/HSI_All/Dataset/Denoise/ICVL_Test/icvl_512_50', size=5, use_2d=True):
    input_trans = T.Compose([G.NdArray2Tensor(use_2d)])
    target_trans = T.Compose([G.NdArray2Tensor(use_2d)])
    ds = get_val_dataset(data_dir=data_dir, size=size, input_trans=input_trans, target_trans=target_trans)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

def get_test_dataset(data_dir='/HDD/Datasets/HSI_All/Dataset/Denoise/ICVL_Test/icvl_512_50', size=None, input_trans=None, target_trans=None):
    return ICVLPreprocessedTestDataset(data_dir=data_dir, size=size, input_transform=input_trans, target_transform=target_trans)
def get_test_loader(data_dir='/HDD/Datasets/HSI_All/Dataset/Denoise/ICVL_Test/icvl_512_50', size=None, use_2d=True):
    input_trans = T.Compose([G.NdArray2Tensor(use_2d)])
    target_trans = T.Compose([G.NdArray2Tensor(use_2d)])
    ds = get_val_dataset(data_dir=data_dir, size=size, input_trans=input_trans, target_trans=target_trans)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    


if __name__ == '__main__':
    print('======================================')
    d = get_val_dataset()
    print(len(d))
    # print(len(d[0][0]))
    # print(d[0][0].shape)
    # d = get_val_loader(use_2d=True)
    # s = next(iter(d))
    # print(s[0].shape)
    print('======================================')