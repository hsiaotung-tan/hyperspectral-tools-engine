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




class ICVLDataset(Dataset):
    def __init__(self, 
                 data_dir:str, 
                 crop_size: Union[List[int], None], 
                 input_transform=None, 
                 target_transform=None, 
                 common_transform=None, 
                 repeat:int=1) -> None:
        super().__init__()
        datadir = Path(data_dir)
        self.files = [datadir / f for f in os.listdir(datadir) if f.endswith(".npy")]
        self.length = len(self.files)
        self.crop_size = crop_size
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.common_transform = common_transform
        self.repeat = repeat

    def __getitem__(self, index) -> Any:
        index = index % self.length
        data:np.ndarray = np.load(self.files[index])
        if self.crop_size:
            if len(self.crop_size)==1:
                h, w = self.crop_size[0], self.crop_size[0]
            else:
                h, w = self.crop_size[0], self.crop_size[1]
            data = G.RandomCrop(croph=h, cropw=w)(data)

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

    

def get_dataset_gaussian_s1(batch_size=16, crop_size=(64, 64), use_2d=True):
    input_trans = T.Compose([N.AddNoise(50), G.NdArray2Tensor(use_2d)])
    target_trans = T.Compose([G.NdArray2Tensor(use_2d)])
    ds = ICVLDataset(data_dir='/HDD/Datasets/HSI_denoising/ICVL/Processed/data', 
                crop_size=crop_size, 
                input_transform=input_trans,
                target_transform=target_trans)
    return ds

def get_dataset_gaussian_s2(batch_size=16, crop_size=(64, 64), use_2d=True):
    input_trans = T.Compose([N.AddNoiseBlind([10, 30, 50, 70]), G.NdArray2Tensor(use_2d)])
    target_trans = T.Compose([G.NdArray2Tensor(use_2d)])
    ds = ICVLDataset(data_dir='/HDD/Datasets/HSI_denoising/ICVL/Processed/data', 
                crop_size=crop_size, 
                input_transform=input_trans,
                target_transform=target_trans)
    return ds

def get_dataset_gaussian_val(batch_size=1, crop_size=(512, 512), use_2d=True):
    input_trans = T.Compose([N.AddNoise(50), G.NdArray2Tensor(use_2d)])
    target_trans = T.Compose([G.NdArray2Tensor(use_2d)])
    ds = ICVLDataset(data_dir='/HDD/Datasets/HSI_denoising/ICVL/Origin/val',
                           crop_size=crop_size, 
                           input_transform=input_trans,
                           target_transform=target_trans)
    return ds

def get_dataset_gaussian_test(batch_size=1, crop_size=(512, 512), use_2d=True):
    input_trans = T.Compose([N.AddNoise(50), G.NdArray2Tensor(use_2d)])
    target_trans = T.Compose([G.NdArray2Tensor(use_2d)])
    ds = ICVLDataset(data_dir='/HDD/Datasets/HSI_denoising/ICVL/Origin/test',
                           crop_size=crop_size, 
                           input_transform=input_trans,
                           target_transform=target_trans)
    return ds


def get_loader_gaussian_s1(train_batch_size=16, train_crop_size=(64, 64), use_2d=True, num_workers=8, pin_memory=True, *args, **kwargs):
    input_trans = T.Compose([N.AddNoise(50), G.NdArray2Tensor(use_2d)])
    target_trans = T.Compose([G.NdArray2Tensor(use_2d)])
    ds = ICVLDataset(data_dir='/HDD/Datasets/HSI_denoising/ICVL/Processed/data', 
                crop_size=train_crop_size, 
                input_transform=input_trans,
                target_transform=target_trans)
    return DataLoader(dataset=ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

def get_loader_gaussian_s2(train_batch_size=16, train_crop_size=(64, 64), use_2d=True, num_workers=8, pin_memory=True, *args, **kwargs):
    input_trans = T.Compose([N.AddNoiseBlind([10, 30, 50, 70]), G.NdArray2Tensor(use_2d)])
    target_trans = T.Compose([G.NdArray2Tensor(use_2d)])
    ds = ICVLDataset(data_dir='/HDD/Datasets/HSI_denoising/ICVL/Processed/data', 
                crop_size=train_crop_size, 
                input_transform=input_trans,
                target_transform=target_trans)
    return DataLoader(dataset=ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

def get_loader_gaussian_val(val_batch_size=1, val_crop_size=(512, 512), use_2d=True, num_workers=8, pin_memory=True, *args, **kwargs):
    input_trans = T.Compose([N.AddNoise(50), G.NdArray2Tensor(use_2d)])
    target_trans = T.Compose([G.NdArray2Tensor(use_2d)])
    ds = ICVLDataset(data_dir='/HDD/Datasets/HSI_denoising/ICVL/Origin/val',
                           crop_size=val_crop_size, 
                           input_transform=input_trans,
                           target_transform=target_trans)
    return DataLoader(dataset=ds, batch_size=val_batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

def get_loader_gaussian_test(test_batch_size=1, test_crop_size=(512, 512), use_2d=True, num_workers=8, pin_memory=True, *args, **kwargs):
    input_trans = T.Compose([N.AddNoise(50), G.NdArray2Tensor(use_2d)])
    target_trans = T.Compose([G.NdArray2Tensor(use_2d)])
    ds = ICVLDataset(data_dir='/HDD/Datasets/HSI_denoising/ICVL/Origin/test',
                           crop_size=test_crop_size, 
                           input_transform=input_trans,
                           target_transform=target_trans)
    return DataLoader(dataset=ds, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)



if __name__ == '__main__':
    # # ds = ICVLDataset('/HDD/Datasets/HSI_denoising/ICVL/Origin/train', (32,), input_transform=G.NdArray2Tensor(False))
    # dm = ICVLDataModule(data_dir='/HDD/Datasets/HSI_denoising/ICVL/Origin', 
    #                     train_crop_size=(32,32), 
    #                     val_crop_size=(32,32), 
    #                     test_crop_size=(32,32), 
    #                     train_batch_size=2, 
    #                     val_batch_size=3, 
    #                     test_batch_size=3, 
    #                     input_transform=N.AddNoise(50), 
    #                     target_transform=None, 
    #                     common_transform=None, 
    #                     train_repeat=1) 
    # # x, y = ds[0]['input'], ds[0]['target']
    # # print(id(x))
    # # print(id(y))
    # dm.setup('test')
    # # dl = dm.train_dataloader()
    # # t = dm.icvl_train
    # # print(next(iter(dl))[1].shape)
    # # print(type(t[0][0]))
    # dl = dm.test_dataloader()
    # print(next(iter(dl))[1].shape)
    l1 = get_loader_gaussian_val()
    print(l1.dataset[0][0].shape)
    


    