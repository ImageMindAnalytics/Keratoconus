from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import SimpleITK as sitk
import nrrd
import os
import sys
import torch

from lightning.pytorch.core import LightningDataModule

import monai
from monai.transforms import (    
    LoadImage,
    LoadImaged
)
from monai.data import ITKReader

from transforms import TrainTransform, EvalTransform

class ImgSegDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img", seg_column="seg"):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column        
        self.seg_column = seg_column

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        seg_path = os.path.join(self.mount_point, self.df.iloc[idx][self.seg_column])

        reader = ITKReader()
        img, _ = reader.get_data(reader.read(img_path))

        # img = sitk.ReadImage(img_path)
        img_t = torch.tensor(img)

        reader = ITKReader()
        seg, _ = reader.get_data(reader.read(seg_path))
        seg_t = torch.tensor(seg).squeeze().to(torch.int64)
        
        img_d = {self.img_column: img_t, self.seg_column: seg_t}
        
        if self.transform:
            return self.transform(img_d)
        return img_d

class ImgSegDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.df_train = pd.read_csv(self.hparams.csv_train)
        self.df_val = pd.read_csv(self.hparams.csv_valid)
        self.df_test = pd.read_csv(self.hparams.csv_test)
        
        self.mount_point = self.hparams.mount_point
        self.batch_size = self.hparams.batch_size
        self.num_workers = self.hparams.num_workers
        self.img_column = self.hparams.img_column
        self.seg_column = self.hparams.seg_column
        self.drop_last = bool(self.hparams.drop_last)

        self.train_transform = TrainTransform(img_key=self.hparams.img_column, seg_key=self.hparams.seg_column)
        self.valid_transform = EvalTransform()
        self.test_transform = EvalTransform()

    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("ImgSegDataModule")
        
        group.add_argument('--batch_size', type=int, default=32)
        group.add_argument('--num_workers', type=int, default=6)
        group.add_argument('--img_column', type=str, default="img")
        group.add_argument('--seg_column', type=str, default="seg")
        group.add_argument('--csv_train', type=str, default=None, required=True)
        group.add_argument('--csv_valid', type=str, default=None, required=True)
        group.add_argument('--csv_test', type=str, default=None, required=True)
        group.add_argument('--mount_point', type=str, default="./")
        group.add_argument('--drop_last', type=int, default=0)

        return parent_parser

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = ImgSegDataset(self.df_train, self.mount_point, img_column=self.img_column, seg_column=self.seg_column, transform=self.train_transform)
        self.val_ds = ImgSegDataset(self.df_val, self.mount_point, img_column=self.img_column, seg_column=self.seg_column, transform=self.valid_transform)
        self.test_ds = ImgSegDataset(self.df_test, self.mount_point, img_column=self.img_column, seg_column=self.seg_column, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)



