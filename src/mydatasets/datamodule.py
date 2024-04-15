import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from mydatasets.dataset_offset import HMSOffsetDataset
from torch.utils.data import DataLoader


class HMSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        train_args: dict,
        val_args: dict,
        batch_size: int,
        batch_size_val: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        if batch_size_val == -1:
            self.batch_size_val = batch_size
        else:
            self.batch_size_val = batch_size_val
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_args = train_args
        self.val_args = val_args

    def setup(self, stage=None):
        self.train_dataset = eval(self.dataset_name)(**self.train_args)
        self.val_dataset = eval(self.dataset_name)(**self.val_args)

    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_val,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return loader
