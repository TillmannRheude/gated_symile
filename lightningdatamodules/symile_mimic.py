import os 

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.symile_mimic import Dataset_SymileMimic

class DataModule_SymileMimic(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size: int = 128,
        split_nr: int = 1,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = 4
        self.split_nr = split_nr

    def setup(self, stage):
        self.ds_train = Dataset_SymileMimic(
            split="train",
            split_nr=self.split_nr,
        )
        self.ds_val = Dataset_SymileMimic(
            split="val",
            split_nr=self.split_nr,
        )
        self.ds_test = Dataset_SymileMimic(
            split="test",
            split_nr=self.split_nr,
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train, batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val, batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test, batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
