import os 

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.symile_mimic import Dataset_SymileMimic

class DataModule_SymileMimic(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size: int = 128,
        split_nr: int = 1,
        permute: bool = False,
        permute_p: float = 1.0,
        permute_mod: list[str] = None,
        permute_seed: int = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.permute = permute
        self.permute_p = permute_p
        self.permute_mod = permute_mod
        self.permute_seed = permute_seed
        self.num_workers = 4
        self.split_nr = split_nr

    def setup(self, stage):
        self.ds_train = Dataset_SymileMimic(
            split="train",
            split_nr=self.split_nr,
            permute=self.permute,
            permute_p=self.permute_p,
            permute_mod=self.permute_mod,
            permute_seed=self.permute_seed,
        )
        self.ds_val = Dataset_SymileMimic(
            split="val",
            split_nr=self.split_nr,
            permute=self.permute,
            permute_p=self.permute_p,
            permute_mod=self.permute_mod,
            permute_seed=self.permute_seed,
        )
        self.ds_test = Dataset_SymileMimic(
            split="test",
            split_nr=self.split_nr,
            permute=self.permute,
            permute_p=self.permute_p,
            permute_mod=self.permute_mod,
            permute_seed=self.permute_seed,
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
