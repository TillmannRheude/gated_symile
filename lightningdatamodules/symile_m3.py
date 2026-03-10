import os 

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.symile_m3 import Dataset_SymileM3

class DataModule_SymileM3(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size: int = 128,
        split_nr: int = 1,
        text_model_id: str = "bert-base-uncased",
        num_langs: int = 5,
    ):
        super().__init__()
        self.batch_size = batch_size

        self.num_workers = len(os.sched_getaffinity(0))
        self.split_nr = split_nr
        self.text_model_id = text_model_id
        self.num_langs = num_langs

    def setup(self, stage):
        self.ds_train = Dataset_SymileM3(split="train", split_nr=self.split_nr, text_model_id=self.text_model_id, num_langs=self.num_langs)
        self.ds_val = Dataset_SymileM3(split="val", split_nr=self.split_nr, text_model_id=self.text_model_id, num_langs=self.num_langs)
        self.ds_test = Dataset_SymileM3(split="test", split_nr=self.split_nr, text_model_id=self.text_model_id, num_langs=self.num_langs)

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

