import os 
from functools import partial

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.mcmed import MCMEDDataset, mcmed_collate_fn

class DataModule_MCMED(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size: int = 128,
        split_nr: int = 1,
        max_waveform_windows: int = 0,
        radiology_model_params: dict = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = 4
        self.split_nr = split_nr
        self.max_waveform_windows = max_waveform_windows
        self.radiology_model_params = radiology_model_params or {}

    def setup(self, stage):
        self.ds_train = MCMEDDataset(
            split_family="chrono",
            split_name="train",
            max_waveform_windows=self.max_waveform_windows,
            #split_nr=self.split_nr,
        )
        self.ds_val = MCMEDDataset(
            split_family="chrono",
            split_name="val",
            max_waveform_windows=self.max_waveform_windows,
            #split_nr=self.split_nr,
        )
        self.ds_test = MCMEDDataset(
            split_family="chrono",
            split_name="test",
            max_waveform_windows=self.max_waveform_windows,
            #split_nr=self.split_nr,
        )

    def train_dataloader(self):
        collate = partial(
            mcmed_collate_fn,
            text_model_id=self.radiology_model_params.get("text_model_id", "bert-base-uncased"),
            text_max_length=int(self.radiology_model_params.get("max_length", 256)),
            text_cache_dir=self.radiology_model_params.get("cache_dir", "/sc-projects/sc-proj-ukb-cvd/projects/data/tmp_hf_cache"),
            text_local_files_only=bool(self.radiology_model_params.get("local_files_only", False)),
        )
        return DataLoader(
            self.ds_train, batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=collate,
        )

    def val_dataloader(self):
        collate = partial(
            mcmed_collate_fn,
            text_model_id=self.radiology_model_params.get("text_model_id", "bert-base-uncased"),
            text_max_length=int(self.radiology_model_params.get("max_length", 256)),
            text_cache_dir=self.radiology_model_params.get("cache_dir", "/sc-projects/sc-proj-ukb-cvd/projects/data/tmp_hf_cache"),
            text_local_files_only=bool(self.radiology_model_params.get("local_files_only", False)),
        )
        return DataLoader(
            self.ds_val, batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate,
        )

    def test_dataloader(self):
        collate = partial(
            mcmed_collate_fn,
            text_model_id=self.radiology_model_params.get("text_model_id", "bert-base-uncased"),
            text_max_length=int(self.radiology_model_params.get("max_length", 256)),
            text_cache_dir=self.radiology_model_params.get("cache_dir", "/sc-projects/sc-proj-ukb-cvd/projects/data/tmp_hf_cache"),
            text_local_files_only=bool(self.radiology_model_params.get("local_files_only", False)),
        )
        return DataLoader(
            self.ds_test, batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate,
        )
