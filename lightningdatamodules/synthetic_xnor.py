import os 

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Subset
import torch

from datasets.synthetic_xnor import SyntheticXNOR

class DataModule_SyntheticXNOR(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size: int = 128,
        n_samples: int = 100000,
        dims_modality: list[int] = [128, 128, 128],
        n_bits: int = 1,
        embed_mode: str = "xnor_only",
        bc_corr_exclusive: bool = False,
        bc_corr_p: float = None,
        bc_corr_split: float = 0.5,
        p_flips: list[float] = [0.10, 0.10],
        p_corrs: list[float] = [0.00, 0.20, 0.20],
        corr_modes: str = "swap_signal",
        signal_scale: float = 3.0,
        distractor_std: float = 1.0,
        a_rule: str = "xor",
        seed: int = 420,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.n_samples = n_samples
        self.dims_modality = dims_modality
        self.n_bits = int(n_bits)
        self.embed_mode = str(embed_mode)
        self.bc_corr_exclusive = bool(bc_corr_exclusive)
        self.bc_corr_p = bc_corr_p
        self.bc_corr_split = float(bc_corr_split)
        self.p_flips = p_flips
        self.p_corrs = p_corrs
        self.corr_modes = corr_modes
        self.signal_scale = signal_scale
        self.distractor_std = distractor_std
        self.a_rule = a_rule
        self.seed = seed
        self.num_workers = 4

    def setup(self, stage):
        self.ds = SyntheticXNOR(
            n_samples=self.n_samples,
            dims_modality=self.dims_modality,
            n_bits=self.n_bits,
            embed_mode=self.embed_mode,
            bc_corr_exclusive=self.bc_corr_exclusive,
            bc_corr_p=self.bc_corr_p,
            bc_corr_split=self.bc_corr_split,
            p_flips=self.p_flips,
            p_corrs=self.p_corrs,
            corr_modes=self.corr_modes,
            signal_scale=self.signal_scale,
            distractor_std=self.distractor_std,
            a_rule=self.a_rule,
            seed=self.seed,
        )

        g = torch.Generator().manual_seed(self.seed)
        n = len(self.ds)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)
        n_test  = n - n_train - n_val

        train_ds, val_ds, test_ds = random_split(self.ds, [n_train, n_val, n_test], generator=g)
        self.ds_train, self.ds_val, self.ds_test = train_ds, val_ds, test_ds

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
