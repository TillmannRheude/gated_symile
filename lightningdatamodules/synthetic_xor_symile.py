import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from datasets.synthetic_xor_symile import SymileBinaryXORDataset


class DataModule_SymileXOR(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 128,
        n_samples: int = 16000,
        p: float = 1.0,
        value_mode: str = "zero_one",
        dim: int = 5,
        signal_scale: float = 1.0,
        distractor_std: float = 0.0,
        seed: int = 420,
    ):
        super().__init__()
        self.batch_size = int(batch_size)
        self.n_samples = int(n_samples)
        self.p = float(p)
        self.value_mode = str(value_mode)
        self.dim = int(dim)
        self.signal_scale = float(signal_scale)
        self.distractor_std = float(distractor_std)
        self.seed = int(seed)
        self.num_workers = 4

    def setup(self, stage=None):
        self.ds = SymileBinaryXORDataset(
            n_samples=self.n_samples,
            p=self.p,
            seed=self.seed,
            value_mode=self.value_mode,
            dim=self.dim,
            signal_scale=self.signal_scale,
            distractor_std=self.distractor_std,
        )

        g = torch.Generator().manual_seed(self.seed)
        n = len(self.ds)
        if n == 16000:
            n_train, n_val, n_test = 10000, 1000, 5000
        else:
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            n_test = n - n_train - n_val
        train_ds, val_ds, test_ds = random_split(self.ds, [n_train, n_val, n_test], generator=g)
        self.ds_train, self.ds_val, self.ds_test = train_ds, val_ds, test_ds

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )
