from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MCMEDDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path = "/sc-resources/dh-mimic/mimic_symile/mcmed_aws/data_preprocessed",
        data_dir: str | Path = None,
        split_family: str = "chrono",
        split_name: str = "train",
        use_waveforms: bool = True,
        max_waveform_windows: int = 0,
        max_rad_reports: int = 0,
        radiology_embedding_dim: int = 768,
        require_all_modalities: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split_family = str(split_family)
        self.split_name = str(split_name)
        self.use_waveforms = bool(use_waveforms)
        self.max_waveform_windows = int(max_waveform_windows)
        self.max_rad_reports = int(max_rad_reports)
        self.radiology_embedding_dim = int(radiology_embedding_dim)
        self.require_all_modalities = bool(require_all_modalities)

        if data_dir is not None:
            # data_dir="/sc-resources/dh-mimic/mimic_symile/mcmed_aws/data_preprocessed/future_rads_cv/fold_0/train",
            self.data_dir = Path(data_dir + f"{split_name}")
        else:
            self.data_dir = self.root_dir / "aggregated_memmap" / f"{self.split_family}_{self.split_name}"
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Aggregated MC-MED split directory not found: {self.data_dir}")

        metadata_path = self.data_dir / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

        self.csn = np.load(self.data_dir / "csn.npy", mmap_mode="r")

        self.numeric_trend_values = np.load(self.data_dir / "numeric_trend_values.npy", mmap_mode="r")
        self.numeric_measure_mask = np.load(self.data_dir / "numeric_measure_mask.npy", mmap_mode="r")
        self.numeric_exists = np.load(self.data_dir / "numeric_exists.npy", mmap_mode="r")

        self.rad_visit_embedding = np.load(self.data_dir / "rad_report_embeddings.npy", mmap_mode="r")
        self.rad_report_mask = np.load(self.data_dir / "rad_report_mask.npy", mmap_mode="r")
        self.rad_exists = np.load(self.data_dir / "rad_exists.npy", mmap_mode="r")

        if self.use_waveforms:
            self.ii_windows = np.load(self.data_dir / "II_windows.npy", mmap_mode="r")
            self.ii_bin_mask = np.load(self.data_dir / "II_window_mask.npy", mmap_mode="r")
            self.ii_exists = np.load(self.data_dir / "II_exists.npy", mmap_mode="r")
        else:
            self.ii_windows = None
            self.ii_bin_mask = None
            self.ii_exists = None

        self.total_samples = int(self.csn.shape[0])

        for name, arr in [
            ("numeric_trend_values", self.numeric_trend_values),
            ("numeric_measure_mask", self.numeric_measure_mask),
            ("numeric_exists", self.numeric_exists),
            ("rad_visit_embedding", self.rad_visit_embedding),
            ("rad_report_mask", self.rad_report_mask),
            ("rad_exists", self.rad_exists),
        ]:
            if int(arr.shape[0]) != self.total_samples:
                raise ValueError(f"{name} has {arr.shape[0]} rows, expected {self.total_samples}.")

        if self.use_waveforms:
            for name, arr in [
                ("II_windows", self.ii_windows),
                ("II_bin_mask", self.ii_bin_mask),
                ("II_exists", self.ii_exists),
            ]:
                if int(arr.shape[0]) != self.total_samples:
                    raise ValueError(f"{name} has {arr.shape[0]} rows, expected {self.total_samples}.")

        if int(self.rad_visit_embedding.shape[-1]) != self.radiology_embedding_dim:
            raise ValueError(
                f"rad_visit_embedding has dim {self.rad_visit_embedding.shape[-1]}, "
                f"expected {self.radiology_embedding_dim}."
            )

        if self.require_all_modalities:
            keep = self.numeric_exists.astype(bool) & self.rad_exists.astype(bool)
            if self.use_waveforms:
                keep = keep & self.ii_exists.astype(bool)
            self.indices = np.flatnonzero(keep)
        else:
            self.indices = np.arange(self.total_samples)
        
        # self.indices = self.indices[:1000]

        self.num_samples = int(len(self.indices))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        idx = int(self.indices[idx])
        csn = int(self.csn[idx])

        trend_values = np.asarray(self.numeric_trend_values[idx], dtype=np.float32).copy()
        measure_mask = np.asarray(self.numeric_measure_mask[idx], dtype=bool).copy()
        numeric_exists = bool(self.numeric_exists[idx])

        if not numeric_exists:
            trend_values = np.full_like(trend_values, np.nan, dtype=np.float32)
            measure_mask = np.zeros_like(measure_mask, dtype=bool)
        else:
            trend_values[~measure_mask] = np.nan

        visit_embedding = np.asarray(self.rad_visit_embedding[idx], dtype=np.float32).copy()
        report_mask = np.asarray(self.rad_report_mask[idx], dtype=bool).copy()
        rad_exists = bool(self.rad_exists[idx])

        if self.max_rad_reports > 0:
            visit_embedding = visit_embedding[: self.max_rad_reports]
            report_mask = report_mask[: self.max_rad_reports]

        if not rad_exists:
            visit_embedding = np.full_like(visit_embedding, np.nan, dtype=np.float32)
            report_mask = np.zeros_like(report_mask, dtype=bool)
        else:
            visit_embedding[~report_mask] = np.nan

        sample = {
            "CSN": csn,
            "numerics": {
                "trend_values": trend_values,
                "measure_mask": measure_mask,
                "bin_counts": measure_mask.sum(axis=1).astype(np.int32),
            },
            "rads": {
                "visit_embedding": visit_embedding,
                "embedding_present": rad_exists,
                "report_mask": report_mask,
            },
            "labels": {},
        }

        if self.use_waveforms:
            windows = np.asarray(self.ii_windows[idx], dtype=np.float32).copy()
            bin_mask = np.asarray(self.ii_bin_mask[idx], dtype=bool).copy()
            waveform_exists = bool(self.ii_exists[idx])

            if self.max_waveform_windows > 0:
                windows = windows[: self.max_waveform_windows]
                bin_mask = bin_mask[: self.max_waveform_windows]

            if not waveform_exists:
                windows = np.full_like(windows, np.nan, dtype=np.float32)
                bin_mask = np.zeros_like(bin_mask, dtype=bool)
            else:
                windows[~bin_mask] = np.nan

            sample["waveforms_II"] = {
                "windows": windows,
                "bin_mask": bin_mask,
            }

        return sample


def mcmed_collate_fn(batch: list[dict]) -> dict:
    collated = {
        "CSN": torch.tensor([sample["CSN"] for sample in batch], dtype=torch.long),
        "numerics": {
            "trend_values": torch.tensor(
                np.stack([sample["numerics"]["trend_values"] for sample in batch]),
                dtype=torch.float32,
            ),
            "measure_mask": torch.tensor(
                np.stack([sample["numerics"]["measure_mask"] for sample in batch]),
                dtype=torch.bool,
            ),
            "bin_counts": torch.tensor(
                np.stack([sample["numerics"]["bin_counts"] for sample in batch]),
                dtype=torch.long,
            ),
        },
        "rads": {
            "visit_embedding": torch.tensor(
                np.stack([sample["rads"]["visit_embedding"] for sample in batch]),
                dtype=torch.float32,
            ),
            "embedding_present": torch.tensor(
                [sample["rads"]["embedding_present"] for sample in batch],
                dtype=torch.bool,
            ),
            "report_mask": torch.tensor(
                np.stack([sample["rads"]["report_mask"] for sample in batch]),
                dtype=torch.bool,
            ),
        },
        "labels": {},
    }

    if "waveforms_II" in batch[0]:
        collated["waveforms_II"] = {
            "windows": torch.tensor(
                np.stack([sample["waveforms_II"]["windows"] for sample in batch]),
                dtype=torch.float32,
            ),
            "bin_mask": torch.tensor(
                np.stack([sample["waveforms_II"]["bin_mask"] for sample in batch]),
                dtype=torch.bool,
            ),
        }

    return collated