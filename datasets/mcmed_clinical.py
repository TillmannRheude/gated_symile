from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MCMEDClinicalDataset(Dataset):
    """
    Dataset for multitask shared demographics memmaps:
      data_preprocessed/aggregated_memmap_multitask_shared_demographics/<split_family>_<split_name>

    Modalities:
      - numerics
      - demographics
      - rads_future: future-window radiology tokens (retrieval target side)
      - rads_input: pre-target radiology tokens (causal input side for probes)
      - rads: alias selected by `rad_for_main` ("future" or "input")
    """

    def __init__(
        self,
        root_dir: str | Path = "/sc-resources/dh-mimic/mimic_symile/mcmed_aws/data_preprocessed",
        data_dir: str | Path | None = None,
        split_family: str = "chrono",
        split_name: str = "train",
        max_rad_reports: int = 0,
        require_all_modalities: bool = False,
        rad_for_main: str = "future",  # "future" for training/retrieval, "input" for probing
        require_both_rad_streams: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split_family = str(split_family)
        self.split_name = str(split_name)

        self.max_rad_reports = int(max_rad_reports)
        self.require_all_modalities = bool(require_all_modalities)
        self.require_both_rad_streams = bool(require_both_rad_streams)

        if rad_for_main not in {"future", "input"}:
            raise ValueError("rad_for_main must be one of {'future', 'input'}")
        self.rad_for_main = rad_for_main

        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = (
                self.root_dir
                / "aggregated_memmap_multitask_shared_demographics"
                / f"{self.split_family}_{self.split_name}"
            )

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Aggregated MC-MED split directory not found: {self.data_dir}")

        metadata_path = self.data_dir / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

        self.csn = np.load(self.data_dir / "csn.npy", mmap_mode="r")

        # Numerics
        self.numeric_trend_values = np.load(self.data_dir / "numeric_trend_values.npy", mmap_mode="r")
        self.numeric_measure_mask = np.load(self.data_dir / "numeric_measure_mask.npy", mmap_mode="r")
        self.numeric_exists = np.load(self.data_dir / "numeric_exists.npy", mmap_mode="r")

        # Demographics
        self.demographics_continuous = np.load(self.data_dir / "demographics_continuous.npy", mmap_mode="r")
        self.demographics_continuous_mask = np.load(self.data_dir / "demographics_continuous_mask.npy", mmap_mode="r")
        self.demographics_categorical = np.load(self.data_dir / "demographics_categorical.npy", mmap_mode="r")
        self.demographics_categorical_mask = np.load(self.data_dir / "demographics_categorical_mask.npy", mmap_mode="r")
        self.demographics_exists = np.load(self.data_dir / "demographics_exists.npy", mmap_mode="r")

        # Future radiology stream (training/retrieval target side)
        self.rad_future_input_ids = np.load(self.data_dir / "rad_input_ids.npy", mmap_mode="r")
        self.rad_future_attention_mask = np.load(self.data_dir / "rad_attention_mask.npy", mmap_mode="r")
        self.rad_future_report_mask = np.load(self.data_dir / "rad_report_mask.npy", mmap_mode="r")
        self.rad_future_exists = np.load(self.data_dir / "rad_exists.npy", mmap_mode="r")

        # Input radiology stream (causal probing side)
        self.rad_input_input_ids = np.load(self.data_dir / "rad_context_input_ids.npy", mmap_mode="r")
        self.rad_input_attention_mask = np.load(self.data_dir / "rad_context_attention_mask.npy", mmap_mode="r")
        self.rad_input_report_mask = np.load(self.data_dir / "rad_context_report_mask.npy", mmap_mode="r")
        self.rad_input_exists = np.load(self.data_dir / "rad_context_exists.npy", mmap_mode="r")

        self.total_samples = int(self.csn.shape[0])

        for name, arr in [
            ("numeric_trend_values", self.numeric_trend_values),
            ("numeric_measure_mask", self.numeric_measure_mask),
            ("numeric_exists", self.numeric_exists),
            ("demographics_continuous", self.demographics_continuous),
            ("demographics_continuous_mask", self.demographics_continuous_mask),
            ("demographics_categorical", self.demographics_categorical),
            ("demographics_categorical_mask", self.demographics_categorical_mask),
            ("demographics_exists", self.demographics_exists),
            ("rad_future_input_ids", self.rad_future_input_ids),
            ("rad_future_attention_mask", self.rad_future_attention_mask),
            ("rad_future_report_mask", self.rad_future_report_mask),
            ("rad_future_exists", self.rad_future_exists),
            ("rad_input_input_ids", self.rad_input_input_ids),
            ("rad_input_attention_mask", self.rad_input_attention_mask),
            ("rad_input_report_mask", self.rad_input_report_mask),
            ("rad_input_exists", self.rad_input_exists),
        ]:
            if int(arr.shape[0]) != self.total_samples:
                raise ValueError(f"{name} has {arr.shape[0]} rows, expected {self.total_samples}.")

        main_rad_exists = (
            self.rad_future_exists.astype(bool) if self.rad_for_main == "future" else self.rad_input_exists.astype(bool)
        )

        if self.require_all_modalities:
            keep = (
                self.numeric_exists.astype(bool)
                & self.demographics_exists.astype(bool)
                & main_rad_exists
            )
            if self.require_both_rad_streams:
                keep = keep & self.rad_future_exists.astype(bool) & self.rad_input_exists.astype(bool)
            self.indices = np.flatnonzero(keep)
        else:
            self.indices = np.arange(self.total_samples)

        self.num_samples = int(len(self.indices))

    def __len__(self) -> int:
        return self.num_samples

    def _slice_rad(self, ids, attn, rep_mask, exists):
        ids = np.asarray(ids, dtype=np.int64).copy()
        attn = np.asarray(attn, dtype=bool).copy()
        rep_mask = np.asarray(rep_mask, dtype=bool).copy()
        exists = bool(exists)

        if self.max_rad_reports > 0:
            ids = ids[: self.max_rad_reports]
            attn = attn[: self.max_rad_reports]
            rep_mask = rep_mask[: self.max_rad_reports]

        if not exists:
            ids[:] = 0
            attn[:] = False
            rep_mask[:] = False
        else:
            ids[~rep_mask] = 0
            attn[~rep_mask] = False

        return ids, attn, rep_mask, exists

    def __getitem__(self, idx: int) -> dict:
        idx = int(self.indices[idx])
        csn = int(self.csn[idx])

        # Numerics
        trend_values = np.asarray(self.numeric_trend_values[idx], dtype=np.float32).copy()
        measure_mask = np.asarray(self.numeric_measure_mask[idx], dtype=bool).copy()
        numeric_exists = bool(self.numeric_exists[idx])

        if not numeric_exists:
            trend_values = np.full_like(trend_values, np.nan, dtype=np.float32)
            measure_mask = np.zeros_like(measure_mask, dtype=bool)
        else:
            trend_values[~measure_mask] = np.nan

        # Demographics
        demo_cont = np.asarray(self.demographics_continuous[idx], dtype=np.float32).copy()
        demo_cont_mask = np.asarray(self.demographics_continuous_mask[idx], dtype=bool).copy()
        demo_cat = np.asarray(self.demographics_categorical[idx], dtype=np.int64).copy()
        demo_cat_mask = np.asarray(self.demographics_categorical_mask[idx], dtype=bool).copy()
        demo_exists = bool(self.demographics_exists[idx])

        if not demo_exists:
            demo_cont[:] = 0.0
            demo_cont_mask[:] = False
            demo_cat[:] = 0
            demo_cat_mask[:] = False
        else:
            demo_cont[~demo_cont_mask] = 0.0
            demo_cat[~demo_cat_mask] = 0

        # Future rad stream
        rf_ids, rf_attn, rf_mask, rf_exists = self._slice_rad(
            self.rad_future_input_ids[idx],
            self.rad_future_attention_mask[idx],
            self.rad_future_report_mask[idx],
            self.rad_future_exists[idx],
        )

        # Input rad stream
        ri_ids, ri_attn, ri_mask, ri_exists = self._slice_rad(
            self.rad_input_input_ids[idx],
            self.rad_input_attention_mask[idx],
            self.rad_input_report_mask[idx],
            self.rad_input_exists[idx],
        )

        # Main rad alias for convenience
        if self.rad_for_main == "future":
            main_ids, main_attn, main_mask, main_exists = rf_ids, rf_attn, rf_mask, rf_exists
        else:
            main_ids, main_attn, main_mask, main_exists = ri_ids, ri_attn, ri_mask, ri_exists

        sample = {
            "CSN": csn,
            "numerics": {
                "trend_values": trend_values,
                "measure_mask": measure_mask,
                "bin_counts": measure_mask.sum(axis=1).astype(np.int32),
                "embedding_present": numeric_exists,
            },
            "demographics": {
                "continuous": demo_cont,
                "continuous_mask": demo_cont_mask,
                "categorical": demo_cat,
                "categorical_mask": demo_cat_mask,
                "embedding_present": demo_exists,
            },
            # Main alias selected by rad_for_main
            "rads": {
                "input_ids": main_ids,
                "attention_mask": main_attn,
                "report_mask": main_mask,
                "embedding_present": main_exists,
                "rad_view": self.rad_for_main,
            },
            # Explicit streams
            "rads_future": {
                "input_ids": rf_ids,
                "attention_mask": rf_attn,
                "report_mask": rf_mask,
                "embedding_present": rf_exists,
            },
            "rads_input": {
                "input_ids": ri_ids,
                "attention_mask": ri_attn,
                "report_mask": ri_mask,
                "embedding_present": ri_exists,
            },
            "labels": {},
        }
        return sample


def mcmed_clinical_collate_fn(batch: list[dict]) -> dict:
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
            "embedding_present": torch.tensor(
                [sample["numerics"]["embedding_present"] for sample in batch],
                dtype=torch.bool,
            ),
        },
        "demographics": {
            "continuous": torch.tensor(
                np.stack([sample["demographics"]["continuous"] for sample in batch]),
                dtype=torch.float32,
            ),
            "continuous_mask": torch.tensor(
                np.stack([sample["demographics"]["continuous_mask"] for sample in batch]),
                dtype=torch.bool,
            ),
            "categorical": torch.tensor(
                np.stack([sample["demographics"]["categorical"] for sample in batch]),
                dtype=torch.long,
            ),
            "categorical_mask": torch.tensor(
                np.stack([sample["demographics"]["categorical_mask"] for sample in batch]),
                dtype=torch.bool,
            ),
            "embedding_present": torch.tensor(
                [sample["demographics"]["embedding_present"] for sample in batch],
                dtype=torch.bool,
            ),
        },
        # Main selected rad view
        "rads": {
            "input_ids": torch.tensor(
                np.stack([sample["rads"]["input_ids"] for sample in batch]),
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                np.stack([sample["rads"]["attention_mask"] for sample in batch]),
                dtype=torch.bool,
            ),
            "report_mask": torch.tensor(
                np.stack([sample["rads"]["report_mask"] for sample in batch]),
                dtype=torch.bool,
            ),
            "embedding_present": torch.tensor(
                [sample["rads"]["embedding_present"] for sample in batch],
                dtype=torch.bool,
            ),
            "rad_view": batch[0]["rads"]["rad_view"],
        },
        "rads_future": {
            "input_ids": torch.tensor(
                np.stack([sample["rads_future"]["input_ids"] for sample in batch]),
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                np.stack([sample["rads_future"]["attention_mask"] for sample in batch]),
                dtype=torch.bool,
            ),
            "report_mask": torch.tensor(
                np.stack([sample["rads_future"]["report_mask"] for sample in batch]),
                dtype=torch.bool,
            ),
            "embedding_present": torch.tensor(
                [sample["rads_future"]["embedding_present"] for sample in batch],
                dtype=torch.bool,
            ),
        },
        "rads_input": {
            "input_ids": torch.tensor(
                np.stack([sample["rads_input"]["input_ids"] for sample in batch]),
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                np.stack([sample["rads_input"]["attention_mask"] for sample in batch]),
                dtype=torch.bool,
            ),
            "report_mask": torch.tensor(
                np.stack([sample["rads_input"]["report_mask"] for sample in batch]),
                dtype=torch.bool,
            ),
            "embedding_present": torch.tensor(
                [sample["rads_input"]["embedding_present"] for sample in batch],
                dtype=torch.bool,
            ),
        },
        # Flat aliases for convenience:
        "rad_input_ids": torch.tensor(
            np.stack([sample["rads"]["input_ids"] for sample in batch]),
            dtype=torch.long,
        ),
        "rad_attention_mask": torch.tensor(
            np.stack([sample["rads"]["attention_mask"] for sample in batch]),
            dtype=torch.bool,
        ),
        "rad_report_mask": torch.tensor(
            np.stack([sample["rads"]["report_mask"] for sample in batch]),
            dtype=torch.bool,
        ),
        "rad_future_input_ids": torch.tensor(
            np.stack([sample["rads_future"]["input_ids"] for sample in batch]),
            dtype=torch.long,
        ),
        "rad_future_attention_mask": torch.tensor(
            np.stack([sample["rads_future"]["attention_mask"] for sample in batch]),
            dtype=torch.bool,
        ),
        "rad_future_report_mask": torch.tensor(
            np.stack([sample["rads_future"]["report_mask"] for sample in batch]),
            dtype=torch.bool,
        ),
        "rad_context_input_ids": torch.tensor(
            np.stack([sample["rads_input"]["input_ids"] for sample in batch]),
            dtype=torch.long,
        ),
        "rad_context_attention_mask": torch.tensor(
            np.stack([sample["rads_input"]["attention_mask"] for sample in batch]),
            dtype=torch.bool,
        ),
        "rad_context_report_mask": torch.tensor(
            np.stack([sample["rads_input"]["report_mask"] for sample in batch]),
            dtype=torch.bool,
        ),
        "demographics_continuous": torch.tensor(
            np.stack([sample["demographics"]["continuous"] for sample in batch]),
            dtype=torch.float32,
        ),
        "demographics_continuous_mask": torch.tensor(
            np.stack([sample["demographics"]["continuous_mask"] for sample in batch]),
            dtype=torch.bool,
        ),
        "demographics_categorical": torch.tensor(
            np.stack([sample["demographics"]["categorical"] for sample in batch]),
            dtype=torch.long,
        ),
        "demographics_categorical_mask": torch.tensor(
            np.stack([sample["demographics"]["categorical_mask"] for sample in batch]),
            dtype=torch.bool,
        ),
        "labels": {},
    }
    return collated