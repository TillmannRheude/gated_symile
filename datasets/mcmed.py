from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset
from transformers import BertTokenizer


_TOKENIZER_CACHE: dict[tuple[str, str | None, bool], BertTokenizer] = {}


def highpass_filter_ecg(signal: np.ndarray, sampling_rate: float, cutoff_hz: float = 0.5, order: int = 3) -> np.ndarray:
    if signal.ndim != 1 or len(signal) < (order * 3 + 1) or sampling_rate <= 0:
        return signal

    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_hz / nyquist
    if normalized_cutoff <= 0 or normalized_cutoff >= 1:
        return signal

    b, a = butter(order, normalized_cutoff, btype="highpass")
    return filtfilt(b, a, signal).astype(np.float32)


def _normalize_batch_texts(batch_texts) -> list[str]:
    normalized = []
    for sample_texts in batch_texts:
        if isinstance(sample_texts, str):
            text = sample_texts.strip()
        elif isinstance(sample_texts, (list, tuple)):
            text = " ".join(str(t).strip() for t in sample_texts if str(t).strip())
        else:
            text = ""
        normalized.append(text if text else "[NO_IMPRESSION]")
    return normalized


def _get_bert_tokenizer(text_model_id: str, cache_dir: str | None, local_files_only: bool) -> BertTokenizer:
    key = (text_model_id, cache_dir, local_files_only)
    tokenizer = _TOKENIZER_CACHE.get(key)
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(
            text_model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        _TOKENIZER_CACHE[key] = tokenizer
    return tokenizer


class MCMEDDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path = "/sc-resources/dh-mimic/mimic_symile/mcmed_aws/data_preprocessed",
        split_family: str = "chrono",
        split_name: str = "train",
        use_waveforms: bool = True,
        max_waveform_windows: int = 0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.use_waveforms = use_waveforms
        self.max_waveform_windows = int(max_waveform_windows)
        self.visits_numeric_columns = [
            "Age",
            "Triage_Temp",
            "Triage_HR",
            "Triage_RR",
            "Triage_SpO2",
            "Triage_SBP",
            "Triage_DBP",
        ]
        self.numerics_columns = [
            "HR",
            "RR",
            "SpO2",
            "SBP",
            "DBP",
            "MAP",
            "Temp",
            "Perf",
            "Pain",
            "LPM_O2",
            "1min_HRV",
            "5min_HRV",
        ]
        self.visits_stats = json.loads(
            (self.root_dir / "metadata" / f"visits_static_stats_{split_family}.json").read_text()
        )
        self.numerics_stats = json.loads(
            (self.root_dir / "metadata" / f"numerics_stats_{split_family}.json").read_text()
        )

        base = pd.read_csv(self.root_dir / "manifests" / f"{split_family}_{split_name}_labels.csv")
        base["CSN"] = base["CSN"].astype("int64")
        base = base[["CSN"]].copy()

        for modality in ["visits_static", "numerics", "rads"]:
            manifest = pd.read_csv(self.root_dir / "manifests" / f"{split_family}_{split_name}_{modality}.csv")
            manifest["CSN"] = manifest["CSN"].astype("int64")
            manifest = manifest[["CSN", "relative_path", "exists"]].rename(
                columns={
                    "relative_path": f"{modality}_path",
                    "exists": f"{modality}_exists",
                }
            )
            base = base.merge(manifest, on="CSN", how="left")

        if use_waveforms:
            for waveform in ["II", "Pleth", "Resp"]:
                manifest_path = self.root_dir / "manifests" / f"{split_family}_{split_name}_waveforms_{waveform}.csv"
                if manifest_path.exists():
                    manifest = pd.read_csv(manifest_path)
                    manifest["CSN"] = manifest["CSN"].astype("int64")
                    manifest = manifest[["CSN", "relative_path", "exists"]].rename(
                        columns={
                            "relative_path": f"waveforms_{waveform}_path",
                            "exists": f"waveforms_{waveform}_exists",
                        }
                    )
                    base = base.merge(manifest, on="CSN", how="left")

        labels_path = self.root_dir / "labels" / "labels.csv"
        labels = pd.read_csv(labels_path)
        labels["CSN"] = labels["CSN"].astype("int64")
        base = base.merge(labels, on="CSN", how="left")

        self.df = base.sort_values("CSN").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        sample = {
            "CSN": int(row["CSN"]),
            "visits_static": {
                "numeric_values": np.full((7,), np.nan, dtype=np.float32),
                "numeric_mask": np.zeros((7,), dtype=bool),
                "categorical_codes": np.full((5,), np.nan, dtype=np.float32),
                "categorical_mask": np.zeros((5,), dtype=bool),
                "chief_complaint": "",
            },
            "numerics": {
                "minute_offsets": np.empty((0,), dtype=np.float32),
                "delta_minutes": np.empty((0,), dtype=np.float32),
                "values": np.empty((0, 12), dtype=np.float32),
                "mask": np.empty((0, 12), dtype=bool),
                "source_codes": np.empty((0, 12), dtype=np.int64),
            },
            "rads": {
                "reports": [],
                "impression_texts": [],
                "study_texts": [],
            },
        }

        if pd.notna(row.get("visits_static_path")) and bool(row.get("visits_static_exists", 0)):
            with np.load(self.root_dir / row["visits_static_path"], allow_pickle=False) as data:
                numeric_values = data["numeric_values"].astype(np.float32)
                numeric_mask = data["numeric_mask"].astype(bool)
                for i, column in enumerate(self.visits_numeric_columns):
                    if numeric_mask[i]:
                        mean = np.float32(self.visits_stats[column]["mean"])
                        std = np.float32(self.visits_stats[column]["std"])
                        numeric_values[i] = (numeric_values[i] - mean) / std
                sample["visits_static"] = {
                    "numeric_values": numeric_values,
                    "numeric_mask": numeric_mask,
                    "categorical_codes": data["categorical_codes"].astype(np.int64),
                    "categorical_mask": data["categorical_mask"].astype(bool),
                    "chief_complaint": str(data["chief_complaint"].item()),
                }

        if pd.notna(row.get("numerics_path")) and bool(row.get("numerics_exists", 0)):
            with np.load(self.root_dir / row["numerics_path"], allow_pickle=False) as data:
                values = data["values"].astype(np.float32)
                mask = data["mask"].astype(bool)
                for i, column in enumerate(self.numerics_columns):
                    mean = np.float32(self.numerics_stats[column]["mean"])
                    std = np.float32(self.numerics_stats[column]["std"])
                    values[:, i] = np.where(mask[:, i], (values[:, i] - mean) / std, values[:, i])
                sample["numerics"] = {
                    "minute_offsets": data["minute_offsets"].astype(np.float32),
                    "delta_minutes": data["delta_minutes"].astype(np.float32),
                    "values": values,
                    "mask": mask,
                    "source_codes": data["source_codes"].astype(np.int64),
                }

        if pd.notna(row.get("rads_path")) and bool(row.get("rads_exists", 0)):
            payload = json.loads((self.root_dir / row["rads_path"]).read_text())
            sample["rads"] = {
                "reports": payload.get("reports", []),
                "impression_texts": [report.get("impression", "") for report in payload.get("reports", [])],
                "study_texts": [report.get("study", "") for report in payload.get("reports", [])],
            }

        if self.use_waveforms:
            for waveform, length in [("II", 5000), ("Pleth", 1250), ("Resp", 625)]:
                key = f"waveforms_{waveform}"
                sample[key] = {
                    "windows": np.empty((0, length, 1), dtype=np.float32),
                    "start_seconds": np.empty((0,), dtype=np.float32),
                    "segment_ids": np.empty((0,), dtype=np.int64),
                    "sampling_rate": np.empty((0,), dtype=np.float32),
                }

                path_col = f"{key}_path"
                exists_col = f"{key}_exists"
                if path_col in row.index and pd.notna(row.get(path_col)) and bool(row.get(exists_col, 0)):
                    with np.load(self.root_dir / row[path_col], allow_pickle=False) as data:
                        windows = data["windows"].astype(np.float32)
                        sampling_rate = data["sampling_rate"].astype(np.float32)
                        start_seconds = data["start_seconds"].astype(np.float32)
                        segment_ids = data["segment_ids"].astype(np.int64)

                        if self.max_waveform_windows > 0 and windows.shape[0] > self.max_waveform_windows:
                            windows = windows[: self.max_waveform_windows]
                            start_seconds = start_seconds[: self.max_waveform_windows]
                            segment_ids = segment_ids[: self.max_waveform_windows]

                        if waveform == "II" and not np.isnan(windows).all():
                            fs = float(sampling_rate[0]) if sampling_rate.size else 0.0
                            for window_idx in range(windows.shape[0]):
                                for channel_idx in range(windows.shape[2]):
                                    signal = windows[window_idx, :, channel_idx]
                                    if np.isnan(signal).all():
                                        continue
                                    windows[window_idx, :, channel_idx] = highpass_filter_ecg(signal, fs)

                        sample[key] = {
                            "windows": windows,
                            "start_seconds": start_seconds,
                            "segment_ids": segment_ids,
                            "sampling_rate": sampling_rate,
                        }

        label_columns = [
            col
            for col in self.df.columns
            if col not in {
                "CSN",
                "visits_static_path",
                "visits_static_exists",
                "numerics_path",
                "numerics_exists",
                "rads_path",
                "rads_exists",
                "waveforms_II_path",
                "waveforms_II_exists",
                "waveforms_Pleth_path",
                "waveforms_Pleth_exists",
                "waveforms_Resp_path",
                "waveforms_Resp_exists",
            }
        ]
        # dict_keys(['MRN', 'Visit_no', 'Visits', 'Triage_acuity', 'ED_dispo', 'DC_dispo', 'Dx_ICD9', 'Dx_ICD10', 'Dx_name', 'Hours_to_next_visit', 'Dispo_class_next_visit', 'ED_LOS', 'Hosp_LOS', 'is_admitted', 'is_discharged', 'hospital_to_home', 'has_next_visit', 'revisit_within_72h', 'revisit_within_7d', 'next_visit_inpatient', 'triage_acuity_level', 'dx_icd10_prefix'])
        sample["labels"] = {col: row[col] for col in label_columns}
        return sample


def mcmed_collate_fn(
    batch: list[dict],
    text_model_id: str = "bert-base-uncased",
    text_max_length: int = 256,
    text_cache_dir: str | None = "/sc-projects/sc-proj-ukb-cvd/projects/data/tmp_hf_cache",
    text_local_files_only: bool = False,
) -> dict:
    collated = {
        "CSN": torch.tensor([sample["CSN"] for sample in batch], dtype=torch.long),
        "visits_static": {
            "numeric_values": torch.tensor(np.stack([sample["visits_static"]["numeric_values"] for sample in batch]), dtype=torch.float32),
            "numeric_mask": torch.tensor(np.stack([sample["visits_static"]["numeric_mask"] for sample in batch]), dtype=torch.bool),
            "categorical_codes": torch.tensor(np.stack([sample["visits_static"]["categorical_codes"] for sample in batch]), dtype=torch.float32),
            "categorical_mask": torch.tensor(np.stack([sample["visits_static"]["categorical_mask"] for sample in batch]), dtype=torch.bool),
            "chief_complaint": [sample["visits_static"]["chief_complaint"] for sample in batch],
        },
        "numerics": {},
        "rads": {
            "reports": [sample["rads"]["reports"] for sample in batch],
            "impression_texts": [sample["rads"]["impression_texts"] for sample in batch],
            "study_texts": [sample["rads"]["study_texts"] for sample in batch],
        },
        "labels": {},
    }

    max_t = max(sample["numerics"]["values"].shape[0] for sample in batch)
    bsz = len(batch)
    collated["numerics"]["minute_offsets"] = torch.full((bsz, max_t), float("nan"), dtype=torch.float32)
    collated["numerics"]["delta_minutes"] = torch.full((bsz, max_t), float("nan"), dtype=torch.float32)
    collated["numerics"]["values"] = torch.full((bsz, max_t, 12), float("nan"), dtype=torch.float32)
    collated["numerics"]["mask"] = torch.zeros((bsz, max_t, 12), dtype=torch.bool)
    collated["numerics"]["source_codes"] = torch.full((bsz, max_t, 12), float("nan"), dtype=torch.float32)
    collated["numerics"]["lengths"] = torch.tensor(
        [sample["numerics"]["values"].shape[0] for sample in batch], dtype=torch.long
    )

    for i, sample in enumerate(batch):
        t = sample["numerics"]["values"].shape[0]
        collated["numerics"]["minute_offsets"][i, :t] = torch.tensor(sample["numerics"]["minute_offsets"], dtype=torch.float32)
        collated["numerics"]["delta_minutes"][i, :t] = torch.tensor(sample["numerics"]["delta_minutes"], dtype=torch.float32)
        collated["numerics"]["values"][i, :t] = torch.tensor(sample["numerics"]["values"], dtype=torch.float32)
        collated["numerics"]["mask"][i, :t] = torch.tensor(sample["numerics"]["mask"], dtype=torch.bool)
        collated["numerics"]["source_codes"][i, :t] = torch.tensor(sample["numerics"]["source_codes"], dtype=torch.float32)

    if "waveforms_II" in batch[0]:
        for waveform, length in [("II", 5000), ("Pleth", 1250), ("Resp", 625)]:
            key = f"waveforms_{waveform}"
            max_n = max(sample[key]["windows"].shape[0] for sample in batch)
            collated[key] = {
                "windows": torch.full((bsz, max_n, length, 1), float("nan"), dtype=torch.float32),
                "start_seconds": torch.full((bsz, max_n), float("nan"), dtype=torch.float32),
                "segment_ids": torch.full((bsz, max_n), float("nan"), dtype=torch.float32),
                "sampling_rate": torch.full((bsz, 1), float("nan"), dtype=torch.float32),
                "lengths": torch.tensor([sample[key]["windows"].shape[0] for sample in batch], dtype=torch.long),
            }
            for i, sample in enumerate(batch):
                n = sample[key]["windows"].shape[0]
                collated[key]["windows"][i, :n] = torch.tensor(sample[key]["windows"], dtype=torch.float32)
                collated[key]["start_seconds"][i, :n] = torch.tensor(sample[key]["start_seconds"], dtype=torch.float32)
                collated[key]["segment_ids"][i, :n] = torch.tensor(sample[key]["segment_ids"], dtype=torch.float32)
                if sample[key]["sampling_rate"].shape[0] > 0:
                    collated[key]["sampling_rate"][i] = torch.tensor(sample[key]["sampling_rate"], dtype=torch.float32)

    tokenizer = _get_bert_tokenizer(
        text_model_id=text_model_id,
        cache_dir=text_cache_dir,
        local_files_only=text_local_files_only,
    )
    normalized_impressions = _normalize_batch_texts(collated["rads"]["impression_texts"])
    tokenized_impressions = tokenizer(
        normalized_impressions,
        padding=True,
        truncation=True,
        max_length=text_max_length,
        return_tensors="pt",
    )
    collated["rads"]["tokenized_impression_texts"] = {
        "input_ids": tokenized_impressions["input_ids"],
        "attention_mask": tokenized_impressions["attention_mask"],
    }

    for key in batch[0]["labels"]:
        values = [sample["labels"][key] for sample in batch]
        first_valid = next((value for value in values if not pd.isna(value)), None)
        if first_valid is None:
            collated["labels"][key] = values
        elif isinstance(first_valid, (bool, np.bool_)):
            collated["labels"][key] = torch.tensor(values, dtype=torch.bool)
        elif isinstance(first_valid, (int, np.integer)):
            collated["labels"][key] = torch.tensor(values, dtype=torch.long)
        elif isinstance(first_valid, (float, np.floating)):
            collated["labels"][key] = torch.tensor(values, dtype=torch.float32)
        else:
            collated["labels"][key] = values

    return collated
