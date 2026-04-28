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
        numerics_trend_schema_path = self.root_dir / "metadata" / "numerics_trend_schema.json"
        waveforms_fixedbins_schema_path = self.root_dir / "metadata" / "waveforms_fixedbins_schema.json"
        self.numerics_trend_schema = json.loads(numerics_trend_schema_path.read_text()) if numerics_trend_schema_path.exists() else None
        self.waveforms_fixedbins_schema = json.loads(waveforms_fixedbins_schema_path.read_text()) if waveforms_fixedbins_schema_path.exists() else None

        base = pd.read_csv(self.root_dir / "manifests" / f"{split_family}_{split_name}_labels.csv")
        base["CSN"] = base["CSN"].astype("int64")
        base = base[["CSN"]].copy()

        self.use_numerics_trend = (self.root_dir / "manifests" / f"{split_family}_{split_name}_numerics_trend.csv").exists()
        numerics_manifest_name = (
            f"{split_family}_{split_name}_numerics_trend.csv"
            if self.use_numerics_trend
            else f"{split_family}_{split_name}_numerics.csv"
        )

        for modality, manifest_name in [
            ("visits_static", f"{split_family}_{split_name}_visits_static.csv"),
            ("numerics", numerics_manifest_name),
            ("rads", f"{split_family}_{split_name}_rads.csv"),
        ]:
            manifest = pd.read_csv(self.root_dir / "manifests" / manifest_name)
            manifest["CSN"] = manifest["CSN"].astype("int64")
            manifest = manifest[["CSN", "relative_path", "exists"]].rename(
                columns={
                    "relative_path": f"{modality}_path",
                    "exists": f"{modality}_exists",
                }
            )
            base = base.merge(manifest, on="CSN", how="left")

        if use_waveforms:
            fixedbins_manifest_path = self.root_dir / "manifests" / f"{split_family}_{split_name}_waveforms_II_fixedbins.csv"
            raw_manifest_path = self.root_dir / "manifests" / f"{split_family}_{split_name}_waveforms_II.csv"
            self.use_waveform_fixedbins = fixedbins_manifest_path.exists()
            manifest_path = fixedbins_manifest_path if self.use_waveform_fixedbins else raw_manifest_path
            if manifest_path.exists():
                manifest = pd.read_csv(manifest_path)
                manifest["CSN"] = manifest["CSN"].astype("int64")
                manifest = manifest[["CSN", "relative_path", "exists"]].rename(
                    columns={
                        "relative_path": "waveforms_II_path",
                        "exists": "waveforms_II_exists",
                    }
                )
                base = base.merge(manifest, on="CSN", how="left")
        else:
            self.use_waveform_fixedbins = False

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

        if self.use_numerics_trend:
            num_bins = int(self.numerics_trend_schema["num_bins"]) if self.numerics_trend_schema is not None else 32
            num_features = int(len(self.numerics_trend_schema["feature_names"])) if self.numerics_trend_schema is not None else 7
            sample["numerics"] = {
                "trend_values": np.zeros((num_bins, 12, num_features), dtype=np.float32),
                "measure_mask": np.zeros((num_bins, 12), dtype=bool),
                "bin_counts": np.zeros((num_bins,), dtype=np.int32),
                "bin_centers_minutes": np.zeros((num_bins,), dtype=np.float32),
                "bin_edges_minutes": np.zeros((num_bins + 1,), dtype=np.float32),
            }
            if pd.notna(row.get("numerics_path")) and bool(row.get("numerics_exists", 0)):
                with np.load(self.root_dir / row["numerics_path"], allow_pickle=False) as data:
                    sample["numerics"] = {
                        "trend_values": data["trend_values"].astype(np.float32),
                        "measure_mask": data["measure_mask"].astype(bool),
                        "bin_counts": data["bin_counts"].astype(np.int32),
                        "bin_centers_minutes": data["bin_centers_minutes"].astype(np.float32),
                        "bin_edges_minutes": data["bin_edges_minutes"].astype(np.float32),
                    }
        elif pd.notna(row.get("numerics_path")) and bool(row.get("numerics_exists", 0)):
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
            waveform_num_bins = int(self.waveforms_fixedbins_schema["num_bins"]) if self.waveforms_fixedbins_schema is not None else 32
            sample["waveforms_II"] = {
                "windows": np.zeros((waveform_num_bins, 5000, 1), dtype=np.float32),
                "bin_mask": np.zeros((waveform_num_bins,), dtype=bool),
                "sampling_rate": np.empty((0,), dtype=np.float32),
                "bin_centers_seconds": np.zeros((waveform_num_bins,), dtype=np.float32),
                "bin_edges_seconds": np.zeros((waveform_num_bins + 1,), dtype=np.float32),
                "windows_per_bin": np.zeros((waveform_num_bins,), dtype=np.int32),
            }

            path_col = "waveforms_II_path"
            exists_col = "waveforms_II_exists"
            if path_col in row.index and pd.notna(row.get(path_col)) and bool(row.get(exists_col, 0)):
                with np.load(self.root_dir / row[path_col], allow_pickle=False) as data:
                    windows = data["windows"].astype(np.float32)
                    sampling_rate = data["sampling_rate"].astype(np.float32)
                    if self.max_waveform_windows > 0 and not self.use_waveform_fixedbins and windows.shape[0] > self.max_waveform_windows:
                        windows = windows[: self.max_waveform_windows]

                    if not np.isnan(windows).all():
                        fs = float(sampling_rate[0]) if sampling_rate.size else 0.0
                        for window_idx in range(windows.shape[0]):
                            for channel_idx in range(windows.shape[2]):
                                signal = windows[window_idx, :, channel_idx]
                                if np.isnan(signal).all():
                                    continue
                                windows[window_idx, :, channel_idx] = highpass_filter_ecg(signal, fs)

                    if self.use_waveform_fixedbins:
                        sample["waveforms_II"] = {
                            "windows": windows,
                            "bin_mask": data["bin_mask"].astype(bool),
                            "sampling_rate": sampling_rate,
                            "bin_centers_seconds": data["bin_centers_seconds"].astype(np.float32),
                            "bin_edges_seconds": data["bin_edges_seconds"].astype(np.float32),
                            "windows_per_bin": data["windows_per_bin"].astype(np.int32),
                        }
                    else:
                        sample["waveforms_II"] = {
                            "windows": windows,
                            "start_seconds": data["start_seconds"].astype(np.float32),
                            "segment_ids": data["segment_ids"].astype(np.int64),
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
            "categorical_codes": torch.tensor(np.stack([sample["visits_static"]["categorical_codes"] for sample in batch]), dtype=torch.long),
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

    bsz = len(batch)
    if "trend_values" in batch[0]["numerics"]:
        collated["numerics"] = {
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
            "bin_centers_minutes": torch.tensor(
                np.stack([sample["numerics"]["bin_centers_minutes"] for sample in batch]),
                dtype=torch.float32,
            ),
            "bin_edges_minutes": torch.tensor(
                np.stack([sample["numerics"]["bin_edges_minutes"] for sample in batch]),
                dtype=torch.float32,
            ),
        }
    else:
        max_t = max(sample["numerics"]["values"].shape[0] for sample in batch)
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
        if "bin_mask" in batch[0]["waveforms_II"]:
            collated["waveforms_II"] = {
                "windows": torch.tensor(
                    np.stack([sample["waveforms_II"]["windows"] for sample in batch]),
                    dtype=torch.float32,
                ),
                "bin_mask": torch.tensor(
                    np.stack([sample["waveforms_II"]["bin_mask"] for sample in batch]),
                    dtype=torch.bool,
                ),
                "sampling_rate": torch.tensor(
                    np.stack([
                        sample["waveforms_II"]["sampling_rate"]
                        if sample["waveforms_II"]["sampling_rate"].shape[0] > 0
                        else np.array([np.nan], dtype=np.float32)
                        for sample in batch
                    ]),
                    dtype=torch.float32,
                ),
                "bin_centers_seconds": torch.tensor(
                    np.stack([sample["waveforms_II"]["bin_centers_seconds"] for sample in batch]),
                    dtype=torch.float32,
                ),
                "bin_edges_seconds": torch.tensor(
                    np.stack([sample["waveforms_II"]["bin_edges_seconds"] for sample in batch]),
                    dtype=torch.float32,
                ),
                "windows_per_bin": torch.tensor(
                    np.stack([sample["waveforms_II"]["windows_per_bin"] for sample in batch]),
                    dtype=torch.long,
                ),
            }
        else:
            max_n = max(sample["waveforms_II"]["windows"].shape[0] for sample in batch)
            collated["waveforms_II"] = {
                "windows": torch.full((bsz, max_n, 5000, 1), float("nan"), dtype=torch.float32),
                "start_seconds": torch.full((bsz, max_n), float("nan"), dtype=torch.float32),
                "segment_ids": torch.full((bsz, max_n), float("nan"), dtype=torch.float32),
                "sampling_rate": torch.full((bsz, 1), float("nan"), dtype=torch.float32),
                "lengths": torch.tensor([sample["waveforms_II"]["windows"].shape[0] for sample in batch], dtype=torch.long),
            }
            for i, sample in enumerate(batch):
                n = sample["waveforms_II"]["windows"].shape[0]
                collated["waveforms_II"]["windows"][i, :n] = torch.tensor(sample["waveforms_II"]["windows"], dtype=torch.float32)
                collated["waveforms_II"]["start_seconds"][i, :n] = torch.tensor(sample["waveforms_II"]["start_seconds"], dtype=torch.float32)
                collated["waveforms_II"]["segment_ids"][i, :n] = torch.tensor(sample["waveforms_II"]["segment_ids"], dtype=torch.float32)
                if sample["waveforms_II"]["sampling_rate"].shape[0] > 0:
                    collated["waveforms_II"]["sampling_rate"][i] = torch.tensor(sample["waveforms_II"]["sampling_rate"], dtype=torch.float32)

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
