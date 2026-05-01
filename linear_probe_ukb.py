import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from helpers import build_datamodule, build_model, set_all_seeds


# Patch TabularPlugin to match the training entrypoints.
try:
    from udm.plugins.tabular.tabular_plugin import TabularPlugin

    if not hasattr(TabularPlugin, "_original_setup"):
        TabularPlugin._original_setup = TabularPlugin._setup

        def patched_setup(self):
            if self.tabular_df is None:
                self.tabular_df = self.load_tabular_df()
            self._original_setup()

        TabularPlugin._setup = patched_setup
except ImportError:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract UKB embeddings from a saved checkpoint and train a sklearn linear probe."
    )
    parser.add_argument("--checkpoint",
        type=Path, 
        default="/sc-projects/sc-proj-ukb-cvd/projects/rhti10/gated_symile/checkpoints/transformer_symile_ukb.ckpt",
        help="Path to the Lightning checkpoint.")
    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Hydra config name under ./config.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override. Repeat this flag for multiple overrides.",
    )
    parser.add_argument(
        "--label-plugin",
        type=str,
        default="labels",
        help="Batch key for the supervised target plugin.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="10y_mortality",
        help="Column inside the label plugin to probe.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["roc_auc", "average_precision", "accuracy", "balanced_accuracy", "f1"],
        default="roc_auc",
        help="Validation metric used to pick the best linear probe.",
    )
    parser.add_argument(
        "--save-embeddings-dir",
        type=Path,
        default=None,
        help="Optional directory to store per-split embeddings as .npz files.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional JSON output path for metrics and chosen hyperparameters.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional dataloader batch size override for extraction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional dataloader num_workers override for extraction.",
    )
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        help="Grid of inverse regularization strengths for LogisticRegression.",
    )
    parser.add_argument(
        "--class-weight",
        nargs="+",
        default=["none", "balanced"],
        help="Grid of class_weight settings. Use 'none' and/or 'balanced'.",
    )
    parser.add_argument("--seed", type=int, default=420, help="Random seed.")
    return parser.parse_args()


def load_cfg(args: argparse.Namespace):
    overrides = list(args.override)
    if not any(o.startswith("dataset_name=") for o in overrides):
        overrides.append("dataset_name=ukb")
    if not any(o.startswith("encoders=") for o in overrides):
        overrides.append("encoders=ukb")
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    if args.batch_size is not None:
        cfg.datamodule.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.datamodule.num_workers = args.num_workers
    return cfg


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but no GPU is available.")
    return torch.device("cuda")


def load_model(cfg, checkpoint_path: Path, device: torch.device):
    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    model = model.to(device)
    model.eval()
    return model


def get_label_column_index(datamodule, label_plugin: str, label_column: str) -> int:
    metadata = datamodule.get_metadata()
    if label_plugin not in metadata:
        raise KeyError(f"Label plugin '{label_plugin}' not found. Available keys: {sorted(metadata.keys())}")
    features = metadata[label_plugin].get("features", [])
    if label_column not in features:
        raise KeyError(
            f"Label column '{label_column}' not found in plugin '{label_plugin}'. Available columns: {features}"
        )
    return int(features.index(label_column))


def get_split_loader(datamodule, split_name: str):
    if split_name == "train":
        return datamodule.train_dataloader(shuffle=False, drop_last=False)
    if split_name == "valid":
        return datamodule.val_dataloader(shuffle=False, drop_last=False)
    if split_name == "test":
        return datamodule.test_dataloader(shuffle=False, drop_last=False)
    raise ValueError(f"Unsupported split: {split_name}")


def move_batch_to_device(batch, device: torch.device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    return batch


def build_concat_features(
    embeddings: List[torch.Tensor],
    normalize_embeddings: bool,
) -> torch.Tensor:
    if normalize_embeddings:
        embeddings = [F.normalize(emb, dim=-1) for emb in embeddings]
    return torch.cat(embeddings, dim=-1)


@torch.no_grad()
def extract_split_representations(
    model,
    datamodule,
    split_name: str,
    label_plugin: str,
    label_column_idx: int,
    normalize_embeddings: bool,
    device: torch.device,
) -> Dict[str, Dict[str, np.ndarray]]:
    dataloader = get_split_loader(datamodule, split_name)
    features_by_representation = {
        "concat": [],
        "transformer_flat": [],
    }
    labels, eids = [], []

    for batch in tqdm(dataloader, desc=f"Extracting {split_name}", leave=True):
        batch = move_batch_to_device(batch, device)
        model_output = model(batch)
        concat_features = build_concat_features(
            embeddings=model_output["embeddings"],
            normalize_embeddings=normalize_embeddings,
        )
        features_by_representation["concat"].append(
            concat_features.detach().cpu().numpy().astype(np.float32, copy=False)
        )

        transformer_flat = model_output.get("transformer_flat")
        if transformer_flat is not None:
            features_by_representation["transformer_flat"].append(
                transformer_flat.detach().cpu().numpy().astype(np.float32, copy=False)
            )

        batch_labels = batch[label_plugin]["tabular_data"][:, label_column_idx]
        labels.append(batch_labels.detach().cpu().numpy().astype(np.float32, copy=False))
        eids.append(np.asarray(batch["eids"]))

    shared_labels = np.concatenate(labels, axis=0)
    shared_eids = np.concatenate(eids, axis=0)

    split_outputs = {}
    for representation_name, feature_chunks in features_by_representation.items():
        if not feature_chunks:
            continue
        split_outputs[representation_name] = {
            "X": np.concatenate(feature_chunks, axis=0),
            "y": shared_labels,
            "eids": shared_eids,
        }
    return split_outputs


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }
    if np.unique(y_true).size < 2:
        metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def build_probe(C: float, class_weight: str, seed: int) -> Pipeline:
    sklearn_class_weight = None if class_weight == "none" else class_weight
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=sklearn_class_weight,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def select_best_probe(
    train_split: Dict[str, np.ndarray],
    valid_split: Dict[str, np.ndarray],
    c_values: Iterable[float],
    class_weights: Iterable[str],
    selection_metric: str,
    seed: int,
) -> Tuple[Pipeline, Dict[str, object], Dict[str, float]]:
    best_probe = None
    best_params = None
    best_valid_metrics = None
    best_score = -np.inf

    y_train = train_split["y"].astype(int)
    y_valid = valid_split["y"].astype(int)
    candidate_settings = [(C, class_weight) for C in c_values for class_weight in class_weights]

    for C, class_weight in tqdm(candidate_settings, desc="Tuning linear probe", leave=True):
        probe = build_probe(C=C, class_weight=class_weight, seed=seed)
        probe.fit(train_split["X"], y_train)
        valid_scores = probe.predict_proba(valid_split["X"])[:, 1]
        valid_metrics = compute_metrics(y_valid, valid_scores)
        score = valid_metrics[selection_metric]
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_probe = probe
            best_params = {"C": float(C), "class_weight": class_weight}
            best_valid_metrics = valid_metrics

    if best_probe is None:
        raise RuntimeError("Failed to select a linear probe. Check that the validation split has both classes.")

    return best_probe, best_params, best_valid_metrics


def maybe_save_embeddings(
    save_dir: Path,
    split_name: str,
    representation_name: str,
    split_data: Dict[str, np.ndarray],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_dir / f"{split_name}_{representation_name}_embeddings.npz",
        X=split_data["X"],
        y=split_data["y"],
        eids=split_data["eids"],
    )


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    set_all_seeds(args.seed)
    device = get_device()

    datamodule = build_datamodule(cfg)
    datamodule.setup(stage="fit")
    label_column_idx = get_label_column_index(datamodule, args.label_plugin, args.label_column)

    model = load_model(cfg, args.checkpoint, device)
    normalize_embeddings = bool(cfg.modelname.embedding_norm)

    split_data = {}
    for split_name in ("train", "valid", "test"):
        print(f"Starting {split_name} extraction on {device}...")
        split_data[split_name] = extract_split_representations(
            model=model,
            datamodule=datamodule,
            split_name=split_name,
            label_plugin=args.label_plugin,
            label_column_idx=label_column_idx,
            normalize_embeddings=normalize_embeddings,
            device=device,
        )
        if args.save_embeddings_dir is not None:
            for representation_name, representation_data in split_data[split_name].items():
                maybe_save_embeddings(
                    args.save_embeddings_dir,
                    split_name,
                    representation_name,
                    representation_data,
                )
        print(f"Finished {split_name} extraction.")

    representation_results = {}
    for representation_name in split_data["train"].keys():
        if representation_name not in split_data["valid"] or representation_name not in split_data["test"]:
            continue

        print(f"Training linear probe for {representation_name}...")
        best_probe, best_params, valid_metrics = select_best_probe(
            train_split=split_data["train"][representation_name],
            valid_split=split_data["valid"][representation_name],
            c_values=args.c_values,
            class_weights=args.class_weight,
            selection_metric=args.selection_metric,
            seed=args.seed,
        )

        test_scores = best_probe.predict_proba(split_data["test"][representation_name]["X"])[:, 1]
        test_metrics = compute_metrics(split_data["test"][representation_name]["y"].astype(int), test_scores)
        representation_results[representation_name] = {
            "best_params": best_params,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "embedding_dim": int(split_data["train"][representation_name]["X"].shape[1]),
        }
        print(f"Finished linear probe for {representation_name}.")

    results = {
        "checkpoint": str(args.checkpoint),
        "selection_metric": args.selection_metric,
        "label_plugin": args.label_plugin,
        "label_column": args.label_column,
        "representations": sorted(representation_results.keys()),
        "n_train": int(next(iter(split_data["train"].values()))["X"].shape[0]),
        "n_valid": int(next(iter(split_data["valid"].values()))["X"].shape[0]),
        "n_test": int(next(iter(split_data["test"].values()))["X"].shape[0]),
        "results_by_representation": representation_results,
    }

    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
