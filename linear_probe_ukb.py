import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import __version__ as SKLEARN_VERSION
from hydra import compose, initialize
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression, SGDClassifier
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


OBJECTIVES = ("clip", "symile", "triangle", "gram", "symile_attention")
DEFAULT_CV_SPLITS_DIR = Path("/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits")


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
        "--run-config",
        type=Path,
        default=None,
        help=(
            "Optional YAML/JSON run config. Use this when different checkpoint variants "
            "need different Hydra overrides."
        ),
    )
    parser.add_argument(
        "--checkpoint-template",
        type=str,
        default=None,
        help=(
            "Optional checkpoint path template for multi-objective runs. "
            "Available fields: {objective}, {split_nr}."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default=None,
        help="Objective/modelname config group to use for this checkpoint.",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=OBJECTIVES,
        default=None,
        help="Run multiple objectives. Requires --checkpoint-template.",
    )
    parser.add_argument(
        "--split-nr",
        type=int,
        choices=range(5),
        metavar="{0,1,2,3,4}",
        default=None,
        help="UKB CV split number. Sets cfg.split_nr and datamodule.splits.",
    )
    parser.add_argument(
        "--cv-splits-dir",
        type=Path,
        default=DEFAULT_CV_SPLITS_DIR,
        help="Directory containing splits_0.yaml ... splits_4.yaml.",
    )
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
        "--probe-type",
        choices=["logreg_liblinear", "logreg_saga", "sgd_logreg"],
        default="logreg_saga",
        help=(
            "Linear probe backend. 'sgd_logreg' is stochastic logistic regression "
            "and is much faster for dense high-dimensional embeddings."
        ),
    )
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[1e-3, 1e-2, 1e-1, 1.0],
        help="Grid of inverse regularization strengths for LogisticRegression.",
    )
    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=[1e-5, 1e-4, 1e-3],
        help="Grid of L2 regularization strengths for SGDClassifier.",
    )
    parser.add_argument(
        "--class-weight",
        nargs="+",
        default=["balanced"],
        help="Grid of class_weight settings. Use 'none' and/or 'balanced'.",
    )
    parser.add_argument(
        "--probe-max-iter",
        type=int,
        default=1000,
        help="Maximum iterations/epochs for the sklearn linear probe.",
    )
    parser.add_argument(
        "--probe-tol",
        type=float,
        default=1e-3,
        help="Optimization tolerance for the sklearn linear probe.",
    )
    parser.add_argument(
        "--probe-n-jobs",
        type=int,
        default=-1,
        help="Number of CPU workers for sklearn probe backends that support it.",
    )
    parser.add_argument("--seed", type=int, default=420, help="Random seed.")
    return parser.parse_args()


def _has_override(overrides: List[str], key: str) -> bool:
    return any(o.startswith(f"{key}=") or o.startswith(f"+{key}=") or o.startswith(f"++{key}=") for o in overrides)


def _is_group_override(override: str, key: str) -> bool:
    return override.startswith(f"{key}=") or override.startswith(f"+{key}=") or override.startswith(f"++{key}=")


def _format_checkpoint_template(
    template: str,
    objective: str,
    split_nr: Optional[int],
    run_name: Optional[str] = None,
) -> Path:
    if split_nr is None and "{split_nr}" in template:
        raise ValueError("--checkpoint-template contains {split_nr}, so --split-nr must be set.")
    return Path(template.format(objective=objective, split_nr=split_nr, run_name=run_name or objective))


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _copy_args(args: argparse.Namespace) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    copied.override = list(args.override)
    return copied


def load_run_config(path: Path) -> Dict[str, object]:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in run config: {path}")
    return config


def build_runs_from_config(args: argparse.Namespace) -> List[Tuple[str, argparse.Namespace, str, Path]]:
    config = load_run_config(args.run_config)
    raw_runs = config.get("runs", config.get("jobs"))
    if not raw_runs:
        raise ValueError(f"Run config must define a non-empty 'runs' list: {args.run_config}")

    common_overrides = _as_list(config.get("common_overrides")) + _as_list(config.get("overrides"))
    default_checkpoint_template = config.get("checkpoint_template", args.checkpoint_template)
    default_split_nr = config.get("split_nr", args.split_nr)

    runs = []
    for idx, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, dict):
            raise ValueError(f"Run config entry {idx} must be a mapping.")

        run_args = _copy_args(args)
        run_args.objectives = None
        run_args.run_config = None
        run_args.override = list(args.override) + common_overrides + _as_list(raw_run.get("overrides"))
        run_args.split_nr = raw_run.get("split_nr", default_split_nr)
        run_args.config_name = raw_run.get("config_name", args.config_name)
        run_args.batch_size = raw_run.get("batch_size", args.batch_size)
        run_args.num_workers = raw_run.get("num_workers", args.num_workers)
        run_args.label_plugin = raw_run.get("label_plugin", args.label_plugin)
        run_args.label_column = raw_run.get("label_column", args.label_column)
        run_args.selection_metric = raw_run.get("selection_metric", args.selection_metric)
        run_args.probe_type = raw_run.get("probe_type", args.probe_type)
        run_args.c_values = _as_list(raw_run.get("c_values")) or args.c_values
        run_args.alpha_values = _as_list(raw_run.get("alpha_values")) or args.alpha_values
        run_args.class_weight = _as_list(raw_run.get("class_weight")) or args.class_weight
        run_args.probe_max_iter = raw_run.get("probe_max_iter", args.probe_max_iter)
        run_args.probe_tol = raw_run.get("probe_tol", args.probe_tol)
        run_args.probe_n_jobs = raw_run.get("probe_n_jobs", args.probe_n_jobs)

        objective = raw_run.get("objective", args.objective)
        if objective is None:
            raise ValueError(f"Run config entry {idx} is missing 'objective'.")
        if objective not in OBJECTIVES:
            raise ValueError(f"Unsupported objective '{objective}' in run config entry {idx}.")
        run_args.objective = objective

        run_name = str(raw_run.get("name", f"{objective}_{idx}"))
        checkpoint = raw_run.get("checkpoint")
        checkpoint_template = raw_run.get("checkpoint_template", default_checkpoint_template)
        if checkpoint is not None:
            checkpoint_path = _format_checkpoint_template(
                str(checkpoint),
                objective=objective,
                split_nr=run_args.split_nr,
                run_name=run_name,
            )
        elif checkpoint_template is not None:
            checkpoint_path = _format_checkpoint_template(
                str(checkpoint_template),
                objective=objective,
                split_nr=run_args.split_nr,
                run_name=run_name,
            )
        else:
            checkpoint_path = args.checkpoint

        runs.append((run_name, run_args, objective, checkpoint_path))

    return runs


def load_cfg(args: argparse.Namespace, objective: Optional[str] = None):
    overrides = list(args.override)
    if not _has_override(overrides, "dataset_name"):
        overrides.append("dataset_name=ukb")
    if not _has_override(overrides, "encoders"):
        overrides.append("encoders=ukb")
    requested_objective = objective if objective is not None else args.objective
    if requested_objective is not None:
        overrides = [o for o in overrides if not _is_group_override(o, "modelname")]
        overrides.append(f"modelname={requested_objective}")
    if args.split_nr is not None:
        split_path = args.cv_splits_dir / f"splits_{args.split_nr}.yaml"
        if not split_path.exists():
            raise FileNotFoundError(f"UKB split file does not exist: {split_path}")
        overrides = [o for o in overrides if not _is_group_override(o, "split_nr")]
        overrides.append(f"split_nr={args.split_nr}")
        if not _has_override(overrides, "datamodule.splits"):
            overrides.append(f"datamodule.splits={split_path}")
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


def eids_to_numpy(eids) -> np.ndarray:
    if torch.is_tensor(eids):
        return eids.detach().cpu().numpy()
    return np.asarray(eids)


def build_concat_features(
    embeddings: List[torch.Tensor],
    normalize_embeddings: bool,
) -> torch.Tensor:
    if normalize_embeddings:
        embeddings = [F.normalize(emb, dim=-1) for emb in embeddings]
    return torch.cat(embeddings, dim=-1)


def add_feature_chunk(
    features_by_representation: Dict[str, List[np.ndarray]],
    representation_name: str,
    features: torch.Tensor,
) -> None:
    features_by_representation.setdefault(representation_name, []).append(
        features.detach().cpu().numpy().astype(np.float32, copy=False)
    )


@torch.no_grad()
def extract_split_representations(
    model,
    datamodule,
    split_name: str,
    label_plugin: str,
    label_column_idx: int,
    normalize_embeddings: bool,
    modality_names: List[str],
    device: torch.device,
) -> Dict[str, Dict[str, np.ndarray]]:
    dataloader = get_split_loader(datamodule, split_name)
    features_by_representation: Dict[str, List[np.ndarray]] = {}
    labels, eids = [], []

    for batch in tqdm(dataloader, desc=f"Extracting {split_name}", leave=True):
        batch = move_batch_to_device(batch, device)
        model_output = model(batch)
        embeddings = model_output["embeddings"]
        if normalize_embeddings:
            embeddings = [F.normalize(emb, dim=-1) for emb in embeddings]

        for modality_name, embedding in zip(modality_names, embeddings):
            add_feature_chunk(features_by_representation, f"unimodal_{modality_name}", embedding)

        concat_features = build_concat_features(
            embeddings=embeddings,
            normalize_embeddings=False,
        )
        add_feature_chunk(features_by_representation, "multimodal_concat", concat_features)

        transformer_flat = model_output.get("transformer_flat")
        if transformer_flat is not None:
            add_feature_chunk(features_by_representation, "after_transformer_flat", transformer_flat)

        batch_labels = batch[label_plugin]["tabular_data"][:, label_column_idx]
        labels.append(batch_labels.detach().cpu().numpy().astype(np.float32, copy=False))
        eids.append(eids_to_numpy(batch["eids"]))

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


def build_probe(
    probe_type: str,
    C: Optional[float],
    alpha: Optional[float],
    class_weight: str,
    seed: int,
    max_iter: int,
    tol: float,
    n_jobs: int,
) -> Pipeline:
    sklearn_class_weight = None if class_weight == "none" else class_weight
    if probe_type == "sgd_logreg":
        clf = SGDClassifier(
            loss=get_sgd_logistic_loss_name(),
            penalty="l2",
            alpha=float(alpha),
            class_weight=sklearn_class_weight,
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
            n_jobs=n_jobs,
            average=True,
        )
    elif probe_type in ("logreg_liblinear", "logreg_saga"):
        solver = "liblinear" if probe_type == "logreg_liblinear" else "saga"
        clf_kwargs = {
            "C": float(C),
            "class_weight": sklearn_class_weight,
            "penalty": "l2",
            "solver": solver,
            "max_iter": max_iter,
            "tol": tol,
            "random_state": seed,
        }
        if solver == "saga":
            clf_kwargs["n_jobs"] = n_jobs
        clf = LogisticRegression(**clf_kwargs)
    else:
        raise ValueError(f"Unsupported probe_type: {probe_type}")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def get_sgd_logistic_loss_name() -> str:
    version = SKLEARN_VERSION.split(".")
    try:
        major, minor = int(version[0]), int(version[1])
    except (IndexError, ValueError):
        return "log_loss"
    return "log_loss" if (major, minor) >= (1, 1) else "log"


def get_probe_scores(probe: Pipeline, X: np.ndarray) -> np.ndarray:
    clf = probe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return probe.predict_proba(X)[:, 1]
    return probe.decision_function(X)


def select_best_probe(
    train_split: Dict[str, np.ndarray],
    valid_split: Dict[str, np.ndarray],
    probe_type: str,
    c_values: Iterable[float],
    alpha_values: Iterable[float],
    class_weights: Iterable[str],
    selection_metric: str,
    seed: int,
    max_iter: int,
    tol: float,
    n_jobs: int,
) -> Tuple[Pipeline, Dict[str, object], Dict[str, float]]:
    best_probe = None
    best_params = None
    best_valid_metrics = None
    best_score = -np.inf

    y_train = train_split["y"].astype(int)
    y_valid = valid_split["y"].astype(int)
    if probe_type == "sgd_logreg":
        candidate_settings = [
            {"alpha": float(alpha), "C": None, "class_weight": class_weight}
            for alpha in alpha_values
            for class_weight in class_weights
        ]
    else:
        candidate_settings = [
            {"C": float(C), "alpha": None, "class_weight": class_weight}
            for C in c_values
            for class_weight in class_weights
        ]

    for params in tqdm(candidate_settings, desc=f"Tuning {probe_type} probe", leave=True):
        probe = build_probe(
            probe_type=probe_type,
            C=params["C"],
            alpha=params["alpha"],
            class_weight=params["class_weight"],
            seed=seed,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
        )
        probe.fit(train_split["X"], y_train)
        valid_scores = get_probe_scores(probe, valid_split["X"])
        valid_metrics = compute_metrics(y_valid, valid_scores)
        score = valid_metrics[selection_metric]
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_probe = probe
            best_params = {"probe_type": probe_type, "class_weight": params["class_weight"]}
            if params["C"] is not None:
                best_params["C"] = float(params["C"])
            if params["alpha"] is not None:
                best_params["alpha"] = float(params["alpha"])
            best_valid_metrics = valid_metrics

    if best_probe is None:
        raise RuntimeError("Failed to select a linear probe. Check that the validation split has both classes.")

    return best_probe, best_params, best_valid_metrics


def maybe_save_embeddings(
    save_dir: Path,
    run_name: str,
    split_nr: Optional[int],
    split_name: str,
    representation_name: str,
    split_data: Dict[str, np.ndarray],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = run_name if split_nr is None else f"{run_name}_split{split_nr}"
    np.savez_compressed(
        save_dir / f"{prefix}_{split_name}_{representation_name}_embeddings.npz",
        X=split_data["X"],
        y=split_data["y"],
        eids=split_data["eids"],
    )


def run_probe(
    args: argparse.Namespace,
    objective: Optional[str],
    checkpoint: Path,
    device: torch.device,
    run_name: Optional[str] = None,
) -> Dict[str, object]:
    cfg = load_cfg(args, objective=objective)
    objective_name = str(cfg.modelname.modelname)
    run_name = run_name or objective_name
    split_nr = int(cfg.split_nr) if args.split_nr is not None else None
    split_file = Path(str(cfg.datamodule.splits))
    modality_names = list(cfg.encoders.modalities)

    datamodule = build_datamodule(cfg)
    datamodule.setup(stage="fit")
    label_column_idx = get_label_column_index(datamodule, args.label_plugin, args.label_column)

    model = load_model(cfg, checkpoint, device)
    normalize_embeddings = bool(cfg.modelname.embedding_norm)

    split_data = {}
    for split_name in ("train", "valid", "test"):
        print(f"[{objective_name}] Starting {split_name} extraction on {device} with {split_file}...")
        split_data[split_name] = extract_split_representations(
            model=model,
            datamodule=datamodule,
            split_name=split_name,
            label_plugin=args.label_plugin,
            label_column_idx=label_column_idx,
            normalize_embeddings=normalize_embeddings,
            modality_names=modality_names,
            device=device,
        )
        if args.save_embeddings_dir is not None:
            for representation_name, representation_data in split_data[split_name].items():
                maybe_save_embeddings(
                    args.save_embeddings_dir,
                    run_name,
                    split_nr,
                    split_name,
                    representation_name,
                    representation_data,
                )
        print(f"[{objective_name}] Finished {split_name} extraction.")

    representation_results = {}
    for representation_name in split_data["train"].keys():
        if representation_name not in split_data["valid"] or representation_name not in split_data["test"]:
            continue

        print(f"[{objective_name}] Training linear probe for {representation_name}...")
        best_probe, best_params, valid_metrics = select_best_probe(
            train_split=split_data["train"][representation_name],
            valid_split=split_data["valid"][representation_name],
            probe_type=args.probe_type,
            c_values=args.c_values,
            alpha_values=args.alpha_values,
            class_weights=args.class_weight,
            selection_metric=args.selection_metric,
            seed=args.seed,
            max_iter=args.probe_max_iter,
            tol=args.probe_tol,
            n_jobs=args.probe_n_jobs,
        )

        test_scores = get_probe_scores(best_probe, split_data["test"][representation_name]["X"])
        test_metrics = compute_metrics(split_data["test"][representation_name]["y"].astype(int), test_scores)
        representation_results[representation_name] = {
            "best_params": best_params,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "embedding_dim": int(split_data["train"][representation_name]["X"].shape[1]),
        }
        print(f"[{objective_name}] Finished linear probe for {representation_name}.")

    return {
        "run_name": run_name,
        "objective": objective_name,
        "checkpoint": str(checkpoint),
        "split_nr": split_nr,
        "split_file": str(split_file),
        "hydra_overrides": list(args.override),
        "probe_type": args.probe_type,
        "c_values": [float(value) for value in args.c_values],
        "alpha_values": [float(value) for value in args.alpha_values],
        "class_weight": list(args.class_weight),
        "selection_metric": args.selection_metric,
        "label_plugin": args.label_plugin,
        "label_column": args.label_column,
        "modalities": modality_names,
        "representations": sorted(representation_results.keys()),
        "n_train": int(next(iter(split_data["train"].values()))["X"].shape[0]),
        "n_valid": int(next(iter(split_data["valid"].values()))["X"].shape[0]),
        "n_test": int(next(iter(split_data["test"].values()))["X"].shape[0]),
        "results_by_representation": representation_results,
    }


def _summary_stats(values: List[float]) -> Dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "stderr": float("nan"),
            "values": [],
        }
    mean = float(np.nanmean(arr))
    if n > 1:
        std = float(np.nanstd(arr, ddof=1))
        stderr = float(std / np.sqrt(n))
    else:
        std = 0.0
        stderr = 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "values": [float(v) for v in arr.tolist()],
    }


def aggregate_results_over_splits(run_results: List[Dict[str, object]]) -> Dict[str, object]:
    """
    Aggregate linear probe metrics over splits.

    Grouping key: (run_name, objective). This supports comparing variants
    like `clip` vs `clip_gp` while also handling per-objective grouping.
    """
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for result in run_results:
        run_name = str(result.get("run_name", "unknown_run"))
        objective = str(result.get("objective", "unknown_objective"))
        grouped.setdefault((run_name, objective), []).append(result)

    aggregated: Dict[str, object] = {}
    for (run_name, objective), items in grouped.items():
        split_nrs = sorted(
            [
                int(item["split_nr"])
                for item in items
                if item.get("split_nr") is not None
            ]
        )
        reps = sorted(
            {
                rep_name
                for item in items
                for rep_name in item.get("results_by_representation", {}).keys()
            }
        )

        by_representation: Dict[str, object] = {}
        for rep_name in reps:
            valid_metric_values: Dict[str, List[float]] = {}
            test_metric_values: Dict[str, List[float]] = {}
            embedding_dims: List[int] = []

            for item in items:
                rep_dict = item.get("results_by_representation", {})
                if rep_name not in rep_dict:
                    continue
                rep_result = rep_dict[rep_name]
                if "embedding_dim" in rep_result:
                    embedding_dims.append(int(rep_result["embedding_dim"]))

                for metric_name, metric_val in rep_result.get("valid_metrics", {}).items():
                    valid_metric_values.setdefault(metric_name, []).append(float(metric_val))
                for metric_name, metric_val in rep_result.get("test_metrics", {}).items():
                    test_metric_values.setdefault(metric_name, []).append(float(metric_val))

            by_representation[rep_name] = {
                "n_splits": len(items),
                "embedding_dim_unique": sorted(set(embedding_dims)),
                "valid_metrics": {
                    metric_name: _summary_stats(metric_vals)
                    for metric_name, metric_vals in sorted(valid_metric_values.items())
                },
                "test_metrics": {
                    metric_name: _summary_stats(metric_vals)
                    for metric_name, metric_vals in sorted(test_metric_values.items())
                },
            }

        key = f"{run_name}__{objective}"
        aggregated[key] = {
            "run_name": run_name,
            "objective": objective,
            "n_splits": len(items),
            "split_nrs": split_nrs,
            "representations": by_representation,
        }

    return aggregated


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    device = get_device()

    if args.run_config is None and args.objectives is not None and args.checkpoint_template is None:
        raise ValueError("--objectives requires --checkpoint-template.")

    if args.run_config is not None:
        run_results = {}
        run_results_list = []
        for run_name, run_args, objective, checkpoint in build_runs_from_config(args):
            run_result = run_probe(
                run_args,
                objective=objective,
                checkpoint=checkpoint,
                device=device,
                run_name=run_name,
            )
            run_results_list.append(run_result)

            run_key = run_name
            if run_key in run_results:
                split_suffix = run_result.get("split_nr")
                if split_suffix is not None:
                    run_key = f"{run_name}_split{split_suffix}"
                else:
                    idx = 1
                    while f"{run_name}_{idx}" in run_results:
                        idx += 1
                    run_key = f"{run_name}_{idx}"
            run_results[run_key] = run_result

        results = {
            "run_config": str(args.run_config),
            "split_nr": args.split_nr,
            "results_by_run": run_results,
            "split_aggregation": aggregate_results_over_splits(run_results_list),
        }
    elif args.objectives is None:
        checkpoint = (
            _format_checkpoint_template(args.checkpoint_template, args.objective, args.split_nr)
            if args.checkpoint_template is not None and args.objective is not None
            else args.checkpoint
        )
        results = run_probe(args, objective=args.objective, checkpoint=checkpoint, device=device)
    else:
        objective_results = {}
        for objective in args.objectives:
            checkpoint = _format_checkpoint_template(args.checkpoint_template, objective, args.split_nr)
            objective_results[objective] = run_probe(args, objective=objective, checkpoint=checkpoint, device=device)
        results = {
            "split_nr": args.split_nr,
            "split_file": str(args.cv_splits_dir / f"splits_{args.split_nr}.yaml") if args.split_nr is not None else None,
            "objectives": list(args.objectives),
            "results_by_objective": objective_results,
        }

    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
