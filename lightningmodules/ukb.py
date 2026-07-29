import torch
import torch.distributed as dist
import wandb
from torchmetrics.functional.classification import binary_auroc, binary_average_precision

from lightningmodules.utils import LightningModuleParent
from losses.retrieval import zeroshot_retrieval_logits
from architecture import ModalityAttentionGate


class UKBModel(LightningModuleParent):
    def __init__(
        self,
        model,
        candidate_idx: int = 0, 
        params_retrival_ds: dict = {
            "batch_size": 128, 
            "split_nr": 1,
        },
        modalities: list = ["nmr", "ehr", "olink"],
        eval_corruption: dict = None,
        missing_strategy: str = "zero_mask",
        modality_dropout_rate: float = 0.0,
        modality_dropout_scope: str = "query",
        **args,
    ):
        super().__init__(**args)
        self.dataset_name = "ukb"
        self.model = model
        self.params_retrival_ds = params_retrival_ds
        self.modalities = modalities
        self.candidate_idx = candidate_idx
        self.emb_dim = int(getattr(self.model.encoders[self.candidate_idx], "emb_dim"))
        self.eval_corruption = dict(eval_corruption or {})
        self.missing_strategy = str(missing_strategy)
        self.modality_dropout_rate = float(modality_dropout_rate)
        self.modality_dropout_scope = str(modality_dropout_scope)
        self._train_feature_means = None

        self.val_step_accuracies = []
        self.test_step_accuracies = []
        self._corruption_epoch_records = {"val": [], "test": []}
        self.register_buffer(
            "_suppression_thresholds",
            torch.full((len(self.modalities),), float("nan"), dtype=torch.float32),
            persistent=True,
        )

        # gate
        self.use_gate = self.params_method["use_gate"]
        if self.use_gate:
            self.gate = ModalityAttentionGate(
                num_modalities=len(self.modalities),
                emb_dim=self.emb_dim,
                d_k=self.params_method["gate_d_k"],
                temperature_init=self.params_method["gate_temp"],
                gate_bias_init=self.params_method["gate_bias_init"],
                gate_strength_init=self.params_method["gate_strength_init"],
                gate_type=self.params_method["gate_type"],
                gate_mode=self.params_method["gate_mode"],
                neutral_type=self.params_method["neutral_type"],
                use_null=self.params_method["use_null"],
                renormalize=self.params_method["renormalize"]
            )
        else:
            self.gate = None

        self.save_hyperparameters()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Older checkpoints predate validation-calibrated suppression thresholds.
        threshold_key = prefix + "_suppression_thresholds"
        if threshold_key not in state_dict:
            state_dict[threshold_key] = self._suppression_thresholds.clone()
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _modality_present(x: torch.Tensor) -> torch.Tensor:
        """
        Treat a modality as "missing" for a sample if ALL features are NaN.
        Returns bool mask of shape (B,).
        """
        if x.dim() == 1:
            return ~torch.isnan(x)
        # works for tabular (B,D) and MRI (B,C,...) alike
        B = x.shape[0]
        return ~torch.isnan(x).reshape(B, -1).all(dim=1)

    def on_fit_start(self):
        if self.missing_strategy == "mean":
            self._ensure_train_feature_means()

    def _ensure_train_feature_means(self) -> None:
        if self._train_feature_means is not None:
            return
        if self.missing_strategy != "mean":
            return
        if not hasattr(self, "trainer") or self.trainer is None or self.trainer.datamodule is None:
            raise RuntimeError("Train-set mean imputation requires an attached trainer/datamodule.")

        sums = None
        counts = None
        device = self.device

        with torch.no_grad():
            for batch in self.trainer.datamodule.train_dataloader():
                mods = self._get_modalities(batch)
                if sums is None:
                    sums = [
                        torch.zeros(int(x.reshape(x.shape[0], -1).shape[1]), device=device, dtype=torch.float64)
                        for x in mods
                    ]
                    counts = [
                        torch.zeros(int(x.reshape(x.shape[0], -1).shape[1]), device=device, dtype=torch.float64)
                        for x in mods
                    ]

                for i, x in enumerate(mods):
                    x = x.to(device=device, dtype=torch.float32).reshape(x.shape[0], -1)
                    finite = ~torch.isnan(x)
                    sums[i] += torch.nan_to_num(x, nan=0.0).double().sum(dim=0)
                    counts[i] += finite.double().sum(dim=0)

        if sums is None or counts is None:
            raise RuntimeError("Could not compute train-set feature means: empty train dataloader.")

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            for i in range(len(sums)):
                dist.all_reduce(sums[i], op=dist.ReduceOp.SUM)
                dist.all_reduce(counts[i], op=dist.ReduceOp.SUM)

        means = []
        for s, c in zip(sums, counts):
            mean = s / c.clamp_min(1.0)
            mean = torch.where(c > 0, mean, torch.zeros_like(mean))
            means.append(mean.float())
        self._train_feature_means = means

    def _dropout_modality_indices(self) -> list[int]:
        if self.modality_dropout_scope == "all":
            return list(range(len(self.modalities)))
        if self.modality_dropout_scope == "query":
            return [i for i in range(len(self.modalities)) if i != self.candidate_idx]
        raise ValueError(f"Unsupported modality_dropout_scope={self.modality_dropout_scope}. Use 'query' or 'all'.")

    def _apply_modality_dropout(self, mods: list[torch.Tensor]) -> list[torch.Tensor]:
        if not self.training or self.modality_dropout_rate <= 0.0:
            return mods

        out = list(mods)
        for i in self._dropout_modality_indices():
            x = out[i]
            present = self._modality_present(x)
            if present.numel() == 0:
                continue
            drop = (torch.rand(present.shape, device=x.device) < self.modality_dropout_rate) & present.to(x.device)
            if not drop.any():
                continue
            x_drop = x.clone()
            x_drop[drop] = float("nan")
            out[i] = x_drop
        return out

    def _apply_missing_strategy(self, mods: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.missing_strategy == "zero_mask":
            return mods
        if self.missing_strategy != "mean":
            raise ValueError(f"Unsupported missing_strategy={self.missing_strategy}. Use 'zero_mask' or 'mean'.")

        self._ensure_train_feature_means()
        out = []
        for i, x in enumerate(mods):
            mean = self._train_feature_means[i].to(device=x.device, dtype=x.dtype)
            original_shape = x.shape
            x_flat = x.reshape(x.shape[0], -1)
            mean = mean.view(1, -1).expand_as(x_flat)
            x_filled = torch.where(torch.isnan(x_flat), mean, x_flat)
            out.append(x_filled.reshape(original_shape))
        return out

    def _prepare_modalities_for_encoder(self, mods: list[torch.Tensor]) -> list[torch.Tensor]:
        mods = self._apply_modality_dropout(mods)
        mods = self._apply_missing_strategy(mods)
        return mods

    def _get_modalities(self, batch):
        modality_type_0 = "tabular_data"
        modality_type_1 = "tabular_data"
        modality_type_2 = "tabular_data"
        x0 = batch[self.modalities[0]][modality_type_0]
        x1 = batch[self.modalities[1]][modality_type_1]
        x2 = batch[self.modalities[2]][modality_type_2]
        return [
            x0,
            x1,
            x2,
            #batch["prs"]["tabular_data"],
            #batch["bloodbio"]["tabular_data"],
            #batch["baselinechars"]["tabular_data"],
            #batch["localenvironment"]["tabular_data"],
            #batch["arterialstiffness"]["tabular_data"],
            #batch["anthropometry"]["tabular_data"],
            #batch["bloodpressure"]["tabular_data"],
            #batch["ecgduringexercise"]["tabular_data"],
            #batch["eyemeasures"]["tabular_data"],
            #batch["bonedensitometry"]["tabular_data"],
            #batch["handgripstrength"]["tabular_data"],
            #batch["spirometry"]["tabular_data"],
            #batch["touchscreen"]["tabular_data"],
            #batch["cognitivefunction"]["tabular_data"],
            #batch["hearingtest"]["tabular_data"],
            #batch["verbalinterview"]["tabular_data"],
            #batch["bloodcount"]["tabular_data"],
            #batch["urineassays"]["tabular_data"],
            #batch["telomeres"]["tabular_data"],
            #batch["infectiousdiseases"]["tabular_data"],
        ]

    def forward(self, batch):
        x = self._get_modalities(batch)
        x = self._prepare_modalities_for_encoder(x)
        return self.model(x)
    
    def build_candidate_bank(self, split):
        r_list, cls_list = [], []
        dl = self.trainer.datamodule.val_dataloader() if split == "val" else self.trainer.datamodule.test_dataloader()

        for batch in dl:
            mods = self._get_modalities(batch)
            mods = self._apply_missing_strategy(mods)
            cls_id = torch.tensor(batch["eids"], device=self.device)

            candidates_raw = mods[self.candidate_idx].to(self.device)
            present_cand = self._modality_present(candidates_raw)
            if present_cand.sum().item() == 0:
                continue

            candidates = candidates_raw[present_cand].float() 
            reps = self.model.encoders[self.candidate_idx](candidates)
            if self.params_method.get("embedding_norm", False):
                reps = torch.nn.functional.normalize(reps, dim=1)

            r_list.append(reps)
            cls_list.append(cls_id[present_cand])

        ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

        # Local tensors (can be empty!)
        if r_list:
            r_local = torch.cat(r_list, dim=0)
            cls_local = torch.cat(cls_list, dim=0)
            emb_dim_local = r_local.shape[1]
        else:
            r_local = None
            cls_local = torch.empty((0,), device=self.device, dtype=torch.long)
            emb_dim_local = -1

        if not ddp:
            if r_local is None:
                return None
            return {"r": r_local, "cls_id": cls_local}

        # --- DDP path: make shapes consistent across ranks ---
        # 1) agree on embedding dim
        emb_dim_t = torch.tensor([emb_dim_local], device=self.device, dtype=torch.long)
        emb_dims = [torch.empty_like(emb_dim_t) for _ in range(dist.get_world_size())]
        dist.all_gather(emb_dims, emb_dim_t)
        emb_dim = int(torch.stack(emb_dims).max().item())
        if emb_dim <= 0:
            return None  # nobody had candidates

        if r_local is None:
            r_local = torch.empty((0, emb_dim), device=self.device, dtype=torch.float32)

        # 2) gather lengths + pad to max_len
        len_t = torch.tensor([r_local.shape[0]], device=self.device, dtype=torch.long)
        lens = [torch.empty_like(len_t) for _ in range(dist.get_world_size())]
        dist.all_gather(lens, len_t)
        lens = [int(x.item()) for x in lens]
        max_len = max(lens)

        if r_local.shape[0] < max_len:
            pad_rows = max_len - r_local.shape[0]
            r_local = torch.cat([r_local, torch.zeros((pad_rows, emb_dim), device=self.device, dtype=r_local.dtype)], dim=0)
            cls_local = torch.cat([cls_local, torch.full((pad_rows,), -1, device=self.device, dtype=cls_local.dtype)], dim=0)

        # 3) all_gather padded tensors
        r_gather = [torch.empty((max_len, emb_dim), device=self.device, dtype=r_local.dtype) for _ in range(dist.get_world_size())]
        cls_gather = [torch.empty((max_len,), device=self.device, dtype=cls_local.dtype) for _ in range(dist.get_world_size())]
        dist.all_gather(r_gather, r_local)
        dist.all_gather(cls_gather, cls_local)

        # 4) unpad + concat
        r_full = torch.cat([r_gather[i][:lens[i]] for i in range(dist.get_world_size())], dim=0)
        cls_full = torch.cat([cls_gather[i][:lens[i]] for i in range(dist.get_world_size())], dim=0)

        return {"r": r_full, "cls_id": cls_full}

    def _cfg_get(self, cfg: dict, key: str, default=None):
        if cfg is None:
            return default
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    @staticmethod
    def _derangement_from_order(order: torch.Tensor) -> torch.Tensor:
        """Return destination->source indices with no fixed points."""
        if order.numel() <= 1:
            return order.clone()
        source_idx = torch.empty_like(order)
        source_idx[order] = torch.roll(order, shifts=-1)
        return source_idx

    def _fixed_eval_derangement(
        self,
        eids: torch.Tensor,
        *,
        seed: int,
        modality_idx: int,
        split: str,
    ) -> torch.Tensor:
        """Create a reproducible within-batch donor mapping."""
        B = int(eids.shape[0])
        if B <= 1:
            return torch.arange(B, device=eids.device)
        split_offset = 0 if split == "val" else 10_000_019
        eid_checksum = int(eids.detach().long().sum().cpu().item() % 2_147_483_647)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + split_offset + 104_729 * int(modality_idx) + eid_checksum)
        order = torch.randperm(B, generator=gen)
        return self._derangement_from_order(order).to(eids.device)

    def prepare_embeddings_for_loss(
        self,
        embeddings: list[torch.Tensor],
        *,
        batch: dict,
        split: str,
    ) -> list[torch.Tensor]:
        """Apply balanced dynamic query-modality misalignment during training."""
        if split != "train" or not self.use_gate or self.modelname != "symile":
            return embeddings
        if str(self._cfg_get(self.eval_corruption, "mode", "none")).lower() != "permute":
            return embeddings
        if not bool(self._cfg_get(self.eval_corruption, "train_enabled", True)):
            return embeddings

        fractions = {
            "nmr": float(self._cfg_get(self.eval_corruption, "train_nmr_fraction", 0.15)),
            "ehr": float(self._cfg_get(self.eval_corruption, "train_ehr_fraction", 0.15)),
        }
        query_modalities = [
            (i, name)
            for i, name in enumerate(self.modalities)
            if i != self.candidate_idx and fractions.get(name, 0.0) > 0.0
        ]
        if not query_modalities:
            return embeddings

        B = int(embeddings[0].shape[0])
        if B <= 1:
            return embeddings

        total_fraction = sum(fractions[name] for _, name in query_modalities)
        if total_fraction > 1.0 + 1e-8:
            raise ValueError(
                "Training corruption fractions must sum to at most 1.0; "
                f"got {total_fraction:.4f}."
            )

        assignment_order = torch.randperm(B, device=embeddings[0].device)
        donor_order = torch.randperm(B, device=embeddings[0].device)
        source_idx = self._derangement_from_order(donor_order)
        corrupted = list(embeddings)
        assigned = 0
        for mod_idx, name in query_modalities:
            count = min(B - assigned, int(round(B * fractions[name])))
            selected = assignment_order[assigned : assigned + count]
            assigned += count
            mask = torch.zeros(B, device=embeddings[0].device, dtype=torch.bool)
            mask[selected] = True
            if mask.any():
                x = corrupted[mod_idx].clone()
                x[mask] = embeddings[mod_idx][source_idx][mask]
                corrupted[mod_idx] = x
            self.log(
                f"train/corruption/{name}_rate",
                mask.float().mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        self.log(
            "train/corruption/clean_rate",
            torch.tensor((B - assigned) / B, device=embeddings[0].device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return corrupted

    def _eval_corruption_is_enabled(self, split: str) -> bool:
        cfg = self.eval_corruption
        mode = str(self._cfg_get(cfg, "mode", "none")).lower()
        if mode in ("none", "false", "off", ""):
            return False
        if mode != "permute":
            raise ValueError(f"Unsupported eval_corruption.mode={mode}. Only 'none' and 'permute' are supported.")

        splits = self._cfg_get(cfg, "splits", ["val", "test"])
        if isinstance(splits, str):
            splits = [splits]
        return split in set(splits)

    def _deterministic_corruption_mask(self, eids: torch.Tensor, fraction: float, seed: int) -> torch.Tensor:
        if fraction <= 0.0:
            return torch.zeros_like(eids, dtype=torch.bool)
        if fraction >= 1.0:
            return torch.ones_like(eids, dtype=torch.bool)

        ids = eids.detach().to(device=eids.device, dtype=torch.long).abs()
        hashed = (ids * 1103515245 + int(seed) * 12345 + 12345) % 1000003
        return (hashed.float() / 1000003.0) < float(fraction)

    def _binary_auroc(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scores = scores.detach().float().flatten()
        labels = labels.detach().bool().flatten()
        n_pos = labels.sum()
        n_neg = (~labels).sum()
        if int(n_pos.item()) == 0 or int(n_neg.item()) == 0:
            return torch.tensor(float("nan"), device=scores.device)
        return binary_auroc(scores, labels.long(), thresholds=None)

    def _binary_auprc(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scores = scores.detach().float().flatten()
        labels = labels.detach().bool().flatten()
        n_pos = labels.sum()
        if int(n_pos.item()) == 0:
            return torch.tensor(float("nan"), device=scores.device)
        return binary_average_precision(scores, labels.long(), thresholds=None)

    def _apply_eval_corruption(self, emb_keep: list[torch.Tensor], eids_keep: torch.Tensor, split: str):
        if not self._eval_corruption_is_enabled(split):
            return emb_keep, None

        cfg = self.eval_corruption
        modality = self._cfg_get(cfg, "modality", None)
        if modality not in self.modalities:
            raise ValueError(f"eval_corruption.modality={modality} is not in modalities={self.modalities}.")

        mod_idx = int(self.modalities.index(modality))
        if mod_idx == self.candidate_idx:
            raise ValueError(
                "UKB eval corruption is currently implemented for query modalities only. "
                f"Got target/candidate modality {modality}."
            )

        fraction = float(self._cfg_get(cfg, "fraction", 0.0))
        seed = int(self._cfg_get(cfg, "seed", 420))
        mask = self._deterministic_corruption_mask(eids_keep, fraction=fraction, seed=seed)
        B = int(eids_keep.shape[0])

        if B <= 1 or int(mask.sum().item()) == 0:
            source_idx = torch.arange(B, device=eids_keep.device)
            return emb_keep, {
                "mod_idx": mod_idx,
                "modality": modality,
                "mask": mask,
                "source_idx": source_idx,
            }

        perm = self._fixed_eval_derangement(
            eids_keep,
            seed=seed,
            modality_idx=mod_idx,
            split=split,
        )

        corrupted = list(emb_keep)
        corrupted_mod = corrupted[mod_idx].clone()
        corrupted_mod[mask] = corrupted[mod_idx][perm][mask]
        corrupted[mod_idx] = corrupted_mod

        return corrupted, {
            "mod_idx": mod_idx,
            "modality": modality,
            "mask": mask,
            "source_idx": perm,
        }

    def _log_eval_corruption_metrics(self, split: str, info: dict, gate_weights: torch.Tensor) -> None:
        if info is None or gate_weights is None or gate_weights.numel() == 0:
            return

        mod_idx = int(info["mod_idx"])
        modality = str(info["modality"])
        mask = info["mask"].to(device=gate_weights.device, dtype=torch.bool)
        w_mod = gate_weights[:, mod_idx].detach().float()

        clean = ~mask
        if mask.any():
            self.log(f"{split}/corruption/{modality}_candidate_avg_gate_permuted_mean", w_mod[mask].mean(), on_step=False, on_epoch=True, sync_dist=True)
        if clean.any():
            self.log(f"{split}/corruption/{modality}_candidate_avg_gate_clean_mean", w_mod[clean].mean(), on_step=False, on_epoch=True, sync_dist=True)

        query_indices = [i for i in range(gate_weights.shape[1]) if i != self.candidate_idx]
        other_query_indices = [i for i in query_indices if i != mod_idx]
        if mask.any() and other_query_indices:
            other_max = gate_weights[:, other_query_indices].detach().float().max(dim=1).values
            suppression_rate = (w_mod[mask] < other_max[mask]).float().mean()
            self.log(f"{split}/corruption/{modality}_candidate_avg_relative_suppression_rate", suppression_rate, on_step=False, on_epoch=True, sync_dist=True)

        self.log(f"{split}/corruption/{modality}_permuted_rate", mask.float().mean(), on_step=False, on_epoch=True, sync_dist=True)

        source_idx = info.get("source_idx", None)
        if source_idx is not None and mask.any():
            row_idx = torch.arange(mask.shape[0], device=mask.device)
            misaligned_rate = (source_idx.to(mask.device)[mask] != row_idx[mask]).float().mean()
            self.log(f"{split}/corruption/{modality}_misaligned_rate", misaligned_rate, on_step=False, on_epoch=True, sync_dist=True)

    def _true_candidate_corruption_records(
        self,
        clean_embeddings: list[torch.Tensor],
        mixed_embeddings: list[torch.Tensor],
        mixed_info: dict,
        eids: torch.Tensor,
        split: str,
    ) -> list[dict]:
        """Build paired clean/corrupted diagnostics at each sample's true target."""
        if not self._eval_corruption_is_enabled(split) or not self.use_gate or self.gate is None:
            return []

        seed = int(self._cfg_get(self.eval_corruption, "seed", 420))
        query_indices = [i for i in range(len(clean_embeddings)) if i != self.candidate_idx]
        names = self.modalities

        with torch.no_grad():
            W_clean = self.gate.compute_W(clean_embeddings)
            clean_gated, clean_w, _ = self.gate.apply_for_target(
                self.candidate_idx,
                clean_embeddings,
                W=W_clean,
            )
            W_mixed = self.gate.compute_W(mixed_embeddings)
            _, mixed_w, _ = self.gate.apply_for_target(
                self.candidate_idx,
                mixed_embeddings,
                W=W_mixed,
            )

            records = []
            for mod_idx in query_indices:
                modality = names[mod_idx]
                source_idx = self._fixed_eval_derangement(
                    eids,
                    seed=seed,
                    modality_idx=mod_idx,
                    split=split,
                )
                counterfactual = list(clean_embeddings)
                counterfactual_mod = clean_embeddings[mod_idx].clone()
                counterfactual_mod[:] = clean_embeddings[mod_idx][source_idx]
                counterfactual[mod_idx] = counterfactual_mod

                W_corrupt = self.gate.compute_W(counterfactual)
                corrupt_gated, corrupt_w, _ = self.gate.apply_for_target(
                    self.candidate_idx,
                    counterfactual,
                    W=W_corrupt,
                )

                clean_weight = clean_w[:, mod_idx].detach().float()
                corrupt_weight = corrupt_w[:, mod_idx].detach().float()
                clean_cos_original = torch.nn.functional.cosine_similarity(
                    clean_gated[mod_idx].detach().float(),
                    clean_embeddings[mod_idx].detach().float(),
                    dim=1,
                )
                corrupt_cos_original = torch.nn.functional.cosine_similarity(
                    corrupt_gated[mod_idx].detach().float(),
                    counterfactual[mod_idx].detach().float(),
                    dim=1,
                )

                if self.gate.neutral is not None:
                    neutral = self.gate.neutral[mod_idx].detach().float().unsqueeze(0)
                    clean_cos_neutral = torch.nn.functional.cosine_similarity(
                        clean_gated[mod_idx].detach().float(),
                        neutral.expand_as(clean_gated[mod_idx]),
                        dim=1,
                    )
                    corrupt_cos_neutral = torch.nn.functional.cosine_similarity(
                        corrupt_gated[mod_idx].detach().float(),
                        neutral.expand_as(corrupt_gated[mod_idx]),
                        dim=1,
                    )
                else:
                    clean_cos_neutral = torch.full_like(clean_weight, float("nan"))
                    corrupt_cos_neutral = torch.full_like(corrupt_weight, float("nan"))

                is_mixed_modality = (
                    mixed_info is not None and int(mixed_info["mod_idx"]) == mod_idx
                )
                if is_mixed_modality:
                    mixed_mask = mixed_info["mask"].detach().bool()
                    mixed_source_idx = mixed_info["source_idx"].detach().long()
                    observed_weight = mixed_w[:, mod_idx].detach().float()
                else:
                    mixed_mask = torch.zeros_like(eids, dtype=torch.bool)
                    mixed_source_idx = torch.arange(eids.shape[0], device=eids.device)
                    observed_weight = clean_weight

                other_query_indices = [i for i in query_indices if i != mod_idx]
                other_idx = other_query_indices[0] if other_query_indices else None
                if other_idx is None:
                    other_clean_weight = torch.full_like(clean_weight, float("nan"))
                    other_corrupt_weight = torch.full_like(clean_weight, float("nan"))
                else:
                    other_clean_weight = clean_w[:, other_idx].detach().float()
                    other_corrupt_weight = corrupt_w[:, other_idx].detach().float()

                eids_cpu = eids.detach().cpu()
                source_eids_cpu = eids[source_idx].detach().cpu()
                mixed_source_eids_cpu = eids[mixed_source_idx].detach().cpu()
                for i in range(int(eids.shape[0])):
                    records.append(
                        {
                            "split": split,
                            "eid": int(eids_cpu[i].item()),
                            "modality": modality,
                            "source_eid": int(source_eids_cpu[i].item()),
                            "is_misaligned": bool(source_eids_cpu[i].item() != eids_cpu[i].item()),
                            "clean_weight": float(clean_weight[i].item()),
                            "corrupted_weight": float(corrupt_weight[i].item()),
                            "delta_weight": float((clean_weight[i] - corrupt_weight[i]).item()),
                            "clean_cos_original": float(clean_cos_original[i].item()),
                            "corrupted_cos_original": float(corrupt_cos_original[i].item()),
                            "clean_cos_neutral": float(clean_cos_neutral[i].item()),
                            "corrupted_cos_neutral": float(corrupt_cos_neutral[i].item()),
                            "other_clean_weight": float(other_clean_weight[i].item()),
                            "other_corrupted_weight": float(other_corrupt_weight[i].item()),
                            "other_delta_weight": float(
                                (other_clean_weight[i] - other_corrupt_weight[i]).item()
                            ),
                            "mixed_is_configured_modality": bool(is_mixed_modality),
                            "mixed_is_permuted": bool(mixed_mask[i].item()),
                            "mixed_source_eid": int(mixed_source_eids_cpu[i].item()),
                            "mixed_weight": float(observed_weight[i].item()),
                        }
                    )
        return records

    def _append_corruption_records(
        self,
        records: list[dict],
        *,
        eids: torch.Tensor,
        pred_eids: torch.Tensor,
        split: str,
    ) -> None:
        if not records:
            return
        pred_by_eid = {
            int(eid): int(pred)
            for eid, pred in zip(
                eids.detach().cpu().tolist(),
                pred_eids.detach().cpu().tolist(),
            )
        }
        for row in records:
            pred = pred_by_eid.get(int(row["eid"]), -1)
            row["pred_eid"] = pred
            row["correct"] = bool(pred == int(row["eid"]))
        self._corruption_epoch_records.setdefault(split, []).extend(records)

    def _gather_corruption_records(self, split: str) -> list[dict]:
        local_records = self._corruption_epoch_records.get(split, [])
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return list(local_records)
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_records)
        return [row for rank_rows in gathered for row in (rank_rows or [])]

    def _log_corruption_epoch_summary(self, split: str) -> None:
        query_indices = [i for i in range(len(self.modalities)) if i != self.candidate_idx]
        if (
            split == "test"
            and self._eval_corruption_is_enabled(split)
            and not torch.isfinite(self._suppression_thresholds[query_indices]).all()
        ):
            raise RuntimeError(
                "Test suppression metrics require thresholds calibrated during "
                "corruption-enabled validation with the same checkpoint."
            )
        records = self._gather_corruption_records(split)
        if self.trainer.is_global_zero and records:
            max_records = int(self._cfg_get(self.eval_corruption, "max_records", 5000))
            quantile = float(self._cfg_get(self.eval_corruption, "suppression_quantile", 0.05))
            for mod_idx, modality in enumerate(self.modalities):
                if mod_idx == self.candidate_idx:
                    continue
                rows = [row for row in records if row["modality"] == modality]
                if not rows:
                    continue

                clean = torch.tensor([row["clean_weight"] for row in rows])
                corrupt = torch.tensor([row["corrupted_weight"] for row in rows])
                paired_scores = torch.cat([-clean, -corrupt])
                paired_labels = torch.cat(
                    [
                        torch.zeros(clean.numel(), dtype=torch.bool),
                        torch.ones(corrupt.numel(), dtype=torch.bool),
                    ]
                )

                if split == "val":
                    self._suppression_thresholds[mod_idx] = torch.quantile(clean, quantile).to(
                        self._suppression_thresholds.device
                    )
                threshold = float(self._suppression_thresholds[mod_idx].item())
                clean_flags = clean < threshold
                corrupt_flags = corrupt < threshold

                self.log(
                    f"{split}/corruption/{modality}_true_candidate_auroc",
                    self._binary_auroc(paired_scores, paired_labels).to(self.device),
                    sync_dist=False,
                    rank_zero_only=True,
                )
                self.log(
                    f"{split}/corruption/{modality}_true_candidate_auprc",
                    self._binary_auprc(paired_scores, paired_labels).to(self.device),
                    sync_dist=False,
                    rank_zero_only=True,
                )
                self.log(
                    f"{split}/corruption/{modality}_paired_delta_mean",
                    (clean - corrupt).mean().to(self.device),
                    sync_dist=False,
                    rank_zero_only=True,
                )
                self.log(
                    f"{split}/corruption/{modality}_paired_delta_positive_rate",
                    ((clean - corrupt) > 0).float().mean().to(self.device),
                    sync_dist=False,
                    rank_zero_only=True,
                )
                self.log(
                    f"{split}/corruption/{modality}_clean_flag_rate",
                    clean_flags.float().mean().to(self.device),
                    sync_dist=False,
                    rank_zero_only=True,
                )
                self.log(
                    f"{split}/corruption/{modality}_suppression_rate",
                    corrupt_flags.float().mean().to(self.device),
                    sync_dist=False,
                    rank_zero_only=True,
                )
                paired_flag_total = int(clean_flags.sum().item() + corrupt_flags.sum().item())
                paired_precision = (
                    float(corrupt_flags.sum().item()) / paired_flag_total
                    if paired_flag_total > 0
                    else float("nan")
                )
                self.log(
                    f"{split}/corruption/{modality}_suppression_precision",
                    torch.tensor(paired_precision, device=self.device),
                    sync_dist=False,
                    rank_zero_only=True,
                )
                self.log(
                    f"{split}/corruption/{modality}_suppression_threshold",
                    torch.tensor(threshold, device=self.device),
                    sync_dist=False,
                    rank_zero_only=True,
                )
                self.log(
                    f"{split}/corruption/{modality}_other_modality_delta_mean",
                    torch.tensor(
                        [row["other_delta_weight"] for row in rows],
                        device=self.device,
                    ).mean(),
                    sync_dist=False,
                    rank_zero_only=True,
                )
                self.log(
                    f"{split}/corruption/{modality}_suppression_specificity_margin",
                    torch.tensor(
                        [
                            row["delta_weight"] - abs(row["other_delta_weight"])
                            for row in rows
                        ],
                        device=self.device,
                    ).mean(),
                    sync_dist=False,
                    rank_zero_only=True,
                )
                for metric_key in (
                    "clean_cos_original",
                    "corrupted_cos_original",
                    "clean_cos_neutral",
                    "corrupted_cos_neutral",
                ):
                    self.log(
                        f"{split}/corruption/{modality}_{metric_key}",
                        torch.tensor(
                            [row[metric_key] for row in rows],
                            device=self.device,
                        ).nanmean(),
                        sync_dist=False,
                        rank_zero_only=True,
                    )
                self.log(
                    f"{split}/corruption/{modality}_misalignment_verified_rate",
                    torch.tensor(
                        sum(bool(row["is_misaligned"]) for row in rows) / len(rows),
                        device=self.device,
                    ),
                    sync_dist=False,
                    rank_zero_only=True,
                )

                mixed_rows = [
                    row for row in rows
                    if row["mixed_is_configured_modality"]
                ]
                mixed_labels = torch.tensor(
                    [row["mixed_is_permuted"] for row in mixed_rows],
                    dtype=torch.bool,
                )
                if mixed_labels.any() and (~mixed_labels).any():
                    mixed_weights = torch.tensor([row["mixed_weight"] for row in mixed_rows])
                    mixed_scores = -mixed_weights
                    self.log(
                        f"{split}/corruption/{modality}_mixed_true_candidate_auroc",
                        self._binary_auroc(mixed_scores, mixed_labels).to(self.device),
                        sync_dist=False,
                        rank_zero_only=True,
                    )
                    self.log(
                        f"{split}/corruption/{modality}_mixed_true_candidate_auprc",
                        self._binary_auprc(mixed_scores, mixed_labels).to(self.device),
                        sync_dist=False,
                        rank_zero_only=True,
                    )
                    mixed_flags = mixed_weights < threshold
                    mixed_tp = int((mixed_flags & mixed_labels).sum().item())
                    mixed_fp = int((mixed_flags & ~mixed_labels).sum().item())
                    self.log(
                        f"{split}/corruption/{modality}_mixed_dataset_flag_rate",
                        mixed_flags.float().mean().to(self.device),
                        sync_dist=False,
                        rank_zero_only=True,
                    )
                    mixed_correct = torch.tensor(
                        [row["correct"] for row in mixed_rows],
                        dtype=torch.float32,
                    )
                    self.log(
                        f"{split}/corruption/{modality}_mixed_permuted_accuracy",
                        mixed_correct[mixed_labels].mean().to(self.device),
                        sync_dist=False,
                        rank_zero_only=True,
                    )
                    self.log(
                        f"{split}/corruption/{modality}_mixed_clean_accuracy",
                        mixed_correct[~mixed_labels].mean().to(self.device),
                        sync_dist=False,
                        rank_zero_only=True,
                    )
                    self.log(
                        f"{split}/corruption/{modality}_mixed_flag_precision",
                        torch.tensor(
                            mixed_tp / max(1, mixed_tp + mixed_fp),
                            device=self.device,
                        ),
                        sync_dist=False,
                        rank_zero_only=True,
                    )

                for row, clean_flag, corrupt_flag in zip(rows, clean_flags, corrupt_flags):
                    row["suppression_threshold"] = threshold
                    row["clean_flag"] = bool(clean_flag.item())
                    row["corrupted_flag"] = bool(corrupt_flag.item())

            rows_for_table = sorted(
                records,
                key=lambda row: (
                    not bool(row.get("corrupted_flag", False)),
                    -float(row["delta_weight"]),
                ),
            )[:max_records]
            if rows_for_table and getattr(self, "logger", None) is not None:
                columns = list(rows_for_table[0].keys())
                table = wandb.Table(
                    columns=columns,
                    data=[[row.get(col) for col in columns] for row in rows_for_table],
                )
                self.logger.experiment.log(
                    {f"{split}/corruption/true_candidate_audit_table": table},
                    commit=False,
                )

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast(self._suppression_thresholds, src=0)
        self._corruption_epoch_records[split] = []

    def on_validation_epoch_start(self):
        self._corruption_epoch_records["val"] = []
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
        if not getattr(self.trainer, "sanity_checking", False):
            self._log_corruption_epoch_summary("val")
        else:
            self._corruption_epoch_records["val"] = []
        return super().on_validation_epoch_end()

    def on_test_epoch_start(self):
        self._corruption_epoch_records["test"] = []
        return super().on_test_epoch_start()

    def on_test_epoch_end(self):
        self._log_corruption_epoch_summary("test")
        return super().on_test_epoch_end()
    
    def retrieval_step(self, batch, embeddings, split):
        bank = getattr(self, "candidate_bank", None)
        if bank is None:
            return []

        mods = self._get_modalities(batch)
        # query modalities are all except candidate modality
        query_indices = [i for i in range(len(mods)) if i != self.candidate_idx]

        # Build per-sample presence masks from raw inputs (NaNs)
        present_masks = []
        for i in query_indices:
            x_raw = mods[i].to(self.device)
            present_masks.append(self._modality_present(x_raw))  # (B,)

        # only evaluate queries that have at least one query modality AND have candidate idx modality present
        present_count = torch.stack([m.float() for m in present_masks], dim=0).sum(dim=0)  # (B,)
        cand_raw = mods[self.candidate_idx].to(self.device)
        present_target = self._modality_present(cand_raw)  # (B,)
        keep = (present_count > 0) & present_target
        if keep.sum().item() == 0:
            return []
        bank_size = int(bank["r"].shape[0])
        num_queries = int(keep.sum().item())
        self._set_retrieval_candidate_scores(num_queries * bank_size)
        y = torch.tensor(batch["eids"]).to(self.device)[keep]
        corruption_info = None
        diagnostic_records = []

        if getattr(self, "use_gate", False) and getattr(self, "gate", None) is not None:
            clean_emb_keep = [embeddings[i][keep] for i in range(len(embeddings))]
            emb_keep, corruption_info = self._apply_eval_corruption(clean_emb_keep, y, split=split)
            diagnostic_records = self._true_candidate_corruption_records(
                clean_emb_keep,
                emb_keep,
                corruption_info,
                y,
                split,
            )
            r_candidates = bank["r"]        # (N, D)
            r_cls_id = bank["cls_id"]       # (N,)
            names = getattr(self, "modalities", [f"m{i}" for i in range(len(embeddings))])

            # -----------------------------------------
            # Candidate-dependent reranker path (SLOW)
            # -----------------------------------------
            cand_dep = True
            if cand_dep and getattr(self.gate, "gate_mode", None) == "attention":
                if self.modelname not in ("symile"):
                    raise ValueError("candidate-dependent gating implemented for symile retrieval only.")

                Bk = emb_keep[0].shape[0]
                D = emb_keep[0].shape[1]
                M_total = len(emb_keep)
                query_indices = [i for i in range(M_total) if i != self.candidate_idx]
                if len(query_indices) == 0:
                    return []

                # chunk size (tune for memory/time)
                chunk_size = int(self.params_method.get("gate_candidate_chunk_size", 256))

                sum_w = torch.zeros((M_total,), device=self.device)
                sum_w_query = torch.zeros((Bk, M_total), device=self.device)
                count_w = 0

                logits_chunks = []
                for s in range(0, r_candidates.shape[0], chunk_size):
                    cand = r_candidates[s : s + chunk_size]  # (Nc, D)
                    Nc = cand.shape[0]

                    # Build pair-batch of size (Bk*Nc)
                    # Target modality tensor becomes the candidate embedding, broadcast across queries.
                    pair_embs = []
                    for m in range(M_total):
                        if m == self.candidate_idx:
                            x = cand.unsqueeze(0).expand(Bk, Nc, D).reshape(Bk * Nc, D)
                        else:
                            x0 = emb_keep[m]  # (Bk, D)
                            x = x0.unsqueeze(1).expand(Bk, Nc, D).reshape(Bk * Nc, D)
                        pair_embs.append(x)

                    # query_mode="target" makes Q_t depend on the target embedding (the candidate)
                    W_pair = self.gate.compute_W(pair_embs)  # (Bk*Nc, M, M)
                    gated_list, w_pair, _ = self.gate.apply_for_target(self.candidate_idx, pair_embs, W=W_pair)

                    max_pairs = 128
                    pair_embs_slice = [x[:max_pairs] for x in pair_embs]
                    gated_slice = [g[:max_pairs] for g in gated_list]
                    self._log_gate_cos_alignment(pair_embs_slice, gated_slice, split=split, names=names)
                    if self.gate.neutral_type is not None and self.gate.neutral_type != "none" and self.gate.neutral_type != "None":
                        self._log_gate_cos_to_neutral(self.gate, gated_slice, split=split, names=names)

                    if hasattr(self.gate, "logit_gate_strength"):
                        alpha = torch.sigmoid(self.gate.logit_gate_strength.detach())
                        self.log(f"{split}/gate_alpha", alpha, on_step=False, on_epoch=True, sync_dist=True)
                        self.log(f"{split}/gate_logit_gate_strength", self.gate.logit_gate_strength.detach(), on_step=False, on_epoch=True, sync_dist=True)

                    # accumulate for logging
                    sum_w += w_pair.detach().sum(dim=0)  # (M_total,)
                    sum_w_query += w_pair.detach().view(Bk, Nc, M_total).sum(dim=1)
                    count_w += int(w_pair.shape[0])

                    # symile retrieval logits:
                    # logit(i,j) = < cand_j , Π_m gated_query_m(i,j) >
                    prod = torch.ones_like(gated_list[query_indices[0]])
                    for qi in query_indices:
                        prod = prod * gated_list[qi]  # (Bk*Nc, D)

                    prod = prod.view(Bk, Nc, D)  # (Bk, Nc, D)
                    raw = torch.einsum("bnd,nd->bn", prod, cand)  # (Bk, Nc)

                    logits_chunks.append(raw)

                logits = torch.cat(logits_chunks, dim=1)  # (Bk, N)

                # match losses/retrieval.py scaling for symile
                M = len(query_indices) + 1
                scale_base = D ** ((M - 1) / 2)
                logits = logits * scale_base

                logits = logits * self.logit_scale.exp()

                pred = r_cls_id[torch.argmax(logits, dim=1)]

                if count_w > 0:
                    mean_w = sum_w / float(count_w)  # (M_total,)
                    for j in range(M_total):
                        self.log(
                            f"{split}/gate_{names[j]}_mean",
                            mean_w[j],
                            on_step=False,
                            on_epoch=True,
                            sync_dist=True,
                        )
                    mean_w_query = sum_w_query / float(r_candidates.shape[0])
                    self._log_eval_corruption_metrics(split, corruption_info, mean_w_query)
                    self._append_corruption_records(
                        diagnostic_records,
                        eids=y,
                        pred_eids=pred,
                        split=split,
                    )

                return (y == pred).float().tolist()

            # -----------------------------------------
            # Existing fast gating path (candidate-independent)
            # -----------------------------------------
            W_keep = self.gate.compute_W(emb_keep)
            gated_list, w_t, _ = self.gate.apply_for_target(self.candidate_idx, emb_keep, W=W_keep)
            names = getattr(self, "modalities", [f"m{i}" for i in range(len(embeddings))])
            self._log_eval_corruption_metrics(split, corruption_info, w_t.detach())
            
            # logging 
            for j in range(len(embeddings)):
                if j == self.candidate_idx:
                    continue
                self.log(f"{split}/gate_{names[j]}_mean", w_t[:, j].mean(), on_step=False, on_epoch=True, sync_dist=True)
            self._log_gate_cos_alignment(emb_keep, gated_list, split=split, names=names)
            self._log_gate_cos_to_neutral(self.gate, gated_list, split=split, names=names)
            if hasattr(self.gate, "logit_gate_strength"):
                alpha = torch.sigmoid(self.gate.logit_gate_strength.detach())
                self.log(f"{split}/gate_alpha", alpha, on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{split}/gate_logit_gate_strength", self.gate.logit_gate_strength.detach(), on_step=False, on_epoch=True, sync_dist=True)

            rep_list = [gated_list[i] for i in query_indices]
        else:
            emb_keep = [embeddings[i][keep] for i in range(len(embeddings))]
            emb_keep, corruption_info = self._apply_eval_corruption(emb_keep, y, split=split)
            rep_list = [emb_keep[i] for i in query_indices]  # list of (B_keep, D)

        r_candidates = bank["r"]
        logits = zeroshot_retrieval_logits(
            r_candidates,
            rep_list,
            self.logit_scale.exp(),
            bias=self.bias,
            modelname=self.modelname,
        )

        r_cls_id = bank["cls_id"]
        pred = r_cls_id[torch.argmax(logits, dim=1)]
        if getattr(self, "use_gate", False) and getattr(self, "gate", None) is not None:
            self._append_corruption_records(
                diagnostic_records,
                eids=y,
                pred_eids=pred,
                split=split,
            )
        return (y == pred).float().tolist()
    
    def training_step(self, batch, batch_idx):
        # get loss + embeddings from the shared step (this also logs train/loss)
        loss, embeddings = self.shared_step(batch, "train", return_embeddings=True)

        return loss
