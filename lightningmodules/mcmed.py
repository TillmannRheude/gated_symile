import torch
import torch.distributed as dist
import torch.nn as nn

from lightningmodules.utils import LightningModuleParent
from losses.retrieval import zeroshot_retrieval_logits
from losses.utils import scale_mip_dvs


class MCMEDModel(LightningModuleParent):
    def __init__(
        self,
        model,
        params_retrival_ds: dict = {
            "batch_size": 128,
            "split_nr": 0,
        },
        candidate_idx: int = 1,
        **args,
    ):
        super().__init__(**args)

        self.dataset_name = "mcmed"
        self.model = model
        self.params_retrival_ds = params_retrival_ds
        self.retrieval_mode = str(self.params_retrival_ds.get("retrieval_mode", "global")).lower()
        self.preselected_num_candidates = int(self.params_retrival_ds.get("preselected_num_candidates", 10))
        self.retrieval_seed = int(self.params_retrival_ds.get("retrieval_seed", 420))
        if self.retrieval_mode not in {"global", "preselected"}:
            raise ValueError(f"Unsupported retrieval_mode={self.retrieval_mode}. Expected 'global' or 'preselected'.")
        if self.preselected_num_candidates < 1:
            raise ValueError(
                f"preselected_num_candidates must be >= 1, got {self.preselected_num_candidates}."
            )

        self.modalities = ["demographics", "rads", "numerics"]  # "waveforms_II",
        self.candidate_idx = int(candidate_idx)
        if self.candidate_idx < 0 or self.candidate_idx >= len(self.modalities):
            raise ValueError(f"candidate_idx must be in [0, {len(self.modalities) - 1}], got {self.candidate_idx}.")

        self.test_step_accuracies = []

        self.use_gate = bool(self.params_method["use_gate"])
        if self.use_gate:
            emb_dim = getattr(self._get_encoder_stack()[self.candidate_idx], "emb_dim", None)
            from architecture import ModalityAttentionGate
            self.gate = ModalityAttentionGate(
                num_modalities=len(self.modalities),
                emb_dim=int(emb_dim),
                d_k=self.params_method["gate_d_k"],
                temperature_init=self.params_method["gate_temp"],
                gate_bias_init=self.params_method["gate_bias_init"],
                gate_strength_init=self.params_method["gate_strength_init"],
                gate_type=self.params_method["gate_type"],
                gate_mode=self.params_method["gate_mode"],
                neutral_type=self.params_method["neutral_type"],
            )
        else:
            self.gate = None

        self.save_hyperparameters()

    def _get_encoder_stack(self):
        encoders = getattr(self.model, "encoders", None)
        if encoders is None and hasattr(self.model, "contrastive_model"):
            encoders = self.model.contrastive_model.encoders
        if encoders is None:
            raise ValueError("Could not locate encoder stack for MCMEDModel.")
        return encoders

    def forward(self, batch):
        x = self._get_modalities(batch)
        return self.model(x)

    def _get_modalities(self, batch):
        return [
            batch["demographics"],
            # batch["waveforms_II"],
            batch["rads"],
            batch["numerics"],
        ]

    def _get_jacobian_encoder_input(self, name: str, x):
        if name == "demographics" and isinstance(x, dict):
            cont = x.get("continuous", None)
            if not torch.is_tensor(cont) or cont.shape[0] == 0:
                return None, None
            cont_mask = x.get("continuous_mask", None)
            cat = x.get("categorical", None)
            cat_mask = x.get("categorical_mask", None)
            if not torch.is_tensor(cont_mask) or cont_mask.shape[0] == 0:
                cont_mask0 = torch.ones_like(cont[0], dtype=torch.bool)
            else:
                cont_mask0 = cont_mask[0].detach().clone()
            if not torch.is_tensor(cat) or cat.shape[0] == 0:
                cat0 = torch.zeros((0,), dtype=torch.long, device=cont.device)
            else:
                cat0 = cat[0].detach().clone()
            if not torch.is_tensor(cat_mask) or cat_mask.shape[0] == 0:
                cat_mask0 = torch.ones_like(cat0, dtype=torch.bool)
            else:
                cat_mask0 = cat_mask[0].detach().clone()

            def _builder(inp):
                return {
                    "continuous": inp.unsqueeze(0),
                    "continuous_mask": cont_mask0.to(device=inp.device, dtype=torch.bool).unsqueeze(0),
                    "categorical": cat0.to(device=inp.device, dtype=torch.long).unsqueeze(0),
                    "categorical_mask": cat_mask0.to(device=inp.device, dtype=torch.bool).unsqueeze(0),
                }

            return cont[0], _builder

        if name == "rads" and isinstance(x, dict):
            visits = x.get("input_ids", None)
            if not torch.is_tensor(visits) or visits.shape[0] == 0:
                return None, None
            attention_mask = x.get("attention_mask", None)
            if torch.is_tensor(attention_mask) and attention_mask.shape[0] > 0:
                attn0 = attention_mask[0].detach().clone()
            else:
                attn0 = torch.zeros_like(visits[0], dtype=torch.bool)

            def _builder(inp):
                report_mask = x.get("report_mask", None)
                if torch.is_tensor(report_mask) and report_mask.shape[0] > 0:
                    report_mask0 = report_mask[0].detach().clone()
                else:
                    report_mask0 = torch.ones(inp.shape[0], dtype=torch.bool)
                return {
                    "input_ids": inp.unsqueeze(0),
                    "attention_mask": attn0.to(device=inp.device, dtype=torch.bool).unsqueeze(0),
                    "report_mask": report_mask0.to(device=inp.device, dtype=torch.bool).unsqueeze(0),
                }

            return visits[0], _builder

        if name == "numerics" and isinstance(x, dict):
            trends = x.get("trend_values", None)
            if not torch.is_tensor(trends) or trends.shape[0] == 0:
                return None, None

            def _builder(inp):
                return {"trend_values": inp.unsqueeze(0)}

            return trends[0], _builder

        return super()._get_jacobian_encoder_input(name, x)

    @staticmethod
    def _tensor_modality_present(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return ~torch.isnan(x)
        batch_size = x.shape[0]
        return ~torch.isnan(x).reshape(batch_size, -1).all(dim=1)

    @staticmethod
    def _waveform_present(batch_waveforms: dict) -> torch.Tensor:
        if "bin_mask" in batch_waveforms:
            return batch_waveforms["bin_mask"].to(dtype=torch.bool).any(dim=1)
        return MCMEDModel._tensor_modality_present(batch_waveforms["windows"])

    @staticmethod
    def _numerics_present(batch_numerics: dict) -> torch.Tensor:
        if "bin_counts" in batch_numerics:
            return batch_numerics["bin_counts"].to(dtype=torch.long).gt(0).any(dim=1)
        return MCMEDModel._tensor_modality_present(batch_numerics["values"])

    def _radiology_present(self, batch_rads: dict) -> torch.Tensor:
        if "embedding_present" in batch_rads:
            return batch_rads["embedding_present"].to(device=self.device, dtype=torch.bool)
        if "report_mask" in batch_rads:
            return batch_rads["report_mask"].to(device=self.device, dtype=torch.bool).any(dim=1)
        return self._tensor_modality_present(batch_rads["visit_embedding"]).to(self.device)
    
    def _demographics_present(self, batch_demographics: dict) -> torch.Tensor:
        if "embedding_present" in batch_demographics:
            return batch_demographics["embedding_present"].to(device=self.device, dtype=torch.bool)
        if "continuous_mask" in batch_demographics:
            cont = batch_demographics["continuous_mask"].to(device=self.device, dtype=torch.bool).any(dim=1)
        else:
            cont = None
        if "categorical_mask" in batch_demographics:
            cat = batch_demographics["categorical_mask"].to(device=self.device, dtype=torch.bool).any(dim=1)
        else:
            cat = None
        if cont is None and cat is None:
            return self._tensor_modality_present(batch_demographics["continuous"]).to(self.device)
        if cont is None:
            return cat
        if cat is None:
            return cont
        return cont | cat

    def _modality_present(self, batch, modality_idx: int) -> torch.Tensor:
        name = self.modalities[int(modality_idx)]
        if name == "waveforms_II":
            return self._waveform_present(batch[name]).to(self.device)
        if name == "demographics":
            return self._demographics_present(batch[name]).to(self.device)
        if name == "rads":
            return self._radiology_present(batch[name]).to(self.device)
        if name == "numerics":
            return self._numerics_present(batch[name]).to(self.device)
        raise ValueError(f"Unknown MC-MED modality: {name}")

    @staticmethod
    def _index_modality(modality, mask: torch.Tensor):
        if torch.is_tensor(modality):
            return modality[mask]
        if isinstance(modality, dict):
            return {
                key: value[mask] if torch.is_tensor(value) and value.shape[0] == mask.shape[0] else value
                for key, value in modality.items()
            }
        raise TypeError(f"Unsupported modality payload type: {type(modality)}")

    def _query_keep_mask(self, batch) -> torch.Tensor:
        target_present = self._modality_present(batch, self.candidate_idx)
        query_indices = [i for i in range(len(self.modalities)) if i != self.candidate_idx]
        query_present = [self._modality_present(batch, i) for i in query_indices]
        present_count = torch.stack([m.float() for m in query_present], dim=0).sum(dim=0)
        return target_present & (present_count > 0)

    @staticmethod
    def _empty_retrieval_counts() -> dict:
        return {
            "count": 0.0,
            "correct_top1": 0.0,
            "correct_top5": 0.0,
            "correct_top10": 0.0,
            "correct_top100": 0.0,
        }

    @staticmethod
    def _topk_retrieval_counts(logits: torch.Tensor, labels: torch.Tensor, csn_bank: torch.Tensor) -> dict:
        out = {"count": float(labels.shape[0])}
        n_candidates = int(logits.shape[1])
        for k in (1, 5, 10, 100):
            k_eff = min(k, n_candidates)
            topk_idx = torch.topk(logits, k=k_eff, dim=1).indices
            if csn_bank.ndim == 1:
                topk_csn = csn_bank[topk_idx]
            else:
                topk_csn = torch.gather(csn_bank, dim=1, index=topk_idx)
            hit = (topk_csn == labels.unsqueeze(1)).any(dim=1)
            out[f"correct_top{k}"] = float(hit.sum().item())
        return out

    @staticmethod
    def _unique_candidates_by_csn(r_bank: torch.Tensor, csn_bank: torch.Tensor):
        """
        Keep one candidate per CSN (first occurrence, stable order).
        """
        if r_bank is None or csn_bank is None or csn_bank.numel() == 0:
            return r_bank, csn_bank

        keep_indices = []
        seen = set()
        csn_cpu = csn_bank.detach().cpu().tolist()
        for idx, csn in enumerate(csn_cpu):
            csn_i = int(csn)
            if csn_i in seen:
                continue
            seen.add(csn_i)
            keep_indices.append(idx)

        if len(keep_indices) == csn_bank.shape[0]:
            return r_bank, csn_bank

        keep_t = torch.tensor(keep_indices, device=csn_bank.device, dtype=torch.long)
        return r_bank.index_select(0, keep_t), csn_bank.index_select(0, keep_t)

    def _apply_preselected_candidates(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        csn_bank: torch.Tensor,
    ):
        """
        Build a per-query preselected candidate set:
        [positive candidate] + [random negatives].
        Returns logits_sel (B_keep, K), labels_sel (B_keep,), csn_sel (B_keep, K).
        """
        n_queries, n_candidates = int(logits.shape[0]), int(logits.shape[1])
        k = min(int(self.preselected_num_candidates), n_candidates)
        if n_queries == 0 or n_candidates == 0:
            return None, None, None

        selected_logits = []
        selected_labels = []
        selected_csns = []
        csn_bank_cpu = csn_bank.detach().cpu()

        for i in range(n_queries):
            y_i = labels[i]
            pos = torch.nonzero(csn_bank == y_i, as_tuple=False).flatten()
            if pos.numel() == 0:
                continue
            pos_idx = int(pos[0].item())

            if k == 1:
                idx = torch.tensor([pos_idx], device=logits.device, dtype=torch.long)
            else:
                neg_idx = torch.arange(n_candidates, device=logits.device, dtype=torch.long)
                neg_idx = neg_idx[neg_idx != pos_idx]
                k_neg = min(k - 1, int(neg_idx.numel()))

                # deterministic per-query sampling for stable eval
                g = torch.Generator(device="cpu")
                y_seed = int(labels[i].detach().cpu().item())
                g.manual_seed(self.retrieval_seed + (104729 * i) + y_seed)
                perm = torch.randperm(int(neg_idx.numel()), generator=g, device="cpu")[:k_neg].to(neg_idx.device)
                neg_pick = neg_idx.index_select(0, perm)
                idx = torch.cat(
                    [torch.tensor([pos_idx], device=logits.device, dtype=torch.long), neg_pick],
                    dim=0,
                )

            selected_logits.append(logits[i].index_select(0, idx))
            selected_labels.append(labels[i])
            selected_csns.append(csn_bank.index_select(0, idx))

        if len(selected_logits) == 0:
            return None, None, None

        return (
            torch.stack(selected_logits, dim=0),
            torch.stack(selected_labels, dim=0),
            torch.stack(selected_csns, dim=0),
        )

    def build_candidate_bank(self, split):
        r_list, csn_list = [], []
        dl = self.trainer.datamodule.val_dataloader() if split == "val" else self.trainer.datamodule.test_dataloader()
        encoder_stack = self._get_encoder_stack()

        for batch in dl:
            batch = self.trainer.strategy.batch_to_device(batch, self.device)
            target_present = self._modality_present(batch, self.candidate_idx)
            if target_present.sum().item() == 0:
                continue

            csn = batch["CSN"].to(self.device)[target_present]
            target_name = self.modalities[self.candidate_idx]
            target_batch = self._index_modality(batch[target_name], target_present)

            reps = encoder_stack[self.candidate_idx](target_batch)
            if self.params_method.get("embedding_norm", False):
                reps = nn.functional.normalize(reps, dim=1)

            r_list.append(reps)
            csn_list.append(csn)

        ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

        if r_list:
            r_local = torch.cat(r_list, dim=0)
            csn_local = torch.cat(csn_list, dim=0)
            emb_dim_local = r_local.shape[1]
        else:
            r_local = None
            csn_local = torch.empty((0,), device=self.device, dtype=torch.long)
            emb_dim_local = -1

        if not ddp:
            if r_local is None:
                return None
            r_unique, csn_unique = self._unique_candidates_by_csn(r_local, csn_local)
            return {"r": r_unique, "csn": csn_unique}

        emb_dim_t = torch.tensor([emb_dim_local], device=self.device, dtype=torch.long)
        emb_dims = [torch.empty_like(emb_dim_t) for _ in range(dist.get_world_size())]
        dist.all_gather(emb_dims, emb_dim_t)
        emb_dim = int(torch.stack(emb_dims).max().item())
        if emb_dim <= 0:
            return None

        if r_local is None:
            r_local = torch.empty((0, emb_dim), device=self.device, dtype=torch.float32)

        len_t = torch.tensor([r_local.shape[0]], device=self.device, dtype=torch.long)
        lens = [torch.empty_like(len_t) for _ in range(dist.get_world_size())]
        dist.all_gather(lens, len_t)
        lens = [int(x.item()) for x in lens]
        max_len = max(lens)

        if r_local.shape[0] < max_len:
            pad_rows = max_len - r_local.shape[0]
            r_local = torch.cat([r_local, torch.zeros((pad_rows, emb_dim), device=self.device, dtype=r_local.dtype)], dim=0)
            csn_local = torch.cat([csn_local, torch.full((pad_rows,), -1, device=self.device, dtype=csn_local.dtype)], dim=0)

        r_gather = [torch.empty((max_len, emb_dim), device=self.device, dtype=r_local.dtype) for _ in range(dist.get_world_size())]
        csn_gather = [torch.empty((max_len,), device=self.device, dtype=csn_local.dtype) for _ in range(dist.get_world_size())]
        dist.all_gather(r_gather, r_local)
        dist.all_gather(csn_gather, csn_local)

        r_full = torch.cat([r_gather[i][:lens[i]] for i in range(dist.get_world_size())], dim=0)
        csn_full = torch.cat([csn_gather[i][:lens[i]] for i in range(dist.get_world_size())], dim=0)

        r_unique, csn_unique = self._unique_candidates_by_csn(r_full, csn_full)
        return {"r": r_unique, "csn": csn_unique}

    def retrieval_step(self, batch, embeddings, split):
        bank = getattr(self, "candidate_bank", None)
        if bank is None:
            return self._empty_retrieval_counts()

        keep = self._query_keep_mask(batch)
        if keep.sum().item() == 0:
            return self._empty_retrieval_counts()

        query_indices = [i for i in range(len(embeddings)) if i != self.candidate_idx]
        emb_keep = [embeddings[i][keep] for i in range(len(embeddings))]
        rep_list = [emb_keep[i] for i in query_indices]
        y = batch["CSN"].to(self.device)[keep]

        csn_candidates = bank["csn"]
        logits_precomputed = None

        if self.use_gate and self.gate is not None:
            if getattr(self.gate, "gate_mode", None) == "attention":
                Bq = emb_keep[self.candidate_idx].shape[0]
                D = emb_keep[self.candidate_idx].shape[1]
                r_candidates = bank["r"]
                chunk_size = int(self.params_method.get("gate_candidate_chunk_size", 256))
                logits_chunks = []

                for s in range(0, r_candidates.shape[0], chunk_size):
                    cand = r_candidates[s : s + chunk_size]
                    nc = cand.shape[0]

                    pair_embs = []
                    for m in range(len(emb_keep)):
                        if m == self.candidate_idx:
                            x = cand.unsqueeze(0).expand(Bq, nc, D).reshape(Bq * nc, D)
                        else:
                            x = emb_keep[m].unsqueeze(1).expand(Bq, nc, D).reshape(Bq * nc, D)
                        pair_embs.append(x)

                    W_pair = self.gate.compute_W(pair_embs)
                    gated_list, w_pair, _ = self.gate.apply_for_target(self.candidate_idx, pair_embs, W=W_pair)

                    self._log_gate_weights(w_pair, set=split)
                    max_pairs = int(self.params_method.get("gate_log_max_pairs", 128))
                    pair_slice = [x[:max_pairs] for x in pair_embs]
                    gated_slice = [g[:max_pairs] for g in gated_list]
                    self._log_gate_cos_alignment(pair_slice, gated_slice, split=split, names=self.modalities)
                    self._log_gate_cos_to_neutral(self.gate, gated_slice, split=split, names=self.modalities)

                    prod = torch.ones_like(gated_list[query_indices[0]])
                    for qi in query_indices:
                        prod = prod * gated_list[qi]
                    raw = (prod * pair_embs[self.candidate_idx]).sum(dim=1).view(Bq, nc)
                    raw = scale_mip_dvs(raw, d=D, M=len(query_indices) + 1)
                    logits_chunks.append(raw)

                logits = torch.cat(logits_chunks, dim=1)
                scale = self.get_logit_scale_exp()
                if scale is not None:
                    logits = scale * logits
                logits_precomputed = logits
            else:
                W = self.gate.compute_W(emb_keep)
                gated_list, w_t, _ = self.gate.apply_for_target(self.candidate_idx, emb_keep, W=W)
                self._log_gate_weights(w_t, set=split)
                self._log_gate_cos_alignment(emb_keep, gated_list, split=split, names=self.modalities)
                self._log_gate_cos_to_neutral(self.gate, gated_list, split=split, names=self.modalities)
                if hasattr(self.gate, "logit_gate_strength"):
                    alpha = torch.sigmoid(self.gate.logit_gate_strength.detach())
                    self.log(f"{split}/gate_alpha", alpha, on_step=False, on_epoch=True, sync_dist=True)
                rep_list = [gated_list[i] for i in query_indices]

        if logits_precomputed is not None:
            logits = logits_precomputed
        elif self.modelname == "symile_attention":
            Bq = int(emb_keep[0].shape[0])
            D = int(emb_keep[0].shape[1])
            Nc = int(bank["r"].shape[0])

            pair_embs = []
            for m in range(len(emb_keep)):
                if m == self.candidate_idx:
                    x = bank["r"].unsqueeze(0).expand(Bq, Nc, D).reshape(Bq * Nc, D)
                else:
                    x = emb_keep[m].unsqueeze(1).expand(Bq, Nc, D).reshape(Bq * Nc, D)
                pair_embs.append(x)

            z = self.model.transformer(pair_embs)
            if z.dim() == 2 and z.shape[1] == 1:
                z = z.squeeze(1)
            elif z.dim() != 1:
                raise ValueError(f"Expected transformer score shape (Bq*Nc,) or (Bq*Nc,1), got {tuple(z.shape)}")

            logits = z.view(Bq, Nc)
            scale = self.get_logit_scale_exp()
            if scale is not None:
                logits = scale * logits
            if self.bias is not None:
                logits = logits + self.bias
        else:
            logits = zeroshot_retrieval_logits(
                bank["r"],
                rep_list,
                self.get_logit_scale_exp(),
                bias=self.bias,
                modelname=self.modelname,
            )

        if self.retrieval_mode == "preselected":
            logits_sel, y_sel, csn_sel = self._apply_preselected_candidates(
                logits=logits,
                labels=y,
                csn_bank=csn_candidates,
            )
            if logits_sel is None:
                return self._empty_retrieval_counts()
            return self._topk_retrieval_counts(logits_sel, y_sel, csn_sel)

        return self._topk_retrieval_counts(logits, y, csn_candidates)
