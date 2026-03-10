import torch
import time
import torch.nn as nn
import torch.distributed as dist
from torchmetrics.classification import BinaryAUROC

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
        **args,
    ):
        super().__init__(**args)
        self.dataset_name = "ukb"
        self.model = model
        self.params_retrival_ds = params_retrival_ds
        self.modalities = modalities
        self.candidate_idx = candidate_idx

        self.val_step_accuracies = []
        self.test_step_accuracies = []

        # linear probe w.r.t. modality prediction
        self.probe_on = True
        self.emb_dim = model.encoders[candidate_idx].emb_dim
        self.probe = nn.Linear(self.emb_dim * 3, 1)
        self.probe_loss = nn.BCEWithLogitsLoss()
        self.probe_metrics = nn.ModuleList([BinaryAUROC() for _ in range(3)])  # train, val, test

        self.unimodal_probe = nn.ModuleList([nn.Linear(self.emb_dim, 1) for _ in range(3)])
        self.unimodal_probe_loss = nn.BCEWithLogitsLoss()
        self.unimodal_probe_metrics = nn.ModuleList([nn.ModuleList([BinaryAUROC() for _ in range(3)]) for _ in range(3)])  # train, val, test for mod0, mod1, mod2

        # gate
        self.use_gate = self.params_method["use_gate"]
        if self.use_gate:
            self.gate = ModalityAttentionGate(
                num_modalities=len(self.modalities),
                emb_dim=self.emb_dim,
                num_heads=self.params_method["gate_num_heads"],
                d_k=self.params_method["gate_d_k"],
                d_null=self.params_method["gate_d_null"],
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

    def clip_and_normalize(
        self,
        x: torch.Tensor,
        clip_percentiles=(1.0, 99.0),
        eps: float = 1e-6,
        use_nonzero: bool = True,
        min_count: int = 100,
    ) -> torch.Tensor:
        """
        x: (H,W,T) or (C,H,W,T) or (B,C,H,W,T). Returns same shape, float32.
        Per-sample, per-channel robust clipping + normalization (batched, no Python loops).
        """
        x_in_ndim = x.ndim
        x = x.float()

        # Canonicalize to (B,C,H,W,T)
        if x.ndim == 3:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 4:
            x = x.unsqueeze(0)
        elif x.ndim != 5:
            raise ValueError(f"Unsupported shape: {tuple(x.shape)}")

        B, C, H, W, T = x.shape
        BC = B * C
        N = H * W * T

        x_flat = x.reshape(BC, N)  # (BC, N)

        q_lo = clip_percentiles[0] / 100.0
        q_hi = clip_percentiles[1] / 100.0

        if use_nonzero:
            mask = x_flat > 0
            cnt = mask.sum(dim=1)  # (BC,)

            # Use only nonzero values via NaN-masking (nanquantile ignores NaNs)
            x_nz = x_flat.masked_fill(~mask, float("nan"))

            # Quantiles on nonzero rows
            lo_nz = torch.nanquantile(x_nz, q_lo, dim=1)
            hi_nz = torch.nanquantile(x_nz, q_hi, dim=1)

            # Fallback quantiles on all values for low-count rows
            lo_all = torch.quantile(x_flat, q_lo, dim=1)
            hi_all = torch.quantile(x_flat, q_hi, dim=1)

            use_all = cnt < min_count
            lo = torch.where(use_all, lo_all, lo_nz)
            hi = torch.where(use_all, hi_all, hi_nz)
        else:
            lo = torch.quantile(x_flat, q_lo, dim=1)
            hi = torch.quantile(x_flat, q_hi, dim=1)

        # Clamp + normalize (broadcast per row)
        lo = lo[:, None]
        hi = hi[:, None]
        denom = (hi - lo).clamp_min(eps)

        x_clip = x_flat.clamp(lo, hi)
        out = (x_clip - lo) / denom

        out = out.reshape(B, C, H, W, T)

        # Restore original dimensionality
        if x_in_ndim == 3:
            return out[0, 0]
        if x_in_ndim == 4:
            return out[0]
        return out

    def _get_modalities(self, batch):
        modality_type_0 = "tabular_data" if "mri_brain" != self.modalities[0] and "mri_heart" != self.modalities[0] else "mri_data"
        modality_type_1 = "tabular_data" if "mri_brain" != self.modalities[1] and "mri_heart" != self.modalities[1] else "mri_data"
        modality_type_2 = "tabular_data" if "mri_brain" != self.modalities[2] and "mri_heart" != self.modalities[2] else "mri_data"

        if self.modalities[0] == "mri_heart" or self.modalities[0] == "mri_brain":
            x0 = self.masked_zscore_keep_zeros(batch[self.modalities[0]][modality_type_0])
        else:
            x0 = batch[self.modalities[0]][modality_type_0]
            
        if self.modalities[1] == "mri_heart" or self.modalities[1] == "mri_brain":
            x1 = self.masked_zscore_keep_zeros(batch[self.modalities[1]][modality_type_1])
        else:
            x1 = batch[self.modalities[1]][modality_type_1]

        if self.modalities[2] == "mri_heart" or self.modalities[2] == "mri_brain":
            x2 = self.masked_zscore_keep_zeros(batch[self.modalities[2]][modality_type_2])
        else:
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

    def masked_zscore_keep_zeros(
        self,
        x: torch.Tensor,
        eps: float = 1e-6,
        min_count: int = 100,
    ) -> torch.Tensor:
        """
        Masked z-score normalization using foreground voxels (x>0) only.
        Background (x<=0) stays 0.

        Works for:
        - (D,H,W)
        - (C,D,H,W)
        - (B,C,D,H,W)

        Returns float32 tensor with same shape.
        """
        threshold = 1e-3 

        x_in = x
        x = x.float()
        ndim = x.ndim

        # Canonicalize to (B,C,D,H,W)
        if ndim == 3:
            x = x.unsqueeze(0).unsqueeze(0)
        elif ndim == 4:
            x = x.unsqueeze(0)
        elif ndim != 5:
            raise ValueError(f"Unsupported shape: {tuple(x_in.shape)}")

        B, C, D, H, W = x.shape
        BC = B * C
        N = D * H * W

        xf = x.reshape(BC, N)                 # (BC, N)
        mask = xf > threshold                 # foreground mask
        cnt = mask.sum(dim=1).clamp_min(1)    # (BC,)

        # Sum and sumsq over foreground only
        xf_masked = xf * mask
        s = xf_masked.sum(dim=1)              # (BC,)
        ss = (xf_masked * xf_masked).sum(dim=1)

        mean_fg = s / cnt
        var_fg = ss / cnt - mean_fg * mean_fg
        std_fg = var_fg.clamp_min(0).sqrt()

        # Fallback for tiny foreground: use all voxels stats
        mean_all = xf.mean(dim=1)
        std_all = xf.std(dim=1, unbiased=False).clamp_min(eps)

        use_all = (cnt < min_count) | (std_fg < eps)
        mean = torch.where(use_all, mean_all, mean_fg)
        std = torch.where(use_all, std_all, std_fg).clamp_min(eps)

        # Apply z-score on foreground only; keep background at 0
        out = (xf - mean[:, None]) / std[:, None]
        out = out * mask.to(out.dtype)

        out = out.reshape(B, C, D, H, W)

        # Restore original dimensionality
        if ndim == 3:
            return out[0, 0]
        if ndim == 4:
            return out[0]
        return out

    def forward(self, batch):
        x = self._get_modalities(batch)

        return self.model(x)
    
    def build_candidate_bank(self, split):
        r_list, cls_list = [], []
        dl = self.trainer.datamodule.val_dataloader() if split == "val" else self.trainer.datamodule.test_dataloader()

        for batch in dl:
            mods = self._get_modalities(batch)
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

        r_list, cls_list = [], []
        dl = self.trainer.datamodule.val_dataloader() if split == "val" else self.trainer.datamodule.test_dataloader()
        for batch in dl:
            mods = self._get_modalities(batch)
            cls_id = torch.tensor(batch["eids"]).to(self.device)

            candidates_raw = mods[self.candidate_idx].to(self.device)  # may contain NaNs
            present_cand = self._modality_present(candidates_raw)
            if present_cand.sum().item() == 0:
                continue

            candidates = candidates_raw[present_cand]
            reps = self.model.encoders[self.candidate_idx](candidates)
            if self.params_method.get("embedding_norm", False):
                reps = nn.functional.normalize(reps, dim=1)

            r_list.append(reps)
            cls_list.append(cls_id[present_cand])

        if not r_list:
            return None

        r_local = torch.cat(r_list)
        cls_local = torch.cat(cls_list)

        r_full = self.all_gather(r_local)
        cls_full = self.all_gather(cls_local)

        if r_full.dim() > 2:
            r_full = r_full.reshape(-1, r_full.shape[-1])
            cls_full = cls_full.reshape(-1)

        return {"r": r_full, "cls_id": cls_full}
    
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

        if getattr(self, "use_gate", False) and getattr(self, "gate", None) is not None:
            emb_keep = [embeddings[i][keep] for i in range(len(embeddings))]
            r_candidates = bank["r"]        # (N, D)
            r_cls_id = bank["cls_id"]       # (N,)
            names = getattr(self, "modalities", [f"m{i}" for i in range(len(embeddings))])

            # -----------------------------------------
            # Candidate-dependent reranker path (SLOW)
            # -----------------------------------------
            cand_dep = True
            if cand_dep and getattr(self.gate, "gate_mode", None) == "attention":
                if self.modelname not in ("symile", "sigmile"):
                    raise ValueError("candidate-dependent gating implemented for symile/sigmile retrieval only.")

                Bk = emb_keep[0].shape[0]
                D = emb_keep[0].shape[1]
                M_total = len(emb_keep)
                query_indices = [i for i in range(M_total) if i != self.candidate_idx]
                if len(query_indices) == 0:
                    return []

                # chunk size (tune for memory/time)
                chunk_size = int(self.params_method.get("gate_candidate_chunk_size", 256))

                sum_w = torch.zeros((M_total,), device=self.device)
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

                    # IMPORTANT: query_mode="target" makes Q_t depend on the target embedding (the candidate)
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
                    count_w += int(w_pair.shape[0])

                    # symile/sigmile retrieval logits:
                    # logit(i,j) = < cand_j , Π_m gated_query_m(i,j) >
                    prod = torch.ones_like(gated_list[query_indices[0]])
                    for qi in query_indices:
                        prod = prod * gated_list[qi]  # (Bk*Nc, D)

                    prod = prod.view(Bk, Nc, D)  # (Bk, Nc, D)
                    raw = torch.einsum("bnd,nd->bn", prod, cand)  # (Bk, Nc)

                    logits_chunks.append(raw)

                logits = torch.cat(logits_chunks, dim=1)  # (Bk, N)

                # match losses/retrieval.py scaling for symile/sigmile
                M = len(query_indices) + 1
                scale_base = D ** ((M - 1) / 2)
                logits = logits * scale_base

                logits = logits * self.logit_scale.exp()
                if self.modelname == "sigmile" and self.bias is not None:
                    logits = logits + self.bias

                y = torch.tensor(batch["eids"]).to(self.device)[keep]
                pred = r_cls_id[torch.argmax(logits, dim=1)]

                if count_w > 0:
                    mean_w = sum_w / float(count_w)  # (M_total,)
                    for j in range(M_total):
                        # usually you care most about the non-candidate modalities,
                        # but you can log all of them if you want
                        self.log(
                            f"{split}/gate_{names[j]}_mean",
                            mean_w[j],
                            on_step=False,
                            on_epoch=True,
                            sync_dist=True,
                        )

                return (y == pred).float().tolist()

            # -----------------------------------------
            # Existing fast gating path (candidate-independent)
            # -----------------------------------------
            W_keep = self.gate.compute_W(emb_keep)
            gated_list, w_t, _ = self.gate.apply_for_target(self.candidate_idx, emb_keep, W=W_keep)
            names = getattr(self, "modalities", [f"m{i}" for i in range(len(embeddings))])
            
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
            rep_list = [embeddings[i][keep] for i in query_indices]  # list of (B_keep, D)
        #rep_list = [embeddings[i][keep] for i in query_indices]

        r_candidates = bank["r"]
        logits = zeroshot_retrieval_logits(
            r_candidates,
            rep_list,
            self.logit_scale.exp(),
            bias=self.bias,
            modelname=self.modelname,
        )

        y = torch.tensor(batch["eids"]).to(self.device)[keep]
        r_cls_id = bank["cls_id"]
        pred = r_cls_id[torch.argmax(logits, dim=1)]
        return (y == pred).float().tolist()

    @torch.no_grad()
    def _inbatch_retrieval_acc_top1(self, batch, embeddings):
        """
        In-batch retrieval: candidates are the SAME batch, so logits are (B,B).
        Returns scalar accuracy in [0,1] or None if invalid.
        """
        # eids are the "class id" we want to retrieve
        eids = torch.tensor(batch["eids"], device=self.device, dtype=torch.long)
        if eids.numel() == 0:
            return None

        # candidates: candidate modality embeddings from this batch
        r_candidates = embeddings[self.candidate_idx]  # (B, d)

        # queries: all other modality embeddings
        query_indices = [i for i in range(len(embeddings)) if i != self.candidate_idx]
        if not query_indices:
            return None

        if getattr(self, "use_gate", False) and getattr(self, "gate", None) is not None:
            gated_list, w_t, _ = self.gate.apply_for_target(self.candidate_idx, embeddings)  # no keep mask here
            rep_list = [gated_list[i] for i in query_indices]
        else:
            rep_list = [embeddings[i] for i in query_indices]

        # logits: (B, B)
        logits = zeroshot_retrieval_logits(
            r_candidates,
            rep_list,
            self.logit_scale.exp(),
            bias=self.bias,
            modelname=self.modelname,
        )

        pred_idx = torch.argmax(logits, dim=1)  # (B,)
        pred_eid = eids[pred_idx]               # (B,)
        return (pred_eid == eids).float().mean()
    
    def training_step(self, batch, batch_idx):
        # get loss + embeddings from the shared step (this also logs train/loss)
        loss, embeddings = self.shared_step(batch, "train", return_embeddings=True)

        acc = self._inbatch_retrieval_acc_top1(batch, embeddings)
        if acc is not None:
            self.log(
                "train/inbatch_acc_top1",
                acc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        # debug
        #emb = embeddings[0]   # MRI embedding
        #print("emb std:", emb.std().item(), "mean cosine offdiag:",
        #    (emb @ emb.t()).fill_diagonal_(0).mean().item())

        return loss
