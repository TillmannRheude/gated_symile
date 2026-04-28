import torch
import torch.distributed as dist
import torch.nn as nn

from lightningmodules.utils import LightningModuleParent
from losses.retrieval import zeroshot_retrieval_logits


class MCMEDModel(LightningModuleParent):
    def __init__(
        self,
        model,
        params_retrival_ds: dict = {
            "batch_size": 128,
            "split_nr": 0,
        },
        **args,
    ):
        super().__init__(**args)

        self.dataset_name = "mcmed"
        self.model = model
        self.params_retrival_ds = params_retrival_ds

        self.modalities = ["waveforms_II", "rads", "numerics"]
        self.candidate_idx = 1  # retrieve rads from (waveforms_II, numerics)

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
        x = [
            batch["waveforms_II"],
            batch["rads"]["tokenized_impression_texts"],
            batch["numerics"],
        ]
        return self.model(x)

    @staticmethod
    def _index_tokenized_batch(tokenized_batch: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return {k: v[mask] for k, v in tokenized_batch.items()}

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

    @staticmethod
    def _normalize_report_texts(batch_texts) -> list[str]:
        normalized = []
        for sample_texts in batch_texts:
            if isinstance(sample_texts, str):
                text = sample_texts.strip()
            elif isinstance(sample_texts, (list, tuple)):
                text = " ".join(str(t).strip() for t in sample_texts if str(t).strip())
            else:
                text = ""
            normalized.append(text)
        return normalized

    def _radiology_present(self, batch_texts) -> torch.Tensor:
        normalized = self._normalize_report_texts(batch_texts)
        return torch.tensor([len(text) > 0 for text in normalized], dtype=torch.bool, device=self.device)

    def _query_keep_mask(self, batch) -> torch.Tensor:
        waveform_present = self._waveform_present(batch["waveforms_II"]).to(self.device)
        numerics_present = self._numerics_present(batch["numerics"]).to(self.device)
        rads_present = self._radiology_present(batch["rads"]["impression_texts"])
        return waveform_present & numerics_present & rads_present

    def build_candidate_bank(self, split):
        r_list, csn_list = [], []
        dl = self.trainer.datamodule.val_dataloader() if split == "val" else self.trainer.datamodule.test_dataloader()
        encoder_stack = self._get_encoder_stack()

        for batch in dl:
            rads_present = self._radiology_present(batch["rads"]["impression_texts"])
            if rads_present.sum().item() == 0:
                continue

            csn = batch["CSN"].to(self.device)[rads_present]
            tokenized = self._index_tokenized_batch(batch["rads"]["tokenized_impression_texts"], rads_present.cpu())

            reps = encoder_stack[self.candidate_idx](tokenized)
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
            return {"r": r_local, "csn": csn_local}

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

        return {"r": r_full, "csn": csn_full}

    def retrieval_step(self, batch, embeddings, split):
        bank = getattr(self, "candidate_bank", None)
        if bank is None:
            return []

        keep = self._query_keep_mask(batch)
        if keep.sum().item() == 0:
            return []

        r_wave = embeddings[0][keep]
        r_rads = embeddings[1][keep]
        r_num = embeddings[2][keep]
        y = batch["CSN"].to(self.device)[keep]

        if self.use_gate and self.gate is not None:
            emb_keep = [r_wave, r_rads, r_num]

            if getattr(self.gate, "gate_mode", None) == "attention":
                Bq = emb_keep[0].shape[0]
                D = emb_keep[0].shape[1]
                r_candidates = bank["r"]
                csn_candidates = bank["csn"]
                chunk_size = int(self.params_method.get("gate_candidate_chunk_size", 256))
                logits_chunks = []

                for s in range(0, r_candidates.shape[0], chunk_size):
                    cand = r_candidates[s : s + chunk_size]
                    nc = cand.shape[0]

                    pair_embs = [
                        emb_keep[0].unsqueeze(1).expand(Bq, nc, D).reshape(Bq * nc, D),
                        cand.unsqueeze(0).expand(Bq, nc, D).reshape(Bq * nc, D),
                        emb_keep[2].unsqueeze(1).expand(Bq, nc, D).reshape(Bq * nc, D),
                    ]

                    W_pair = self.gate.compute_W(pair_embs)
                    gated_list, w_pair, _ = self.gate.apply_for_target(self.candidate_idx, pair_embs, W=W_pair)

                    self._log_gate_weights(w_pair, set=split)
                    max_pairs = int(self.params_method.get("gate_log_max_pairs", 128))
                    pair_slice = [x[:max_pairs] for x in pair_embs]
                    gated_slice = [g[:max_pairs] for g in gated_list]
                    self._log_gate_cos_alignment(pair_slice, gated_slice, split=split, names=self.modalities)
                    self._log_gate_cos_to_neutral(self.gate, gated_slice, split=split, names=self.modalities)

                    prod = gated_list[0] * gated_list[2]
                    raw = (prod * pair_embs[1]).sum(dim=1).view(Bq, nc)
                    logits_chunks.append(raw)

                logits = torch.cat(logits_chunks, dim=1)
                scale = self.get_logit_scale_exp()
                if scale is not None:
                    logits = scale * logits
                pred = csn_candidates[torch.argmax(logits, dim=1)]
                return (y == pred).float().tolist()

            W = self.gate.compute_W(emb_keep)
            gated_list, w_t, _ = self.gate.apply_for_target(self.candidate_idx, emb_keep, W=W)
            self._log_gate_weights(w_t, set=split)
            self._log_gate_cos_alignment(emb_keep, gated_list, split=split, names=self.modalities)
            self._log_gate_cos_to_neutral(self.gate, gated_list, split=split, names=self.modalities)
            if hasattr(self.gate, "logit_gate_strength"):
                alpha = torch.sigmoid(self.gate.logit_gate_strength.detach())
                self.log(f"{split}/gate_alpha", alpha, on_step=False, on_epoch=True, sync_dist=True)
            rep_list = [gated_list[0], gated_list[2]]
        else:
            rep_list = [r_wave, r_num]

        if self.modelname == "symile_attention":
            Bq = int(r_wave.shape[0])
            D = int(r_wave.shape[1])
            Nc = int(bank["r"].shape[0])

            wave_pair = r_wave.unsqueeze(1).expand(Bq, Nc, D).reshape(Bq * Nc, D)
            rads_pair = bank["r"].unsqueeze(0).expand(Bq, Nc, D).reshape(Bq * Nc, D)
            num_pair = r_num.unsqueeze(1).expand(Bq, Nc, D).reshape(Bq * Nc, D)

            z = self.model.transformer([wave_pair, rads_pair, num_pair])
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

        pred = bank["csn"][torch.argmax(logits, dim=1)]
        return (y == pred).float().tolist()
