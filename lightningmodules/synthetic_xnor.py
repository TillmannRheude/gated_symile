import torch
import torch.distributed as dist
from typing import Optional

from architecture import ModalityAttentionGate
from lightningmodules.utils import LightningModuleParent
from losses.retrieval import zeroshot_retrieval_logits
from losses.utils import scale_mip_dvs


class SyntheticXNORModel(LightningModuleParent):
    def __init__(
        self,
        model,
        params_retrival_ds: dict = None,
        **args,
    ):
        super().__init__(**args)

        self.dataset_name = "synthetic_xnor"
        self.model = model

        if params_retrival_ds is None:
            params_retrival_ds = {"batch_size": 128, "split_nr": 0}
        self.params_retrival_ds = params_retrival_ds

        self.modalities = ["A", "B", "C"]
        self.candidate_idx = 0  # retrieve A from (B,C)

        # required by LightningModuleParent.test_step aggregation path
        self.test_step_accuracies = []

        # gate 
        self.use_gate = bool(self.params_method["use_gate"])
        if self.use_gate:
            emb_dim = getattr(self.model.encoders[0], "emb_dim", None)
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

    def forward(self, x):
        if isinstance(x, dict):
            a = x["A"]
            b = x["B"]
            c = x["C"]
        else:
            a, b, c = x

        return self.model([a, b, c])

    def retrieval_step(self, batch, embeddings, split: str):
        """
        In-batch retrieval: for each query i we retrieve A_i from candidates {A_j}.
        Query is (B_i, C_i). Correct if argmax_j logit(i,j) == i.
        """
        r_a, r_b, r_c = embeddings
        if r_a.numel() == 0:
            return []

        # If training uses pair sampling with gating, make validation candidate-dependent too:
        # for each query i and candidate j, compute gate weights from (A_j, B_i, C_i).
        if (
            self.use_gate
            and self.gate is not None
            and self.params_method.get("negative_sampling") == "pair"
        ):
            B = int(r_a.shape[0])
            D = int(r_a.shape[1])

            # Build all (query i, candidate j) triples as a flat batch of size B*B.
            a_pair = r_a[None, :, :].expand(B, B, D).reshape(B * B, D)                 # (B*B, D)
            b_pair = r_b[:, None, :].expand(B, B, D).reshape(B * B, D)                 # (B*B, D)
            c_pair = r_c[:, None, :].expand(B, B, D).reshape(B * B, D)                 # (B*B, D)
            pair_embs = [a_pair, b_pair, c_pair]

            W_pair = self.gate.compute_W(pair_embs)
            gated_list, w_t, _ = self.gate.apply_for_target(self.candidate_idx, pair_embs, W=W_pair)

            # Log weights/alignments on the positive (i==j) pairs only (avoid O(B^2) logging).
            w_pos = w_t.view(B, B, -1).diagonal(dim1=0, dim2=1).transpose(0, 1).contiguous()  # (B, M)
            self._log_gate_weights(w_pos, set=split)
            self._log_gate_w_diff_bc(batch, w_pos, split=split)
            # Also log cosine similarities on positive (i==j) pairs.
            a_pos = r_a
            b_pos = r_b
            c_pos = r_c
            ga_pos = gated_list[0].view(B, B, D).diagonal(dim1=0, dim2=1).transpose(0, 1).contiguous()  # (B, D)
            gb_pos = gated_list[1].view(B, B, D).diagonal(dim1=0, dim2=1).transpose(0, 1).contiguous()  # (B, D)
            gc_pos = gated_list[2].view(B, B, D).diagonal(dim1=0, dim2=1).transpose(0, 1).contiguous()  # (B, D)
            emb_pos = [a_pos, b_pos, c_pos]
            gated_pos = [ga_pos, gb_pos, gc_pos]
            self._log_gate_cos_alignment(emb_pos, gated_pos, split=split, names=self.modalities)
            self._log_gate_cos_to_neutral(self.gate, gated_pos, split=split, names=self.modalities)

            # Candidate-dependent symile score for target A:
            # score(i,j) = <A_j, gated_B(i,j) ⊙ gated_C(i,j)>
            prod = gated_list[1] * gated_list[2]  # (B*B, D)
            raw = (prod * a_pair).sum(dim=1).view(B, B)  # (B, B)

            raw = scale_mip_dvs(raw, d=D, M=3)
            scale = self.get_logit_scale_exp()
            logits = raw if scale is None else scale * raw

            pred = torch.argmax(logits, dim=1)
            y = torch.arange(B, device=pred.device, dtype=pred.dtype)
            return (pred == y).float().tolist()

        rep_list = [r_b, r_c]
        if self.use_gate and self.gate is not None:
            emb = [r_a, r_b, r_c]
            W = self.gate.compute_W(emb)
            gated_list, w_t, _ = self.gate.apply_for_target(self.candidate_idx, emb, W=W)

            # lightweight logging
            self._log_gate_weights(w_t, set=split)
            self._log_gate_w_diff_bc(batch, w_t, split=split)
            self._log_gate_cos_alignment(emb, gated_list, split=split, names=self.modalities)
            self._log_gate_cos_to_neutral(self.gate, gated_list, split=split, names=self.modalities)
            if hasattr(self.gate, "logit_gate_strength"):
                alpha = torch.sigmoid(self.gate.logit_gate_strength.detach())
                self.log(f"{split}/gate_alpha", alpha, on_step=False, on_epoch=True, sync_dist=True)

            rep_list = [gated_list[1], gated_list[2]]

        if self.modelname == "symile_attention":
            B = int(r_a.shape[0])
            D = int(r_a.shape[1])

            # Build all query-candidate triplets:
            # row i = query (B_i, C_i), column j = candidate A_j
            a_pair = r_a[None, :, :].expand(B, B, D).reshape(B * B, D)   # candidates A_j
            b_pair = r_b[:, None, :].expand(B, B, D).reshape(B * B, D)   # query B_i
            c_pair = r_c[:, None, :].expand(B, B, D).reshape(B * B, D)   # query C_i

            # TransformerSymile expects a list of modality embeddings and returns one score per triplet
            z = self.model.transformer([a_pair, b_pair, c_pair])

            if z.dim() == 2 and z.shape[1] == 1:
                z = z.squeeze(1)
            elif z.dim() != 1:
                raise ValueError(f"Expected transformer score shape (B*B,) or (B*B,1), got {tuple(z.shape)}")

            logits = z.view(B, B)
            scale = self.get_logit_scale_exp()
            if scale is not None:
                logits = scale * logits

            if self.bias is not None:
                logits = logits + self.bias

            pred = torch.argmax(logits, dim=1)
            y = torch.arange(B, device=pred.device, dtype=pred.dtype)
            return (pred == y).float().tolist()
        else:
            logits = zeroshot_retrieval_logits(
                r_a,
                rep_list,
                self.get_logit_scale_exp(),
                bias=self.bias,
                modelname=self.modelname,
            )
        pred = torch.argmax(logits, dim=1)
        y = torch.arange(r_a.shape[0], device=pred.device, dtype=pred.dtype)
        return (pred == y).float().tolist()

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
        mask = mask.to(device=values.device, dtype=torch.bool)
        if mask.numel() == 0 or values.numel() == 0:
            return None

        v = values[mask]
        if v.numel() == 0:
            return None

        v_sum = v.float().sum()
        v_count = torch.tensor(float(v.numel()), device=values.device, dtype=torch.float32)

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(v_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(v_count, op=dist.ReduceOp.SUM)

        if float(v_count.item()) <= 0.0:
            return None
        return v_sum / v_count

    def _log_gate_w_diff_bc(self, batch: dict, w: torch.Tensor, split: str) -> None:
        """
        Logs the mean of w_B - w_C under three conditions:
          - clean:  B clean AND C clean
          - corrB:  B corrupted AND C clean
          - corrC:  C corrupted AND B clean
        """
        if w is None or w.numel() == 0:
            return

        try:
            idx_b = self.modalities.index("B")
            idx_c = self.modalities.index("C")
        except ValueError:
            return

        corr_b = batch.get("corr_b", None)
        corr_c = batch.get("corr_c", None)
        if corr_b is None or corr_c is None:
            return

        corr_b = corr_b.to(device=w.device).view(-1)
        corr_c = corr_c.to(device=w.device).view(-1)
        if corr_b.numel() != w.shape[0] or corr_c.numel() != w.shape[0]:
            return

        diff = (w[:, idx_b] - w[:, idx_c]).detach()
        mask_clean = (corr_b == 0) & (corr_c == 0)
        mask_corr_b = (corr_b != 0) & (corr_c == 0)
        mask_corr_c = (corr_c != 0) & (corr_b == 0)

        mean_clean = self._masked_mean(diff, mask_clean)
        mean_corr_b = self._masked_mean(diff, mask_corr_b)
        mean_corr_c = self._masked_mean(diff, mask_corr_c)

        if mean_clean is not None:
            self.log(
                f"{split}/gate_w_diff_BC_clean",
                mean_clean,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
        if mean_corr_b is not None:
            self.log(
                f"{split}/gate_w_diff_BC_corrB",
                mean_corr_b,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
        if mean_corr_c is not None:
            self.log(
                f"{split}/gate_w_diff_BC_corrC",
                mean_corr_c,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
    
    
