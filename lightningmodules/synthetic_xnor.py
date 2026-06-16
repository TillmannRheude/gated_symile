import torch
import torch.distributed as dist
import torch.nn.functional as F
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
        regularizer_params: dict = None,
        **args,
    ):
        super().__init__(**args)

        self.dataset_name = "synthetic_xnor"
        self.model = model

        if regularizer_params is None:
            regularizer_params = {
                "m": 0.5, 
                "M": 5.0, 
                "w_low": 10.0, 
                "w_high": 1.0,
                "lam": 1.0
            }
        self.regularizer_params = regularizer_params

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

        self._analysis_batch = None

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

        if split == "val" and self._analysis_batch is None:
            self._analysis_batch = {
                "A": batch["A"].detach().cpu(),
                "B": batch["B"].detach().cpu(),
                "C": batch["C"].detach().cpu(),
            }

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

    def _local_isometry_penalty(
        self,
        batch: dict,
        num_vecs: int = 2,
        every_n_steps: int = 1,
        num_modalities: int = 3,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Stochastic local-isometry penalty:
        ((||J(x)v||^2 / ||v||^2) - c)^2
        computed with JVP, cheap compared to full Jacobian SVD.
        """
        if every_n_steps > 1 and (int(self.global_step) % every_n_steps) != 0:
            return torch.zeros((), device=self.device)

        # synthetic keys
        keys = ["A", "B", "C"][: max(1, int(num_modalities))]
        encoders = (
            self.model.contrastive_model.encoders
            if hasattr(self.model, "contrastive_model")
            else self.model.encoders
        )

        penalties = []
        for i, k in enumerate(keys):
            x = batch[k]
            if x.ndim != 2 or x.shape[0] == 0:
                continue

            x0 = x[:].detach().to(self.device).requires_grad_(True)  # (1, Din)
            encoder = encoders[i]

            def f(inp):
                return encoder(inp)  # (1, Dout)

            mod_pen = torch.zeros((), device=x0.device)
            r_vals = []
            for _ in range(int(num_vecs)):
                v = torch.randn_like(x0)
                v = v / (v.norm(dim=1, keepdim=True) + eps)
                _, jv = torch.autograd.functional.jvp(
                    f, (x0,), (v,), create_graph=True, strict=False
                )
                r = jv.pow(2).sum(dim=1) / (v.pow(2).sum(dim=1) + eps)  # shape (B,)
                r_vals.append(r)
            
            m = self.regularizer_params["m"]
            M = self.regularizer_params["M"]
            w_low = self.regularizer_params["w_low"]
            w_high = self.regularizer_params["w_high"]

            # stack all sampled directional gains
            r_all = torch.stack(r_vals, dim=0)  # shape (num_vecs, B)
            r_min = r_all.min(dim=0).values
            r_max = r_all.max(dim=0).values
            r_lo = m * m
            r_hi = M * M
            pen_low = w_low * torch.relu(r_lo - r_min).pow(2)
            pen_high = w_high * torch.relu(r_max - r_hi).pow(2)
            mod_pen = mod_pen + (pen_low + pen_high).mean()
            penalties.append(mod_pen)

        return torch.stack(penalties).mean()

    def _exact_jacobian_spectral_penalty(
        self,
        batch: dict,
        every_n_steps: int = 20,
        num_modalities: int = 1,
        samples_per_modality: int = 1,
        smin_target: float = 0.5,
        smax_target: float = 3.0,
        cond_target: float = 20.0,
        w_smin: float = 1.0,
        w_smax: float = 0.1,
        w_cond: float = 0.5,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Exact Jacobian-spectrum hinge penalty on tiny subsets.
        Expensive: use every_n_steps > 1 and small samples_per_modality.
        """
        if every_n_steps > 1 and (int(self.global_step) % every_n_steps) != 0:
            return torch.zeros((), device=self.device)

        keys = ["A", "B", "C"][: max(1, int(num_modalities))]
        encoders = (
            self.model.contrastive_model.encoders
            if hasattr(self.model, "contrastive_model")
            else self.model.encoders
        )

        penalties = []

        for i, k in enumerate(keys):
            x = batch[k]
            if x.ndim != 2 or x.shape[0] == 0:
                continue

            encoder = encoders[i]
            n = min(int(samples_per_modality), int(x.shape[0]))

            for b in range(n):
                x0 = x[b].detach().to(self.device).clone().requires_grad_(True)  # (Din,)

                def f(inp):
                    # inp: (Din,) -> encoder expects batch -> (1, Dout) -> (Dout,)
                    return encoder(inp.unsqueeze(0)).squeeze(0)

                jac = torch.autograd.functional.jacobian(
                    f, x0, vectorize=True, create_graph=True
                )  # (Dout, Din)

                s = torch.linalg.svdvals(jac.float())
                smax = s.max()
                smin = s.min()
                cond = smax / (smin + eps)

                p_smin = torch.relu(torch.tensor(smin_target, device=smin.device) - smin).pow(2)
                p_smax = torch.relu(smax - torch.tensor(smax_target, device=smax.device)).pow(2)
                p_cond = torch.relu(cond - torch.tensor(cond_target, device=cond.device)).pow(2)

                pen = (w_smin * p_smin) + (w_smax * p_smax) + (w_cond * p_cond)
                penalties.append(pen)

        if len(penalties) == 0:
            return torch.zeros((), device=self.device)

        return torch.stack(penalties).mean()

    """ 
    def training_step(self, batch, batch_idx):
        # base objective from parent (already handles symile/transformer variants)
        base_loss, _ = super().shared_step(batch, set="train", return_embeddings=True)

        iso_pen = self._local_isometry_penalty(
            batch,
            num_vecs=32,          # your 1–2 vectors request
            every_n_steps=1,    # compute every few steps
            num_modalities=3,    # keep cheap initially
        )
        lam = 1.0
        total_loss = base_loss + lam * iso_pen

        self.log("train/iso_penalty", iso_pen.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_total", total_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return total_loss

        jac_pen = self._exact_jacobian_spectral_penalty(
            batch,
            every_n_steps=20,      # important for cost
            num_modalities=3,      # start cheap
            samples_per_modality=128,
            smin_target=0.5,
            smax_target=3.0,
            cond_target=20.0,
            w_smin=1.0,
            w_smax=0.1,
            w_cond=0.5,
        )
        lam = 1e-2
        total_loss = base_loss + lam * jac_pen
        self.log("train/jac_exact_penalty", jac_pen.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_total", total_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return total_loss
    """  

    def _center(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=0, keepdim=True)

    def _linear_cka(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._center(x.float())
        y = self._center(y.float())
        hsic_xy = torch.linalg.norm(x.T @ y, ord="fro").pow(2)
        hsic_xx = torch.linalg.norm(x.T @ x, ord="fro")
        hsic_yy = torch.linalg.norm(y.T @ y, ord="fro")
        denom = hsic_xx * hsic_yy + 1e-8
        return hsic_xy / denom

    def _subspace_cosine(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._center(x.float())
        y = self._center(y.float())
        qx, _ = torch.linalg.qr(x, mode="reduced")
        qy, _ = torch.linalg.qr(y, mode="reduced")
        s = torch.linalg.svdvals(qx.T @ qy)
        return s.mean()

    def _procrustes_error(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._center(x.float())
        y = self._center(y.float())
        x = x / (torch.linalg.norm(x) + 1e-8)
        y = y / (torch.linalg.norm(y) + 1e-8)
        m = x.T @ y
        u, _, vh = torch.linalg.svd(m, full_matrices=False)
        r = u @ vh
        return torch.linalg.norm(x @ r - y, ord="fro")

    def _pairwise_distance_corr(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.float()
        y = y.float()
        dx = torch.pdist(x)
        dy = torch.pdist(y)
        dx = dx - dx.mean()
        dy = dy - dy.mean()
        denom = torch.linalg.norm(dx) * torch.linalg.norm(dy) + 1e-8
        return torch.dot(dx, dy) / denom

    def _encoder_jacobian_stats(self, encoder, x0: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = x0.detach().clone().to(self.device).requires_grad_(True)

        def f(inp):
            return encoder(inp.unsqueeze(0)).squeeze(0)

        jac = torch.autograd.functional.jacobian(f, x0, vectorize=True)
        s = torch.linalg.svdvals(jac.float())
        s_max = s.max()
        s_min = s.min()
        return {
            "jac_smax": s_max,
            "jac_smean": s.mean(),
            "jac_smin": s_min,
            "jac_cond": s_max / (s_min + 1e-8),
            #"jac_frob": torch.linalg.norm(jac.float(), ord="fro"),
        }

    def analyze_geometry(self, split: str = "val") -> None:
        if self._analysis_batch is None or not self.trainer.is_global_zero:
            return

        batch = {k: v.to(self.device) for k, v in self._analysis_batch.items()}
        with torch.no_grad():
            model_output = self.forward(batch)
            embeddings = model_output["embeddings"]

        names = ["A", "B", "C"]
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                xi = embeddings[i].detach()
                xj = embeddings[j].detach()
                ni, nj = names[i], names[j]
                self.log(f"{split}/analysis_cka_{ni}_{nj}", self._linear_cka(xi, xj), on_step=False, on_epoch=True, sync_dist=False)
                self.log(f"{split}/analysis_subspace_cos_{ni}_{nj}", self._subspace_cosine(xi, xj), on_step=False, on_epoch=True, sync_dist=False)
                self.log(f"{split}/analysis_procrustes_{ni}_{nj}", self._procrustes_error(xi, xj), on_step=False, on_epoch=True, sync_dist=False)
                self.log(f"{split}/analysis_distcorr_{ni}_{nj}", self._pairwise_distance_corr(xi, xj), on_step=False, on_epoch=True, sync_dist=False)

        encoders = self.model.contrastive_model.encoders if hasattr(self.model, "contrastive_model") else self.model.encoders
        for name, encoder, x0 in zip(names, encoders, [batch["A"][0], batch["B"][0], batch["C"][0]]):
            stats = self._encoder_jacobian_stats(encoder, x0)
            for key, value in stats.items():
                self.log(f"{split}/analysis_{name}_{key}", value, on_step=False, on_epoch=True, sync_dist=False)

    def on_validation_epoch_start(self):
        self._analysis_batch = None
        super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
        if not getattr(self.trainer, "sanity_checking", False):
            self.analyze_geometry(split="val")
        super().on_validation_epoch_end()

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


class SyntheticXNORBimodalModel(LightningModuleParent):
    def __init__(
        self,
        model,
        params_retrival_ds: dict = None,
        **args,
    ):
        if "params_method" in args and bool(args["params_method"].get("use_gate", False)):
            params_method = dict(args["params_method"])
            params_method["use_gate"] = False
            args["params_method"] = params_method
        super().__init__(**args)

        self.dataset_name = "synthetic_xnor"
        self.model = model

        if params_retrival_ds is None:
            params_retrival_ds = {"batch_size": 128, "split_nr": 0}
        self.params_retrival_ds = params_retrival_ds

        self.modalities = ["A", "B"]
        self.candidate_idx = 0  # retrieve A from B
        self.test_step_accuracies = []
        self._analysis_batch = None

        # Bimodal synthetic path only supports CLIP for now.
        self.loss = bimodal_clip

        self.save_hyperparameters()

    def forward(self, x):
        if isinstance(x, dict):
            a = x["A"]
            b = x["B"]
        else:
            a, b = x
        return self.model([a, b])

    def retrieval_step(self, batch, embeddings, split: str):
        r_a, r_b = embeddings
        if r_a.numel() == 0:
            return []

        if split == "val" and self._analysis_batch is None:
            self._analysis_batch = {
                "A": batch["A"].detach().cpu(),
                "B": batch["B"].detach().cpu(),
            }

        logits = zeroshot_retrieval_logits(
            r_a,
            [r_b],
            self.get_logit_scale_exp(),
            bias=self.bias,
            modelname="clip",
        )
        pred = torch.argmax(logits, dim=1)
        y = torch.arange(r_a.shape[0], device=pred.device, dtype=pred.dtype)
        return (pred == y).float().tolist()

    def _center(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=0, keepdim=True)

    def _linear_cka(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._center(x.float())
        y = self._center(y.float())
        hsic_xy = torch.linalg.norm(x.T @ y, ord="fro").pow(2)
        hsic_xx = torch.linalg.norm(x.T @ x, ord="fro")
        hsic_yy = torch.linalg.norm(y.T @ y, ord="fro")
        denom = hsic_xx * hsic_yy + 1e-8
        return hsic_xy / denom

    def _subspace_cosine(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._center(x.float())
        y = self._center(y.float())
        qx, _ = torch.linalg.qr(x, mode="reduced")
        qy, _ = torch.linalg.qr(y, mode="reduced")
        s = torch.linalg.svdvals(qx.T @ qy)
        return s.mean()

    def _procrustes_error(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._center(x.float())
        y = self._center(y.float())
        x = x / (torch.linalg.norm(x) + 1e-8)
        y = y / (torch.linalg.norm(y) + 1e-8)
        m = x.T @ y
        u, _, vh = torch.linalg.svd(m, full_matrices=False)
        r = u @ vh
        return torch.linalg.norm(x @ r - y, ord="fro")

    def _pairwise_distance_corr(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.float()
        y = y.float()
        dx = torch.pdist(x)
        dy = torch.pdist(y)
        dx = dx - dx.mean()
        dy = dy - dy.mean()
        denom = torch.linalg.norm(dx) * torch.linalg.norm(dy) + 1e-8
        return torch.dot(dx, dy) / denom

    def _encoder_jacobian_stats(self, encoder, x0: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = x0.detach().clone().to(self.device).requires_grad_(True)

        def f(inp):
            return encoder(inp.unsqueeze(0)).squeeze(0)

        jac = torch.autograd.functional.jacobian(f, x0, vectorize=True)
        s = torch.linalg.svdvals(jac.float())
        s_max = s.max()
        s_min = s.min()
        return {
            "jac_smax": s_max,
            "jac_smean": s.mean(),
            "jac_smin": s_min,
            "jac_cond": s_max / (s_min + 1e-8),
            "jac_frob": torch.linalg.norm(jac.float(), ord="fro"),
        }

    def analyze_geometry(self, split: str = "val") -> None:
        if self._analysis_batch is None or not self.trainer.is_global_zero:
            return

        batch = {k: v.to(self.device) for k, v in self._analysis_batch.items()}
        with torch.no_grad():
            model_output = self.forward(batch)
            embeddings = model_output["embeddings"]

        x_a, x_b = embeddings
        self.log(f"{split}/analysis_cka_A_B", self._linear_cka(x_a, x_b), on_step=False, on_epoch=True, sync_dist=False)
        self.log(
            f"{split}/analysis_subspace_cos_A_B",
            self._subspace_cosine(x_a, x_b),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            f"{split}/analysis_procrustes_A_B",
            self._procrustes_error(x_a, x_b),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            f"{split}/analysis_distcorr_A_B",
            self._pairwise_distance_corr(x_a, x_b),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

        encoders = self.model.contrastive_model.encoders if hasattr(self.model, "contrastive_model") else self.model.encoders
        for name, encoder, x0 in zip(["A", "B"], encoders, [batch["A"][0], batch["B"][0]]):
            stats = self._encoder_jacobian_stats(encoder, x0)
            for key, value in stats.items():
                self.log(f"{split}/analysis_{name}_{key}", value, on_step=False, on_epoch=True, sync_dist=False)

    def on_validation_epoch_start(self):
        self._analysis_batch = None
        super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
        if not getattr(self.trainer, "sanity_checking", False):
            self.analyze_geometry(split="val")
        super().on_validation_epoch_end()
