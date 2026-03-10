import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class ModalityAttentionGate(nn.Module):
    def __init__(
        self,
        num_modalities: int,
        emb_dim: int,
        num_heads: int = 4,
        d_k: int = 256,
        d_null: int = 64,
        temperature_init: float = 1.0,
        eps: float = 1e-6,
        gate_bias_init: float = 0.0,
        gate_type: str = "softmax",
        gate_mode: str = "matrix",
        gate_strength_init: float = -6.0,
        renormalize: bool = True,
        use_null: bool = True,
        neutral_type: str = "random_trainable",  # "ones", "random_frozen", "random_trainable", None
        # Debug plotting (OFF by default)
        debug_plot_pca: bool = False,
        debug_plot_pca_target: int = 0,
        debug_plot_pca_names: list[str] = None,
        debug_plot_pca_dir: str = "outputs/gate_pca",
        debug_plot_pca_every: int = 0,
        debug_plot_pca_max_points: int = 512,
        debug_plot_pca_only_positives: bool = True,
    ):
        super().__init__()
        self.M = int(num_modalities)
        self.D = int(emb_dim)
        self.d_k = int(d_k)
        self.eps = float(eps)
        self.renormalize = bool(renormalize)
        self.gate_type = gate_type
        self.gate_mode = gate_mode  # "attention" or "matrix"

        # matrix baseline
        if self.gate_mode == "matrix":
            self.W_global_raw = nn.Parameter(torch.zeros(self.M, self.M))
            with torch.no_grad():
                self.W_global_raw.zero_()
                if self.gate_type == "sigmoid":
                    # init to a pass-through prior (e.g., 0.9). Use 0.5 for neutral.
                    w0 = 0.5
                    init_logit = math.log(w0 / (1.0 - w0))
                    for t in range(self.M):
                        for m in range(self.M):
                            if m != t:
                                self.W_global_raw[t, m] = init_logit
                elif self.gate_type == "softmax":
                    # uniform (0.5/0.5 for M=3) + tiny noise to break symmetry
                    self.W_global_raw.add_(0.01 * torch.randn_like(self.W_global_raw))
                #p_ecg = 0.8
                #gap = math.log(p_ecg / (1.0 - p_ecg))  # ~1.386
                #self.W_global_raw[0, 1] = +gap  # ecg
                #self.W_global_raw[0, 2] = 0.0

        # NULL gating
        self.use_null = use_null
        if self.use_null:
            self.null_logit = nn.Parameter(torch.full((self.M,), float(0.0)))  # (M,)
        else:
            self.null_logit = None

        # Neutral directions ...
        self.neutral_type = neutral_type  
        if self.neutral_type == "ones":
            self.neutral = nn.Parameter(torch.ones(self.M, self.D) / math.sqrt(self.D), requires_grad=False)
        elif self.neutral_type == "random_frozen":
            neutral = F.normalize(torch.randn(self.M, self.D), dim=1, eps=1e-6)
            self.neutral = nn.Parameter(neutral, requires_grad=False)
        elif self.neutral_type == "random_trainable":
            neutral = F.normalize(torch.randn(self.M, self.D), dim=1, eps=1e-6)
            self.neutral = nn.Parameter(neutral, requires_grad=True)
        elif self.neutral_type is None or self.neutral_type == "None" or self.neutral_type == "none":
            self.neutral = None

        # temperature, bias, gate strength
        self.log_temp = nn.Parameter(torch.tensor(float(temperature_init)).log())
        self.logit_gate_strength = nn.Parameter(torch.tensor(float(gate_strength_init)))

        # attention stuff 
        #self.num_heads = int(num_heads)
        #assert self.d_k % self.num_heads == 0
        #self.d_head = self.d_k // self.num_heads
        if self.gate_mode == "attention":
            self.q_proj = nn.ModuleList([nn.Linear(self.D, self.d_k, bias=False) for _ in range(self.M)])
            self.k_proj = nn.ModuleList([nn.Linear(self.D, self.d_k, bias=False) for _ in range(self.M)])
            self.null_head = nn.Sequential(
                nn.LayerNorm(self.D),
                nn.Linear(self.D, self.d_k),
                nn.ReLU(),
                nn.Linear(self.d_k, 1),
            )
        else:
            self.q_proj = None
            self.k_proj = None
            self.null_head = None

        # Debug plotting config (no side effects unless enabled)
        self.debug_plot_pca = bool(debug_plot_pca)
        self.debug_plot_pca_target = int(debug_plot_pca_target)  # None if debug_plot_pca_target is None else int(debug_plot_pca_target)
        self.debug_plot_pca_names = None if debug_plot_pca_names is None else list(debug_plot_pca_names)
        self.debug_plot_pca_dir = str(debug_plot_pca_dir)
        self.debug_plot_pca_every = int(debug_plot_pca_every)
        self.debug_plot_pca_max_points = int(debug_plot_pca_max_points)
        self.debug_plot_pca_only_positives = bool(debug_plot_pca_only_positives)
        self._debug_plot_calls = 0

    def compute_W(self, embeddings: list[torch.Tensor]) -> torch.Tensor:

        if self.gate_mode == "matrix":
            B = embeddings[0].shape[0]
            device = embeddings[0].device

            # Temperature
            temp = F.softplus(self.log_temp) + 0.1

            # Weight matrix 
            raw = self.W_global_raw.to(device)  # (M, M)
            W = raw.new_zeros((B, self.M, self.M))
            for t in range(self.M):
                mask = torch.ones(self.M, device=device, dtype=torch.bool)
                mask[t] = False

                s_t = raw[t, mask][None, :].expand(B, -1)  # (B, M-1)
                if self.gate_type == "softmax":
                    if self.use_null:
                        null_raw = (self.null_logit[t] / temp).expand(B)  # (B,)   
                        s_aug = torch.cat([s_t, null_raw[:, None]], dim=1)  # (B, (M-1)+1)
                        p_aug = torch.softmax(s_aug, dim=1)
                        w_t = p_aug[:, :-1]  # sums to (1 - p_null)
                    else:
                        w_t = torch.softmax(s_t, dim=1)  # sums to 1
                elif self.gate_type == "sigmoid":
                    w_t = torch.sigmoid(s_t)
                    if self.use_null:
                        p_null = torch.sigmoid(self.null_logit[t] / temp)
                        w_t = w_t * (1.0 - p_null)

                W[:, t, mask] = w_t
                W[:, t, t] = 1.0

            return W
        
        if self.gate_mode == "attention":
            B = embeddings[0].shape[0]
            device = embeddings[0].device
            temp = (F.softplus(self.log_temp) + 0.1).float()

            E = torch.stack(embeddings, dim=1).float()  # (B,M,D) fp32

            Q_list = [self.q_proj[t](E[:, t, :]) for t in range(self.M)]
            K_list = [self.k_proj[m](E[:, m, :]) for m in range(self.M)]
            Q = F.normalize(torch.stack(Q_list, dim=1), dim=-1, eps=self.eps)  # fp32
            K = F.normalize(torch.stack(K_list, dim=1), dim=-1, eps=self.eps)  # fp32
            scores = torch.einsum("btd,bmd->btm", Q, K) / temp  # fp32

            W = torch.zeros((B, self.M, self.M), device=device, dtype=torch.float32)
            for t in range(self.M):
                mask = torch.ones(self.M, device=device, dtype=torch.bool)
                mask[t] = False

                s_t = scores[:, t, mask]  # fp32

                null_raw = (self.null_head(E[:, t, :]).squeeze(-1) + self.null_logit[t].float()) / temp  # fp32

                if self.gate_type == "softmax":
                    if self.use_null:
                        s_aug = torch.cat([s_t, null_raw[:, None]], dim=1)
                        p_aug = torch.softmax(s_aug, dim=1)
                        w_t = p_aug[:, :-1]  # fp32
                    else:
                        w_t = torch.softmax(s_t, dim=1)
                elif self.gate_type == "sigmoid":
                    w_t = torch.sigmoid(s_t)  # fp32
                    if self.use_null:
                        p_null = torch.sigmoid(null_raw)  # fp32
                        w_t = w_t * (1.0 - p_null[:, None])

                w_t = w_t.to(dtype=W.dtype)
                W[:, t, mask] = w_t
                W[:, t, t] = 1.0

            return W

    def apply_for_target(self, target_idx: int, embeddings: list[torch.Tensor], W: torch.Tensor = None):
        if W is None:
            W = self.compute_W(embeddings)  # (B,M,M)
        w = W[:, int(target_idx), :]  # (B,M)

        E = torch.stack(embeddings, dim=1)  # (B,M,D)
        N = self.neutral

        has_neutral = self.neutral_type is not None and self.neutral_type != "none" and self.neutral_type != "None"
        if has_neutral:
            if self.neutral_type == "random_trainable":
                N = F.normalize(N, dim=1, eps=self.eps)
            N = N[None, :, :] 
            # neutral gate
            G = w[:, :, None] * E + (1.0 - w[:, :, None]) * N  # (B,M,D)
        else:
            # "No neutral" ablation: only attenuate modalities by their gate weights (no neutral interpolation).
            # with renormalization enabled, pure scaling becomes direction-preserving and the gate
            # becomes (almost) a no-op for cosine/dot-product objectives. So we skip renormalization here.
            G = w[:, :, None] * E

        # gate strength
        alpha = torch.sigmoid(self.logit_gate_strength)
        G = (1.0 - alpha) * E + alpha * G

        if self.renormalize and has_neutral:
            G = F.normalize(G, dim=2, eps=self.eps)

        gated_list = [G[:, m, :] for m in range(self.M)]
        self._maybe_save_pca_plot(target_idx=int(target_idx), embeddings=embeddings, gated_list=gated_list, w=w)
        return gated_list, w, W

    def _maybe_save_pca_plot(
        self,
        target_idx: int,
        embeddings: list[torch.Tensor],
        gated_list: list[torch.Tensor],
        w: torch.Tensor,
    ) -> None:
        """
        Debug utility: saves a PCA(2) plot of embeddings before/after gating plus neutral vectors.
        Off by default. Enable by passing `debug_plot_pca=True` or setting `self.debug_plot_pca=True`.
        """
        if not getattr(self, "debug_plot_pca", False):
            return
        target_filter = getattr(self, "debug_plot_pca_target", 0)
        if target_filter is not None and int(target_idx) != int(target_filter):
            return

        self._debug_plot_calls += 1
        every = int(getattr(self, "debug_plot_pca_every", 0))
        if every > 0 and (self._debug_plot_calls % every) != 0:
            return

        try:
            import os
            import math
            import numpy as np
            import matplotlib.pyplot as plt
        except Exception:
            # If matplotlib isn't available, silently skip (debug-only feature).
            return

        if embeddings is None or gated_list is None:
            return
        if len(embeddings) != len(gated_list):
            return
        if len(embeddings) == 0:
            return

        with torch.no_grad():
            # Shapes
            B = int(embeddings[0].shape[0])
            M = int(len(embeddings))
            D = int(embeddings[0].shape[1])
            if B == 0 or D == 0:
                return

            # Sample points to keep plotting cheap
            max_points = int(getattr(self, "debug_plot_pca_max_points", 512))
            max_points = max(32, max_points)
            idx = torch.arange(B, device=embeddings[0].device)

            # Optionally restrict to "positives" only:
            # - if B is a perfect square, assume a full (Bk,Bk) candidate grid and take the diagonal.
            # - otherwise, try to infer the pair-block length (K+1) from repeats and take the first
            #   element in each block (column 0), which corresponds to positives in pair sampling.
            if getattr(self, "debug_plot_pca_only_positives", False) and B > 0:
                root = int(B ** 0.5)
                if root * root == B:
                    idx = (torch.arange(root, device=idx.device) * (root + 1)).to(torch.long)
                else:
                    block = None
                    if len(embeddings) > 1 and embeddings[1].shape[0] == B:
                        diffs = (embeddings[1][1:] - embeddings[1][:-1]).abs().sum(dim=1)
                        nz = torch.nonzero(diffs > 0, as_tuple=False)
                        if nz.numel() > 0:
                            block = int(nz[0].item()) + 1
                    if block is not None and block > 0 and (B % block) == 0:
                        idx = torch.arange(0, B, block, device=idx.device, dtype=torch.long)

            if idx.numel() > max_points:
                # deterministic subsample: spread indices
                step = max(1, int(idx.numel()) // max_points)
                idx = idx[::step][:max_points]

            E = torch.stack([e[idx] for e in embeddings], dim=1)  # (b,M,D)
            G = torch.stack([g[idx] for g in gated_list], dim=1)  # (b,M,D)

            # Neutral vectors (M,D)
            N = getattr(self, "neutral", None)
            if N is None:
                return
            N = N.to(device=E.device, dtype=E.dtype)
            if getattr(self, "neutral_type", None) == "random_trainable":
                N = F.normalize(N, dim=1, eps=self.eps)

            b = int(E.shape[0])

            # Prepare annotations (computed once; displayed on the whole figure)
            alpha = float(torch.sigmoid(self.logit_gate_strength.detach()).cpu().item()) if hasattr(self, "logit_gate_strength") else float("nan")
            w_mean = w.detach().float().mean(dim=0).cpu().numpy() if w is not None else None
            p_null = None
            if (
                getattr(self, "gate_type", None) == "softmax"
                and getattr(self, "use_null", False)
                and w_mean is not None
            ):
                # compute_W sets W[t,t]=1, and distributes mass across other modalities (excluding null).
                # p_null is the leftover mass for (m!=t) weights.
                other_sum = float(np.sum([w_mean[m] for m in range(M) if m != int(target_idx)]))
                p_null = max(0.0, 1.0 - other_sum)

            # Plot
            os.makedirs(self.debug_plot_pca_dir, exist_ok=True)
            fname = f"gate_pca_target{int(target_idx)}_call{int(self._debug_plot_calls)}.png"
            out_path = os.path.join(self.debug_plot_pca_dir, fname)

            names = getattr(self, "debug_plot_pca_names", None)
            if not names or len(names) != M:
                names = [f"m{m}" for m in range(M)]

            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            colors = (colors * ((M + len(colors) - 1) // len(colors)))[:M]

            # One figure with M panels (one per modality), each with PCA fit on that modality's ORIG points.
            fig, axes = plt.subplots(1, M, figsize=(5.5 * M, 5.5), sharex=False, sharey=False)
            if M == 1:
                axes = [axes]

            def _pca2_fit_and_proj(x_fit: torch.Tensor, x_list: list[torch.Tensor]) -> list[np.ndarray]:
                x_fit_cpu = x_fit.detach().cpu().float()
                mu = x_fit_cpu.mean(dim=0, keepdim=True)
                Xc = x_fit_cpu - mu
                try:
                    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
                    P = Vh[:2].T  # (D,2)
                except Exception:
                    return None

                out = []
                for x in x_list:
                    x_cpu = x.detach().cpu().float()
                    out.append(((x_cpu - mu) @ P).numpy())
                return out

            for m in range(M):
                ax = axes[m]

                e_m = E[:, m, :]  # (b,D)
                g_m = G[:, m, :]  # (b,D)
                n_m = N[m : m + 1, :]  # (1,D)

                proj = _pca2_fit_and_proj(e_m, [e_m, g_m, n_m])
                if proj is None:
                    continue
                z_e_m, z_g_m, z_n_m = proj

                ax.scatter(z_e_m[:, 0], z_e_m[:, 1], s=18, c=colors[m], alpha=0.30, marker="o")
                ax.scatter(z_g_m[:, 0], z_g_m[:, 1], s=18, c=colors[m], alpha=0.85, marker="x")
                ax.scatter(
                    z_n_m[0, 0],
                    z_n_m[0, 1],
                    s=260,
                    c=colors[m],
                    alpha=1.0,
                    marker="*",
                    edgecolors="black",
                    linewidths=0.6,
                )
                ax.annotate("neutral", (z_n_m[0, 0], z_n_m[0, 1]), textcoords="offset points", xytext=(6, 6), fontsize=10)

                ax.set_title(names[m], fontsize=13)
                ax.set_xlabel("PC1")
                if m == 0:
                    ax.set_ylabel("PC2")
                ax.grid(True, linestyle="--", alpha=0.25)

            # Figure-level title + annotation
            title = f"Gate PCA per modality (target={int(target_idx)})"
            if w_mean is not None:
                other = [f"w({names[m]}→target)={w_mean[m]:.2f}" for m in range(M) if m != int(target_idx)]
                w_str = ", ".join(other) if other else "w_mean=N/A"
            else:
                w_str = "w_mean=N/A"
            extra = f"α={alpha:.2f}"
            if p_null is not None and not math.isnan(p_null):
                extra += f", p_null≈{p_null:.2f}"
            fig.suptitle(f"{title}\n{extra} | {w_str}", fontsize=14, y=1.02)

            # Legend (markers only; colors are obvious per panel title)
            from matplotlib.lines import Line2D
            marker_handles = [
                Line2D([0], [0], color="black", marker="o", linestyle="None", markersize=8, label="orig"),
                Line2D([0], [0], color="black", marker="x", linestyle="None", markersize=8, label="gated"),
                Line2D([0], [0], color="black", marker="*", linestyle="None", markersize=12, label="neutral"),
            ]
            axes[-1].legend(handles=marker_handles, fontsize=10, loc="best", framealpha=0.9)

            fig.tight_layout()
            fig.savefig(out_path, dpi=160)
            plt.close(fig)


class Contrastive_Model(nn.Module):
    def __init__(
        self,
        encoders: nn.ModuleList = nn.ModuleList([]),
    ):
        super().__init__()

        self.encoders = encoders

    def forward(
        self, 
        x: list = [torch.Tensor, torch.Tensor, torch.Tensor],
        ):

        # debug: random shuffle of the last (labs) modality
        #perm = torch.randperm(x[-1].shape[0], device=x[-1].device)
        #x[-1] = x[-1][perm]  # .detach()

        embeddings = [self.encoders[i](x[i]) for i in range(len(x))]

        return {
            "embeddings": embeddings
        }
