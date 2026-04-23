import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class ModalityAttentionGate(nn.Module):
    def __init__(
        self,
        num_modalities: int,
        emb_dim: int,
        d_k: int = 256,
        temperature_init: float = 1.0,
        eps: float = 1e-6,
        gate_bias_init: float = 0.0,
        gate_type: str = "softmax",
        gate_mode: str = "matrix",
        gate_strength_init: float = -6.0,
        renormalize: bool = True,
        use_null: bool = True,
        neutral_type: str = "random_trainable",
    ):
        super().__init__()
        self.M = int(num_modalities)
        self.D = int(emb_dim)
        self.d_k = int(d_k)
        self.eps = float(eps)
        self.renormalize = bool(renormalize)
        self.gate_type = gate_type
        self.gate_mode = gate_mode

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

        # NULL gating
        self.use_null = use_null
        if self.use_null:
            self.null_logit = nn.Parameter(torch.full((self.M,), float(0.0)))  # (M,)
        else:
            self.null_logit = None

        # Neutral directions
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

                if self.use_null:
                    null_raw = (self.null_head(E[:, t, :]).squeeze(-1) + self.null_logit[t].float()) / temp  # fp32
                else:
                    null_raw = (self.null_head(E[:, t, :]).squeeze(-1) / temp)  # fp32

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
        return gated_list, w, W

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
        embeddings = [self.encoders[i](x[i]) for i in range(len(x))]
        return {
            "embeddings": embeddings
        }
