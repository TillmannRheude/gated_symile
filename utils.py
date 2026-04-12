import math
import torch
import torch.nn as nn


class SigmoidSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()

        self.norm_first = True
        self.batch_first = True

        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        gates = torch.sigmoid(scores)
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)
        gates = self.dropout(gates)

        out = torch.matmul(gates, v)  # [B, H, T, Hd]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        out = x + self.alpha * out
        return out

class SigmoidTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = SigmoidSelfAttention(d_model, nhead, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class PatchEncoder_CXR(nn.Module):
    def __init__(self, image_size=320, patch_size=64, in_channels=3, emb_dim=256):
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.emb_dim = int(emb_dim)

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
            )

        self.patch_dim = self.in_channels * self.patch_size * self.patch_size
        self.proj = nn.Linear(self.patch_dim, self.emb_dim)

    def forward(self, x):
        # x: [B, 3, 320, 320]
        B, C, H, W = x.shape
        if C != self.in_channels or H != self.image_size or W != self.image_size:
            raise ValueError(
                f"Expected input of shape [B,{self.in_channels},{self.image_size},{self.image_size}], got {tuple(x.shape)}"
            )

        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)          # [B, C, H/p, W/p, p, p]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()   # [B, H/p, W/p, C, p, p]
        x = x.view(B, -1, self.patch_dim)              # [B, T, C*p*p]
        x = self.proj(x)                               # [B, T, D]
        return x

class PatchEncoder_ECG(nn.Module):
    def __init__(
        self,
        input_size=(5000, 12),
        patch_size=(250, 12),
        in_channels=1,
        emb_dim=256,
    ):
        super().__init__()
        self.input_size = tuple(input_size)
        self.patch_size = tuple(patch_size)
        self.in_channels = int(in_channels)
        self.emb_dim = int(emb_dim)

        if len(self.input_size) != 2 or len(self.patch_size) != 2:
            raise ValueError("input_size and patch_size must be 2D tuples")
        if self.input_size[0] % self.patch_size[0] != 0 or self.input_size[1] % self.patch_size[1] != 0:
            raise ValueError(
                f"input_size {self.input_size} must be divisible by patch_size {self.patch_size}"
            )

        self.patch_dim = self.in_channels * self.patch_size[0] * self.patch_size[1]
        self.proj = nn.Linear(self.patch_dim, self.emb_dim)

    def forward(self, x):
        # x: [B, 1, 5000, 12]
        B, C, H, W = x.shape
        if C != self.in_channels or (H, W) != self.input_size:
            raise ValueError(
                f"Expected input of shape [B,{self.in_channels},{self.input_size[0]},{self.input_size[1]}], got {tuple(x.shape)}"
            )

        ph, pw = self.patch_size
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)      # [B, C, H/ph, W/pw, ph, pw]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()   # [B, H/ph, W/pw, C, ph, pw]
        x = x.view(B, -1, self.patch_dim)              # [B, T, C*ph*pw]
        x = self.proj(x)                               # [B, T, D]
        return x


def mimetic_init_svd_(
    module: nn.Module,
    alpha1: float = 0.7,
    beta1: float = 0.7,
    alpha2: float = 0.4,
    beta2: float = 0.4
) -> None:
    """
    Applies mimetic initialization using SVD factorization.

    [1] A. Trockman and J. Z. Kolter, “Mimetic initialization of self-attention layers” 
    """
    if isinstance(module, (nn.MultiheadAttention)):
        print("applying mimetic_init_svd_ to multi head attention")
        embed_dim = module.embed_dim
        device = module.in_proj_weight.device
        dtype = module.in_proj_weight.dtype
        num_heads = module.num_heads
        head_dim = embed_dim // num_heads

        with torch.no_grad():
            eye = torch.eye(embed_dim, device=device, dtype=dtype)
            Z1 = torch.randn(embed_dim, embed_dim, device=device, dtype=dtype) * (1/embed_dim)

            # W_Q and W_K source calculation (embed_dim, embed_dim)
            W_Q_W_KT = (alpha1 * Z1) - (beta1 * eye)
            U_1, S_1, V_1T = torch.linalg.svd(W_Q_W_KT, full_matrices=True) # U_1, V_1T are embed_dim x embed_dim
            S_1 = torch.diag(torch.sqrt(S_1))

            # Construct W_V and W_proj from SVD of W_Q_W_KT
            W_V = U_1 @ S_1
            W_proj = V_1T @ (S_1**0.5)

            # Process each head separately for Q and K with a new Z2
            W_Q = torch.zeros(embed_dim, embed_dim, device=device, dtype=dtype)
            W_K = torch.zeros(embed_dim, embed_dim, device=device, dtype=dtype)
            for h in range(num_heads):
                Z2 = torch.randn(embed_dim, embed_dim, device=device, dtype=dtype) * (1/embed_dim)
                W_V_W_Tproj = (alpha2 * Z2) + (beta2 * eye)
                U_2, S_2, V_2T = torch.linalg.svd(W_V_W_Tproj, full_matrices=False)
                S_2 = torch.diag(torch.sqrt(S_2))

                head_W_Q = U_2[:, :head_dim] @ (S_2[:head_dim, :head_dim]**0.5) # (d, k) @ (k, k) -> (d, k)
                head_W_K = V_2T.T[:, :head_dim] @ (S_2[:head_dim, :head_dim]**0.5) # (d, k) @ (k, k) -> (d, k)

                # Assign to the appropriate slice of the final weight matrices
                start_idx = h * head_dim
                end_idx = (h + 1) * head_dim
                W_Q[:, start_idx:end_idx] = head_W_Q
                W_K[:, start_idx:end_idx] = head_W_K

            # Assign concatenated/calculated module weights
            module.in_proj_weight.data[:embed_dim] = W_Q
            module.in_proj_weight.data[embed_dim:2*embed_dim] = W_K
            module.in_proj_weight.data[2*embed_dim:] = W_V
            if isinstance(module.out_proj, nn.Linear):
                module.out_proj.weight.data = W_proj
            else: 
                module.out_proj.data = W_proj

            # Zero Bias Initialization
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)

            print("mimetic_init_svd_ applied to multi head attention")
