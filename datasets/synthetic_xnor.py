import torch
from torch.utils.data import Dataset
from typing import Dict


class SyntheticXNOR(Dataset):
    def __init__(
        self, 
        n_samples: int = 100000,
        dims_modality: list[int] = [128, 128, 128],
        feature_layout: str = "1d",
        seq_len: int = 4,
        n_bits: int = 1,
        embed_mode: str = "xnor_only",
        bc_corr_exclusive: bool = False,
        bc_corr_p: float = None,
        bc_corr_split: float = 0.5,
        p_flips: list[float] = [0.10, 0.10],
        p_corrs: list[float] = [0.00, 0.20, 0.20],
        corr_modes: str = "swap_signal",
        signal_scale: float = 3.0,
        distractor_std: float = 1.0,
        a_rule: str = "xor",
        seed: int = 420
    ):
        super().__init__()

        self.n_samples = n_samples
        self.dims_modality = dims_modality
        self.feature_layout = str(feature_layout)
        self.seq_len = int(seq_len)
        self.n_bits = int(n_bits)
        self.embed_mode = str(embed_mode)
        self.bc_corr_exclusive = bool(bc_corr_exclusive)
        self.bc_corr_p = None if bc_corr_p is None else float(bc_corr_p)
        self.bc_corr_split = float(bc_corr_split)
        self.p_flips = p_flips
        self.p_corrs = p_corrs
        self.corr_modes = corr_modes
        self.signal_scale = signal_scale
        self.distractor_std = distractor_std
        self.a_rule = a_rule
        self.seed = seed

        g = torch.Generator().manual_seed(int(seed))
        N = int(n_samples)

        if self.n_bits < 1:
            raise ValueError(f"n_bits must be >= 1, got {self.n_bits}")
        if self.feature_layout not in ("1d", "2d"):
            raise ValueError(f"feature_layout must be '1d' or '2d', got {self.feature_layout}")
        if self.feature_layout == "2d" and self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1 when feature_layout='2d', got {self.seq_len}")
        if self.embed_mode not in ("xnor_only", "u_v_uv"):
            raise ValueError(f"Unknown embed_mode: {self.embed_mode} (expected 'xnor_only' or 'u_v_uv')")
        if not (0.0 <= self.bc_corr_split <= 1.0):
            raise ValueError(f"bc_corr_split must be in [0,1], got {self.bc_corr_split}")

        self.signal_dim = self.n_bits if self.embed_mode == "xnor_only" else 3 * self.n_bits
        effective_dims = [self._effective_dim(int(d)) for d in self.dims_modality]
        if any(d < self.signal_dim for d in effective_dims):
            raise ValueError(
                f"All dims_modality must be >= signal_dim. Got dims_modality={self.dims_modality}, "
                f"signal_dim={self.signal_dim} (embed_mode={self.embed_mode}, n_bits={self.n_bits})"
            )

        # latent bits (N, K)
        u = torch.bernoulli(torch.full((N, self.n_bits), 0.5), generator=g).to(torch.int64)
        v = torch.bernoulli(torch.full((N, self.n_bits), 0.5), generator=g).to(torch.int64)
        self.u, self.v = u, v

        # a label bits
        if a_rule == "xor":
            a_bits = (u ^ v).to(torch.int64)
        elif a_rule == "xnor":
            a_bits = (1 - (u ^ v)).to(torch.int64)
        else:
            raise ValueError(f"Unknown a_rule: {a_rule}")

        # store both bit-vector and an integer label
        self.y_bits = a_bits
        if self.n_bits == 1:
            self.y = a_bits.view(-1)
        else:
            # pack bits little-endian into an int64 label in [0, 2^K)
            weights = (2 ** torch.arange(self.n_bits, dtype=torch.int64))
            self.y = (a_bits * weights[None, :]).sum(dim=1).to(torch.int64)

        # noisy bits for B and C (BSC bit-flip)
        b_bits = self._flip_bits(u, float(p_flips[0]), g)
        c_bits = self._flip_bits(v, float(p_flips[1]), g)

        # embed to vectors (signal in first signal_dim coords, rest distractors)
        if self.embed_mode == "xnor_only":
            A = self._embed_bits(a_bits, self._effective_dim(int(dims_modality[0])), g)
            B = self._embed_bits(b_bits, self._effective_dim(int(dims_modality[1])), g)
            C = self._embed_bits(c_bits, self._effective_dim(int(dims_modality[2])), g)
        else:
            # Redundant A signal to make per-sample gating meaningful:
            #   A = [u, v, u*v] where u*v is XOR/XNOR in {-1,+1} space
            #   B = [u, 1, u],  C = [1, v, v]  =>  B⊙C = [u, v, u*v]
            ones = torch.ones_like(u, dtype=torch.int64)
            uv_bits = (u ^ v).to(torch.int64) if a_rule == "xor" else (1 - (u ^ v)).to(torch.int64)

            A = self._embed_blocks([u, v, uv_bits], self._effective_dim(int(dims_modality[0])), g)
            B = self._embed_blocks([b_bits, ones, b_bits], self._effective_dim(int(dims_modality[1])), g)
            C = self._embed_blocks([ones, c_bits, c_bits], self._effective_dim(int(dims_modality[2])), g)

        # corruption masks (fixed once)
        mask_a = (torch.rand(N, generator=g) < float(p_corrs[0]))
        if self.bc_corr_exclusive:
            if self.bc_corr_p is None:
                pb = float(p_corrs[1])
                pc = float(p_corrs[2])
                if abs(pb - pc) > 1e-9:
                    raise ValueError(
                        "bc_corr_exclusive=True requires either bc_corr_p to be set, "
                        f"or p_corrs[1] == p_corrs[2]. Got p_corrs={p_corrs}."
                    )
                p_bc = pb
            else:
                p_bc = float(self.bc_corr_p)

            mask_bc = (torch.rand(N, generator=g) < p_bc)
            choose_b = (torch.rand(N, generator=g) < self.bc_corr_split)
            mask_b = mask_bc & choose_b
            mask_c = mask_bc & (~choose_b)
        else:
            mask_b = (torch.rand(N, generator=g) < float(p_corrs[1]))
            mask_c = (torch.rand(N, generator=g) < float(p_corrs[2]))

        # swap indices for swap-signal corruption
        swap_a = torch.randperm(N, generator=g)
        swap_b = torch.randperm(N, generator=g)
        swap_c = torch.randperm(N, generator=g)

        A = self._corrupt(A, mask_a, swap_a, g)
        B = self._corrupt(B, mask_b, swap_b, g)
        C = self._corrupt(C, mask_c, swap_c, g)

        self.A = self._to_feature_layout(A, int(dims_modality[0]))
        self.B = self._to_feature_layout(B, int(dims_modality[1]))
        self.C = self._to_feature_layout(C, int(dims_modality[2]))
        self.corr_a, self.corr_b, self.corr_c = mask_a, mask_b, mask_c
        corr_sum = mask_a.to(torch.int64) + mask_b.to(torch.int64) + mask_c.to(torch.int64)
        corr_mod = torch.zeros(N, dtype=torch.int64)
        corr_mod[corr_sum == 0] = 0
        corr_mod[corr_sum > 1] = 4
        corr_mod[(corr_sum == 1) & mask_a] = 1
        corr_mod[(corr_sum == 1) & mask_b] = 2
        corr_mod[(corr_sum == 1) & mask_c] = 3
        self.corr_mod = corr_mod

    def __len__(self) -> int:
        return int(self.n_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "A": self.A[idx].float(),
            "B": self.B[idx].float(),
            "C": self.C[idx].float(),
            "y": self.y[idx].long(),
            "corr_a": self.corr_a[idx].long(),
            "corr_b": self.corr_b[idx].long(),
            "corr_c": self.corr_c[idx].long(),
            # 0=none, 1=A only, 2=B only, 3=C only, 4=multiple corrupted
            "corr_mod": self.corr_mod[idx].long(),
        }

    def _flip_bits(self, bit: torch.Tensor, p: float, g: torch.Generator) -> torch.Tensor:
        if p <= 0.0:
            return bit
        flip = (torch.rand(bit.shape, generator=g) < p).to(torch.int64)
        return (bit ^ flip).to(torch.int64)

    def _embed_bits(self, bit: torch.Tensor, d: int, g: torch.Generator) -> torch.Tensor:
        """
        bit: (N, K) int64 in {0,1}
        Embeds K bits into the first K coordinates as {-scale, +scale} and fills
        the remaining coordinates with Gaussian distractors.
        """
        if bit.dim() == 1:
            bit = bit[:, None]
        if bit.shape[1] != self.n_bits:
            raise ValueError(f"Expected bit.shape[1] == n_bits ({self.n_bits}), got {bit.shape}")

        X = torch.randn((bit.shape[0], d), generator=g) * float(self.distractor_std)
        # {0,1}->{-1,+1} on the signal coordinates
        X[:, : self.n_bits] = (2.0 * bit.float() - 1.0) * float(self.signal_scale)
        return X

    def _corrupt(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        swap_idx: torch.Tensor,
        g: torch.Generator,
    ) -> torch.Tensor:
        if mask.sum().item() == 0:
            return X

        mode = self.corr_modes
        if mode == "swap_signal":
            # swap only signal coords: misleading but still “looks like” valid signal
            X = X.clone()
            X[mask, : self.signal_dim] = X[swap_idx[mask], : self.signal_dim]
            return X

        if mode == "gaussian":
            # replace full vector: uninformative
            X = X.clone()
            n = int(mask.sum().item())
            X[mask] = torch.randn((n, X.shape[1]), generator=g) * float(self.distractor_std)
            return X

        raise ValueError(f"Unknown corr_mode: {mode}")

    def _embed_blocks(self, blocks: list[torch.Tensor], d: int, g: torch.Generator) -> torch.Tensor:
        """
        Embeds multiple bit-blocks into the first `signal_dim` coordinates.
        Each block is mapped {0,1}->{-scale,+scale}. Remaining dims are distractors.

        blocks: list of (N, K_i) int64 in {0,1}, with sum(K_i) == signal_dim
        """
        if not blocks:
            raise ValueError("blocks must be non-empty")
        N = int(blocks[0].shape[0])
        widths = []
        for b in blocks:
            if b.dim() == 1:
                b = b[:, None]
            if int(b.shape[0]) != N:
                raise ValueError("All blocks must share the same N")
            widths.append(int(b.shape[1]))

        total = int(sum(widths))
        if total != int(self.signal_dim):
            raise ValueError(f"Sum of block widths must equal signal_dim={self.signal_dim}, got {widths} (sum={total})")
        if int(d) < total:
            raise ValueError(f"Embedding dim d must be >= signal_dim={total}, got d={d}")

        X = torch.randn((N, d), generator=g) * float(self.distractor_std)
        start = 0
        for b, w in zip(blocks, widths):
            if b.dim() == 1:
                b = b[:, None]
            X[:, start : start + w] = (2.0 * b.float() - 1.0) * float(self.signal_scale)
            start += w
        return X

    def _effective_dim(self, d: int) -> int:
        if self.feature_layout == "2d":
            return int(d) * int(self.seq_len)
        return int(d)

    def _to_feature_layout(self, x: torch.Tensor, d_mod: int) -> torch.Tensor:
        if self.feature_layout == "2d":
            x = x[:, :int(d_mod)]  # keep first token-sized block
            return x.unsqueeze(1).repeat(1, int(self.seq_len), 1)
        return x
        if self.feature_layout == "2d":
            return x.view(x.shape[0], int(self.seq_len), int(d_mod))
        return x


class SyntheticXNORBimodal(Dataset):
    def __init__(
        self,
        n_samples: int = 100000,
        dims_modality: list[int] = [128, 128],
        feature_layout: str = "1d",
        seq_len: int = 4,
        n_bits: int = 1,
        embed_mode: str = "xnor_only",
        ab_corr_exclusive: bool = False,
        ab_corr_p: float = None,
        ab_corr_split: float = 0.5,
        p_flips: list[float] = [0.10],
        p_corrs: list[float] = [0.00, 0.20],
        corr_modes: str = "swap_signal",
        signal_scale: float = 3.0,
        distractor_std: float = 1.0,
        a_rule: str = "xor",
        seed: int = 420,
    ):
        super().__init__()

        self.n_samples = n_samples
        self.dims_modality = dims_modality
        self.feature_layout = str(feature_layout)
        self.seq_len = int(seq_len)
        self.n_bits = int(n_bits)
        self.embed_mode = str(embed_mode)
        self.ab_corr_exclusive = bool(ab_corr_exclusive)
        self.ab_corr_p = None if ab_corr_p is None else float(ab_corr_p)
        self.ab_corr_split = float(ab_corr_split)
        self.p_flips = p_flips
        self.p_corrs = p_corrs
        self.corr_modes = corr_modes
        self.signal_scale = signal_scale
        self.distractor_std = distractor_std
        self.a_rule = a_rule
        self.seed = seed

        g = torch.Generator().manual_seed(int(seed))
        N = int(n_samples)

        if self.n_bits < 1:
            raise ValueError(f"n_bits must be >= 1, got {self.n_bits}")
        if self.feature_layout not in ("1d", "2d"):
            raise ValueError(f"feature_layout must be '1d' or '2d', got {self.feature_layout}")
        if self.feature_layout == "2d" and self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1 when feature_layout='2d', got {self.seq_len}")
        if len(self.dims_modality) != 2:
            raise ValueError(f"dims_modality must have length 2, got {self.dims_modality}")
        if self.embed_mode not in ("xnor_only", "u_v_uv"):
            raise ValueError(f"Unknown embed_mode: {self.embed_mode} (expected 'xnor_only' or 'u_v_uv')")
        if not (0.0 <= self.ab_corr_split <= 1.0):
            raise ValueError(f"ab_corr_split must be in [0,1], got {self.ab_corr_split}")

        self.signal_dim = self.n_bits if self.embed_mode == "xnor_only" else 3 * self.n_bits
        effective_dims = [self._effective_dim(int(d)) for d in self.dims_modality]
        if any(d < self.signal_dim for d in effective_dims):
            raise ValueError(
                f"All dims_modality must be >= signal_dim. Got dims_modality={self.dims_modality}, "
                f"signal_dim={self.signal_dim} (embed_mode={self.embed_mode}, n_bits={self.n_bits})"
            )

        u = torch.bernoulli(torch.full((N, self.n_bits), 0.5), generator=g).to(torch.int64)
        v = torch.bernoulli(torch.full((N, self.n_bits), 0.5), generator=g).to(torch.int64)

        if a_rule == "xor":
            a_bits = (u ^ v).to(torch.int64)
        elif a_rule == "xnor":
            a_bits = (1 - (u ^ v)).to(torch.int64)
        else:
            raise ValueError(f"Unknown a_rule: {a_rule}")

        self.y_bits = a_bits
        if self.n_bits == 1:
            self.y = a_bits.view(-1)
        else:
            weights = (2 ** torch.arange(self.n_bits, dtype=torch.int64))
            self.y = (a_bits * weights[None, :]).sum(dim=1).to(torch.int64)

        if self.embed_mode == "xnor_only":
            b_bits = self._flip_bits(a_bits, float(p_flips[0]), g)
            A = self._embed_bits(a_bits, self._effective_dim(int(dims_modality[0])), g)
            B = self._embed_bits(b_bits, self._effective_dim(int(dims_modality[1])), g)
        else:
            uv_bits = (u ^ v).to(torch.int64) if a_rule == "xor" else (1 - (u ^ v)).to(torch.int64)
            bu = self._flip_bits(u, float(p_flips[0]), g)
            bv = self._flip_bits(v, float(p_flips[0]), g)
            buv = self._flip_bits(uv_bits, float(p_flips[0]), g)
            A = self._embed_blocks([u, v, uv_bits], self._effective_dim(int(dims_modality[0])), g)
            B = self._embed_blocks([bu, bv, buv], self._effective_dim(int(dims_modality[1])), g)

        if self.ab_corr_exclusive:
            if self.ab_corr_p is None:
                pa = float(p_corrs[0])
                pb = float(p_corrs[1])
                if abs(pa - pb) > 1e-9:
                    raise ValueError(
                        "ab_corr_exclusive=True requires either ab_corr_p to be set, "
                        f"or p_corrs[0] == p_corrs[1]. Got p_corrs={p_corrs}."
                    )
                p_ab = pa
            else:
                p_ab = float(self.ab_corr_p)

            mask_ab = (torch.rand(N, generator=g) < p_ab)
            choose_a = (torch.rand(N, generator=g) < self.ab_corr_split)
            mask_a = mask_ab & choose_a
            mask_b = mask_ab & (~choose_a)
        else:
            mask_a = (torch.rand(N, generator=g) < float(p_corrs[0]))
            mask_b = (torch.rand(N, generator=g) < float(p_corrs[1]))
        swap_a = torch.randperm(N, generator=g)
        swap_b = torch.randperm(N, generator=g)

        A = self._corrupt(A, mask_a, swap_a, g)
        B = self._corrupt(B, mask_b, swap_b, g)

        self.A = self._to_feature_layout(A, int(dims_modality[0]))
        self.B = self._to_feature_layout(B, int(dims_modality[1]))
        self.corr_a, self.corr_b = mask_a, mask_b
        corr_sum = mask_a.to(torch.int64) + mask_b.to(torch.int64)
        corr_mod = torch.zeros(N, dtype=torch.int64)
        corr_mod[corr_sum == 0] = 0
        corr_mod[corr_sum > 1] = 3
        corr_mod[(corr_sum == 1) & mask_a] = 1
        corr_mod[(corr_sum == 1) & mask_b] = 2
        self.corr_mod = corr_mod

    def __len__(self) -> int:
        return int(self.n_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "A": self.A[idx].float(),
            "B": self.B[idx].float(),
            "y": self.y[idx].long(),
            "corr_a": self.corr_a[idx].long(),
            "corr_b": self.corr_b[idx].long(),
            # 0=none, 1=A only, 2=B only, 3=both corrupted
            "corr_mod": self.corr_mod[idx].long(),
        }

    def _flip_bits(self, bit: torch.Tensor, p: float, g: torch.Generator) -> torch.Tensor:
        if p <= 0.0:
            return bit
        flip = (torch.rand(bit.shape, generator=g) < p).to(torch.int64)
        return (bit ^ flip).to(torch.int64)

    def _embed_bits(self, bit: torch.Tensor, d: int, g: torch.Generator) -> torch.Tensor:
        if bit.dim() == 1:
            bit = bit[:, None]
        if bit.shape[1] != self.n_bits:
            raise ValueError(f"Expected bit.shape[1] == n_bits ({self.n_bits}), got {bit.shape}")

        X = torch.randn((bit.shape[0], d), generator=g) * float(self.distractor_std)
        X[:, : self.n_bits] = (2.0 * bit.float() - 1.0) * float(self.signal_scale)
        return X

    def _corrupt(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        swap_idx: torch.Tensor,
        g: torch.Generator,
    ) -> torch.Tensor:
        if mask.sum().item() == 0:
            return X

        mode = self.corr_modes
        if mode == "swap_signal":
            X = X.clone()
            X[mask, : self.signal_dim] = X[swap_idx[mask], : self.signal_dim]
            return X

        if mode == "gaussian":
            X = X.clone()
            n = int(mask.sum().item())
            X[mask] = torch.randn((n, X.shape[1]), generator=g) * float(self.distractor_std)
            return X

        raise ValueError(f"Unknown corr_mode: {mode}")

    def _embed_blocks(self, blocks: list[torch.Tensor], d: int, g: torch.Generator) -> torch.Tensor:
        if not blocks:
            raise ValueError("blocks must be non-empty")
        N = int(blocks[0].shape[0])
        widths = []
        for b in blocks:
            if b.dim() == 1:
                b = b[:, None]
            if int(b.shape[0]) != N:
                raise ValueError("All blocks must share the same N")
            widths.append(int(b.shape[1]))

        total = int(sum(widths))
        if total != int(self.signal_dim):
            raise ValueError(f"Sum of block widths must equal signal_dim={self.signal_dim}, got {widths} (sum={total})")
        if int(d) < total:
            raise ValueError(f"Embedding dim d must be >= signal_dim={total}, got d={d}")

        X = torch.randn((N, d), generator=g) * float(self.distractor_std)
        start = 0
        for b, w in zip(blocks, widths):
            if b.dim() == 1:
                b = b[:, None]
            X[:, start : start + w] = (2.0 * b.float() - 1.0) * float(self.signal_scale)
            start += w
        return X

    def _effective_dim(self, d: int) -> int:
        if self.feature_layout == "2d":
            return int(d) * int(self.seq_len)
        return int(d)

    def _to_feature_layout(self, x: torch.Tensor, d_mod: int) -> torch.Tensor:
        if self.feature_layout == "2d":
            return x.view(x.shape[0], int(self.seq_len), int(d_mod))
        return x
