import torch
from torch.utils.data import Dataset
from typing import Dict, Optional


class SymileBinaryXORDataset(Dataset):
    """
    Three-modality synthetic dataset matching the Symile paper's 5D setup.

    For each sample, draw binary vectors a, b in {0, 1}^dim with independent
    Bernoulli(0.5) coordinates. Then draw a sample-level indicator i ~
    Bernoulli(p). The third modality is:

        c_j = (a_j XOR b_j)^i * a_j^(1-i)

    Equivalently:
      - if i = 1, c = a XOR b
      - if i = 0, c = a

    Zero-shot task from the paper:
        predict b from a and c.

    Returns:
        {
            "A": a,
            "B": b,
            "C": c,
            "y": b,
            "is_xor": indicator whether sample used XOR rule
        }
    """

    def __init__(
        self,
        n_samples: int = 16000,
        p: float = 1.0,
        seed: int = 420,
        value_mode: str = "zero_one",  # "zero_one" or "minus_plus"
        dim: int = 5,
        signal_scale: float = 1.0,
        distractor_std: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}.")
        if value_mode not in ("zero_one", "minus_plus"):
            raise ValueError("value_mode must be 'zero_one' or 'minus_plus'.")
        if dim < 1:
            raise ValueError("dim must be >= 1.")

        self.n_samples = int(n_samples)
        self.p = float(p)
        self.seed = int(seed)
        self.value_mode = value_mode
        self.dim = int(dim)

        g = torch.Generator(device="cpu").manual_seed(self.seed)

        a = torch.bernoulli(torch.full((self.n_samples, self.dim), 0.5), generator=g).long()
        b = torch.bernoulli(torch.full((self.n_samples, self.dim), 0.5), generator=g).long()

        is_xor = (torch.rand(self.n_samples, generator=g) < self.p).long()
        xor_c = a ^ b
        c = torch.where(is_xor[:, None].bool(), xor_c, a).long()

        self.y = b.clone()

        if value_mode == "minus_plus":
            a = 2.0 * a.float() - 1.0
            b = 2.0 * b.float() - 1.0
            c = 2.0 * c.float() - 1.0
        else:
            a = a.float()
            b = b.float()
            c = c.float()

        self.A = signal_scale * a
        self.B = signal_scale * b
        self.C = signal_scale * c
        self.is_xor = is_xor

        if float(distractor_std) != 0.0:
            noise_a = torch.randn(self.n_samples, self.dim, generator=g) * distractor_std
            noise_b = torch.randn(self.n_samples, self.dim, generator=g) * distractor_std
            noise_c = torch.randn(self.n_samples, self.dim, generator=g) * distractor_std
            self.A = self.A + noise_a
            self.B = self.B + noise_b
            self.C = self.C + noise_c

        if device is not None:
            self.A = self.A.to(device)
            self.B = self.B.to(device)
            self.C = self.C.to(device)
            self.y = self.y.to(device)
            self.is_xor = self.is_xor.to(device)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "A": self.A[idx],
            "B": self.B[idx],
            "C": self.C[idx],
            "y": self.y[idx],
            "is_xor": self.is_xor[idx],
        }
