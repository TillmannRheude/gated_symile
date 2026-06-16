import torch
import torch.nn.functional as F

from architecture import ModalityAttentionGate
from lightningmodules.utils import LightningModuleParent
from losses.retrieval import zeroshot_retrieval_logits
from losses.utils import apply_logit_scale, scale_mip_dvs


class SymileXORModel(LightningModuleParent):
    def __init__(
        self,
        model,
        params_retrival_ds: dict = None,
        candidate_idx: int = 1,
        **args,
    ):
        super().__init__(**args)

        self.dataset_name = "symile_xor"
        self.model = model
        self.params_retrival_ds = params_retrival_ds or {"batch_size": 128, "split_nr": 0}

        self.modalities = ["A", "B", "C"]
        self.candidate_idx = int(candidate_idx)  # paper task: retrieve B from (A, C)

        self.test_step_accuracies = []

        self.use_gate = bool(self.params_method["use_gate"])
        if self.use_gate:
            emb_dim = getattr(self._get_encoder_stack()[0], "emb_dim", None)
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
            raise ValueError("Could not locate encoder stack for SymileXORModel.")
        return encoders

    def _targets_to_bits(self, y: torch.Tensor) -> torch.Tensor:
        y = y.float()
        if y.ndim == 1:
            y = y[:, None]
        if y.numel() == 0:
            return y.long()

        if float(y.min().item()) < 0.0:
            return (y > 0.0).long()

        scale = float(y.max().item())
        if scale <= 0.0:
            raise ValueError("Could not infer candidate scale from Symile XOR targets.")
        return (y > (0.5 * scale)).long()

    def _bits_to_class_index(self, bits: torch.Tensor) -> torch.Tensor:
        if bits.ndim != 2:
            raise ValueError(f"Expected bit matrix of shape (N, K), got {tuple(bits.shape)}.")
        weights = (2 ** torch.arange(bits.shape[1], device=bits.device, dtype=torch.long))
        return (bits.long() * weights.unsqueeze(0)).sum(dim=1)

    def _candidate_raw_inputs(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        y = batch["y"].to(self.device)
        bits = self._targets_to_bits(y)
        n_bits = int(bits.shape[1])
        n_classes = int(2 ** n_bits)

        class_ids = torch.arange(n_classes, device=self.device, dtype=torch.long)
        cand_bits = ((class_ids[:, None] >> torch.arange(n_bits, device=self.device, dtype=torch.long)) & 1).long()

        y_float = y.float()
        if float(y_float.min().item()) < 0.0:
            scale = float(y_float.abs().max().item())
            cand_raw = (2.0 * cand_bits.float() - 1.0) * scale
        else:
            scale = float(y_float.max().item())
            if scale <= 0.0:
                scale = 1.0
            cand_raw = cand_bits.float() * scale

        return cand_raw, self._bits_to_class_index(bits)

    def _encode_candidate_bank(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        cand_raw, labels = self._candidate_raw_inputs(batch)
        encoder_b = self._get_encoder_stack()[self.candidate_idx]
        cand_emb = encoder_b(cand_raw)
        if self.params_method["embedding_norm"]:
            cand_emb = F.normalize(cand_emb, dim=-1)
        return cand_emb, labels

    def _compute_logits(self, batch: dict, embeddings) -> tuple[torch.Tensor, torch.Tensor]:
        r_a, r_b, r_c = embeddings
        emb_all = [r_a, r_b, r_c]
        cand_emb, labels = self._encode_candidate_bank(batch)
        batch_size = int(r_a.shape[0])

        if self.use_gate and self.gate is not None:
            raise NotImplementedError(
                "Fixed-candidate Symile XOR currently supports use_gate=False only."
            )

        if self.modelname == "symile_attention":
            emb_dim = int(cand_emb.shape[1])
            pair_embs = []
            for m, emb_m in enumerate(emb_all):
                if m == self.candidate_idx:
                    pair_embs.append(
                        cand_emb.unsqueeze(0).expand(batch_size, cand_emb.shape[0], emb_dim).reshape(batch_size * cand_emb.shape[0], emb_dim)
                    )
                else:
                    pair_embs.append(
                        emb_m[:, None, :].expand(batch_size, cand_emb.shape[0], emb_dim).reshape(batch_size * cand_emb.shape[0], emb_dim)
                    )

            logits = self.model.transformer(pair_embs)
            if logits.dim() == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            elif logits.dim() != 1:
                raise ValueError(
                    f"Expected transformer score shape (B*C,) or (B*C,1), got {tuple(logits.shape)}"
                )
            logits = logits.view(batch_size, cand_emb.shape[0])
            logits = apply_logit_scale(logits, self.get_logit_scale_exp())
            if self.bias is not None:
                logits = logits + self.bias
            return logits, labels

        rep_list = [emb_all[m] for m in range(len(emb_all)) if m != self.candidate_idx]
        logits = zeroshot_retrieval_logits(
            cand_emb,
            rep_list,
            self.get_logit_scale_exp(),
            bias=self.bias,
            modelname=self.modelname,
        )
        return logits, labels

    def forward(self, x):
        if isinstance(x, dict):
            a = x["A"]
            b = x["B"]
            c = x["C"]
        else:
            a, b, c = x
        return self.model([a, b, c])

    def shared_step(
        self,
        batch: dict,
        set: str = "train",
        return_embeddings: bool = False,
    ):
        model_output = self.forward(batch)
        embeddings = model_output["embeddings"]

        if self.params_method["embedding_norm"]:
            embeddings = [F.normalize(emb, dim=-1) for emb in embeddings]

        logits, labels = self._compute_logits(batch, embeddings)
        loss = F.cross_entropy(logits, labels)

        self.log(f"{set}/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        if self.bias is not None:
            self.log(f"{set}/bias", self.bias.detach(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
        if self.logit_scale is not None:
            self.log(
                f"{set}/logit_scale_exp",
                self.get_logit_scale_exp().detach(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                prog_bar=False,
            )

        if return_embeddings:
            return loss, embeddings
        return loss

    def retrieval_step(self, batch, embeddings, split: str):
        if embeddings[0].numel() == 0:
            return []
        logits, labels = self._compute_logits(batch, embeddings)
        pred = torch.argmax(logits, dim=1)
        return (pred == labels).float().tolist()
