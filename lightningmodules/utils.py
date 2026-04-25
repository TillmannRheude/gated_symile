import torch
import schedulefree
import math 
import time

import torch.nn as nn 
import pytorch_lightning as pl
import torch.distributed as dist
import torch.distributed.nn.functional as distnn
import torch.nn.functional as F

from torchmetrics.aggregation import MinMetric

from losses.clip import clip
from losses.symile import symile, symile_gated
from losses.triangle import triangle
from losses.gram import gram

class LightningModuleParent(pl.LightningModule):
    def __init__(
        self,
        modelname: str = "symile",
        params_optimizer: dict = {
            "name": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "eps": 1e-8,
            "betas": (0.9, 0.999),
            "warmup_steps": 1000,
        },
        params_method: dict = {
            "modelname": "symile",
            "negative_sampling": "n_squared",
            "logit_scale_init": 2.65926,
            "batch_size": 128,
            "embedding_norm": True,
            "bias_init_mult": 1.0,
            "use_gate": False,
            "gate_temp": 0.1,
            "pair_num_negatives": 128,
        },
        **kwargs,
    ) -> None: 
        super().__init__()

        self.params_optimizer = params_optimizer
        self.params_method = params_method
        self.modelname = modelname
        self.use_gate = params_method["use_gate"]

        # Walltime logging (disabled only if explicitly turned off).
        # Logged as epoch-mean `train/step_walltime_s` every `step_walltime_every` batches.
        self.log_step_walltime = bool(self.params_method.get("log_step_walltime", True))
        self.step_walltime_every = int(self.params_method.get("step_walltime_every", 50))

        # Temperature Tau
        self.logit_scale = nn.Parameter(torch.ones([]) * params_method["logit_scale_init"])

        #freeze_logit_scale = True  # DEBUG toggle
        #if freeze_logit_scale:
        #    self.logit_scale.requires_grad_(False)

        self.bias = None
        self._bias_needs_global_init = False

        # Logging variables
        self.best_losses = {
            "train": float('inf'),
            "val": float('inf'),
            "test": float('inf'),
        }
        self.best_acc_top1s = {
            "train": 0.0,
            "val": 0.0,
            "test": 0.0,
        }
        self.val_loss_best = MinMetric()

        self.val_step_accuracies = []

        # Loss functions
        if modelname == "symile":
            self.loss = symile if not self.use_gate else symile_gated
        elif modelname == "clip":
            self.loss = clip
        elif modelname == "triangle":
            self.loss = triangle
        elif modelname == "gram":
            self.loss = gram

        self.save_hyperparameters()

    def forward(
        self, 
        x: list = [torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return self.model(x)
    
    def shared_step(
        self, 
        batch: dict,
        set: str = "train",
        return_embeddings: bool = False,
    ) -> torch.tensor:
        model_output = self.forward(batch)
        embeddings = model_output["embeddings"]

        if self.params_method["embedding_norm"]:
            embeddings = [nn.functional.normalize(emb, dim=1) for emb in embeddings]
        
        # DDP for Symile n^2
        if (
            set == "train"
            and self.modelname == "symile"
            and self.params_method["negative_sampling"] in ["n_squared", "pair"]
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            r_a_local, r_b_local, r_c_local = embeddings

            # global candidate pool; 
            # gather with grad: matches the naive "global-batch" objective: other-rank embeddings participate as candidates/negatives AND receive gradient contributions from this rank's anchors during backprop. 
            # just gather: still use all global negatives in the forward pass, but only local embeddings get updated by the candidate-side terms.
            r_a_all = self._all_gather_with_grad(r_a_local)
            r_b_all = self._all_gather_with_grad(r_b_local)
            r_c_all = self._all_gather_with_grad(r_c_local)

            #world_size = dist.get_world_size()
            #rank = dist.get_rank()
            #ra_list = [torch.zeros_like(r_a_local) for _ in range(world_size)]
            #rb_list = [torch.zeros_like(r_b_local) for _ in range(world_size)]
            #rc_list = [torch.zeros_like(r_c_local) for _ in range(world_size)]
            #dist.all_gather(ra_list, r_a_local.detach())
            #dist.all_gather(rb_list, r_b_local.detach())
            #dist.all_gather(rc_list, r_c_local.detach())
            #ra_list[rank] = r_a_local  # restore local tensor to keep local grads
            #rb_list[rank] = r_b_local
            #rc_list[rank] = r_c_local
            #r_a_all = torch.cat(ra_list, dim=0)
            #r_b_all = torch.cat(rb_list, dim=0)
            #r_c_all = torch.cat(rc_list, dim=0)

            # labels are the global indices of the local anchors
            b_local = r_a_local.shape[0]
            rank = dist.get_rank()
            labels = torch.arange(rank * b_local, rank * b_local + b_local, device=self.device)

            if self.use_gate:
                if self.params_method["negative_sampling"] == "pair":
                    loss = self.loss(
                        r_a_local, r_b_local, r_c_local,
                        logit_scale=self.logit_scale.exp(),
                        negative_sampling=self.params_method["negative_sampling"],
                        gate=self.gate,
                        bias=self.bias,
                        labels=labels,
                        candidates=(r_a_all, r_b_all, r_c_all),
                        pair_num_negatives=self.params_method["pair_num_negatives"],
                    )
                else:
                    loss = self.loss(
                        r_a_local, r_b_local, r_c_local,
                        logit_scale=self.logit_scale.exp(),
                        negative_sampling=self.params_method["negative_sampling"], 
                        gate=self.gate,
                        bias=self.bias,
                        labels=labels,
                        candidates=(r_a_all, r_b_all, r_c_all),
                    )
            else:
                if self.params_method["negative_sampling"] == "pair":
                    loss = self.loss(
                        r_a_local, r_b_local, r_c_local,
                        logit_scale=self.logit_scale.exp(),
                        negative_sampling=self.params_method["negative_sampling"],
                        bias=None,
                        labels=labels,
                        candidates=(r_a_all, r_b_all, r_c_all),
                        pair_num_negatives=self.params_method["pair_num_negatives"],
                    )
                else:
                    loss = self.loss(
                        r_a_local, r_b_local, r_c_local,
                        logit_scale=self.logit_scale.exp(),
                        negative_sampling=self.params_method["negative_sampling"],
                        bias=None,
                        labels=labels,
                        candidates=(r_a_all, r_b_all, r_c_all),
                    )
        else:
            if self.use_gate:
                if self.params_method["negative_sampling"] == "pair":
                    loss = self.loss(
                        *embeddings,
                        logit_scale=self.logit_scale.exp(),
                        negative_sampling=self.params_method["negative_sampling"],
                        gate=self.gate,
                        bias=self.bias,
                        pair_num_negatives=self.params_method["pair_num_negatives"],
                    )
                else:
                    loss = self.loss(
                        *embeddings,
                        logit_scale=self.logit_scale.exp(),
                        negative_sampling=self.params_method["negative_sampling"],
                        gate=self.gate,
                        bias=self.bias,
                    )
            else:
                if self.params_method["negative_sampling"] == "pair":
                    loss = self.loss(
                        *embeddings,
                        logit_scale=self.logit_scale.exp(),
                        negative_sampling=self.params_method["negative_sampling"],
                        bias=None,
                        pair_num_negatives=self.params_method["pair_num_negatives"],
                    )
                else:
                    loss = self.loss(
                        *embeddings,
                        logit_scale=self.logit_scale.exp(),
                        negative_sampling=self.params_method["negative_sampling"],
                        bias=self.bias,
                    )

        self.log(f"{set}/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        if self.bias is not None:
            # log the scalar bias and the effective temperature scale
            self.log(f"{set}/bias", self.bias.detach(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            self.log(f"{set}/logit_scale_exp", self.logit_scale.exp().detach(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)

        if return_embeddings:
            return loss, embeddings
        return loss

    def compute_loss_from_outputs(self, outputs: dict, batch: dict, set: str):
        embeddings = outputs["embeddings"]

        if self.params_method["embedding_norm"]:
            embeddings = [nn.functional.normalize(emb, dim=1) for emb in embeddings]

        # your existing DDP symile n^2 special-case
        if (
            set == "train"
            and self.modelname == "symile"
            and self.params_method["negative_sampling"] == "n_squared"
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            r_a_local, r_b_local, r_c_local = embeddings
            r_a_all = self._all_gather_with_grad(r_a_local)
            r_b_all = self._all_gather_with_grad(r_b_local)
            r_c_all = self._all_gather_with_grad(r_c_local)

            b_local = r_a_local.shape[0]
            rank = dist.get_rank()
            labels = torch.arange(rank * b_local, rank * b_local + b_local, device=self.device)

            loss = symile(
                r_a_local, r_b_local, r_c_local,
                logit_scale=self.logit_scale.exp(),
                negative_sampling="n_squared",
                bias=None,
                labels=labels,
                candidates=(r_a_all, r_b_all, r_c_all),
            )
            return loss

        # default loss
        return self.loss(
            *embeddings,
            logit_scale=self.logit_scale.exp(),
            negative_sampling=self.params_method["negative_sampling"],
            bias=self.bias,
        )

    def _log_gate_weights(self, gate_weights: torch.Tensor, set: str):
        """
        gate_weights: (B, M) in [0,1]
        Logs nice scalar summaries per modality + optional histograms (TensorBoard).
        """
        w = gate_weights.detach()
        if w.numel() == 0:
            return

        # modality names if available (UKBModel defines self.modalities)
        names = getattr(self, "modalities", None)
        M = w.shape[1]
        if not names or len(names) != M:
            names = [f"m{i}" for i in range(M)]

        # per-modality summaries
        for i, name in enumerate(names):
            wi = w[:, i]
            self.log(f"{set}/gate_w/{name}_mean", wi.mean(), on_step=False, on_epoch=True, sync_dist=True)

    def _log_gate_cos_alignment(
        self,
        embeddings: list[torch.Tensor],
        gated_list: list[torch.Tensor],
        split: str,
        names: list[str] = None,
    ):
        """
        Logs mean cosine similarity cos(gated_m, orig_m) per modality.
        embeddings[m], gated_list[m]: (B, D)
        """
        if embeddings is None or gated_list is None:
            return
        if len(embeddings) != len(gated_list):
            return

        M = len(embeddings)
        if names is None or len(names) != M:
            names = [f"m{i}" for i in range(M)]

        with torch.no_grad():
            for m in range(M):
                e = embeddings[m]
                g = gated_list[m]
                if e.numel() == 0 or g.numel() == 0:
                    continue
                # both are typically already normalized, but make it robust
                cos_m = F.cosine_similarity(
                    F.normalize(g, dim=1),
                    F.normalize(e, dim=1),
                    dim=1,
                ).mean()
                self.log(
                    f"{split}/gate_cos_{names[m]}",
                    cos_m,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
    
    def _log_gate_cos_to_neutral(
        self,
        gate,  # ModalityAttentionGate
        gated_list: list[torch.Tensor],
        split: str,
        names: list[str] = None,
    ):
        """
        Logs mean cosine similarity cos(gated_m, neutral_m) per modality.
        gated_list[m]: (B, D)
        gate.neutral: (M, D)
        """
        if gate is None or gated_list is None:
            return
        M = len(gated_list)
        if names is None or len(names) != M:
            names = [f"m{i}" for i in range(M)]

        if gate.neutral is None:
            return
        # gate.neutral is on (M,D); broadcast to (B,D) per modality
        N = gate.neutral.to(device=gated_list[0].device, dtype=gated_list[0].dtype)

        with torch.no_grad():
            for m in range(M):
                g = gated_list[m]
                if g.numel() == 0:
                    continue
                n = N[m].unsqueeze(0).expand(g.shape[0], -1)  # (B,D)
                cos_m = F.cosine_similarity(
                    F.normalize(g, dim=1),
                    F.normalize(n, dim=1),
                    dim=1,
                ).mean()
                self.log(
                    f"{split}/gate_cos_neutral_{names[m]}",
                    cos_m,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def training_step(self, batch, batch_idx):
        if self.automatic_optimization:
            return self.shared_step(batch, "train")

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        # Optional walltime logging (intended for UKB gating runtime comparisons).
        if getattr(self.trainer, "sanity_checking", False):
            return

        if not bool(getattr(self, "log_step_walltime", False)):
            return

        # Default: log occasionally to limit overhead; can be overridden per module.
        every = int(getattr(self, "step_walltime_every", 50))
        if every > 1 and (int(batch_idx) % every) != 0:
            self._walltime_t0 = None
            return

        self._walltime_t0 = time.perf_counter()

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        t0 = getattr(self, "_walltime_t0", None)
        if t0 is None:
            return

        # Ensure we include GPU work (forward+backward+optimizer) in the walltime.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        dt = float(time.perf_counter() - float(t0))
        self._walltime_t0 = None

        self.log(
            "train/step_walltime_s",
            dt,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        
    def validation_step(self, batch, batch_idx):
        loss, embeddings = self.shared_step(batch, "val", return_embeddings=True)

        if hasattr(self, "retrieval_step"):
            accs = self.retrieval_step(batch, embeddings, split="val") or []

            ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

            # local stats (can be zero-length)
            correct = float(sum(accs))
            count = float(len(accs))

            stats = torch.tensor([correct, count], device=self.device, dtype=torch.float32)
            if ddp:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)

            # store per-step global stats on rank 0 only
            if self.trainer.is_global_zero and count > 0:
                self.val_step_accuracies.append((float(stats[0].item()), float(stats[1].item())))

        return loss

    def test_step(self, batch, batch_idx):
        loss, embeddings = self.shared_step(batch, "test", return_embeddings=True)

        if hasattr(self, "retrieval_step"):
            accs = self.retrieval_step(batch, embeddings, split="test") or []

            ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

            correct = float(sum(accs))
            count = float(len(accs))

            stats = torch.tensor([correct, count], device=self.device, dtype=torch.float32)
            if ddp:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)

            if self.trainer.is_global_zero and count > 0:
                self.test_step_accuracies.append((float(stats[0].item()), float(stats[1].item())))

        return loss

    def run_zeroshot_retrieval(self, set: str = "val"):
        if self.dataset_name == "symile_mimic":
            acc_dict = self.zeroshot_retrieval(f"{set}", split_nr=self.params_retrival_ds["split_nr"])
            acc_top1 = acc_dict["acc@top1"]
        elif self.dataset_name == "symile_m3":
            acc_return = self.zeroshot_retrieval(f"{set}", split_nr=self.params_retrival_ds["split_nr"])
            acc_top1 = acc_return
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported, yet.")
        
        # get best metrics over epochs 
        current_set_loss = self.trainer.callback_metrics[f"{set}/loss_epoch"].item()
        if current_set_loss < self.best_losses[set]:
            self.best_losses[set] = current_set_loss
        current_set_acc_top1 = acc_top1
        if current_set_acc_top1 > self.best_acc_top1s[set]:
            self.best_acc_top1s[set] = current_set_acc_top1

        self.log(f"{set}/min_loss", self.best_losses[set], sync_dist=True, prog_bar=True)

        return acc_dict

    def on_train_epoch_start(self):
        self.set_optimizer_mode(mode="train")

    def on_train_start(self):
        self.set_optimizer_mode(mode="train")

    def on_validation_start(self):
        self.set_optimizer_mode(mode="eval")
    
    def on_validation_epoch_start(self):
        if hasattr(self, "build_candidate_bank"):
            self.candidate_bank = self.build_candidate_bank(split="val")
        else:
            self.candidate_bank = None

    def on_validation_epoch_end(self):
        # During sanity check, it's normal to have no candidate bank / no valid queries
        if getattr(self.trainer, "sanity_checking", False):
            self.val_step_accuracies.clear()
            return

        current = self.trainer.callback_metrics.get("val/loss")
        self.val_loss_best.update(current)
        self.log("val/min_loss", self.val_loss_best.compute(), prog_bar=True, sync_dist=True)

        ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

        # -------------------------
        # Case 1: UKB-style retrieval (per-batch accuracies collected into a Python list)
        # -------------------------
        if hasattr(self, "retrieval_step"):
            # Only rank 0 has the list populated (see validation_step).
            if self.trainer.is_global_zero and self.val_step_accuracies:
                total_correct = sum(c for c, n in self.val_step_accuracies)
                total_count = sum(n for c, n in self.val_step_accuracies)
                acc_top1 = (total_correct / total_count) if total_count > 0 else float("nan")
            else:
                acc_top1 = float("nan")
            self.val_step_accuracies.clear()

            acc_t = torch.tensor(acc_top1, device=self.device)
            if ddp:
                dist.broadcast(acc_t, src=0)

            acc_val = float(acc_t.item())
            if acc_val == acc_val:  # not NaN
                if acc_val > self.best_acc_top1s["val"]:
                    self.best_acc_top1s["val"] = acc_val

            # Log on ALL ranks so EarlyStopping sees the metric everywhere
            self.log("val/acc_top1", acc_t, prog_bar=True, sync_dist=False, rank_zero_only=False)
            self.log(
                "val/max_acc_top1",
                torch.tensor(self.best_acc_top1s["val"], device=self.device),
                prog_bar=True,
                sync_dist=False,
                rank_zero_only=False,
            )
            return

        # -------------------------
        # Case 2: MIMIC-style retrieval (run_zeroshot_retrieval computes acc dict)
        # -------------------------
        if hasattr(self, "run_zeroshot_retrieval"):
            if self.trainer.is_global_zero:
                acc_dict = self.run_zeroshot_retrieval("val")
                acc_top1 = float(acc_dict["acc@top1"])
            else:
                acc_top1 = float("nan")

            acc_t = torch.tensor(acc_top1, device=self.device)
            if ddp:
                dist.broadcast(acc_t, src=0)

            acc_val = float(acc_t.item())
            if acc_val == acc_val:  # not NaN
                if acc_val > self.best_acc_top1s["val"]:
                    self.best_acc_top1s["val"] = acc_val

            # Log on ALL ranks so EarlyStopping sees the metric everywhere
            self.log("val/acc_top1", acc_t, prog_bar=True, sync_dist=False, rank_zero_only=False)
            self.log(
                "val/max_acc_top1",
                torch.tensor(self.best_acc_top1s["val"], device=self.device),
                prog_bar=True,
                sync_dist=False,
                rank_zero_only=False,
            )

            return

        return

    def on_test_start(self):
        self.configure_optimizers()
        self.set_optimizer_mode(mode="eval")

    def on_test_epoch_start(self):
        if hasattr(self, "build_candidate_bank"):
            self.candidate_bank = self.build_candidate_bank(split="test")
        else:
            self.candidate_bank = None

    def on_test_epoch_end(self):
        # Case 1: models with per-batch retrieval_step accumulation (e.g., UKB)
        if hasattr(self, "test_step_accuracies") and self.test_step_accuracies:
            if not self.trainer.is_global_zero:
                self.test_step_accuracies.clear()
                return

            total_correct = sum(c for c, n in self.test_step_accuracies)
            total_count = sum(n for c, n in self.test_step_accuracies)
            acc_top1 = (total_correct / total_count) if total_count > 0 else float("nan")

            self.test_step_accuracies.clear()
            self.log("test/acc_top1", acc_top1, sync_dist=False, rank_zero_only=False, prog_bar=False)
            return

        # Case 2: models with run_zeroshot_retrieval (e.g., MIMIC)
        if hasattr(self, "run_zeroshot_retrieval") and not hasattr(self, "retrieval_step"):
            acc_dict = self.run_zeroshot_retrieval("test")
            acc_top1 = acc_dict["acc@top1"]
            self.log("test/acc_top1", acc_top1, sync_dist=True, prog_bar=False)
            return

    def _all_gather_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            return x
        xs = distnn.all_gather(x) 
        return torch.cat(xs, dim=0)   # (B_global, d)

    def _apply_loss_mask(self, embeddings: list[torch.Tensor], mask: torch.Tensor) -> list[torch.Tensor]:
        # mask: (B,) bool
        mask = mask.to(device=embeddings[0].device, dtype=torch.bool)
        return [e[mask] for e in embeddings]

    def set_optimizer_mode(
            self, 
            mode: str ="train",
            just_optim: bool = False,
            return_optim: bool = False
    ) -> None: 
        optim = self.optimizers()
        
        if mode == "train" and "schedulefree" in self.params_optimizer["name"]:
            optim.train()  
        elif mode == "eval" and "schedulefree" in self.params_optimizer["name"]:
            optim.eval()
        
        if return_optim:
            return optim

    def configure_optimizers(self):
        if self.params_method["use_gate"]:
            gate_params = []
            base_params = []
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if "gate" in name:
                    gate_params.append(p)
                else:
                    base_params.append(p)
            param_groups = [
                {"params": base_params, "lr": self.params_optimizer["lr"], "weight_decay": self.params_optimizer["weight_decay"]},
                {"params": gate_params, "lr": self.params_optimizer["lr"] * self.params_optimizer["lr_gate_mul"], "weight_decay": 0.0},
            ]
        else: 
            param_groups = [
                {"params": self.parameters(), "lr": self.params_optimizer["lr"], "weight_decay": self.params_optimizer["weight_decay"]},
            ]

        if self.params_optimizer["name"] == "schedulefree_adamw":
            optimizer = schedulefree.AdamWScheduleFree(
                param_groups, 
                eps=self.params_optimizer["eps"],
                warmup_steps=self.params_optimizer["warmup_steps"],
                betas=self.params_optimizer["betas"]
            )
        elif self.params_optimizer["name"] == "schedulefree_sgd":
            optimizer = schedulefree.SGDScheduleFree(
                param_groups, 
                lr=self.params_optimizer["lr"],
                weight_decay=self.params_optimizer["weight_decay"], 
                momentum=self.params_optimizer["momentum"],
                warmup_steps=self.params_optimizer["warmup_steps"],
            )
        elif self.params_optimizer["name"] == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.params_optimizer["lr"],
                weight_decay=self.params_optimizer["weight_decay"]
            )
        elif self.params_optimizer["name"] == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups, 
                #lr=self.params_optimizer["lr"],
                #weight_decay=self.params_optimizer["weight_decay"], 
                #eps=self.params_optimizer["eps"],
                #betas=self.params_optimizer["betas"]
            )
        else:
            raise ValueError(f"Optimizer {self.params_optimizer['name']} not supported.")
        
        return optimizer
