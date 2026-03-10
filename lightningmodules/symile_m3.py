import torch 
import torch.nn as nn
import numpy as np

from lightningmodules.utils import LightningModuleParent
from torch.utils.data import DataLoader

from losses.retrieval import zeroshot_retrieval_logits
from datasets.symile_m3 import Dataset_SymileM3

class SymileM3Model(LightningModuleParent):
    def __init__(
        self, 
        model,
        params_retrival_ds: dict = {
            "batch_size": 128,
            "split_nr": 1,
        },
        **args
    ):
        super().__init__(**args)

        self.dataset_name = "symile_m3"

        self.model = model
        self.params_retrival_ds = params_retrival_ds  # unused 

        # for logging attributes and metrics
        self.run_info = {}
        self.val_step_accuracies = []
        self.test_step_accuracies = []

        # used during testing if saving representations
        self.r_a_test_save = torch.empty(0)
        self.r_i_test_save = torch.empty(0)
        self.r_t_test_save = torch.empty(0)

        self.save_hyperparameters()
    
    def forward(
        self, 
        x: list = [torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = [
            x["audio"],
            x["image"],
            x["text"],
        ]
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        loss, embeddings = self.shared_step(batch, "val", return_embeddings=True)
        accuracies = self.batch_retrieval(
            r_i=self.model.encoders[1](batch["image"].to(self.device),
                                       batch["image_missing"].to(self.device)),
            r_a=embeddings[0],
            r_t=embeddings[2],
            cls_id=batch["cls_id"].to(self.device),
            all_observed=batch["all_observed"].to(self.device),
        )
        acc_tensor = torch.tensor(accuracies, device=self.device)
        gathered = self.all_gather(acc_tensor)
        if self.trainer.is_global_zero:
            self.val_step_accuracies.extend(gathered.flatten().tolist())
        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        mean_acc = sum(self.val_step_accuracies) / len(self.val_step_accuracies)
        self.log("val/acc", mean_acc, sync_dist=True, prog_bar=True, rank_zero_only=True)
        self.log("val/acc_top1", mean_acc, sync_dist=True, prog_bar=False, rank_zero_only=True)
        self.val_step_accuracies.clear()

    def batch_retrieval(self, r_i, r_a, r_t, cls_id, all_observed):
        mask = all_observed == 1
        r_i = r_i[mask]
        r_a = r_a[mask]
        r_t = r_t[mask]
        cls_id = cls_id[mask]
        if r_i.numel() == 0:
            return []
        logits = zeroshot_retrieval_logits(
            r_i, [r_a, r_t], self.logit_scale.exp(), bias=self.bias, modelname=self.modelname
        )
        pred = cls_id[torch.argmax(logits, dim=1)]
        return (cls_id == pred).float().tolist()

    def build_candidate_bank(self, split):
        r_i_list, cls_list = [], []
        dl = self.trainer.datamodule.val_dataloader() if split == "val" else self.trainer.datamodule.test_dataloader()
        for x in dl:
            mask = x["all_observed"] == 1
            image = x["image"][mask].to(self.device)
            image_missing = x["image_missing"][mask].to(self.device)
            cls_id = x["cls_id"][mask].to(self.device)
            r_i = self.model.encoders[1](image, image_missing)
            if self.params_method.get("embedding_norm", False):
                r_i = nn.functional.normalize(r_i, dim=1)
            r_i_list.append(r_i)
            cls_list.append(cls_id)
        if not r_i_list:
            return None
        r_i_local = torch.cat(r_i_list)
        cls_local = torch.cat(cls_list)
        r_i_full = self.all_gather(r_i_local)
        cls_full = self.all_gather(cls_local)

        if r_i_full.dim() > 2:  # shape (world, n_local, d)
            r_i_full = r_i_full.reshape(-1, r_i_full.shape[-1])
            cls_full = cls_full.reshape(-1)
            
        return {"r_i": r_i_full, "cls_id": cls_full}

    def retrieval_step(self, batch, embeddings, split):
        bank = getattr(self, "candidate_bank", None)
        if bank is None:
            return []
        mask = batch["all_observed"].to(self.device) == 1
        if not mask.any():
            return []
        r_a = embeddings[0][mask]
        r_t = embeddings[2][mask]
        y   = batch["cls_id"].to(self.device)[mask]
        logits = zeroshot_retrieval_logits(
            bank["r_i"], [r_a, r_t], self.logit_scale.exp(), bias=self.bias, modelname=self.modelname
        )
        pred = bank["cls_id"][torch.argmax(logits, dim=1)]
        return (y == pred).float().tolist()

    """
    Original version: 
    Precomputed candidate bank over the entire validation set.
    This is used to speed up the evaluation process and to 
    query every val batch agains all val images. 
    """
    def og_save_candidate_image_representations(self, split):
        """
        Computes all image representations for the specified dataset split.
        (either 'val' or 'test') and then saves these representations for later use.
        Note that this method is only called during validation and testing; therefore,
        during evaluation, we only look at samples where all modalities are observed.
        """
        r_i_list = []
        cls_id_list = []

        # get dataloader
        if split == "val":
            dl = self.trainer.datamodule.val_dataloader()
        elif split == "test":
            if self.trainer.datamodule is None:
                dl = getattr(self, "test_dataloader", None)
            else:
                dl = self.trainer.datamodule.test_dataloader()

        # loop through dataloader
        for x in dl:
            # only look at samples where all modalities are observed
            mask = x["all_observed"] == 1

            image = x["image"][mask].to(self.device)
            image_missing = x["image_missing"][mask].to(self.device)
            cls_id = x["cls_id"][mask]

            r_i = self.model.encoders[1](image, image_missing)
            if self.params_method.get("embedding_norm", False):
                r_i = nn.functional.normalize(r_i, dim=1)

            r_i_list.append(r_i)
            cls_id_list.append(cls_id)

        # save reps
        if split == "val":
            self.r_i_val = torch.cat(r_i_list)
            self.r_i_cls_id_val = torch.cat(cls_id_list).to(self.device)
        elif split == "test":
            self.r_i_test = torch.cat(r_i_list)
            self.r_i_cls_id_test = torch.cat(cls_id_list).to(self.device)

    def og_zeroshot_retrieval(self, r_a, r_t, batch, split, split_nr):
        """
        Perform zeroshot retrieval to predict images given audio and text representations.

        Args:
            r_a (torch.Tensor): Learned audio representations of shape (batch_sz, d).
            r_t (torch.Tensor): Learned text representations of shape (batch_sz, d).
            batch (dict): A dictionary containing the input batch. Refer to the `forward` method
                for detailed descriptions of the keys and their shapes.
            split (str): The dataset split to process ('val' or 'test').

        Returns:
            list: A list of accuracies for each sample in the batch.
        """
        # get candidate image representations and class ids
        if split == "val":
            r_i = self.r_i_val
            r_i_cls_id = self.r_i_cls_id_val
        elif split == "test":
            r_i = self.r_i_test
            r_i_cls_id = self.r_i_cls_id_test
        else:
            return 

        mask = batch["all_observed"] == 1

        if split == "test":
            assert mask.all(), "All values should be observed in test set."

        r_a = r_a[mask]
        r_t = r_t[mask]

        # logits is a tensor of shape (num_samples_all_observed, num_candidates)
        # where each element in a row is the score for the corresponding image candidate.
        logits = zeroshot_retrieval_logits(
            r_i, 
            [r_a, r_t], 
            self.logit_scale.exp(),
            bias=self.bias,
            modelname=self.modelname
        )  # .cpu()

        # pred_idx is a tensor of length batch_sz where each element is the
        # index of the r_i (across the whole all-observed eval set) that maximizes the score.
        pred_idx = torch.argmax(logits, dim=1)

        # for each index in pred_idx, we get the class id (label) that corresponds
        # to the r_i at that index; so pred is a tensor of length batch_sz where
        # each element is the predicted label
        pred = r_i_cls_id[pred_idx]

        y = batch["cls_id"][mask]

        accuracies = (y == pred).float().tolist()

        return accuracies