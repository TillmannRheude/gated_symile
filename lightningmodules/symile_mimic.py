import torch 
import torch.nn as nn
import numpy as np

from lightningmodules.utils import LightningModuleParent
from torch.utils.data import DataLoader

from datasets.symile_mimic import Dataset_SymileMimic
from losses.retrieval import zeroshot_retrieval_logits
from losses.utils import scale_mip_dvs
from architecture import ModalityAttentionGate

class SymileMIMICModel(LightningModuleParent):
    def __init__(
        self, 
        model,
        params_retrival_ds: dict = {
            "batch_size": 128,
            "split_nr": 0, 
        },
        **args
    ):
        super().__init__(**args)

        self.dataset_name = "symile_mimic"

        self.model = model
        self.params_retrival_ds = params_retrival_ds

        self.modalities = ["cxr", "ecg", "labs"]
        self.candidate_idx = 0  # retrieve cxr from (ecg,labs)

        self.use_gate = self.params_method["use_gate"]
        if self.use_gate:
            self.emb_dim = model.encoders[self.candidate_idx].emb_dim
            self.gate = ModalityAttentionGate(
                num_modalities=len(self.modalities),
                emb_dim=self.emb_dim,
                num_heads=self.params_method["gate_num_heads"],
                d_k=self.params_method["gate_d_k"],
                d_null=self.params_method["gate_d_null"],
                temperature_init=self.params_method["gate_temp"],
                gate_bias_init=self.params_method["gate_bias_init"],
                gate_strength_init=self.params_method["gate_strength_init"],
                gate_type=self.params_method["gate_type"],
                gate_mode=self.params_method["gate_mode"],
            )
        else:
            self.gate = None

        self.save_hyperparameters()
    
    def forward(
        self, 
        x: list = [torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        cxr = x["cxr"]
        ecg = x["ecg"]
        labs = x["labs"]

        #if self.training:  # only corrupt during training
        #B = ecg.shape[0]
        #perm = torch.randperm(B, device=ecg.device)
        #labs = labs[perm]

        return self.model([cxr, ecg, labs])
        x = [
            x["cxr"],
            x["ecg"],
            x["labs"],
        ]
        return self.model(x)

    def zeroshot_retrieval(self, split, split_nr, bootstrap=False):
        mode = self.params_method["mimic_retrieval_mode"]

        if mode == "global":
            return self._zeroshot_retrieval_global(split=split, split_nr=split_nr)

        return self._zeroshot_retrieval_preselected(split=split, split_nr=split_nr, bootstrap=bootstrap)

    def _encode_split_embeddings(self, split: str, split_nr: int):
        """
        Encodes (cxr, ecg, labs) for the given split (train/val/test), returning
        normalized embeddings (if enabled) and hadm_id.
        """
        batch_sz = int(self.params_retrival_ds["batch_size"])
        ds = Dataset_SymileMimic(split=split, split_nr=split_nr)

        r_c, r_e, r_l = [], [], []
        hadm_id = []

        for batch in DataLoader(ds, batch_size=batch_sz, shuffle=False, drop_last=False, generator=torch.Generator()):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            model_output = self.forward(batch)
            embeddings = model_output["embeddings"]

            r_c.append(embeddings[0])
            r_e.append(embeddings[1])
            r_l.append(embeddings[2])
            hadm_id.append(batch["hadm_id"])

        r_c = torch.cat(r_c, dim=0)
        r_e = torch.cat(r_e, dim=0)
        r_l = torch.cat(r_l, dim=0)
        hadm_id = torch.cat(hadm_id, dim=0)

        if self.params_method["embedding_norm"]:
            r_c = nn.functional.normalize(r_c, dim=1)
            r_e = nn.functional.normalize(r_e, dim=1)
            r_l = nn.functional.normalize(r_l, dim=1)

        return {"r_c": r_c, "r_e": r_e, "r_l": r_l, "hadm_id": hadm_id}

    @staticmethod
    def _dedup_by_id(emb: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Deduplicate embeddings by integer id, keeping the first occurrence.
        Returns (emb_unique, ids_unique).
        """
        ids_list = ids.detach().cpu().tolist()
        seen = set()
        keep_idx = []
        for i, v in enumerate(ids_list):
            if v in seen:
                continue
            seen.add(v)
            keep_idx.append(i)
        keep = torch.tensor(keep_idx, device=emb.device, dtype=torch.long)
        return emb.index_select(0, keep), ids.index_select(0, keep)

    def _zeroshot_retrieval_global(self, split: str, split_nr: int):
        """
        UKB-like global candidate bank evaluation for MIMIC:
          - candidates: all CXRs in the split (dedup by hadm_id)
          - queries: all (ecg, labs) in the split
          - prediction is correct if predicted hadm_id == query hadm_id
        """
        eval_split = "val" if split == "val" else "test"
        enc = self._encode_split_embeddings(eval_split, split_nr)

        cand_r_c, cand_id = self._dedup_by_id(enc["r_c"], enc["hadm_id"])
        query_r_e = enc["r_e"]
        query_r_l = enc["r_l"]
        query_id = enc["hadm_id"]

        #Nc = cand_r_c.shape[0]
        #print(f"[MIMIC global] N_candidates={Nc}, chance≈{1.0/Nc:.6g}")


        use_gate = getattr(self, "use_gate", False) and getattr(self, "gate", None) is not None
        gate_mode = getattr(self.gate, "gate_mode", None) if use_gate else None
        cand_dep_gate = bool(
            use_gate
            and gate_mode == "attention"
        )

        # Candidate-independent gate: gate queries once, then do matrix logits.
        if use_gate and not cand_dep_gate:
            dummy_cxr = torch.zeros_like(query_r_e)
            emb_q = [dummy_cxr, query_r_e, query_r_l]
            W_q = self.gate.compute_W(emb_q)
            gated_list, w_t, _ = self.gate.apply_for_target(self.candidate_idx, emb_q, W=W_q)
            query_r_e = gated_list[1]
            query_r_l = gated_list[2]

        # Compute logits (Bq, Nc)
        if cand_dep_gate:
            Bq, D = query_r_e.shape
            Nc = cand_r_c.shape[0]
            logits = torch.empty((Bq, Nc), device=self.device, dtype=torch.float32)

            q_chunk = int(self.params_method.get("mimic_global_query_chunk_size", 64))
            c_chunk = int(self.params_method.get("gate_candidate_chunk_size", 256))

            # only log once per epoch
            did_log = False

            for qs in range(0, Bq, q_chunk):
                qe = min(Bq, qs + q_chunk)
                ecg_q = query_r_e[qs:qe]
                labs_q = query_r_l[qs:qe]
                Bk = ecg_q.shape[0]

                for cs in range(0, Nc, c_chunk):
                    ce = min(Nc, cs + c_chunk)
                    cand = cand_r_c[cs:ce]  # (Nc_chunk, D)
                    Nc_k = cand.shape[0]

                    pair_embs = [
                        cand.unsqueeze(0).expand(Bk, Nc_k, D).reshape(Bk * Nc_k, D),
                        ecg_q.unsqueeze(1).expand(Bk, Nc_k, D).reshape(Bk * Nc_k, D),
                        labs_q.unsqueeze(1).expand(Bk, Nc_k, D).reshape(Bk * Nc_k, D),
                    ]

                    W_pair = self.gate.compute_W(pair_embs)
                    gated_list, w_pair, _ = self.gate.apply_for_target(self.candidate_idx, pair_embs, W=W_pair)

                    if not did_log:
                        max_pairs = int(self.params_method.get("gate_log_max_pairs", 128))
                        pair_slice = [x[:max_pairs] for x in pair_embs]
                        gated_slice = [g[:max_pairs] for g in gated_list]
                        self._log_gate_cos_alignment(pair_slice, gated_slice, split="val", names=self.modalities)
                        self._log_gate_cos_to_neutral(self.gate, gated_slice, split="val", names=self.modalities)
                        if hasattr(self.gate, "logit_gate_strength"):
                            alpha = torch.sigmoid(self.gate.logit_gate_strength.detach())
                            self.log("val/gate_alpha", alpha, on_step=False, on_epoch=True, sync_dist=True)
                        did_log = True

                    prod = gated_list[1] * gated_list[2]  # (Bk*Nc_k, D)
                    prod = prod.view(Bk, Nc_k, D)
                    raw = torch.einsum("bnd,nd->bn", prod, cand)  # (Bk, Nc_k)

                    raw = scale_mip_dvs(raw, d=D, M=3)
                    out = self.logit_scale.exp() * raw
                    if self.modelname == "sigmile" and self.bias is not None:
                        out = out + self.bias

                    logits[qs:qe, cs:ce] = out

            logits = logits.cpu()
        else:
            logits = zeroshot_retrieval_logits(
                cand_r_c,
                [query_r_e, query_r_l],
                self.logit_scale.exp(),
                bias=self.bias,
                modelname=self.modelname,
            ).cpu()

        # Map hadm_id -> candidate index
        cand_id_list = cand_id.detach().cpu().tolist()
        id_to_idx = {int(v): i for i, v in enumerate(cand_id_list)}
        query_id_list = query_id.detach().cpu().tolist()

        pos_idx = []
        keep_q = []
        for i, qid in enumerate(query_id_list):
            j = id_to_idx.get(int(qid), None)
            if j is None:
                continue
            keep_q.append(i)
            pos_idx.append(j)

        if not keep_q:
            return {
                "acc@top1": float("nan"),
                "acc@top3": float("nan"),
                "acc@top5": float("nan"),
                "rank_mean": float("nan"),
                "rank_median": float("nan"),
                "rank_p95": float("nan"),
                "hard_margin_mean": float("nan"),
                "hard_margin_p5": float("nan"),
                "hard_margin_p50": float("nan"),
                "hard_margin_p95": float("nan"),
                "pos_minus_meanneg_mean": float("nan"),
                "pos_prob_mean": float("nan"),
                "pos_prob_p50": float("nan"),
                "pos_prob_p95": float("nan"),
                "auroc_mean": float("nan"),
                "auroc_p50": float("nan"),
                "auroc_p95": float("nan"),
            }

        keep_q = torch.tensor(keep_q, dtype=torch.long)
        pos_idx = torch.tensor(pos_idx, dtype=torch.long)
        logits = logits.index_select(0, keep_q)

        # metrics
        pred_idx = torch.argmax(logits, dim=1)
        pred_id = cand_id.cpu().index_select(0, pred_idx)
        true_id = query_id.cpu().index_select(0, keep_q)

        correct_pred_top1 = int((pred_id == true_id).sum().item())
        Bk = int(true_id.shape[0])

        correct_pred_top3 = 0
        correct_pred_top5 = 0
        for k in (3, 5):
            kk = min(k, logits.shape[1])
            _, topk = torch.topk(logits, kk, dim=1)
            hit = topk.eq(pos_idx.unsqueeze(1)).any(dim=1)
            if k == 3:
                correct_pred_top3 = int(hit.sum().item())
            else:
                correct_pred_top5 = int(hit.sum().item())

        # diagnostics: rank/margins/auroc (1 pos per query by construction)
        ranks = []
        hard_margins = []
        pos_minus_meanneg = []
        pos_probs = []
        aurocs = []
        for i in range(Bk):
            row = logits[i]
            j = int(pos_idx[i].item())
            pos_logit = row[j]
            neg = torch.cat([row[:j], row[j + 1 :]])
            if neg.numel() > 0:
                max_neg = torch.max(neg)
                mean_neg = torch.mean(neg)
                hard_margins.append((pos_logit - max_neg).item())
                pos_minus_meanneg.append((pos_logit - mean_neg).item())
                rank = 1 + torch.sum(neg > pos_logit).item()
            else:
                hard_margins.append(float("inf"))
                pos_minus_meanneg.append(float("inf"))
                rank = 1
            ranks.append(float(rank))
            aurocs.append(float(self._pairwise_auroc_1pos(pos_logit, neg).item()))

            if self.modelname in ["symile", "clip"]:
                pos_probs.append(torch.nn.functional.softmax(row.unsqueeze(0), dim=1)[0, j].item())
            else:
                pos_probs.append(torch.sigmoid(row[j]).item())

        retrieval_acc_top1 = correct_pred_top1 / Bk
        retrieval_acc_top3 = correct_pred_top3 / Bk
        retrieval_acc_top5 = correct_pred_top5 / Bk

        ranks_t = torch.tensor(ranks, dtype=torch.float32)
        margins_t = torch.tensor(hard_margins, dtype=torch.float32)
        pos_minus_meanneg_t = torch.tensor(pos_minus_meanneg, dtype=torch.float32)
        pos_prob_t = torch.tensor(pos_probs, dtype=torch.float32)
        aurocs_t = torch.tensor(aurocs, dtype=torch.float32)

        return {
            "acc@top1": retrieval_acc_top1,
            "acc@top3": retrieval_acc_top3,
            "acc@top5": retrieval_acc_top5,
            "rank_mean": torch.mean(ranks_t).item(),
            "rank_median": torch.quantile(ranks_t, 0.5).item(),
            "rank_p95": torch.quantile(ranks_t, 0.95).item(),
            "hard_margin_mean": torch.mean(margins_t).item(),
            "hard_margin_p5": torch.quantile(margins_t, 0.05).item(),
            "hard_margin_p50": torch.quantile(margins_t, 0.5).item(),
            "hard_margin_p95": torch.quantile(margins_t, 0.95).item(),
            "pos_minus_meanneg_mean": torch.mean(pos_minus_meanneg_t).item(),
            "pos_prob_mean": torch.mean(pos_prob_t).item(),
            "pos_prob_p50": torch.quantile(pos_prob_t, 0.5).item(),
            "pos_prob_p95": torch.quantile(pos_prob_t, 0.95).item(),
            "auroc_mean": torch.mean(aurocs_t).item(),
            "auroc_p50": torch.quantile(aurocs_t, 0.5).item(),
            "auroc_p95": torch.quantile(aurocs_t, 0.95).item(),
        }

    def _zeroshot_retrieval_preselected(self, split, split_nr, bootstrap=False):
        """
        Calculates zero-shot retrieval accuracy for a given dataset split ('val'
        or 'test'), where the task is to retrieve the true corresponding CXR
        image for each query ECG and labs pair.

        Args:
            split (str): The dataset split to evaluate ('val' or 'test').
            bootstrap (bool): Whether to bootstrap resample the test retrieval dataset.

        Returns:
            dict: Retrieval metrics including accuracies and ranking/separation diagnostics.
        """
        if split == "val":
            split = "val_retrieval"
        if split == "test":
            split = "test"
        retrieval_ds = self.get_retrieval_dataset(split, split_nr)

        if bootstrap:
            retrieval_ds = self.resample_retrieval_ds(retrieval_ds)

        # get query data (positive samples)
        mask = retrieval_ds["label"] == 1
        query_r_c = retrieval_ds["r_c"][mask]
        query_r_e = retrieval_ds["r_e"][mask]
        query_r_l = retrieval_ds["r_l"][mask]
        use_gate = getattr(self, "use_gate", False) and getattr(self, "gate", None) is not None
        gate_mode = getattr(self.gate, "gate_mode", None) if use_gate else None

        # Candidate-dependent gating toggle (True when using attention gate).
        cand_dep_gate = bool(
            use_gate
            and gate_mode == "attention"
        )

        # candidate independent gate 
        if use_gate and not cand_dep_gate:
            dummy_cxr = torch.zeros_like(query_r_e)  # (num_queries, d)

            emb_q = [dummy_cxr, query_r_e, query_r_l]
            W_q = self.gate.compute_W(emb_q)
            gated_list, w_t, _ = self.gate.apply_for_target(self.candidate_idx, emb_q, W=W_q)
            p_null = 1.0 - (w_t[:, 1] + w_t[:, 2])
            self.log("val/gate_p_null_mean", p_null.mean(), on_step=False, on_epoch=True, sync_dist=True)

            self.log("val/gate_ecg_mean",  w_t[:, 1].mean(), on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/gate_labs_mean", w_t[:, 2].mean(), on_step=False, on_epoch=True, sync_dist=True)

            self._log_gate_cos_alignment(emb_q, gated_list, split="val", names=self.modalities)
            self._log_gate_cos_to_neutral(self.gate, gated_list, split="val", names=self.modalities)

            query_r_e = gated_list[1]
            query_r_l = gated_list[2]

        query_hadm_id = retrieval_ds["hadm_id"][mask]

        correct_pred_top1 = 0
        correct_pred_top3 = 0
        correct_pred_top5 = 0
        print_warning = False

        # Added collectors for diagnostics
        ranks = []
        pos_probs = []
        hard_margins = []          # pos - max_neg
        pos_minus_meanneg = []     # pos - mean_neg
        candidates_per_query = []
        aurocs = []


        # loop through each query sample
        for ix, true_hadm_id in enumerate(query_hadm_id):
            r_c = query_r_c[ix] # (d,)
            r_e = query_r_e[ix] # (d,)
            r_l = query_r_l[ix] # (d,)

            # find negative candidates for this query, and add to positive candidate
            mask = (retrieval_ds["label_hadm_id"] == true_hadm_id) & (retrieval_ds["label"] == 0)
            neg_r_c = retrieval_ds["r_c"][mask] # (candidate_n - 1, d)
            r_c = torch.cat([r_c.unsqueeze(0), neg_r_c], dim=0) # (candidate_n, d)

            candidate_label = torch.zeros(len(r_c), dtype=torch.long)
            candidate_label[0] = 1

            assert torch.sum(candidate_label) == 1 and torch.count_nonzero(candidate_label) == 1, \
                "candidate_label must have exactly one 1 and all other elements as 0."

            # Candidate-dependent gate: gate the query modalities (ecg,labs) conditioned on each candidate CXR.
            if cand_dep_gate:
                Nc, D = r_c.shape
                ecg = r_e.unsqueeze(0).expand(Nc, D)
                labs = r_l.unsqueeze(0).expand(Nc, D)
                pair_embs = [r_c, ecg, labs]

                chunk_size = int(self.params_method.get("gate_candidate_chunk_size", 256))
                raw_chunks = []
                sum_w = torch.zeros((3,), device=self.device)
                count_w = 0

                for s in range(0, Nc, chunk_size):
                    e = min(Nc, s + chunk_size)
                    pair_chunk = [x[s:e] for x in pair_embs]

                    W_pair = self.gate.compute_W(pair_chunk)  # (chunk, 3, 3)
                    gated_list, w_pair, _ = self.gate.apply_for_target(self.candidate_idx, pair_chunk, W=W_pair)

                    # lightweight diagnostics: only log cos sims once per epoch (first query + first chunk)
                    if ix == 0 and s == 0:
                        max_pairs = int(self.params_method.get("gate_log_max_pairs", 128))
                        pair_slice = [x[:max_pairs] for x in pair_chunk]
                        gated_slice = [g[:max_pairs] for g in gated_list]
                        self._log_gate_cos_alignment(pair_slice, gated_slice, split="val", names=self.modalities)
                        self._log_gate_cos_to_neutral(self.gate, gated_slice, split="val", names=self.modalities)
                        if hasattr(self.gate, "logit_gate_strength"):
                            alpha = torch.sigmoid(self.gate.logit_gate_strength.detach())
                            self.log("val/gate_alpha", alpha, on_step=False, on_epoch=True, sync_dist=True)

                    # accumulate for logging (optional, cheap)
                    sum_w += w_pair.detach().sum(dim=0)
                    count_w += int(w_pair.shape[0])

                    prod = gated_list[1] * gated_list[2]  # (chunk, D)
                    raw = (prod * pair_chunk[0]).sum(dim=1)  # (chunk,)
                    raw_chunks.append(raw)

                raw = torch.cat(raw_chunks, dim=0).unsqueeze(0)  # (1, Nc)
                logits = scale_mip_dvs(raw, d=D, M=3)
                logits = self.logit_scale.exp() * logits
                if self.modelname == "sigmile" and self.bias is not None:
                    logits = logits + self.bias

                if count_w > 0:
                    mean_w = sum_w / float(count_w)
                    self.log("val/gate_ecg_mean",  mean_w[1], on_step=False, on_epoch=True, sync_dist=True)
                    self.log("val/gate_labs_mean", mean_w[2], on_step=False, on_epoch=True, sync_dist=True)

                logits = logits.cpu()
            else:
                logits = zeroshot_retrieval_logits(
                    r_c,
                    [r_e, r_l],
                    self.logit_scale.exp(),
                    bias=self.bias,
                    modelname=self.modelname,
                ).cpu()

            # Diagnostics: compute rank and margins
            # logits shape: (1, n)
            pos_logit = logits[0, 0]
            neg_logits = logits[0, 1:]
            auc_q = self._pairwise_auroc_1pos(pos_logit, neg_logits)
            aurocs.append(auc_q.item())
            if neg_logits.numel() > 0:
                max_neg = torch.max(neg_logits)
                mean_neg = torch.mean(neg_logits)
                hard_margins.append((pos_logit - max_neg).item())
                pos_minus_meanneg.append((pos_logit - mean_neg).item())
                # Strict rank: 1 + count(neg > pos); ties are treated as equal (optimistic)
                rank = 1 + torch.sum(neg_logits > pos_logit).item()
            else:
                # Degenerate case: only positive candidate
                hard_margins.append(float('inf'))
                pos_minus_meanneg.append(float('inf'))
                rank = 1
            ranks.append(float(rank))

            # Also collect a unified positive softmax probability for comparability
            if self.modelname in ["symile", "clip"]:
                pos_probs.append(torch.nn.functional.softmax(logits, dim=1)[0, 0].item())
            else:   
                # Sigmoid family: independent probabilities per candidate
                pos_probs.append(torch.sigmoid(logits)[0, 0].item())

            # find all indices with the maximum value; if multiple indices have
            # the same max value, randomly select one of them (note: must use
            # np.random.choice instead of torch.randint to avoid altering the global random seed)
            max_value = torch.max(logits)
            max_indices = (logits == max_value).nonzero(as_tuple=True)[1]

            if len(max_indices) > 1:
                print_warning = True

            pred_ix = max_indices[np.random.choice(len(max_indices))].item()
            true_ix = torch.nonzero(candidate_label, as_tuple=True)[0].item()

            if pred_ix == true_ix:
                correct_pred_top1 += 1

            # Top-3 accuracy
            k = min(3, len(r_c))
            _, topk_indices = torch.topk(logits, k, dim=1)
            if true_ix in topk_indices[0]:
                correct_pred_top3 += 1

            # Top-5 accuracy
            k = min(5, len(r_c))
            _, topk_indices = torch.topk(logits, k, dim=1)
            if true_ix in topk_indices[0]:
                correct_pred_top5 += 1

        retrieval_acc_top1 = correct_pred_top1 / len(query_hadm_id)
        retrieval_acc_top3 = correct_pred_top3 / len(query_hadm_id)
        retrieval_acc_top5 = correct_pred_top5 / len(query_hadm_id)
        aurocs_t = torch.tensor(aurocs, dtype=torch.float32)
        auroc_mean = torch.mean(aurocs_t).item()
        auroc_p50 = torch.quantile(aurocs_t, 0.5).item()
        auroc_p95 = torch.quantile(aurocs_t, 0.95).item()

        # Aggregate diagnostics
        ranks_t = torch.tensor(ranks, dtype=torch.float32)
        margins_t = torch.tensor(hard_margins, dtype=torch.float32)
        pos_minus_meanneg_t = torch.tensor(pos_minus_meanneg, dtype=torch.float32)
        pos_prob_t = torch.tensor(pos_probs, dtype=torch.float32)

        # Ranking metrics
        rank_mean = torch.mean(ranks_t).item()
        rank_median = torch.quantile(ranks_t, 0.5).item()
        rank_p95 = torch.quantile(ranks_t, 0.95).item()

        # Separation metrics
        hard_margin_mean = torch.mean(margins_t).item()
        hard_margin_p5 = torch.quantile(margins_t, 0.05).item()
        hard_margin_p50 = torch.quantile(margins_t, 0.5).item()
        hard_margin_p95 = torch.quantile(margins_t, 0.95).item()

        pos_minus_meanneg_mean = torch.mean(pos_minus_meanneg_t).item()

        # Positive probability distribution (aligned with loss type)
        pos_prob_mean = torch.mean(pos_prob_t).item()
        pos_prob_p50 = torch.quantile(pos_prob_t, 0.5).item()
        pos_prob_p95 = torch.quantile(pos_prob_t, 0.95).item()

        if print_warning:
            print("\nWARNING: Multiple indices with max value. Random index selected.\n")

        return {
            "acc@top1": retrieval_acc_top1,
            "acc@top3": retrieval_acc_top3,
            "acc@top5": retrieval_acc_top5,
            # Added ranking metrics
            "rank_mean": rank_mean,
            "rank_median": rank_median,
            "rank_p95": rank_p95,
            # Added separation metrics
            "hard_margin_mean": hard_margin_mean,
            "hard_margin_p5": hard_margin_p5,
            "hard_margin_p50": hard_margin_p50,
            "hard_margin_p95": hard_margin_p95,
            "pos_minus_meanneg_mean": pos_minus_meanneg_mean,
            # Positive probability distribution (aligned with loss type)
            "pos_prob_mean": pos_prob_mean,
            "pos_prob_p50": pos_prob_p50,
            "pos_prob_p95": pos_prob_p95,
            # AUROC metrics
            "auroc_mean": auroc_mean,
            "auroc_p50": auroc_p50,
            "auroc_p95": auroc_p95,
        }

    def get_retrieval_dataset(self, set: str = "val", split_nr: int = 1):
        """
        Retrieves and encodes the evaluation data (queries and candidates) for
        the specified dataset split. Each sample in the dataset is either a
        positive or a negative candidate (according to its `label`). All positive
        candidates serve as queries. Therefore the total size of the evaluation
        set is evaluation_n = num_queries * num_candidates.

        Args:
            split (str): The dataset split to evaluate ('val' or 'test').

        Returns:
            dict: A dictionary containing the encoded query data with the following keys:
                - "r_c" (torch.Tensor): Encoded representations of the CXR data (evaluation_n, d).
                - "r_e" (torch.Tensor): Encoded representations of the ECG data (evaluation_n, d).
                - "r_l" (torch.Tensor): Encoded representations of the laboratory test data (evaluation_n, d).
                - "hadm_id" (torch.Tensor): Tensor containing the hospital admission ID for each sample (evaluation_n,).
                - "label_hadm_id" (torch.Tensor): Hospital admission ID indicating the true corresponding CXR for which
                        this sample is a candidate (evaluation_n,). For positive candidates, `hamd_id` = `label_hadm_id`.
                - "label" (torch.Tensor): Tensor containing the label (1 or 0) to indicate whether the sample is a
                        positive or negative candidate (evaluation_n,).
        """
        batch_sz = self.params_retrival_ds["batch_size"]
        retrieval_ds = Dataset_SymileMimic(split=set, split_nr=split_nr)

        r_c, r_e, r_l = [], [], []
        hadm_id = []
        label_hadm_id = []
        label = []

        for batch in DataLoader(retrieval_ds, batch_size=batch_sz, shuffle=False, drop_last=False, generator=torch.Generator()):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            model_output = self.forward(batch)
            embeddings = model_output["embeddings"]

            r_c.append(embeddings[0])
            r_e.append(embeddings[1])
            r_l.append(embeddings[2])

            hadm_id.append(batch["hadm_id"])
            label_hadm_id.append(batch["label_hadm_id"])
            label.append(batch["label"])

        r_c = torch.cat(r_c, dim=0)
        r_e = torch.cat(r_e, dim=0)
        r_l = torch.cat(r_l, dim=0)
        hadm_id = torch.cat(hadm_id, dim=0)
        label_hadm_id = torch.cat(label_hadm_id, dim=0)
        label = torch.cat(label, dim=0)

        if self.params_method["embedding_norm"]:
            r_c = nn.functional.normalize(r_c, dim=1)
            r_e = nn.functional.normalize(r_e, dim=1)
            r_l = nn.functional.normalize(r_l, dim=1)

        assert len(r_c) == len(r_e) == len(r_l) == len(retrieval_ds), \
            "r_c, r_e, r_l, and retrieval_ds should have the same length"

        return {
            "r_c": r_c, 
            "r_e": r_e, 
            "r_l": r_l, 
            "hadm_id": hadm_id,
            "label_hadm_id": label_hadm_id,
            "label": label,
        }

    def resample_retrieval_ds(self, ds):
        # get all query samples
        mask = ds["label"] == 1
        query_r_c = ds["r_c"][mask]
        query_r_e = ds["r_e"][mask]
        query_r_l = ds["r_l"][mask]
        query_hadm_id = ds["hadm_id"][mask]
        query_label_hadm_id = ds["label_hadm_id"][mask]
        query_label = ds["label"][mask]

        # randomly sample from the query subset with replacement
        n_samples = len(query_label)
        sample_indices = torch.randint(0, n_samples, (n_samples,), dtype=torch.long)

        # apply the sampled indices consistently across all keys
        sampled_r_c = query_r_c[sample_indices]
        sampled_r_e = query_r_e[sample_indices]
        sampled_r_l = query_r_l[sample_indices]
        sampled_hadm_id = query_hadm_id[sample_indices]
        sampled_label_hadm_id = query_label_hadm_id[sample_indices]
        sampled_label = query_label[sample_indices]

        # get the negative candidate samples
        negative_mask = ds["label"] == 0
        negative_r_c = ds["r_c"][negative_mask]
        negative_r_e = ds["r_e"][negative_mask]
        negative_r_l = ds["r_l"][negative_mask]
        negative_hadm_id = ds["hadm_id"][negative_mask]
        negative_label_hadm_id = ds["label_hadm_id"][negative_mask]
        negative_label = ds["label"][negative_mask]

        # combine positive and negative samples
        final_r_c = torch.cat([sampled_r_c, negative_r_c])
        final_r_e = torch.cat([sampled_r_e, negative_r_e])
        final_r_l = torch.cat([sampled_r_l, negative_r_l])
        final_hadm_id = torch.cat([sampled_hadm_id, negative_hadm_id])
        final_label_hadm_id = torch.cat([sampled_label_hadm_id, negative_label_hadm_id])
        final_label = torch.cat([sampled_label, negative_label])

        return {"r_c": final_r_c,
                "r_e": final_r_e,
                "r_l": final_r_l,
                "hadm_id": final_hadm_id,
                "label_hadm_id": final_label_hadm_id,
                "label": final_label}
