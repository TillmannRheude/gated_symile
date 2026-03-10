import torch
import torch.nn.functional as F
from losses.utils import scale_mip_dvs


def compute_logits_neg_sampling_n(x, y, z):
    """
    Computes the logits for anchor modality x with batch_sz - 1 negatives for
    each positive - or (batch_sz^2 - batch_sz) total negatives.

    If batch_sz is n, then returned logits have size (n, n) with n positive
    multilinear inner products and (n^2 - n) negative multilinear inner products.

    Positive multilinear inner products (MIPs) are along the diagonal of the
    square logits matrix. For example, the second row of `logits` might be:

    [ MIP(x[1], y[3], z[2]) MIP(x[1], y[1], z[1]) MIP(x[1], y[0], z[1]) MIP(x[1], y[2], z[3]) ].

    Notice that only the second element is the positive MIP; all others are negative.
    There is a small chance of a false negative MIP.

    Args:
        x (torch.Tensor): Representation vector of size (batch_sz, d_r).
        y (torch.Tensor): Representation vector of size (batch_sz, d_r).
        z (torch.Tensor): Representation vector of size (batch_sz, d_r).
    Returns:
        logits (torch.Tensor): Logits for x of size (batch_sz, batch_sz).
    """
    # shuffle rows of y and z
    y_shuff = y[torch.randperm(y.shape[0])]
    z_shuff = z[torch.randperm(z.shape[0])]
    logits_x = x @ torch.t(y_shuff * z_shuff) # (batch_sz, batch_sz)
    MIP_of_pos_triples = (x * y * z).sum(axis=1) # (batch_sz)
    # insert positive triples along diagonal of shuffled logits
    return torch.where(torch.eye(n=x.shape[0]).to(x.device) > 0.5, MIP_of_pos_triples, logits_x)


def compute_logits_neg_sampling_n_squared(x, y, z):
    """
    Computes the logits for anchor modality x with batch_sz^2 - 1 negatives for
    each positive.

    If batch size is n, then returned logits have size (n, n^2) with n positive
    multilinear inner products and (n^3 - n) negative multilinear inner products.

    Positive multilinear inner products (MIP) are along the main diagonal of the
    (non-square) logits matrix. For example, if n = 4, then the second row of
    `logits` is:

    [ MIP(x[1], y[0], z[0]) MIP(x[1], y[1], z[1]) MIP(x[1], y[2], z[2]) MIP(x[1], y[3], z[3])
      MIP(x[1], y[0], z[3]) MIP(x[1], y[1], z[0]) MIP(x[1], y[2], z[1]) MIP(x[1], y[3], z[2])
      MIP(x[1], y[0], z[2]) MIP(x[1], y[1], z[3]) MIP(x[1], y[2], z[0]) MIP(x[1], y[3], z[1])
      MIP(x[1], y[0], z[1]) MIP(x[1], y[1], z[2]) MIP(x[1], y[2], z[3]) MIP(x[1], y[3], z[0])  ]

    Notice that only the second element is the positive MIP; all others are negative.

    Args:
        x (torch.Tensor): Representation vector of size (batch_sz, d_r).
        y (torch.Tensor): Representation vector of size (batch_sz, d_r).
        z (torch.Tensor): Representation vector of size (batch_sz, d_r).
    Returns:
        logits (torch.Tensor): Logits for x of size (batch_sz, batch_sz^2).
    """
    y_z = []
    for i in range(y.shape[0]):
        y_z.append(y * z)
        z = torch.roll(z, shifts=1, dims=0)

    # concatenate elements in y_z so that y_z has shape (n^2, d) where each row
    # is a different element-wise product of a row from y and a row from z
    y_z = torch.cat(y_z, 0)

    # return logits with shape (n, n^2) where each row is the multilinear inner
    # product between that row in x and each row from y_z
    logits = x @ y_z.T
    return logits


def symile_gated(
    r_a, r_b, r_c,
    logit_scale,
    negative_sampling,
    gate,                 # ModalityAttentionGate
    bias=None,
    labels=None,
    candidates=None,      # optional tuple (c_a, c_b, c_c)
    pair_num_negatives=None,
):
    d = r_a.shape[-1]
    M = 3

    emb_local = [r_a, r_b, r_c]

    if negative_sampling == "pair":
        # pair_num_negatives = 128  # r_a.shape[0] // 16

        # candidate pools (local or provided global pool)
        if candidates is None:
            c_a, c_b, c_c = r_a, r_b, r_c
        else:
            c_a, c_b, c_c = candidates
        cand_pools = [c_a, c_b, c_c]

        B = r_a.shape[0]
        if labels is None:
            # local in-batch positives
            labels = torch.arange(B, device=r_a.device)

        # clamp K to available negatives
        def _loss_for_target(t: int) -> torch.Tensor:
            c_t = cand_pools[t]               # (N, d)
            N = c_t.shape[0]
            if N < 2:
                # degenerate: no negatives
                return (r_a.sum() * 0.0) + 0.0

            K = int(pair_num_negatives)
            K = max(1, min(K, N - 1))

            pos = labels.to(device=r_a.device, dtype=torch.long)  # (B,)
            if pos.min().item() < 0 or pos.max().item() >= N:
                raise ValueError(f"Labels out of range for target {t}: got [{int(pos.min())},{int(pos.max())}] but N={N}")

            # sample negatives uniformly without replacement constraint (cheap + fine)
            neg = torch.randint(0, N - 1, (B, K), device=r_a.device, dtype=torch.long)  # in [0, N-2]
            neg = neg + (neg >= pos[:, None]).long()  # shift to skip pos -> now in [0, N-1] \ {pos}

            idx = torch.cat([pos[:, None], neg], dim=1)  # (B, K+1), positive is column 0

            # build pair-batch embeddings of shape (B*(K+1), d) per modality
            D = r_a.shape[1]
            KK = K + 1
            pair_embs = []
            for m in range(3):
                if m == t:
                    x = cand_pools[m][idx.reshape(-1)]  # (B*(K+1), d)
                else:
                    x0 = emb_local[m]  # (B, d)
                    x = x0[:, None, :].expand(B, KK, D).reshape(B * KK, D)
                pair_embs.append(x)

            # candidate-conditioned gate: queries derived from target embedding
            W_pair = gate.compute_W(pair_embs)  # requires your gate support this
            gated_list, _, _ = gate.apply_for_target(t, pair_embs, W=W_pair)

            # symile score for target t: dot( cand_t , Π_{m!=t} gated_m )
            prod = torch.ones_like(gated_list[0])
            for m in range(3):
                if m == t:
                    continue
                prod = prod * gated_list[m]  # (B*(K+1), d)

            cand_t = pair_embs[t]  # (B*(K+1), d)
            raw = (prod * cand_t).sum(dim=1).view(B, KK)  # (B, K+1)

            # DVS scaling + temperature
            raw = scale_mip_dvs(raw, d=d, M=M)
            logits = logit_scale * raw
            if bias is not None:
                logits = logits + bias  # safe no-op for symile

            y = torch.zeros((B,), device=r_a.device, dtype=torch.long)  # positive at column 0
            return F.cross_entropy(logits, y)

        loss_a = _loss_for_target(0)
        loss_b = _loss_for_target(1)
        loss_c = _loss_for_target(2)
        return (loss_a + loss_b + loss_c) / 3.0

    # --> candidate-independent gated symile
    W_local = gate.compute_W(emb_local)

    # For each anchor/target t in {0,1,2}, get gated local triplet
    (ga0, gb0, gc0), w0, _ = gate.apply_for_target(0, emb_local, W_local)
    (ga1, gb1, gc1), w1, _ = gate.apply_for_target(1, emb_local, W_local)
    (ga2, gb2, gc2), w2, _ = gate.apply_for_target(2, emb_local, W_local)

    if candidates is None:
        cand0 = (ga0, gb0, gc0)
        cand1 = (ga1, gb1, gc1)
        cand2 = (ga2, gb2, gc2)
    else:
        c_a, c_b, c_c = candidates
        emb_cand = [c_a, c_b, c_c]
        W_cand = gate.compute_W(emb_cand)
        (cga0, cgb0, cgc0), _, _ = gate.apply_for_target(0, emb_cand, W_cand)
        (cga1, cgb1, cgc1), _, _ = gate.apply_for_target(1, emb_cand, W_cand)
        (cga2, cgb2, cgc2), _, _ = gate.apply_for_target(2, emb_cand, W_cand)
        cand0 = (cga0, cgb0, cgc0)
        cand1 = (cga1, cgb1, cgc1)
        cand2 = (cga2, cgb2, cgc2)

    if negative_sampling == "n":
        raw_a = compute_logits_neg_sampling_n(ga0, gb0, gc0)
        raw_b = compute_logits_neg_sampling_n(gb1, ga1, gc1)
        raw_c = compute_logits_neg_sampling_n(gc2, ga2, gb2)
    elif negative_sampling == "n_squared":
        # loss for anchor a uses candidate pools of b and c under target=0 gating
        raw_a = compute_logits_neg_sampling_n_squared(ga0, cand0[1], cand0[2])
        raw_b = compute_logits_neg_sampling_n_squared(gb1, cand1[0], cand1[2])
        raw_c = compute_logits_neg_sampling_n_squared(gc2, cand2[0], cand2[1])
    else:
        raise ValueError("negative_sampling must be either 'n' or 'n_squared'.")

    raw_a = scale_mip_dvs(raw_a, d, M)
    raw_b = scale_mip_dvs(raw_b, d, M)
    raw_c = scale_mip_dvs(raw_c, d, M)

    logits_a = logit_scale * raw_a
    logits_b = logit_scale * raw_b
    logits_c = logit_scale * raw_c

    if labels is None:
        labels = torch.arange(logits_a.shape[0], device=r_a.device)

    loss_a = F.cross_entropy(logits_a, labels)
    loss_b = F.cross_entropy(logits_b, labels)
    loss_c = F.cross_entropy(logits_c, labels)
    return (loss_a + loss_b + loss_c) / 3.0

def symile(
    r_a, 
    r_b, 
    r_c, 
    logit_scale, 
    negative_sampling, 
    bias=None, 
    labels=None, 
    candidates=None,
    pair_num_negatives=None,
):
    """
    Computes the Symile loss for a batch of representations. The final Symile
    loss is an average of the loss terms where each modality is treated as the
    anchor in turn.

    The argument `negative_sampling` can take on one of two values:
        - `n` (for O(n)): draws n - 1 negative samples for each positive
        - `n_squared` (for O(n^2)): draws n^2 - 1 negative samples for each positive

    Args:
        r_a, r_b, r_c (torch.Tensor): Representation vectors each of size (batch_sz, d_r).
        logit_scale (torch.Tensor): Learned temperature parameter.
        negative_sampling (str): Specifies the negative sampling strategy.
                                 Must be either `n` or `n_squared`.
        bias (torch.Tensor, optional): Bias parameter for SigLIP loss. Not used in Symile loss.

    Returns:
        (torch.Tensor): Average over the losses where each modality is treated
                        as the anchor in turn.
    """
    d = r_a.shape[-1]
    M = 3

    if candidates is None:
        c_a, c_b, c_c = r_a, r_b, r_c
    else:
        c_a, c_b, c_c = candidates
    cand_pools = [c_a, c_b, c_c]
    emb_local = [r_a, r_b, r_c]

    if negative_sampling == "pair":
        print(f"pair_num_negatives: {pair_num_negatives}")
        # pair_num_negatives = 128

        B = r_a.shape[0]
        if labels is None:
            # local in-batch positives
            labels = torch.arange(B, device=r_a.device)

        def _loss_for_target(t: int) -> torch.Tensor:
            c_t = cand_pools[t]  # (N, d)
            N = c_t.shape[0]
            if N < 2:
                return (r_a.sum() * 0.0) + 0.0

            K = int(pair_num_negatives)
            K = max(1, min(K, N - 1))

            pos = labels.to(device=r_a.device, dtype=torch.long)  # (B,)
            if pos.min().item() < 0 or pos.max().item() >= N:
                raise ValueError(
                    f"Labels out of range for target {t}: got [{int(pos.min())},{int(pos.max())}] but N={N}"
                )

            # sample negatives uniformly (cheap; no strict w/o-replacement)
            neg = torch.randint(0, N - 1, (B, K), device=r_a.device, dtype=torch.long)  # [0, N-2]
            neg = neg + (neg >= pos[:, None]).long()  # shift to skip pos -> now in [0,N-1]\{pos}

            idx = torch.cat([pos[:, None], neg], dim=1)  # (B, K+1), positive in col 0
            KK = K + 1

            # gather target candidates: (B, K+1, d)
            cand_t = c_t[idx]  # (B, KK, d)

            # broadcast the other (local) modalities across the KK candidates
            prod = torch.ones((B, KK, d), device=r_a.device, dtype=r_a.dtype)
            for m in range(3):
                if m == t:
                    continue
                x0 = emb_local[m]  # (B, d)
                prod = prod * x0[:, None, :].expand(B, KK, d)

            raw = (prod * cand_t).sum(dim=2)  # (B, KK)

            # DVS scaling + temperature
            raw = scale_mip_dvs(raw, d=d, M=M)
            logits = logit_scale * raw
            if bias is not None:
                logits = logits + bias  # no-op for symile, but harmless

            y = torch.zeros((B,), device=r_a.device, dtype=torch.long)
            return F.cross_entropy(logits, y)

        loss_a = _loss_for_target(0)
        loss_b = _loss_for_target(1)
        loss_c = _loss_for_target(2)
        return (loss_a + loss_b + loss_c) / 3.0

    if negative_sampling == "n":
        raw_a = compute_logits_neg_sampling_n(r_a, r_b, r_c)
        raw_b = compute_logits_neg_sampling_n(r_b, r_a, r_c)
        raw_c = compute_logits_neg_sampling_n(r_c, r_a, r_b)
    elif negative_sampling == "n_squared":
        raw_a = compute_logits_neg_sampling_n_squared(r_a, c_b, c_c)
        raw_b = compute_logits_neg_sampling_n_squared(r_b, c_a, c_c)
        raw_c = compute_logits_neg_sampling_n_squared(r_c, c_a, c_b)
    else:
        raise ValueError("negative_sampling must be either 'n' or 'n_squared'.")
    
    # DVS scaling
    raw_a = scale_mip_dvs(raw_a, d, M)
    raw_b = scale_mip_dvs(raw_b, d, M)
    raw_c = scale_mip_dvs(raw_c, d, M)
    # Temperature
    logits_a = logit_scale * raw_a
    logits_b = logit_scale * raw_b
    logits_c = logit_scale * raw_c

    if labels is None:
        labels = torch.arange(logits_a.shape[0], device=r_a.device)

    loss_a = F.cross_entropy(logits_a, labels)
    loss_b = F.cross_entropy(logits_b, labels)
    loss_c = F.cross_entropy(logits_c, labels)
    return (loss_a + loss_b + loss_c) / 3.0

