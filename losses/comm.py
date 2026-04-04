import torch
import torch.nn.functional as F


def _sample_modality_dropout_mask(batch_size, num_modalities, keep_prob, device, dtype):
    """
    Samples a per-sample modality dropout mask while guaranteeing that at least
    one modality remains active for every sample.
    """
    mask = (torch.rand(batch_size, num_modalities, device=device) < keep_prob).to(dtype=dtype)

    # Ensure at least one modality is kept per sample.
    empty = mask.sum(dim=1) == 0
    if empty.any():
        rand_idx = torch.randint(0, num_modalities, (int(empty.sum().item()),), device=device)
        mask[empty] = 0.0
        mask[empty, rand_idx] = 1.0

    return mask


def _augment_modalities(embeddings, feature_dropout=0.1, modality_dropout=0.0, noise_std=0.0):
    """
    Applies simple stochastic multimodal augmentations directly to modality
    embeddings. This keeps the implementation plug-and-play within the current
    codebase, where losses receive modality embeddings rather than raw inputs.
    """
    if len(embeddings) == 0:
        raise ValueError("_augment_modalities requires at least one modality embedding.")

    batch_size = embeddings[0].shape[0]
    num_modalities = len(embeddings)
    device = embeddings[0].device
    dtype = embeddings[0].dtype

    keep_prob = 1.0 - float(modality_dropout)
    modality_mask = _sample_modality_dropout_mask(batch_size, num_modalities, keep_prob, device, dtype)

    augmented = []
    for idx, emb in enumerate(embeddings):
        x = emb

        if feature_dropout > 0.0:
            x = F.dropout(x, p=float(feature_dropout), training=True)

        if noise_std > 0.0:
            x = x + (float(noise_std) * torch.randn_like(x))

        x = x * modality_mask[:, idx].unsqueeze(1)
        augmented.append(x)

    return augmented, modality_mask


def compute_comm_logits(z_1, z_2, logit_scale):
    """
    CLIP/SimCLR-style logits between two augmented multimodal views.
    """
    z_1 = F.normalize(z_1, dim=1)
    z_2 = F.normalize(z_2, dim=1)
    return logit_scale * (z_1 @ z_2.t())


def _resolve_multimodal_views(r_a, r_b, r_c, **kwargs):
    """
    Resolves the two multimodal views used by the CoMM objective.

    The model must compute fused multimodal views and pass them as kwargs using
    `z_view1` and `z_view2`.
    """
    z_view1 = kwargs.get("z_view1", None)
    z_view2 = kwargs.get("z_view2", None)
    if z_view1 is not None and z_view2 is not None:
        return z_view1, z_view2

    raise ValueError("comm(...) requires `z_view1` and `z_view2` to be provided by the model.")


def comm(r_a, r_b, r_c, logit_scale, negative_sampling=None, bias=None, **kwargs):
    """
    Plug-and-play CoMM-style objective adapted to the current codebase.

    The model must compute two fused multimodal views and pass them via
    `z_view1` and `z_view2` in kwargs.

    Notes:
      - `negative_sampling` and `bias` are ignored, mirroring `clip(...)`.
      - Augmentations/fusion belong in the model path, not inside this loss.
    """
    z_1, z_2 = _resolve_multimodal_views(r_a, r_b, r_c, **kwargs)

    logits = compute_comm_logits(z_1, z_2, logit_scale)
    labels = torch.arange(logits.shape[0], device=logits.device)

    loss_forward = F.cross_entropy(logits, labels)
    loss_backward = F.cross_entropy(logits.t(), labels)

    return 0.5 * (loss_forward + loss_backward)
