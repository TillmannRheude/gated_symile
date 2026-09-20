import torch
import itertools

from losses.utils import apply_logit_scale, scale_mip_dvs
from losses.gram import compute_gramian_volume_matrix
from losses.triangle import compute_triangle_area_matrix

def zeroshot_retrieval_logits(
    r_x: torch.Tensor,
    rep_list: list[torch.Tensor],
    logit_scale_exp: torch.Tensor,
    bias: torch.Tensor = None,
    modelname: str = "symile",
):
    """
    Computes logits for zeroshot retrieval based on the specified loss function.

    Returns:
        Tensor: Logits for zeroshot retrieval, of shape (batch_sz, num_candidates).
    """

    if modelname in ["symile"]:
        # logits (batch_sz, n), matrix where each row i is [ MIP(r_x[i], r_y[i], r_z[0]) ... MIP(r_x[i], r_y[i], r_z[n-1]) ]
        product = torch.ones_like(rep_list[0])
        for r in rep_list:
            product *= r

        logits = product @ torch.t(r_x)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        d = r_x.shape[-1]
        M = len(rep_list) + 1
        logits = scale_mip_dvs(logits, d, M)

    elif modelname in ["clip"]:
        # logits (batch_sz, n) matrix where each row i is [ r_x[i]^T r_z[0] + r_z[0]^T r_y[i]   + r_x[i]^T r_y[i] ... r_x[i]^T r_z[n-1] + r_z[n-1]^T r_y[i] + r_x[i]^T r_y[i] ]
        for i in range(len(rep_list)):
            rep_list[i] = rep_list[i].unsqueeze(0) if rep_list[i].dim() == 1 else rep_list[i]  # (batch_sz, d)

        pairwise_sum_with_r_x = torch.zeros_like(rep_list[0] @ torch.t(r_x))  # (batch_sz, num_candidates)
        for r in rep_list:
            pairwise_sum_with_r_x += r @ torch.t(r_x)

        pairwise_sum_without_r_x = torch.zeros((rep_list[0].shape[0], 1), device=rep_list[0].device)  # (batch_sz, 1)
        for x, y in itertools.combinations(rep_list, 2):
            pairwise_sum_without_r_x += torch.diagonal(x @ torch.t(y)).unsqueeze(dim=1)

        logits = pairwise_sum_with_r_x + pairwise_sum_without_r_x

    elif modelname in ["triangle"]:
        # Match the same retrieval API as symile/clip:
        #   r_x = candidate bank of the target modality, shape (N, D)
        #   rep_list = [query_context_1, query_context_2], each (B, D)
        #
        # Row i scores candidate j via the triangle formed by:
        #   r_x[j], rep_list[0][i], rep_list[1][i]
        #
        # Smaller area = better, so logits = -area with shape (B, N).
        if len(rep_list) != 2:
            raise ValueError(
                "For modelname='triangle', rep_list must contain exactly 2 tensors: "
                "[query_context_modality_1, query_context_modality_2]."
            )

        cand_rep = r_x.unsqueeze(0) if r_x.dim() == 1 else r_x
        query_b = rep_list[0].unsqueeze(0) if rep_list[0].dim() == 1 else rep_list[0]
        query_c = rep_list[1].unsqueeze(0) if rep_list[1].dim() == 1 else rep_list[1]

        cand_norm2 = torch.sum(cand_rep * cand_rep, dim=1).unsqueeze(0)     # (1, N)
        query_b_norm2 = torch.sum(query_b * query_b, dim=1, keepdim=True)    # (B, 1)
        query_c_norm2 = torch.sum(query_c * query_c, dim=1, keepdim=True)    # (B, 1)

        b_cand = query_b @ cand_rep.t()                                       # (B, N)
        c_cand = query_c @ cand_rep.t()                                       # (B, N)
        b_c = torch.sum(query_b * query_c, dim=1, keepdim=True)               # (B, 1)

        u_norm = cand_norm2 + query_b_norm2 - (2.0 * b_cand)
        v_norm = cand_norm2 + query_c_norm2 - (2.0 * c_cand)
        uv_dot = cand_norm2 - b_cand - c_cand + b_c

        det_term = torch.clamp(u_norm * v_norm - uv_dot * uv_dot, min=0.0)
        area = 0.5 * det_term
        logits = -area

    elif modelname in ["gram"]:
        # Match the same retrieval API as symile/clip:
        #   r_x = candidate bank of the target modality, shape (N, D)
        #   rep_list = [query_context_1, query_context_2], each (B, D)
        #
        # Row i scores candidate j via the Gramian volume formed by:
        #   r_x[j], rep_list[0][i], rep_list[1][i]
        #
        # Smaller volume = better, so logits = -volume with shape (B, N).
        if len(rep_list) != 2:
            raise ValueError(
                "For modelname='gram', rep_list must contain exactly 2 tensors: "
                "[query_context_modality_1, query_context_modality_2]."
            )

        cand_rep = r_x.unsqueeze(0) if r_x.dim() == 1 else r_x
        query_1 = rep_list[0].unsqueeze(0) if rep_list[0].dim() == 1 else rep_list[0]
        query_2 = rep_list[1].unsqueeze(0) if rep_list[1].dim() == 1 else rep_list[1]

        volume = compute_gramian_volume_matrix(cand_rep, query_1, query_2).t()
        logits = -volume

    elif modelname in ["comm"]:
        # CoMM retrieval uses a fused query representation produced upstream
        # (e.g. by the multimodal transformer) and compares it to the candidate
        # bank via dot product.
        #
        #   r_x = candidate bank of the target modality, shape (N, D)
        #   rep_list = [fused_query_rep], shape (B, D)
        #
        # Returns logits of shape (B, N).
        if len(rep_list) != 1:
            raise ValueError(
                "For modelname='comm', rep_list must contain exactly 1 tensor: "
                "[fused_query_representation]."
            )

        cand_rep = r_x.unsqueeze(0) if r_x.dim() == 1 else r_x
        query_rep = rep_list[0].unsqueeze(0) if rep_list[0].dim() == 1 else rep_list[0]
        logits = query_rep @ cand_rep.t()

    elif modelname in ["symile_attention"]:
        # Learned-score Symile variant:
        #   r_x is unused here because the model is expected to have already
        #   produced learned compatibility logits against the relevant candidates.
        #
        #   rep_list = [z], where z has shape (B,), (B, 1), or (B, C)
        if len(rep_list) != 1:
            raise ValueError(
                "For modelname='symile_attention', rep_list must contain exactly 1 tensor: "
                "[learned_compatibility_logits]."
            )

        logits = rep_list[0]
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
        if logits.dim() != 2:
            raise ValueError(
                f"For modelname='symile_attention', expected logits of shape (B,), (B,1), or (B,C); got {tuple(logits.shape)}."
            )

    else:
        raise ValueError(f"Unsupported modelname: {modelname}")

    assert logits.dim() == 2, "Logits must be a 2D tensor."

    logits = apply_logit_scale(logits, logit_scale_exp)

    return logits
