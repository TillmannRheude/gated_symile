import torch
import itertools

from losses.utils import scale_mip_dvs

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

    else:
        raise ValueError(f"Unsupported modelname: {modelname}")

    assert logits.dim() == 2, "Logits must be a 2D tensor."

    logits = logit_scale_exp * logits

    return logits
