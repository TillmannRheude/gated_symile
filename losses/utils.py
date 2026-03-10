def scale_mip_dvs(score_matrix, d, M):
    """
    Scale raw multi-way MIP scores using generalized Deterministic Variance Scaling.
    scale_base = d ** ((M - 1) / 2)  (valid for any M >= 2)
    """
    scale_base = d ** ((M - 1) / 2)
    return score_matrix * scale_base