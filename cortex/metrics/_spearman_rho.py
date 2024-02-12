import numpy as np
from scipy import stats


def spearman_rho(scores: np.ndarray, targets: np.ndarray):
    """
    Compute the Spearman's rank correlation coefficient between scores and targets,
    averaged across the last dimension.
    """
    if scores.ndim == 1:
        return stats.spearmanr(targets, scores).correlation
    spearman_rho = 0.0
    for idx in range(targets.shape[-1]):
        spearman_rho += stats.spearmanr(targets[..., idx], scores[..., idx]).correlation / targets.shape[-1]
    return spearman_rho
