from typing import Optional

import torch


def occlusion(
    score_fn: callable,
    tok_idxs: torch.LongTensor,
    null_value: int,
    is_excluded: Optional[torch.BoolTensor] = None,
):
    scores = []
    for i in range(tok_idxs.size(-1)):
        if torch.all(is_excluded[..., i]):
            scores.append(torch.full_like(tok_idxs[..., 0].float(), -float("inf")))
            continue
        occluded = tok_idxs.clone()
        occluded[..., i] = null_value
        scores.append(score_fn(occluded))
    return torch.stack(scores, dim=-1)


def approximate_occlusion(
    score_fn: callable,
    tok_embeddings: torch.FloatTensor,
    null_embedding: torch.FloatTensor,
    is_excluded: Optional[torch.BoolTensor] = None,
):
    """
    First-order Taylor expansion of the occlusion score.
    """
    tok_embeddings = torch.nn.Parameter(tok_embeddings)
    score = score_fn(tok_embeddings).sum()
    score.backward()
    emb_grad = tok_embeddings.grad

    perturbation = null_embedding - tok_embeddings

    score_delta = (emb_grad * perturbation).sum(-1)

    score_delta = torch.where(is_excluded, torch.full_like(score_delta, -float("inf")), score_delta)
    return score_delta
