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
