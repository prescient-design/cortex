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


def greedy_occlusion_search(
    tok_idxs: torch.LongTensor,
    score_fn: callable,
    null_value: int,
    num_coordinates: int,
    is_excluded: Optional[torch.BoolTensor] = None,
    take_second_prob: float = 0.5,
):
    """
    Greedy coordinate selection based on sensitivity of `score_fn` to pointwise occlusion.
    `score_fn` should be a callable that takes a tensor of token indices and returns a batch of scalar scores.
    At each iteration, each coordinate is occluded and the score_fn is evaluated on the resulting tensor.
    For each element in the batch, the coordinate with the highest score is selected and remains occluded.
    This process is repeated until `num_coordinates` coordinates are selected.
    Returns a tensor of indices of selected coordinates.
    """

    num_feasible = (~is_excluded).float().sum(-1)
    assert torch.all(num_feasible >= num_coordinates), "Not enough feasible coordinates"
    is_selected = torch.zeros_like(tok_idxs, dtype=torch.bool)
    for _ in range(num_coordinates):
        scores = occlusion(
            score_fn=score_fn, tok_idxs=tok_idxs, null_value=null_value, is_excluded=is_excluded + is_selected
        )
        # don't select already selected coordinates
        scores = scores.masked_fill(is_selected, -float("inf"))
        # don't select excluded coordinates
        if is_excluded is not None:
            scores = scores.masked_fill(is_excluded, -float("inf"))

        _, sorted_idxs = torch.sort(scores, dim=-1, descending=True)
        best_coord = sorted_idxs[..., 0]
        second_best = sorted_idxs[..., 1]
        second_available = (scores > -float("inf")).sum(-1) > 1
        take_second = (torch.rand_like(best_coord.float()) < take_second_prob) * second_available
        best_coord = torch.where(take_second, second_best, best_coord)

        is_selected.scatter_(-1, best_coord.unsqueeze(-1), True)
        tok_idxs = torch.where(is_selected, null_value, tok_idxs)
    select_coord = torch.where(is_selected)[1].view(*tok_idxs.shape[:-1], num_coordinates)
    return select_coord
