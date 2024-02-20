from typing import Optional

import torch

from cortex.attribution import occlusion
from cortex.model.tree import NeuralTree, NeuralTreeOutput, fetch_task_outputs


def greedy_occlusion_selection(
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
    print(select_coord)
    return select_coord


class NOSCoordinateScore(object):
    r"""
    Wrapper for a `cortex` model that computes the following score:
    $$ s_i = (1 - \lambda) (v(x_{-i}) - v(x)) - \lambda \logp(x_i \mid x_{-i}) $$
    where $x_i$ is the $i$th token in the input sequence $x$, $x_{-i}$ is the sequence with the $i$th token occluded,
    $v$ is a value function, and $\lambda$ is a hyperparameter.
    """

    def __init__(
        self,
        model: NeuralTree,
        value_fn,
        logp_fn,
        x_instances,
        lambda_val: float,
        root_key: str,
    ):
        self.model = model
        self.value_fn = value_fn
        self.logp_fn = logp_fn
        self._x_instances = x_instances
        with torch.inference_mode():
            self._ref_values = value_fn(model(x_instances))
        self._lambda_val = lambda_val
        self._root_key = root_key

    def __call__(self, x_occluded):
        with torch.inference_mode():
            model_output = self.model(x_occluded)
        values = self.value_fn(model_output)
        values = values - self._ref_values  # want to change positions with large change in value
        if self._lambda_val == 0.0:
            return values
        logp = self.logp_fn(
            model_output, self._x_instances, x_occluded, self._root_key
        )  # want to change positions with low probability
        return (1 - self._lambda_val) * values - self._lambda_val * logp


def mlm_conditional_log_likelihood(
    tree_output: NeuralTreeOutput,
    x_instances,
    x_occluded,
    root_key: str,
):
    """
    Compute the MLM conditional log-likelihood of the masked tokens in `x_occluded` given the unmasked tokens in `x_instances`.
    """
    task_outputs = fetch_task_outputs(tree_output, root_key)
    token_probs = task_outputs["logits"].log_softmax(-1)  # (ensemble_size, batch_size, seq_len, vocab_size)
    is_occluded = x_instances != x_occluded
    token_cll = token_probs.gather(-1, x_instances[None, ..., None]).squeeze(-1)  # (ensemble_size, batch_size, seq_len)
    infill_cll = (token_cll * is_occluded).sum(-1) / is_occluded.sum(-1)  # (ensemble_size, batch_size)
    return infill_cll.mean(0)


def mlm_pseudo_log_likelihood(
    tok_idxs: torch.LongTensor,
    null_value: int,
    model: NeuralTree,
    root_key: str,
    is_excluded: Optional[torch.BoolTensor] = None,
):
    """
    Compute the MLM pseudo-log-likelihood of the full tok_idxs sequence
    """
    scores = []
    for coord_idx in range(tok_idxs.size(-1)):
        if is_excluded is not None and torch.all(is_excluded[..., coord_idx]):
            scores.append(torch.full_like(tok_idxs[..., 0].float(), 0.0))
            continue
        occluded = tok_idxs.clone()
        occluded[..., coord_idx] = null_value
        with torch.inference_mode():
            model_output = model(occluded, leaf_keys=[f"{root_key}_0"])
        scores.append(mlm_conditional_log_likelihood(model_output, tok_idxs, occluded, root_key))
    is_included = 1.0 - is_excluded.float() if is_excluded is not None else torch.ones_like(scores)
    scores = is_included * torch.stack(scores, dim=-1)
    return scores.sum(-1) / is_included.sum(-1)
