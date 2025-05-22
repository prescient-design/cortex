from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch.nn import functional as F

from cortex.corruption._abstract_corruption import CorruptionProcess
from cortex.model.branch import BranchNodeOutput
from cortex.model.leaf import ClassifierLeaf, LeafNodeOutput
from cortex.model.root import RootNodeOutput

# avoids circular import
if TYPE_CHECKING:
    from cortex.model.tree import NeuralTree, NeuralTreeOutput


@dataclass
class DenoisingLanguageModelLeafOutput(LeafNodeOutput):
    logits: torch.Tensor


class DenoisingLanguageModelLeaf(ClassifierLeaf):
    """
    Leaf node which transforms branch sequence features to discrete sequence logits.

    Can optionally apply a corruption process to the masked tokens during training,
    which serves as a form of data augmentation to increase sample diversity and
    potentially improve embedding quality. This is particularly useful with
    biologically-informed corruption processes like BLOSUM62-based substitutions
    for protein sequences.
    """

    def __init__(
        self,
        *args,
        corruption_process: Optional[CorruptionProcess] = None,
        corruption_rate: float = 0.1,
        layernorm: bool = True,
        **kwargs,
    ):
        """
        Initialize the DenoisingLanguageModelLeaf.

        Args:
            corruption_process: Optional corruption process to apply to masked targets during training
            corruption_rate: Fixed rate at which to apply corruption to masked targets (default: 0.1)
            *args: Additional positional arguments to pass to the parent class
            **kwargs: Additional keyword arguments to pass to the parent class
        """
        super().__init__(*args, layernorm=layernorm, **kwargs)
        self.corruption_process = corruption_process
        self.corruption_rate = corruption_rate

    def forward(self, branch_outputs: BranchNodeOutput, *args, **kwargs) -> DenoisingLanguageModelLeafOutput:
        """
        Args:
            branch_outputs: SeqCNNBranchOutput
        Returns:
            outputs: DenoisingLanguageModelLeafOutput
        """
        branch_features = branch_outputs.branch_features
        logits = self.encoder(branch_features)
        outputs = DenoisingLanguageModelLeafOutput(logits=logits)
        return outputs

    def loss(
        self,
        leaf_outputs: DenoisingLanguageModelLeafOutput,
        root_outputs: RootNodeOutput,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        masked_logits, masked_targets = self.format_outputs(leaf_outputs, root_outputs)
        return self.loss_fn(masked_logits, masked_targets)

    def format_outputs(
        self, leaf_outputs: DenoisingLanguageModelLeafOutput, root_outputs: RootNodeOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = leaf_outputs.logits
        tgt_tok_idxs = root_outputs.tgt_tok_idxs.to(logits.device)

        is_corrupted = root_outputs.is_corrupted.to(logits.device)
        masked_logits = torch.masked_select(logits, is_corrupted[..., None]).view(-1, logits.shape[-1])
        masked_tok_idxs = torch.masked_select(tgt_tok_idxs, is_corrupted).to(masked_logits.device)

        # Apply data augmentation if corruption_process is provided and we're in training mode
        if self.corruption_process is not None and self.training:
            # Reshape to match expected input format for corruption process
            # The corruption process expects a batch of sequences
            batch_size = masked_tok_idxs.size(0)

            # Apply the corruption with fixed rate
            corrupted_tok_idxs, _ = self.corruption_process(
                masked_tok_idxs.view(batch_size, 1), corrupt_frac=self.corruption_rate
            )

            # Reshape back to the original shape
            masked_tok_idxs = corrupted_tok_idxs.view(-1)

        return masked_logits, masked_tok_idxs

    def evaluate(
        self,
        leaf_outputs: DenoisingLanguageModelLeafOutput,
        root_outputs: RootNodeOutput,
        *args,
        **kwargs,
    ) -> dict:
        # The model is already in eval mode during evaluation, so no corruption will be applied
        logits, targets = self.format_outputs(leaf_outputs, root_outputs)
        pred_class = logits.argmax(-1)
        correct = pred_class.eq(targets)
        log_prob = F.log_softmax(logits, dim=-1)
        perplexity = 2 ** (-1 * log_prob / math.log(2))

        metrics = {
            "nll": self.loss_fn(logits, targets).item(),
            "acc": correct.float().mean().item(),
            "perplexity": perplexity.mean().item(),
        }

        return metrics


def format_denoising_lm_ensemble_output(
    leaf_outputs: list[DenoisingLanguageModelLeafOutput],
    root_outputs: list[RootNodeOutput],
    task_key: str,
):
    res = {}
    logits = [l_out.logits for l_out in leaf_outputs]
    tgt_tok_idxs = [r_out.tgt_tok_idxs for r_out in root_outputs]
    is_corrupted = [r_out.is_corrupted for r_out in root_outputs]

    masked_logits = torch.stack(
        [
            torch.masked_select(lgt, mask[..., None].to(lgt.device)).view(-1, lgt.shape[-1])
            for lgt, mask in zip(logits, is_corrupted)
        ]
    )
    masked_tok_idxs = torch.stack([torch.masked_select(tgt, mask) for tgt, mask in zip(tgt_tok_idxs, is_corrupted)]).to(
        masked_logits.device
    )

    res[f"{task_key}_logits"] = masked_logits
    res[f"{task_key}_targets"] = masked_tok_idxs
    return res


def mlm_conditional_log_likelihood(
    tree_output: NeuralTreeOutput,
    x_instances,
    x_occluded,
    root_key: str,
):
    """
    Compute the MLM conditional log-likelihood of the masked tokens in `x_occluded` given the unmasked tokens in `x_instances`.
    """
    task_outputs = tree_output.fetch_task_outputs(root_key)
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
