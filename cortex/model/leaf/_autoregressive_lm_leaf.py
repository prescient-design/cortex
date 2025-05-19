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
    from cortex.model.tree import NeuralTreeOutput


@dataclass
class AutoregressiveLanguageModelLeafOutput(LeafNodeOutput):
    logits: torch.Tensor


class AutoregressiveLanguageModelLeaf(ClassifierLeaf):
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
        Initialize the AutoregressiveLanguageModelLeaf.

        Args:
            corruption_process: Optional corruption process to apply to masked targets during training
            corruption_rate: Fixed rate at which to apply corruption to masked targets (default: 0.1)
            *args: Additional positional arguments to pass to the parent class
            **kwargs: Additional keyword arguments to pass to the parent class
        """
        super().__init__(*args, layernorm=layernorm, **kwargs)
        self.corruption_process = corruption_process
        self.corruption_rate = corruption_rate

    def forward(self, branch_outputs: BranchNodeOutput, *args, **kwargs) -> AutoregressiveLanguageModelLeafOutput:
        """
        Args:
            branch_outputs: TransforerBranchOutput  (is_causal should be true)
        Returns:
            outputs: AutoregressiveLanguageModelLeafOutput
        """
        branch_features = branch_outputs.branch_features
        logits = self.encoder(branch_features)
        outputs = AutoregressiveLanguageModelLeafOutput(logits=logits)
        return outputs

    def loss(
        self,
        leaf_outputs: AutoregressiveLanguageModelLeafOutput,
        root_outputs: RootNodeOutput,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        masked_logits, masked_targets = self.format_outputs(leaf_outputs, root_outputs)
        return self.loss_fn(masked_logits, masked_targets)

    def format_outputs(
        self, leaf_outputs: AutoregressiveLanguageModelLeafOutput, root_outputs: RootNodeOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = leaf_outputs.logits[..., :-1, :]
        tgt_tok_idxs = root_outputs.tgt_tok_idxs.to(logits.device)[..., 1:]

        # Apply data augmentation if corruption_process is provided and we're in training mode
        if self.corruption_process is not None and self.training:
            # Apply the corruption with fixed rate
            tgt_tok_idxs, _ = self.corruption_process(tgt_tok_idxs, corrupt_frac=self.corruption_rate)

        return logits.flatten(0, -2), tgt_tok_idxs.flatten()

    def evaluate(
        self,
        leaf_outputs: AutoregressiveLanguageModelLeafOutput,
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


def format_autoregressive_lm_ensemble_output(
    leaf_outputs: list[AutoregressiveLanguageModelLeafOutput],
    root_outputs: list[RootNodeOutput],
    task_key: str,
):
    res = {}
    logits = [l_out.logits.flatten(0, -2) for l_out in leaf_outputs]
    tgt_tok_idxs = [r_out.tgt_tok_idxs.flatten() for r_out in root_outputs]

    res[f"{task_key}_logits"] = torch.stack(logits)
    res[f"{task_key}_targets"] = torch.stack(tgt_tok_idxs)

    return res


def autoregressive_log_likelihood(
    tree_output: NeuralTreeOutput,
    x_instances,
    root_key: str,
):
    """
    Compute the autoregressive log-likelihood of the tokens in `x_instances`.
    """
    task_outputs = tree_output.fetch_task_outputs(root_key)
    token_probs = task_outputs["logits"].log_softmax(-1)  # (ensemble_size, batch_size, seq_len, vocab_size)
    token_cll = token_probs.gather(-1, x_instances[None, ..., None]).squeeze(-1)  # (ensemble_size, batch_size, seq_len)
    return token_cll.mean(dim=(0, -1))
