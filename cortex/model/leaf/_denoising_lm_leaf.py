import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from cortex.model.branch import BranchNodeOutput
from cortex.model.leaf import ClassifierLeaf, LeafNodeOutput
from cortex.model.root import RootNodeOutput


@dataclass
class DenoisingLanguageModelLeafOutput(LeafNodeOutput):
    logits: torch.Tensor


class DenoisingLanguageModelLeaf(ClassifierLeaf):
    """
    Leaf node which transforms branch sequence features to discrete sequence logits
    """

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

        return masked_logits, masked_tok_idxs

    def evaluate(
        self,
        leaf_outputs: DenoisingLanguageModelLeafOutput,
        root_outputs: RootNodeOutput,
        *args,
        **kwargs,
    ) -> dict:
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
