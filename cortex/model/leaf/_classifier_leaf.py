from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor, distributions, nn
from torch.nn import functional as F

from cortex.model.branch import BranchNodeOutput
from cortex.model.leaf import LeafNode, LeafNodeOutput
from cortex.model.root import RootNodeOutput


def cross_entropy_with_per_sample_smoothing(logits, targets, alpha=0.0):
    """Custom cross entropy loss function that supports per-sample label smoothing.

    Args:
        logits: Predicted logits of shape [batch_size, num_classes]
        targets: Ground truth labels of shape [batch_size]
        alpha: Float or tensor of shape [batch_size] specifying smoothing value per sample

    Returns:
        Loss tensor averaged across the batch
    """
    num_classes = logits.size(-1)

    # Convert scalar to tensor if needed
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.full((logits.size(0),), alpha, device=logits.device, dtype=torch.float)

    # Create one-hot encoded targets
    targets_one_hot = F.one_hot(targets, num_classes).float()

    # Apply label smoothing in a vectorized way
    # Expand alpha to match targets_one_hot dimensions for proper broadcasting
    alpha = alpha.view(-1, 1)
    uniform_dist = torch.ones_like(targets_one_hot) / num_classes
    smooth_targets = (1 - alpha) * targets_one_hot + alpha * uniform_dist

    # Compute loss using standard cross entropy with the smoothed targets
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(smooth_targets * log_probs).sum(dim=-1)

    return loss.mean()


@dataclass
class ClassifierLeafOutput(LeafNodeOutput):
    logits: torch.Tensor


def check_probs(probs: torch.Tensor, dim: int = -1) -> bool:
    """
    Check that the probabilities are valid
    """
    if torch.any(probs < 0) or torch.any(probs > 1):
        raise ValueError("Probabilities must be between 0 and 1")

    if not torch.allclose(probs.sum(dim=dim), torch.ones(probs.shape[:-1], device=probs.device)):
        raise ValueError("Probabilities must sum to 1")

    return True


class ClassifierLeaf(LeafNode):
    """
    Leaf node which transforms pooled branch features to discrete classifier logits
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        branch_key: str,
        num_layers: int = 0,
        last_layer_bias: bool = True,
        label_smoothing: Union[float, str] = 0.0,
        root_key: Optional[str] = None,
        layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.branch_key = branch_key
        self.root_key = root_key

        # testing out normalizing the penultimate activations
        encoder_modules = [nn.LayerNorm(in_dim, bias=False)] if layernorm else []
        if num_layers >= 1:
            for _ in range(num_layers):
                encoder_modules.extend(
                    [
                        nn.Linear(in_dim, in_dim),
                        nn.ReLU(inplace=True),
                    ]
                )
        encoder_modules.append(nn.Linear(in_dim, num_classes, bias=last_layer_bias))
        self.encoder = nn.Sequential(*encoder_modules)
        self.loss_fn = cross_entropy_with_per_sample_smoothing
        self.label_smoothing = label_smoothing

    def forward(self, branch_outputs: BranchNodeOutput) -> ClassifierLeafOutput:
        """
        Args:
            branch_outputs: {
                'branch_features': torch.Tensor,
                'branch_mask': torch.Tensor,
                'pooled_features': torch.Tensor
            }
        Returns:
            outputs: {'logits': torch.Tensor}
        """
        outputs = ClassifierLeafOutput(
            logits=self.encoder(branch_outputs.pooled_features),
        )
        return outputs

    def tie_last_layer_weight(self, weight):
        last_layer = self.encoder[-1]
        last_layer.weight = weight

    def untie_last_layer_weight(self):
        last_layer = self.encoder[-1]
        last_layer.weight = nn.Parameter(last_layer.weight.clone())

    def class_probs(
        self,
        branch_outputs: BranchNodeOutput,
    ):
        logits = self(branch_outputs).logits
        return logits.softmax(dim=-1)

    def sample(self, branch_outputs: BranchNodeOutput, num_samples: int):
        logits = self(branch_outputs).logits
        dist = distributions.Categorical(logits=logits)
        return dist.sample((num_samples,))

    def _preprocess_targets(self, targets: Tensor, device: torch.device):
        if not torch.is_tensor(targets):
            targets = torch.tensor(targets, dtype=torch.int64)
        targets = targets.to(device)
        return targets

    def loss(
        self,
        leaf_outputs: ClassifierLeafOutput,
        root_outputs: RootNodeOutput,
        targets: Tensor,
        *args,
        **kwargs,
    ):
        logits = leaf_outputs.logits

        if self.label_smoothing == "corrupt_frac" and hasattr(root_outputs, "corrupt_frac"):
            alpha = root_outputs.corrupt_frac
        else:
            alpha = self.label_smoothing

        targets = self._preprocess_targets(targets, logits.device)
        return self.loss_fn(logits, targets, alpha=alpha)

    def evaluate(self, outputs: ClassifierLeafOutput, targets: Tensor):
        logits = outputs.logits
        targets = self._preprocess_targets(targets, logits.device)
        pred_class = logits.argmax(-1)
        correct = pred_class.eq(targets)

        metrics = {
            "nll": self.loss_fn(logits, targets, alpha=0.0).item(),  # No smoothing during evaluation
            "acc": correct.float().mean().item(),
        }

        if self.num_classes == 2:
            num_total = pred_class.size(0)
            num_positive = targets.float().sum().item()
            num_negative = num_total - num_positive
            num_false_positive = (targets.eq(0) * pred_class.eq(1)).float().sum().item()
            num_false_negative = (targets.eq(1) * pred_class.eq(0)).float().sum().item()
            metrics["false_positive"] = num_false_positive / num_negative
            metrics["false_negative"] = num_false_negative / num_positive

        return metrics

    def initialize(self) -> None:
        """
        initialize leaf weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def format_classifier_ensemble_output(leaf_outputs: ClassifierLeafOutput, task_key: str):
    res = {}
    class_probs = torch.stack([l_out.logits.softmax(-1) for l_out in leaf_outputs])
    check_probs(class_probs)
    res[f"{task_key}_class_probs"] = class_probs
    res[f"{task_key}_top_1"] = class_probs.argmax(-1)
    return res
