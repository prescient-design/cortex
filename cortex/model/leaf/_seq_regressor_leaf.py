from typing import Optional

import numpy as np
import torch

from cortex.model.branch import BranchNodeOutput
from cortex.model.leaf import RegressorLeaf, RegressorLeafOutput
from cortex.model.root import RootNodeOutput


class SequenceRegressorLeaf(RegressorLeaf):
    def forward(self, branch_outputs: BranchNodeOutput) -> RegressorLeafOutput:
        # clip start/end token features since they will never be labeled
        res = self.encoder(branch_outputs.branch_features[..., 1:-1, :])
        return self.transform_output(res)

    def loss(
        self,
        leaf_outputs: RegressorLeafOutput,
        root_outputs: RootNodeOutput,
        targets: torch.Tensor,
        position_mask: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self.label_smoothing == "corrupt_frac" and hasattr(root_outputs, "corrupt_frac"):
            label_smoothing = root_outputs.corrupt_frac
        else:
            label_smoothing = self.label_smoothing
        canon_param = leaf_outputs.canon_param

        if position_mask is None:
            mask_shape = canon_param.shape[1:-1]
            is_labeled = torch.full(mask_shape, False, dtype=torch.bool, device=canon_param.device)
        else:
            position_mask = torch.from_numpy(position_mask).to(canon_param.device)
            is_labeled = adjust_sequence_mask(position_mask, canon_param[0])

        labeled_params = torch.stack(
            [
                torch.masked_select(canon_param[0], is_labeled[..., None]),
                torch.masked_select(canon_param[1], is_labeled[..., None]),
            ],
            dim=0,
        ).view(2, -1, self.out_dim)
        assert labeled_params.size(-2) == targets.shape[-2]
        return self.loss_from_canon_param(labeled_params, targets, label_smoothing)


def adjust_sequence_mask(mask: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
    padded_position_mask = torch.full(tgt_tensor.shape[:-1], False, dtype=torch.bool, device=tgt_tensor.device)
    padded_len = padded_position_mask.size(-1)
    unpadded_len = mask.size(-1)
    if padded_len < unpadded_len:
        padded_position_mask[..., :padded_len] = mask[..., :padded_len]
    else:
        padded_position_mask[..., :unpadded_len] = mask
    return padded_position_mask
