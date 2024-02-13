from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from scipy import stats

from cortex.metrics import spearman_rho
from cortex.model.elemental import DDPStandardize
from cortex.model.leaf import SequenceRegressorLeaf, adjust_sequence_mask
from cortex.task._regression import RegressionTask


class SequenceRegressionTask(RegressionTask):
    def format_inputs(self, batch: OrderedDict, corrupt_frac: float = 0.0) -> dict:
        inputs = {}
        for root_key, input_cols in self.input_map.items():
            inputs[root_key] = {
                "inputs": np.concatenate([np.array(batch[col]).reshape(-1, 1) for col in input_cols], axis=-1),
                "corrupt_frac": corrupt_frac,
            }
        return inputs

    def format_targets(self, batch: OrderedDict) -> dict:
        # get non-nan values from target array, will be 2D array
        # assumes one outcome_col.
        if self.out_dim > 1:
            raise NotImplementedError("SequenceRegressionTask only supports one outcome_col")
        target_label = self.outcome_cols[0]
        padded_targets = np.vstack(pad_sequence_labels(batch[target_label]))
        target_is_nan = np.isnan(padded_targets)

        res = {
            self.leaf_key: {
                "targets": np.expand_dims(padded_targets[~target_is_nan], axis=-1),
                "position_mask": ~target_is_nan,
            }
        }
        return res

    def create_leaf(self, in_dim: int, branch_key: str) -> SequenceRegressorLeaf:
        """
        Create the leaf node for this task to be added to a `NeuralTree` object.
        """
        label_smoothing = "corrupt_frac" if self.corrupt_train_inputs else 0.0
        outcome_transform = DDPStandardize(m=self.out_dim)
        return SequenceRegressorLeaf(
            in_dim=in_dim,
            out_dim=self.out_dim,
            branch_key=branch_key,
            outcome_transform=outcome_transform,
            label_smoothing=label_smoothing,
            root_key=self.root_key,
        )

    def compute_eval_metrics(self, ensemble_output: dict, targets: dict, task_key: str) -> dict:
        unpadded_position_mask = torch.from_numpy(targets["position_mask"])
        targets = targets["targets"]

        loc_key = f"{task_key}_mean"
        scale_key = f"{task_key}_st_dev"
        loc = ensemble_output[loc_key].cpu()
        scale = ensemble_output[scale_key].cpu()

        padded_position_mask = adjust_sequence_mask(unpadded_position_mask, loc)

        new_shape = list(loc.shape[:-3]) + [-1, self.out_dim]
        loc = torch.masked_select(loc, padded_position_mask[..., None]).reshape(*new_shape).numpy()
        scale = torch.masked_select(scale, padded_position_mask[..., None]).reshape(*new_shape).numpy()

        diff = targets - loc.mean(0)
        norm_factor = np.linalg.norm(np.abs(targets) + np.abs(loc.mean(0)), axis=-1, keepdims=True)
        norm_diff = diff / np.clip(norm_factor, a_min=1e-6, a_max=None)
        rmse = np.sqrt((diff**2).mean())
        nrmse = np.sqrt((norm_diff**2).mean())
        task_metrics = {
            "nll": -stats.norm.logpdf(targets, loc, scale).mean().item(),
            "rmse": rmse.item(),
            "mae": np.abs(diff).mean().item(),
            "nrmse": nrmse.item(),
            "s_rho": spearman_rho(loc.mean(0), targets),
            "avg_pred_st_dev": scale.mean().item(),
            "avg_pred_mean": loc.mean().item(),
        }
        return task_metrics


def pad_sequence_labels(data: pd.Series) -> np.ndarray:
    max_len = data.apply(len).max()

    def _pad_target(array):
        padding = np.array([float("NaN")] * (max_len - len(array)))
        return np.concatenate([array, padding])

    return data.apply(_pad_target).values
