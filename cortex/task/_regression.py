from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from botorch.models.transforms.outcome import OutcomeTransform
from scipy import stats

from cortex.data.data_module import TaskDataModule
from cortex.metrics import spearman_rho
from cortex.model.elemental import DDPStandardize
from cortex.model.leaf import RegressorLeaf
from cortex.task._abstract_task import BaseTask


class RegressionTask(BaseTask):
    """
    Regression task
    """

    def __init__(
        self,
        data_module: TaskDataModule,
        input_map: dict[str, str],
        outcome_cols: list[str],
        leaf_key: str,
        corrupt_train_inputs: bool = False,
        corrupt_inference_inputs: bool = False,
        root_key: Optional[str] = None,
        nominal_label_var: float = 0.25**2,
        **kwargs,
    ) -> None:
        super().__init__(
            data_module=data_module,
            input_map=input_map,
            leaf_key=leaf_key,
            corrupt_train_inputs=corrupt_train_inputs,
            corrupt_inference_inputs=corrupt_inference_inputs,
        )
        self.outcome_cols = outcome_cols
        self.out_dim = len(self.outcome_cols)
        self.root_key = root_key
        self.nominal_label_var = nominal_label_var

    def fit_transform(self, outcome_transform: OutcomeTransform, device: torch.device, dtype: torch.dtype) -> None:
        """
        Fit an `OutcomeTransform` object to the training data.
        """
        outcome_transform.train()
        train_df = self.data_module.datasets["train_val"]._data.iloc[self.data_module.datasets["train"].indices]
        outcomes = self.format_targets(train_df)[self.leaf_key]["targets"]
        outcomes = torch.tensor(outcomes, device=device, dtype=dtype)
        outcome_transform(outcomes)
        outcome_transform.eval()

    def format_batch(self, batch: OrderedDict, corrupt_frac: float = None) -> dict:
        """
        Format a batch of data for a `NeuralTree` object
        """
        return {
            # "df": batch,
            "root_inputs": self.format_inputs(batch, corrupt_frac=corrupt_frac),
            "leaf_targets": self.format_targets(batch),
        }

    def format_inputs(self, batch: OrderedDict, corrupt_frac: float = 0.0) -> dict:
        """
        Format input DataFrame for a `NeuralTree` object
        """
        inputs = {}
        for root_key, input_cols in self.input_map.items():
            inputs[root_key] = {
                "inputs": np.concatenate([np.array(batch[col]).reshape(-1, 1) for col in input_cols], axis=-1),
                "corrupt_frac": corrupt_frac,
            }
        return inputs

    def format_targets(self, batch: OrderedDict) -> dict:
        """
        Format target DataFrame for a `NeuralTree` object
        """
        targets = {
            self.leaf_key: {
                "targets": np.concatenate(
                    [np.array(batch[col]).astype(float).reshape(-1, 1) for col in self.outcome_cols], axis=-1
                )
            }
        }
        return targets

    def create_leaf(self, in_dim: int, branch_key: str) -> RegressorLeaf:
        """
        Create the leaf node for this task to be added to a `NeuralTree` object.
        """
        label_smoothing = "corrupt_frac" if self.corrupt_train_inputs else 0.0
        outcome_transform = DDPStandardize(m=self.out_dim)
        return RegressorLeaf(
            in_dim=in_dim,
            out_dim=self.out_dim,
            branch_key=branch_key,
            outcome_transform=outcome_transform,
            label_smoothing=label_smoothing,
            root_key=self.root_key,
            nominal_label_var=self.nominal_label_var,
        )

    def compute_eval_metrics(self, ensemble_output: dict, targets: dict, task_key: str) -> dict:
        targets = targets["targets"]
        loc_key = f"{task_key}_mean"
        scale_key = f"{task_key}_st_dev"
        loc = ensemble_output[loc_key].cpu().numpy()
        scale = ensemble_output[scale_key].cpu().numpy()
        task_metrics = regression_metrics(loc, scale, targets)
        return task_metrics


def regression_metrics(loc, scale, targets):
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
