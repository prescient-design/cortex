from collections import OrderedDict
from typing import Optional

import numpy as np
from lightning import LightningDataModule
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

from cortex.model.leaf import ClassifierLeaf
from cortex.task._abstract_task import BaseTask


class ClassificationTask(BaseTask):
    """
    Binary or multiclass classification
    """

    def __init__(
        self,
        data_module: LightningDataModule,
        input_map: dict[str, str],
        leaf_key: str,
        class_col: str,
        num_classes: int,
        root_key: Optional[str] = None,
        corrupt_train_inputs: bool = False,
        corrupt_inference_inputs: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            data_module=data_module,
            input_map=input_map,
            leaf_key=leaf_key,
            corrupt_train_inputs=corrupt_train_inputs,
            corrupt_inference_inputs=corrupt_inference_inputs,
        )
        self.class_col = class_col
        self.num_classes = num_classes
        self.root_key = root_key

        if "mask_inputs" in kwargs:
            msg = "mask_inputs is deprecated, specify input corruptions at the root node level"
            raise DeprecationWarning(msg)

    def format_batch(self, batch: OrderedDict, corrupt_frac: float = None) -> dict:
        """
        Format a batch of data for a `NeuralTree` object
        """
        return {
            "root_inputs": self.format_inputs(batch, corrupt_frac=corrupt_frac),
            "leaf_targets": self.format_targets(batch),
        }

    def format_inputs(self, batch: OrderedDict, corrupt_frac: Optional[float] = None) -> dict:
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
                "targets": np.array(batch[self.class_col]).reshape(-1),
            }
        }
        return targets

    def create_leaf(self, in_dim: int, branch_key: str) -> ClassifierLeaf:
        """
        Create the leaf node for this task to be added to a `NeuralTree` object.
        """
        label_smoothing = "corrupt_frac" if self.corrupt_train_inputs else 0.0
        return ClassifierLeaf(
            in_dim=in_dim,
            num_classes=self.num_classes,
            branch_key=branch_key,
            label_smoothing=label_smoothing,
            root_key=self.root_key,
        )

    def compute_eval_metrics(self, ensemble_output: dict, targets: dict, task_key: str) -> dict:
        targets = targets["targets"]
        cp_key = f"{task_key}_class_probs"
        labels = list(range(self.num_classes))
        # multi-class predictions
        avg_class_probs = ensemble_output[cp_key].mean(0).cpu().numpy()
        if avg_class_probs.shape[-1] > 2:
            task_metrics = multiclass_classification_metrics(avg_class_probs, targets, labels)
        elif avg_class_probs.shape[-1] == 2:
            task_metrics = binary_classification_metrics(avg_class_probs, targets, labels)
        else:
            raise ValueError("Invalid number of classes")

        return task_metrics


def binary_classification_metrics(avg_class_probs: np.ndarray, targets: np.ndarray, labels: list[int]):
    top_1 = avg_class_probs.argmax(-1)
    # ROC AUC is undefined if there is only one class in the test set
    try:
        roc = roc_auc_score(targets, avg_class_probs[..., 1])
    except ValueError:
        roc = float("NaN")
    task_metrics = {
        "bin_acc": accuracy_score(targets, top_1),
        "bin_auc": roc,
        "bin_nll": log_loss(targets, avg_class_probs, labels=labels),
        "bin_precision": precision_score(y_true=targets, y_pred=top_1),
        "bin_recall": recall_score(y_true=targets, y_pred=top_1),
        "bin_f1": f1_score(y_true=targets, y_pred=top_1),
        "bin_avg_precision": average_precision_score(y_true=targets, y_score=avg_class_probs[..., 1]),
    }
    return task_metrics


def multiclass_classification_metrics(avg_class_probs: np.ndarray, targets: np.ndarray, labels: list[int]):
    task_metrics = {
        "mc_top_1_acc": top_k_accuracy_score(y_true=targets, y_score=avg_class_probs, k=1, labels=labels),
        "mc_ovo_auc": roc_auc_score(targets, avg_class_probs, multi_class="ovo", labels=labels),
        "mc_nll": log_loss(targets, avg_class_probs, labels=labels),
    }
    return task_metrics
