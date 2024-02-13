from abc import ABC, abstractmethod
from collections import OrderedDict

import pandas as pd

from cortex.data.data_module import TaskDataModule
from cortex.model.leaf import LeafNode


class BaseTask(ABC):
    def __init__(
        self,
        data_module: TaskDataModule,
        input_map: dict[str, str],
        leaf_key: str,
        corrupt_train_inputs: bool = False,
        corrupt_inference_inputs: bool = False,
        **kwargs,
    ) -> None:
        """
        Base class for tasks
        """
        self.data_module = data_module
        self.input_map = input_map
        self.leaf_key = leaf_key
        self._dataloaders = {
            "train": iter(self.data_module.train_dataloader()),
            "val": iter(self.data_module.val_dataloader()),
            "test": iter(self.data_module.test_dataloader()),
        }
        self.corrupt_train_inputs = corrupt_train_inputs
        self.corrupt_inference_inputs = corrupt_inference_inputs

    def sample_minibatch(self, split: str = "train", as_df: bool = False) -> dict | pd.DataFrame:
        """
        Return a random minibatch of data formatted for a `NeuralTree` object
        """
        try:
            batch = next(self._dataloaders[split])
        except StopIteration:
            dataset = self.data_module.datasets[split]
            if dataset is not None and len(dataset) > 0:
                self._dataloaders[split] = iter(self.data_module.get_dataloader(split))
                batch = next(self._dataloaders[split])
            else:
                return None

        if as_df:
            return batch

        if split == "train" and self.corrupt_inputs:
            corrupt_frac = None
        else:
            corrupt_frac = 0.0

        return self.format_batch(batch, corrupt_frac=corrupt_frac)

    def format_batch(self, batch: OrderedDict, corrupt_frac: float = None) -> dict:
        """
        Format a batch of data for a `NeuralTree` object
        """
        return {
            "root_inputs": self.format_inputs(batch, corrupt_frac=corrupt_frac),
            "leaf_targets": {},
        }

    @abstractmethod
    def format_inputs(self, batch: OrderedDict) -> dict:
        """
        Format input DataFrame for a `NeuralTree` object
        """
        raise NotImplementedError

    @abstractmethod
    def create_leaf(self, **kwargs) -> LeafNode:
        raise NotImplementedError

    @abstractmethod
    def compute_eval_metrics(self, **kwargs) -> dict:
        raise NotImplementedError
