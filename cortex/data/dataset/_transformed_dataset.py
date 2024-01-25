from typing import Optional

import hydra
import pandas as pd
from omegaconf import DictConfig
from torch.nn import Sequential
from torch.utils.data import ConcatDataset, Dataset
from cortex.datasets._dataframe_dataset import DataFrameDataset

from collections import OrderedDict
from typing import Any

class TransformedDataset(DataFrameDataset):
    def __init__(
        self,
        base_dataset: Dataset | DictConfig,
        columns: Optional[list[None]] = None,
        preprocessing_transforms: Optional[list] = None,
        runtime_transforms: Optional[list] = None,
    ):
        if isinstance(base_dataset, DictConfig):
            base_dataset = hydra.utils.instantiate(base_dataset)
        if isinstance(base_dataset, ConcatDataset):
            data = pd.concat(
                [dataset._data for dataset in base_dataset.datasets], ignore_index=True
            )
        else:
            data = base_dataset._data.reset_index(drop=True)

        preprocessing_transforms = preprocessing_transforms or []
        if len(preprocessing_transforms) > 0:
            self._preprocessing_transforms = Sequential(*preprocessing_transforms)
        else:
            self._preprocessing_transforms = None

        super().__init__(
            self._preprocess(data),
            columns,
        )
        assert all([c in self._data.columns for c in self.columns])

        runtime_transforms = runtime_transforms or []
        if len(runtime_transforms) > 0:
            self._runtime_transforms = Sequential(*runtime_transforms)
        else:
            self._runtime_transforms = None

    def _preprocess(self, data) -> pd.DataFrame:
        if self._preprocessing_transforms is not None:
            data = self._preprocessing_transforms(data).reset_index(drop=True)
        return data

    def __getitem__(self, index) -> OrderedDict[str, Any]:
        item = self._fetch_item(index)
        if self._runtime_transforms is not None:
            item = self._runtime_transforms(item)
        return self._format_item(item)
