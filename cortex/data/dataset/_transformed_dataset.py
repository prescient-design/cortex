from collections import OrderedDict
from typing import Any, Optional

import pandas as pd
from torch.nn import Sequential

from cortex.data.dataset import DataFrameDataset


class TransformedDataset(DataFrameDataset):
    def __init__(
        self,
        preprocessing_transforms: Optional[list] = None,
        runtime_transforms: Optional[list] = None,
        *args,
        **kwargs,
    ):
        # if isinstance(base_dataset, DictConfig):
        #     base_dataset = hydra.utils.instantiate(base_dataset)
        # if isinstance(base_dataset, ConcatDataset):
        #     data = pd.concat([dataset._data for dataset in base_dataset.datasets], ignore_index=True)
        # else:
        #     data = base_dataset._data.reset_index(drop=True)

        preprocessing_transforms = preprocessing_transforms or []
        if len(preprocessing_transforms) > 0:
            self._preprocessing_transforms = Sequential(*preprocessing_transforms)
        else:
            self._preprocessing_transforms = None

        super().__init__(*args, **kwargs)
        self._data = self._preprocess(self._data)

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
