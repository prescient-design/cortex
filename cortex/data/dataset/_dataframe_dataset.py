"""
Simple Pytorch Dataset for reading from a dataframe.
"""

from collections import OrderedDict
from typing import Any

import pandas as pd
from torch.utils.data import Dataset


class DataFrameDataset(Dataset):
    """
    Simple dataset that retrieves data from a dataframe.
    """

    def __init__(self, data: pd.DataFrame, columns: list[str], dedup: bool = True):
        self.columns = columns
        self._data = data.reset_index(drop=True)

        if dedup:
            self._data.drop_duplicates(subset=self.columns, inplace=True, ignore_index=True)

    def __len__(self):
        return len(self.df)

    def _fetch_item(self, index) -> pd.DataFrame:
        # check if int or slice
        if isinstance(index, int):
            item = self._data.iloc[index : index + 1]
        else:
            item = self._data.iloc[index]

    def _format_item(self, item: pd.DataFrame) -> OrderedDict[str, Any]:
        if len(item) == 1:
            return OrderedDict([(c, item[c].iloc[0]) for c in self.columns])
        return OrderedDict([(c, item[c]) for c in self.columns])

    def __getitem__(self, index) -> OrderedDict[str, Any]:
        item = self._fetch_item(index)
        return self._format_item(item)


def ordered_dict_collator(
    batch: list[OrderedDict[str, Any]],
) -> OrderedDict[str, Any]:
    """
    Collates a batch of OrderedDicts into a single OrderedDict.
    """
    return OrderedDict([(key, [item[key] for item in batch]) for key in batch[0].keys()])
