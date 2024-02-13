import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset

from cortex.io import download_and_extract_archive

T = TypeVar("T")


class DataFrameDataset(Dataset):
    _data: DataFrame
    _name: str = "temp"
    _target: str = "data.csv"
    columns = None

    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = False,
        download_source: Optional[str] = None,
        dedup: bool = True,
        train: bool = True,
        random_seed: int = 0xDEADBEEF,
        **kwargs: Any,
    ) -> None:
        """
        :param root: Root directory where the dataset subdirectory exists or,
            if :attr:`download` is ``True``, the directory where the dataset
            subdirectory will be created and the dataset downloaded.
        """
        if isinstance(root, str):
            root = Path(root).resolve()
        self._root = root

        path = self._root / self._name

        if os.path.exists(path / self._target):
            pass
        elif download:
            if download_source is None:
                raise ValueError("If `download` is `True`, `download_source` must be provided.")
            download_and_extract_archive(
                resource=download_source,
                source=path,
                destination=path,
                name=f"{self._name}.tar.gz",
                remove_archive=True,
            )
        else:
            raise ValueError(
                f"Dataset not found at {path}. " "If `download` is `True`, the dataset will be downloaded."
            )
        self._data = self._read_data(path, dedup=dedup, train=train, random_seed=random_seed, **kwargs)

    def _read_data(self, path: str, dedup: bool, train: bool, random_seed: int, **kwargs: Any) -> DataFrame:
        if self._target.endswith(".csv"):
            data = pd.read_csv(path / self._target, **kwargs)
        elif self._target.endswith(".parquet"):
            data = pd.read_parquet(path / self._target, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {self._target}")

        if self.columns is None:
            self.columns = list(data.columns)

        if dedup:
            data.drop_duplicates(inplace=True)

        # split data into train and test using random seed
        train_indices = data.sample(frac=0.8, random_state=random_seed).index
        test_indices = data.index.difference(train_indices)

        select_indices = train_indices if train else test_indices
        return data.loc[select_indices].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._data)

    def _fetch_item(self, index) -> pd.DataFrame:
        # check if int or slice
        if isinstance(index, int):
            item = self._data.iloc[index : index + 1]
        else:
            item = self._data.iloc[index]
        return item

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
    res = OrderedDict([(key, [item[key] for item in batch]) for key in batch[0].keys()])
    res["batch_size"] = len(batch)
    return res
