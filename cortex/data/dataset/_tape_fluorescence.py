import json
from typing import Any

import numpy as np
import pandas as pd

from cortex.data.dataset._data_frame_dataset import DataFrameDataset

_DOWNLOAD_URL = "http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/fluorescence.tar.gz"


def tokenize_gfp_df(data: pd.DataFrame) -> pd.DataFrame:
    raw_seqs = data["primary"]
    tokenized_seqs = []
    for seq in raw_seqs:
        tokenized_seqs.append(" ".join(seq))
    data["tokenized_seq"] = tokenized_seqs
    return data


class TAPEFluorescenceDataset(DataFrameDataset):
    _name = "tape_fluorescence"
    _target = "fluorescence"
    columns = [
        "tokenized_seq",
        "log_fluorescence",
    ]

    def __init__(self, root: str, download: bool = False, download_source: str = _DOWNLOAD_URL, **kwargs):
        super().__init__(root=root, download=download, download_source=download_source, **kwargs)

    def _read_data(self, path: str, dedup: bool, train: bool, random_seed: int, **kwargs: Any) -> pd.DataFrame:
        if train:
            paths = [
                path / "fluorescence" / "fluorescence_train.json",
                path / "fluorescence" / "fluorescence_valid.json",
            ]
        else:
            paths = [path / "fluorescence" / "fluorescence_test.json"]

        dfs = []
        for p in paths:
            # with open(p, "r") as f:
            data = json.loads(p.read_text())
            dfs.append(pd.DataFrame.from_records(data))

        data = pd.concat(dfs, ignore_index=True)

        data.loc[:, "log_fluorescence"] = np.array([val[0] for val in data["log_fluorescence"].values])
        data = tokenize_gfp_df(data)

        return data
