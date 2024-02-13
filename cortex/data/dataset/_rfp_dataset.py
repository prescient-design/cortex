import pandas as pd

from cortex.data.dataset._data_frame_dataset import DataFrameDataset

_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/samuelstanton/lambo/main/lambo/assets/fpbase/rfp_known_structures.tar.gz"
)


def tokenize_rfp_df(data: pd.DataFrame) -> pd.DataFrame:
    raw_seqs = data["foldx_seq"]
    tokenized_seqs = []
    for seq in raw_seqs:
        tokenized_seqs.append(" ".join(seq))
    data["tokenized_seq"] = tokenized_seqs
    return data


class RedFluorescentProteinDataset(DataFrameDataset):
    _name = "rfp"
    _target = "rfp_known_structures.csv"
    columns = [
        "tokenized_seq",
        "foldx_total_energy",
        "SASA",
    ]

    def __init__(self, root: str, download: bool = False, download_source: str = _DOWNLOAD_URL, **kwargs):
        super().__init__(root=root, download=download, download_source=download_source, **kwargs)
        self._data = tokenize_rfp_df(self._data)
