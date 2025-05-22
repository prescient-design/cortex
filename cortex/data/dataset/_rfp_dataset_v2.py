import pandas as pd
from typing import Optional, List

from cortex.data.dataset._cortex_dataset import SequenceDataset

_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/samuelstanton/lambo/main/lambo/assets/fpbase/rfp_known_structures.tar.gz"
)


def tokenize_rfp_df(data: pd.DataFrame) -> pd.DataFrame:
    """Tokenize RFP sequences for dataloader processing."""
    raw_seqs = data["foldx_seq"]
    tokenized_seqs = []
    for seq in raw_seqs:
        tokenized_seqs.append(" ".join(seq))
    data["tokenized_seq"] = tokenized_seqs
    return data


class RedFluorescentProteinDatasetV2(SequenceDataset):
    """
    Updated RFP dataset using CortexDataset pattern with transform separation.

    Moves tokenization to dataloader for parallel execution.
    """

    _name = "rfp"
    _target = "rfp_known_structures.csv"
    columns = [
        "tokenized_seq",
        "foldx_total_energy",
        "SASA",
    ]

    def __init__(
        self,
        root: str,
        tokenizer_transform,
        max_len: int = 512,
        pad_tok_idx: int = 0,
        download: bool = False,
        download_source: str = _DOWNLOAD_URL,
        train_transforms: Optional[List] = None,
        eval_transforms: Optional[List] = None,
        corruption_transforms: Optional[List] = None,
        **kwargs,
    ):
        # Initialize SequenceDataset with tokenization moved to dataloader
        super().__init__(
            tokenizer_transform=tokenizer_transform,
            max_len=max_len,
            pad_tok_idx=pad_tok_idx,
            train_transforms=train_transforms,
            eval_transforms=eval_transforms,
            corruption_transforms=corruption_transforms,
            root=root,
            download=download,
            download_source=download_source,
            **kwargs,
        )

        # Apply RFP-specific preprocessing
        self._data = tokenize_rfp_df(self._data)

    def _fetch_item(self, index):
        """Fetch raw sequence data for tokenization in dataloader."""
        item = self._data.iloc[index].to_dict()

        # The sequence will be tokenized by dataloader transforms
        # Return raw sequence for tokenization
        if "tokenized_seq" in item:
            # Convert space-separated tokens back to raw sequence for proper tokenization
            item["sequence"] = item["tokenized_seq"].replace(" ", "")

        return item
