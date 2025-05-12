import pandas as pd

from cortex.data.dataset import TAPEFluorescenceDataset, TAPEStabilityDataset
from cortex.data.dataset._data_frame_dataset import DataFrameDataset


# hack to combine TAPE datasets for self-supervised training
class TAPECombinedDataset(DataFrameDataset):
    columns = [
        "tokenized_seq",
        "partition",
    ]

    def __init__(self, root: str, download: bool = False, **kwargs):
        fluorescence_data = TAPEFluorescenceDataset(root=root, download=download, **kwargs)._data
        stability_data = TAPEStabilityDataset(root=root, download=download, **kwargs)._data

        fluorescence_data["partition"] = "fluorescence"
        stability_data["partition"] = "stability"
        self._data = pd.concat([fluorescence_data[self.columns], stability_data[self.columns]], ignore_index=True)
