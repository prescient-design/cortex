import numpy as np
import numpy.typing as npt
import pandas as pd

from cortex.data.dataset._data_frame_dataset import DataFrameDataset


class NumpyDataset(DataFrameDataset):
    """
    Create a DataFrameDataset from a dictionary of numpy arrays stored in memory.
    Useful if not reading data from disk.
    """

    def __init__(
        self,
        data: dict[str, npt.NDArray],
        train: bool = True,
        random_seed: int = 0xDEADBEEF,
    ) -> None:
        total_len = len(data[list(data.keys())[0]])
        if not all(len(arr) == total_len for arr in data.values()):
            raise ValueError("All arrays must have the same length.")

        # randomly split the data into train and test sets
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(total_len)
        split = int(total_len * 0.8)
        if train:
            indices = indices[:split]
        else:
            indices = indices[split:]

        self._data = pd.DataFrame({k: v[indices] for k, v in data.items()})
        self.columns = list(data.keys())
