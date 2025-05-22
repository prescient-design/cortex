from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd
from torch.nn import Sequential

from cortex.data.dataset._data_frame_dataset import DataFrameDataset


class CortexDataset(DataFrameDataset, ABC):
    """
    Base dataset class for cortex with transform separation.

    Moves tokenization and preprocessing from model forward pass to dataloader
    for parallel execution and improved GPU utilization.

    Key principles:
    - dataloader_transforms: Run in dataloader workers (tokenization, padding)
    - model_transforms: Run on GPU during forward pass (corruption, embeddings)
    """

    def __init__(
        self,
        dataloader_transforms: Optional[list] = None,
        model_transforms: Optional[list] = None,
        preprocessing_transforms: Optional[list] = None,
        *args,
        **kwargs,
    ):
        # Dataloader transforms: tokenization, padding (parallel execution)
        dataloader_transforms = dataloader_transforms or []
        if len(dataloader_transforms) > 0:
            self._dataloader_transforms = Sequential(*dataloader_transforms)
        else:
            self._dataloader_transforms = None

        # Model transforms: corruption, embedding operations (GPU execution)
        model_transforms = model_transforms or []
        if len(model_transforms) > 0:
            self._model_transforms = Sequential(*model_transforms)
        else:
            self._model_transforms = None

        # Preprocessing transforms: data cleaning, preprocessing
        preprocessing_transforms = preprocessing_transforms or []
        if len(preprocessing_transforms) > 0:
            self._preprocessing_transforms = Sequential(*preprocessing_transforms)
        else:
            self._preprocessing_transforms = None

        super().__init__(*args, **kwargs)
        self._data = self._preprocess(self._data)

    def _preprocess(self, data) -> pd.DataFrame:
        """Apply preprocessing transforms to raw data."""
        if self._preprocessing_transforms is not None:
            data = self._preprocessing_transforms(data).reset_index(drop=True)
        return data

    def __getitem__(self, index) -> Dict[str, Any]:
        """
        Get item with dataloader transforms applied.
        Returns pre-tokenized data ready for GPU processing.
        """
        item = self._fetch_item(index)

        # Apply dataloader transforms (tokenization, padding)
        if self._dataloader_transforms is not None:
            item = self._dataloader_transforms(item)

        return self._format_item(item)

    def apply_model_transforms(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply model transforms (corruption, embeddings) on GPU.
        Called by root nodes during forward pass.
        """
        if self._model_transforms is not None:
            batch = self._model_transforms(batch)
        return batch

    @abstractmethod
    def get_dataloader_transforms(self) -> list:
        """Return list of transforms to run in dataloader workers."""
        pass

    @abstractmethod
    def get_model_transforms(self) -> list:
        """Return list of transforms to run on GPU during forward pass."""
        pass


class SequenceDataset(CortexDataset):
    """
    Dataset for sequence data with tokenization moved to dataloader.
    """

    def __init__(
        self,
        tokenizer_transform,
        max_len: int,
        pad_tok_idx: int,
        train_transforms: Optional[list] = None,
        eval_transforms: Optional[list] = None,
        corruption_transforms: Optional[list] = None,
        *args,
        **kwargs,
    ):
        self.tokenizer_transform = tokenizer_transform
        self.max_len = max_len
        self.pad_tok_idx = pad_tok_idx

        # Import transforms
        from cortex.transforms import PadTransform, ToTensor

        # Build dataloader transforms (parallel execution)
        dataloader_transforms = []

        # Add training/eval specific transforms
        train_transforms = train_transforms or []
        eval_transforms = eval_transforms or []

        # Add shared transforms that should run in dataloader
        shared_dataloader_transforms = [
            tokenizer_transform,  # Tokenization
            ToTensor(padding_value=pad_tok_idx),  # Convert to tensor
            PadTransform(max_length=max_len, pad_value=pad_tok_idx),  # Padding
        ]

        # For now, use shared transforms (training vs eval distinction handled in root)
        dataloader_transforms.extend(shared_dataloader_transforms)

        # Model transforms (corruption, etc.) - run on GPU
        model_transforms = corruption_transforms or []

        super().__init__(
            *args,
            dataloader_transforms=dataloader_transforms,
            model_transforms=model_transforms,
            **kwargs,
        )

        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms

    def get_dataloader_transforms(self) -> list:
        """Return tokenization and padding transforms for dataloader."""
        return list(self._dataloader_transforms) if self._dataloader_transforms else []

    def get_model_transforms(self) -> list:
        """Return corruption and embedding transforms for GPU."""
        return list(self._model_transforms) if self._model_transforms else []

    def set_training_mode(self, training: bool):
        """Switch between training and evaluation transforms."""
        # For now, this is a placeholder
        # In full implementation, would rebuild transforms based on mode
        pass
