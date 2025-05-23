"""
HuggingFace-compatible task data module with efficient tokenization.
"""

from typing import Optional

from datasets import Dataset, DatasetDict, IterableDataset
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class HFTaskDataModule(LightningDataModule):
    """
    Task data module for HuggingFace datasets with efficient tokenization.

    Tokenization is performed using dataset.map() which:
    - Processes data lazily without loading everything into memory
    - Caches results to disk for reuse
    - Supports multiprocessing for faster preprocessing
    """

    def __init__(
        self,
        dataset_config: DictConfig,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        text_field: str = "sequence",
        label_field: str = "label",
        add_spaces_between_chars: bool = True,
        tokenization_batch_size: int = 1000,
        tokenization_num_proc: Optional[int] = None,
        cache_dir: Optional[str] = None,
        skip_task_setup: bool = False,
    ):
        super().__init__()

        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.text_field = text_field
        self.label_field = label_field
        self.add_spaces_between_chars = add_spaces_between_chars
        self.tokenization_batch_size = tokenization_batch_size
        self.tokenization_num_proc = tokenization_num_proc
        self.cache_dir = cache_dir

        # Will be set by build_tree
        self._tokenizer_config = None

        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Keep track of whether tokenization has been applied
        self._tokenized = False

        if not skip_task_setup:
            self.setup()

    def setup(self, stage: Optional[str] = None):
        """Load and optionally tokenize HuggingFace dataset."""
        import hydra

        # Only load dataset if not already loaded
        if self.train_dataset is None and self.val_dataset is None and self.test_dataset is None:
            # Check if dataset_config is already instantiated by Hydra
            if isinstance(self.dataset_config, (DatasetDict, Dataset, IterableDataset)):
                # Already loaded by Hydra
                dataset = self.dataset_config
            elif hasattr(self.dataset_config, "_target_") and self.dataset_config._target_ == "lambda: small_dataset":
                # Handle test case
                dataset = eval(self.dataset_config._target_)()
            else:
                # Need to instantiate
                dataset = hydra.utils.instantiate(self.dataset_config)

            # Handle different dataset types
            if isinstance(dataset, DatasetDict):
                self.train_dataset = dataset.get("train")
                self.val_dataset = dataset.get("validation", dataset.get("val"))
                self.test_dataset = dataset.get("test", self.val_dataset)
            elif isinstance(dataset, Dataset):
                # Single dataset - need to split
                splits = dataset.train_test_split(test_size=0.2, seed=42)
                self.train_dataset = splits["train"]
                self.val_dataset = splits["test"]
                self.test_dataset = splits["test"]
            elif isinstance(dataset, IterableDataset):
                # Streaming dataset
                self.train_dataset = dataset
                self.val_dataset = dataset
                self.test_dataset = dataset

        # Apply tokenization if tokenizer config is available
        if self._tokenizer_config and not self._tokenized:
            self._tokenize_datasets()

    def _tokenize_datasets(self):
        """Apply tokenization to all datasets using HF dataset.map()."""
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(**self._tokenizer_config)

        # Get max length from config
        max_length = self._tokenizer_config.get("max_length", 512)

        def tokenize_function(examples):
            """Tokenize a batch of examples."""
            # Get sequences
            sequences = examples[self.text_field]

            # Add spaces between characters if needed (e.g., for protein sequences)
            if self.add_spaces_between_chars:
                sequences = [" ".join(seq) for seq in sequences]

            # Tokenize
            tokenized = tokenizer(
                sequences,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors=None,  # Return lists for dataset storage
            )

            # Preserve labels if they exist
            if self.label_field in examples:
                tokenized[self.label_field] = examples[self.label_field]

            return tokenized

        # Determine columns to remove (text that's been tokenized)
        remove_columns = [self.text_field]

        # Apply tokenization to each dataset
        if self.train_dataset and not isinstance(self.train_dataset, IterableDataset):
            self.train_dataset = self.train_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=self.tokenization_batch_size,
                num_proc=self.tokenization_num_proc,
                remove_columns=remove_columns,
                desc="Tokenizing train dataset",
                cache_file_name=f"{self.cache_dir}/train_tokenized.arrow" if self.cache_dir else None,
            )
            self.train_dataset.set_format("torch")

        if self.val_dataset and not isinstance(self.val_dataset, IterableDataset):
            self.val_dataset = self.val_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=self.tokenization_batch_size,
                num_proc=self.tokenization_num_proc,
                remove_columns=remove_columns,
                desc="Tokenizing validation dataset",
                cache_file_name=f"{self.cache_dir}/val_tokenized.arrow" if self.cache_dir else None,
            )
            self.val_dataset.set_format("torch")

        if self.test_dataset and not isinstance(self.test_dataset, IterableDataset):
            self.test_dataset = self.test_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=self.tokenization_batch_size,
                num_proc=self.tokenization_num_proc,
                remove_columns=remove_columns,
                desc="Tokenizing test dataset",
                cache_file_name=f"{self.cache_dir}/test_tokenized.arrow" if self.cache_dir else None,
            )
            self.test_dataset.set_format("torch")

        self._tokenized = True

    def train_dataloader(self):
        """Create training dataloader."""
        if self.train_dataset is None:
            return None

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True if not isinstance(self.train_dataset, IterableDataset) else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
