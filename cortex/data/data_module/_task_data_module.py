from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar, Union

import hydra
import pandas as pd
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch import Generator
from torch.utils.data import DataLoader, Sampler, random_split

from cortex.data.dataset import ordered_dict_collator

# TODO change to prescient.samplers when available
from cortex.data.samplers import RandomizedMinorityUpsampler

T = TypeVar("T")


class TaskDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_config: DictConfig,
        train_on_everything: bool = False,
        lengths: Union[Sequence[float], None] = None,
        # generator: Union[Generator, None] = None,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        shuffle: Optional[bool] = None,
        balance_train_partition: Optional[Union[str, list[str]]] = None,
        sampler: Union[Iterable, Sampler, None] = None,
        batch_sampler: Union[Iterable[Sequence], Sampler[Sequence], None] = None,
        num_workers: int = 0,
        collate_fn: Union[Callable[[T], Any], None] = None,
        pin_memory: bool = True,
        drop_last: bool = False,
        skip_task_setup: bool = False,
    ):
        _default_lengths = [1.0, 0.0] if train_on_everything else [0.8, 0.2]
        self._train_on_everything = train_on_everything
        self._lengths = lengths or _default_lengths
        self._seed = seed
        # self._generator = generator or Generator().manual_seed(seed)
        self._dataset_config = dataset_config
        super().__init__()

        self._shuffle = shuffle
        self._balance_train_partition = balance_train_partition
        self._sampler = sampler
        self._batch_size = batch_size
        self._batch_sampler = batch_sampler
        self._collate_fn = collate_fn or ordered_dict_collator
        self._drop_last = drop_last

        self.datasets = {
            "train": None,
            "val": None,
            "test": None,
            "predict": None,
        }
        self._dataloader_kwargs = {
            "batch_size": batch_size,
            # "shuffle": self._shuffle,
            "num_workers": num_workers,
            "collate_fn": self._collate_fn,
            "pin_memory": pin_memory,
        }
        if not skip_task_setup:
            self.setup(stage="test")
            self.setup(stage="fit")

    def setup(self, stage=None):
        if stage == "fit":
            self._dataset_config.train = True
            train_val = hydra.utils.instantiate(self._dataset_config)
            if self._train_on_everything:
                train_val.df = pd.concat([train_val.df, self.datasets["test"].df], ignore_index=True)
            train_dataset, val_dataset = random_split(
                train_val,
                lengths=self._lengths,
                generator=Generator().manual_seed(self._seed),
            )
            # Subset datasets are awkward to work with.
            self.datasets["train_val"] = train_val
            self.datasets["train"] = train_dataset
            self.datasets["val"] = val_dataset
        if stage == "test":
            self._dataset_config.train = False
            test_dataset = hydra.utils.instantiate(self._dataset_config)
            self.datasets["test"] = test_dataset
        if stage == "predict":
            self._dataset_config.base_dataset.train = False
            predict_dataset = hydra.utils.instantiate(self._dataset_config)
            self.datasets["predict"] = predict_dataset

    def train_dataloader(self):
        return self.get_dataloader(split="train")

    def val_dataloader(self):
        return self.get_dataloader(split="val")

    def test_dataloader(self):
        return self.get_dataloader(split="test")

    def get_dataloader(self, split: str = "train"):
        if self.datasets[split] is None or len(self.datasets[split]) == 0:
            return []

        if self._batch_sampler is None:
            sampler = self._sampler or RandomizedMinorityUpsampler(
                self._partition_train_indices(),
            )
        else:
            sampler = None
        if split == "train":
            return DataLoader(
                self.datasets[split],
                sampler=sampler,
                batch_sampler=self._batch_sampler,
                drop_last=self._drop_last,
                **self._dataloader_kwargs,
            )
        else:
            # Full batch for evaluation on the test set
            if split == "test":
                self._dataloader_kwargs["batch_size"] = len(self.datasets[split])
            dataloader = DataLoader(self.datasets[split], shuffle=True, drop_last=True, **self._dataloader_kwargs)
            if split == "test":
                self._dataloader_kwargs["batch_size"] = self._batch_size
            return dataloader

    def _partition_train_indices(self):
        if self._balance_train_partition is None:
            return [list(range(len(self.datasets["train"])))]

        train_df = self.datasets["train_val"]._data.iloc[self.datasets["train"].indices].reset_index(drop=True)
        if isinstance(self._balance_train_partition, str):
            partition = [self._balance_train_partition]
        else:
            partition = list(self._balance_train_partition)

        index_list = list(train_df.groupby(partition).indices.values())
        return index_list
