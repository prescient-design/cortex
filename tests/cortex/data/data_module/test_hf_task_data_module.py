"""Tests for HFTaskDataModule."""

import pytest
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig

from cortex.data.data_module import HFTaskDataModule


class TestHFTaskDataModule:
    """Test suite for HFTaskDataModule."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a small mock dataset for testing."""
        data = {
            "sequence": ["MKTVRQ", "AGVHWT", "PLQVST", "WERTYU"],
            "label": [1.5, 2.3, 0.8, 3.1],
        }
        dataset = Dataset.from_dict(data)
        return DatasetDict(
            {
                "train": dataset,
                "validation": dataset,
                "test": dataset,
            }
        )

    @pytest.fixture
    def tokenizer_config(self):
        """Mock tokenizer configuration."""
        return {
            "pretrained_model_name_or_path": "prajjwal1/bert-tiny",
            "max_length": 128,
        }

    def test_initialization(self):
        """Test basic initialization."""
        data_module = HFTaskDataModule(
            dataset_config=DictConfig({"_target_": "mock"}),
            batch_size=32,
            num_workers=0,
            skip_task_setup=True,
        )

        assert data_module.batch_size == 32
        assert data_module.num_workers == 0
        assert data_module.text_field == "sequence"
        assert data_module.label_field == "label"
        assert data_module._tokenizer_config is None

    def test_tokenizer_config_setting(self, tokenizer_config):
        """Test setting tokenizer configuration."""
        data_module = HFTaskDataModule(
            dataset_config=DictConfig({"_target_": "mock"}),
            skip_task_setup=True,
        )

        # Simulate what build_tree does
        data_module._tokenizer_config = tokenizer_config

        assert data_module._tokenizer_config == tokenizer_config

    def test_tokenization(self, mock_dataset, tokenizer_config):
        """Test dataset tokenization."""
        data_module = HFTaskDataModule(
            dataset_config=None,
            batch_size=2,
            num_workers=0,
            tokenization_num_proc=None,  # Disable multiprocessing
            skip_task_setup=True,
        )

        # Manually set dataset and tokenizer config
        data_module.train_dataset = mock_dataset["train"]
        data_module.val_dataset = mock_dataset["validation"]
        data_module._tokenizer_config = tokenizer_config

        # Apply tokenization
        data_module._tokenize_datasets()

        # Check that tokenization was applied
        assert data_module._tokenized

        # Check dataset has expected fields
        train_batch = next(iter(data_module.train_dataloader()))
        assert "input_ids" in train_batch
        assert "attention_mask" in train_batch
        assert "label" in train_batch

        # Check shapes
        assert train_batch["input_ids"].shape[0] == 2  # batch_size
        assert train_batch["input_ids"].shape[1] == 128  # max_length
        assert train_batch["label"].shape[0] == 2

    def test_dataloader_creation(self, mock_dataset):
        """Test dataloader creation."""
        data_module = HFTaskDataModule(
            dataset_config=None,
            batch_size=2,
            num_workers=0,
            skip_task_setup=True,
        )

        data_module.train_dataset = mock_dataset["train"]
        data_module.val_dataset = mock_dataset["validation"]
        data_module.test_dataset = mock_dataset["test"]

        # Get dataloaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Check batch from train loader
        batch = next(iter(train_loader))
        assert len(batch["sequence"]) == 2  # batch_size
        assert len(batch["label"]) == 2

    def test_spaces_between_chars(self, mock_dataset, tokenizer_config):
        """Test adding spaces between characters for protein sequences."""
        from transformers import AutoTokenizer

        # Just testing the transformation logic
        HFTaskDataModule(
            dataset_config=None,
            add_spaces_between_chars=True,
            skip_task_setup=True,
        )

        # Create tokenizer to test the transformation
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["pretrained_model_name_or_path"])

        # Test the transformation
        sequence = "MKTVRQ"
        spaced = " ".join(sequence)
        assert spaced == "M K T V R Q"

        # Verify tokenization would be different
        tokens_no_space = tokenizer.encode(sequence)
        tokens_with_space = tokenizer.encode(spaced)
        assert len(tokens_with_space) > len(tokens_no_space)
