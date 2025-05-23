"""Tests for RegressionTask."""

import numpy as np
import pytest
import torch

from cortex.data.data_module import HFTaskDataModule, TaskDataModule
from cortex.model.leaf import RegressorLeaf
from cortex.task import RegressionTask


class TestRegressionTask:
    """Test suite for RegressionTask."""

    @pytest.fixture
    def mock_data_module(self):
        """Create a mock data module."""
        from unittest.mock import Mock

        data_module = Mock(spec=TaskDataModule)
        # Mock the dataloader methods to return empty iterators
        data_module.train_dataloader.return_value = iter([])
        data_module.val_dataloader.return_value = iter([])
        data_module.test_dataloader.return_value = iter([])

        return data_module

    @pytest.fixture
    def hf_data_module(self):
        """Create a mock HuggingFace data module."""
        from unittest.mock import Mock

        data_module = Mock(spec=HFTaskDataModule)
        # Mock the dataloader methods to return empty iterators
        data_module.train_dataloader.return_value = iter([])
        data_module.val_dataloader.return_value = iter([])
        data_module.test_dataloader.return_value = iter([])

        return data_module

    @pytest.fixture
    def hf_batch(self):
        """Create a mock HuggingFace tokenized batch."""
        return {
            "input_ids": torch.randint(0, 1000, (4, 128)),
            "attention_mask": torch.ones(4, 128),
            "token_type_ids": torch.zeros(4, 128, dtype=torch.long),
            "label": torch.tensor([1.5, 2.3, 0.8, 3.1]),
        }

    @pytest.fixture
    def legacy_batch(self):
        """Create a mock legacy column-based batch."""
        return {
            "col1": [1.0, 2.0, 3.0, 4.0],
            "col2": [0.5, 1.5, 2.5, 3.5],
            "target": [1.5, 2.3, 0.8, 3.1],
        }

    def test_initialization(self, mock_data_module):
        """Test basic initialization."""
        task = RegressionTask(
            data_module=mock_data_module,
            input_map={"features": ["col1", "col2"]},
            outcome_cols=["target"],
            leaf_key="prediction",
        )

        assert task.data_module == mock_data_module
        assert task.input_map == {"features": ["col1", "col2"]}
        assert task.outcome_cols == ["target"]
        assert task.leaf_key == "prediction"
        assert task.out_dim == 1
        assert task.root_key is None
        assert task.nominal_label_var == 0.25**2

    def test_initialization_with_root_key(self, mock_data_module):
        """Test initialization with root key."""
        task = RegressionTask(
            data_module=mock_data_module,
            input_map={"protein": []},
            outcome_cols=["fluorescence"],
            leaf_key="fluorescence_pred",
            root_key="protein",
            nominal_label_var=0.01,
        )

        assert task.root_key == "protein"
        assert task.nominal_label_var == 0.01

    def test_format_inputs_hf(self, hf_data_module, hf_batch):
        """Test format_inputs with HuggingFace tokenized inputs."""
        task = RegressionTask(
            data_module=hf_data_module,
            input_map={"protein": []},  # Empty for HF inputs
            outcome_cols=["label"],
            leaf_key="fluorescence",
            root_key="protein",
        )

        formatted = task.format_inputs(hf_batch, corrupt_frac=0.0)

        # Check structure
        assert "protein" in formatted
        assert "input_ids" in formatted["protein"]
        assert "attention_mask" in formatted["protein"]
        assert "token_type_ids" in formatted["protein"]

        # Check tensors
        assert torch.equal(formatted["protein"]["input_ids"], hf_batch["input_ids"])
        assert torch.equal(formatted["protein"]["attention_mask"], hf_batch["attention_mask"])

    def test_format_inputs_legacy(self, mock_data_module, legacy_batch):
        """Test format_inputs with legacy column-based inputs."""
        task = RegressionTask(
            data_module=mock_data_module,
            input_map={"features": ["col1", "col2"]},
            outcome_cols=["target"],
            leaf_key="prediction",
        )

        formatted = task.format_inputs(legacy_batch, corrupt_frac=0.0)

        # Check structure
        assert "features" in formatted
        assert "inputs" in formatted["features"]
        assert "corrupt_frac" in formatted["features"]

        # Check array shape
        inputs = formatted["features"]["inputs"]
        assert inputs.shape == (4, 2)  # 4 samples, 2 features
        assert inputs[0, 0] == 1.0
        assert inputs[0, 1] == 0.5

    def test_format_targets_hf_tensor(self, hf_data_module, hf_batch):
        """Test format_targets with HuggingFace tensor labels."""
        task = RegressionTask(
            data_module=hf_data_module,
            input_map={"protein": []},
            outcome_cols=["label"],
            leaf_key="fluorescence",
        )

        formatted = task.format_targets(hf_batch)

        # Check structure
        assert "fluorescence" in formatted
        assert "targets" in formatted["fluorescence"]

        # Check array
        targets = formatted["fluorescence"]["targets"]
        assert isinstance(targets, np.ndarray)
        assert targets.shape == (4, 1)
        np.testing.assert_array_equal(targets.flatten(), hf_batch["label"].numpy())

    def test_format_targets_hf_numpy(self, hf_data_module):
        """Test format_targets with numpy array labels."""
        batch = {"label": np.array([1.5, 2.3, 0.8, 3.1])}

        task = RegressionTask(
            data_module=hf_data_module,
            input_map={"protein": []},
            outcome_cols=["label"],
            leaf_key="fluorescence",
        )

        formatted = task.format_targets(batch)
        targets = formatted["fluorescence"]["targets"]
        assert targets.shape == (4, 1)
        np.testing.assert_array_equal(targets.flatten(), batch["label"])

    def test_format_targets_legacy(self, mock_data_module, legacy_batch):
        """Test format_targets with legacy column-based targets."""
        task = RegressionTask(
            data_module=mock_data_module,
            input_map={"features": ["col1", "col2"]},
            outcome_cols=["target"],
            leaf_key="prediction",
        )

        formatted = task.format_targets(legacy_batch)

        # Check structure
        assert "prediction" in formatted
        assert "targets" in formatted["prediction"]

        # Check array
        targets = formatted["prediction"]["targets"]
        assert targets.shape == (4, 1)
        assert targets[0, 0] == 1.5

    def test_format_targets_multiple_outcomes(self, mock_data_module):
        """Test format_targets with multiple outcome columns."""
        batch = {
            "target1": [1.0, 2.0, 3.0],
            "target2": [0.5, 1.5, 2.5],
        }

        task = RegressionTask(
            data_module=mock_data_module,
            input_map={"features": []},
            outcome_cols=["target1", "target2"],
            leaf_key="multi_pred",
        )

        formatted = task.format_targets(batch)
        targets = formatted["multi_pred"]["targets"]

        assert targets.shape == (3, 2)
        assert targets[0, 0] == 1.0
        assert targets[0, 1] == 0.5

    def test_format_batch_complete(self, hf_data_module, hf_batch):
        """Test complete format_batch with HuggingFace inputs."""
        task = RegressionTask(
            data_module=hf_data_module,
            input_map={"protein": []},
            outcome_cols=["label"],
            leaf_key="fluorescence",
            root_key="protein",
        )

        formatted = task.format_batch(hf_batch, corrupt_frac=0.0)

        # Check top-level structure
        assert "root_inputs" in formatted
        assert "leaf_targets" in formatted

        # Check root inputs
        assert "protein" in formatted["root_inputs"]
        assert "input_ids" in formatted["root_inputs"]["protein"]

        # Check leaf targets
        assert "fluorescence" in formatted["leaf_targets"]
        assert "targets" in formatted["leaf_targets"]["fluorescence"]

    def test_corruption_handling(self, hf_data_module, hf_batch):
        """Test corruption fraction handling."""
        task = RegressionTask(
            data_module=hf_data_module,
            input_map={"protein": []},
            outcome_cols=["label"],
            leaf_key="fluorescence",
            root_key="protein",
            corrupt_train_inputs=True,
        )

        # Test with corruption
        formatted = task.format_inputs(hf_batch, corrupt_frac=0.15)
        assert formatted["protein"].get("corrupt_frac") == 0.15

        # Test without corruption
        formatted = task.format_inputs(hf_batch, corrupt_frac=0.0)
        assert "corrupt_frac" not in formatted["protein"]

    def test_create_leaf(self, mock_data_module):
        """Test leaf node creation."""
        task = RegressionTask(
            data_module=mock_data_module,
            input_map={"features": ["col1", "col2"]},
            outcome_cols=["target1", "target2"],
            leaf_key="prediction",
            root_key="features",
            corrupt_train_inputs=True,
            nominal_label_var=0.1,
        )

        leaf = task.create_leaf(in_dim=64, branch_key="branch_0")

        assert isinstance(leaf, RegressorLeaf)
        assert leaf.in_dim == 64
        assert leaf.out_dim == 2  # Two outcome columns
        assert leaf.branch_key == "branch_0"
        assert leaf.root_key == "features"
        assert leaf.nominal_label_var == 0.1
        assert leaf.label_smoothing == "corrupt_frac"

    def test_create_leaf_no_corruption(self, mock_data_module):
        """Test leaf node creation without corruption."""
        task = RegressionTask(
            data_module=mock_data_module,
            input_map={"features": ["col1"]},
            outcome_cols=["target"],
            leaf_key="prediction",
            corrupt_train_inputs=False,
        )

        leaf = task.create_leaf(in_dim=32, branch_key="branch_0")

        assert leaf.label_smoothing == 0.0

    def test_mixed_input_map(self, mock_data_module):
        """Test that multiple root keys are supported in legacy mode."""
        batch = {
            "feat1": [1.0, 2.0],
            "feat2": [3.0, 4.0],
            "seq1": [5.0, 6.0],
            "seq2": [7.0, 8.0],
        }

        task = RegressionTask(
            data_module=mock_data_module,
            input_map={
                "features": ["feat1", "feat2"],
                "sequences": ["seq1", "seq2"],
            },
            outcome_cols=["target"],
            leaf_key="prediction",
        )

        formatted = task.format_inputs(batch)

        assert "features" in formatted
        assert "sequences" in formatted
        assert formatted["features"]["inputs"].shape == (2, 2)
        assert formatted["sequences"]["inputs"].shape == (2, 2)
