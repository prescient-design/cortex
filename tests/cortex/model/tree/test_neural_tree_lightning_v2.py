"""
Tests for NeuralTreeLightningV2 module.

Comprehensive testing of the modernized Lightning integration including:
- Multi-task training patterns
- Callback integration (weight averaging)
- HuggingFace model compatibility
- Lightning 2.x features
"""

from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from cortex.model.branch import TransformerBranch
from cortex.model.callbacks import WeightAveragingCallback
from cortex.model.leaf import ClassifierLeaf
from cortex.model.root import TransformerRootV2, TransformerRootV3
from cortex.model.tree import NeuralTreeLightningV2
from cortex.model.trunk import SumTrunk


@pytest.fixture
def mock_task():
    """Create a mock task for testing."""
    task = Mock()
    task.format_batch.return_value = {
        "root_inputs": {"transformer": {"tgt_tok_idxs": torch.randint(0, 100, (2, 10))}},
        "leaf_targets": {"test_task": {"targets": torch.randint(0, 2, (2,))}},
    }
    return task


@pytest.fixture
def simple_neural_tree_v2():
    """Create a simple neural tree for testing."""
    # Create mock tokenizer transform with nested tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.vocab = {f"token_{i}": i for i in range(100)}
    mock_tokenizer.padding_idx = 0

    mock_tokenizer_transform = Mock()
    mock_tokenizer_transform.tokenizer = mock_tokenizer

    # Create root node (v2 or v3)
    root_nodes = nn.ModuleDict(
        {
            "transformer": TransformerRootV2(
                vocab_size=100,
                d_model=32,
                num_layers=1,
                num_heads=2,
                max_len=64,
                tokenizer_transform=mock_tokenizer_transform,
            )
        }
    )

    # Create trunk node
    trunk_node = SumTrunk(in_dims=[64], out_dim=64)  # TransformerRootV2 default out_dim=64

    # Create branch node
    branch_nodes = nn.ModuleDict(
        {
            "transformer": TransformerBranch(
                in_dim=64,
                out_dim=32,
                num_blocks=1,
                num_heads=2,
            )
        }
    )

    # Create leaf node
    leaf_nodes = nn.ModuleDict(
        {
            "test_task_leaf": ClassifierLeaf(
                in_dim=32,
                num_classes=2,
                branch_key="transformer",
                num_layers=1,
            )
        }
    )

    # Create Lightning module
    module = NeuralTreeLightningV2(
        root_nodes=root_nodes,
        trunk_node=trunk_node,
        branch_nodes=branch_nodes,
        leaf_nodes=leaf_nodes,
        optimizer_config=DictConfig(
            {
                "_target_": "torch.optim.Adam",
                "lr": 1e-3,
            }
        ),
    )

    return module


@pytest.fixture
def neural_tree_with_v3_root():
    """Create neural tree with v3 root for torch.compile testing."""

    # Create mock tokenizer for v3 root
    mock_tokenizer = Mock()
    mock_tokenizer.vocab = {f"token_{i}": i for i in range(100)}
    mock_tokenizer.padding_idx = 0

    mock_tokenizer_transform = Mock()
    mock_tokenizer_transform.tokenizer = mock_tokenizer

    root_nodes = nn.ModuleDict(
        {
            "transformer": TransformerRootV3(
                vocab_size=100,
                d_model=32,
                num_layers=1,
                num_heads=2,
                max_len=64,
                tokenizer_transform=mock_tokenizer_transform,
                corruption_type="mask",
                corruption_kwargs={"vocab_size": 100, "mask_token_id": 0},
            )
        }
    )

    # Create trunk node
    trunk_node = SumTrunk(in_dims=[64], out_dim=64)  # TransformerRootV3 default out_dim=64

    # Create branch node
    branch_nodes = nn.ModuleDict(
        {
            "transformer": TransformerBranch(
                in_dim=64,
                out_dim=32,
                num_blocks=1,
                num_heads=2,
            )
        }
    )

    leaf_nodes = nn.ModuleDict(
        {
            "test_task_leaf": ClassifierLeaf(
                in_dim=32,
                num_classes=2,
                branch_key="transformer",
                num_layers=1,
            )
        }
    )

    module = NeuralTreeLightningV2(
        root_nodes=root_nodes,
        trunk_node=trunk_node,
        branch_nodes=branch_nodes,
        leaf_nodes=leaf_nodes,
    )

    return module


def test_neural_tree_lightning_v2_initialization():
    """Test basic initialization of NeuralTreeLightningV2."""
    module = NeuralTreeLightningV2()

    assert isinstance(module, NeuralTreeLightningV2)
    assert module.automatic_optimization is False
    assert hasattr(module, "training_step_outputs")
    assert hasattr(module, "validation_step_outputs")
    assert isinstance(module.task_dict, dict)


def test_configure_optimizers_with_config(simple_neural_tree_v2):
    """Test optimizer configuration with provided config."""
    config = simple_neural_tree_v2.configure_optimizers()

    assert isinstance(config, torch.optim.Adam)
    assert config.param_groups[0]["lr"] == 1e-3


def test_configure_optimizers_with_scheduler():
    """Test optimizer and scheduler configuration."""
    # Create module with some parameters
    leaf_nodes = nn.ModuleDict(
        {"test_leaf": ClassifierLeaf(in_dim=32, num_classes=2, branch_key="test_branch", num_layers=1)}
    )

    module = NeuralTreeLightningV2(
        leaf_nodes=leaf_nodes,
        optimizer_config=DictConfig(
            {
                "_target_": "torch.optim.Adam",
                "lr": 1e-3,
            }
        ),
        scheduler_config=DictConfig(
            {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 10,
                "gamma": 0.1,
            }
        ),
    )

    config = module.configure_optimizers()

    assert "optimizer" in config
    assert "lr_scheduler" in config
    assert isinstance(config["optimizer"], torch.optim.Adam)
    assert "scheduler" in config["lr_scheduler"]


def test_training_step_multi_task(simple_neural_tree_v2, mock_task):
    """Test multi-task training step."""
    # Setup task
    simple_neural_tree_v2.task_dict = {"test_task": mock_task}

    # Create multi-task batch
    batch = {
        "test_task_leaf": {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "batch_size": 2,
        }
    }

    # Mock optimizer
    with patch.object(simple_neural_tree_v2, "optimizers") as mock_opt:
        mock_optimizer = Mock()
        mock_opt.return_value = mock_optimizer

        # Mock manual_backward
        with patch.object(simple_neural_tree_v2, "manual_backward"):
            metrics = simple_neural_tree_v2.training_step(batch, 0)

    # Verify training step behavior
    assert isinstance(metrics, dict)
    assert "test_task/train_loss" in metrics
    assert "test_task/train_batch_size" in metrics
    assert len(simple_neural_tree_v2.training_step_outputs) == 1

    # Verify optimizer calls
    mock_optimizer.zero_grad.assert_called()
    mock_optimizer.step.assert_called()


def test_validation_step_multi_task(simple_neural_tree_v2, mock_task):
    """Test multi-task validation step."""
    # Setup task
    simple_neural_tree_v2.task_dict = {"test_task": mock_task}

    # Create batch
    batch = {
        "test_task_leaf": {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "batch_size": 2,
        }
    }

    metrics = simple_neural_tree_v2.validation_step(batch, 0)

    assert isinstance(metrics, dict)
    assert "test_task/val_loss" in metrics
    assert "test_task/val_batch_size" in metrics
    assert len(simple_neural_tree_v2.validation_step_outputs) == 1


def test_epoch_end_processing(simple_neural_tree_v2):
    """Test epoch end metric processing."""
    # Add mock training outputs
    simple_neural_tree_v2.training_step_outputs = [
        {"task1/train_loss": 0.5, "task1/train_batch_size": 2},
        {"task1/train_loss": 0.4, "task1/train_batch_size": 2},
    ]

    # Mock logging
    with patch.object(simple_neural_tree_v2, "log_dict") as mock_log:
        simple_neural_tree_v2.on_train_epoch_end()

    # Verify outputs are cleared
    assert len(simple_neural_tree_v2.training_step_outputs) == 0

    # Verify logging was called
    mock_log.assert_called()


def test_freeze_backbone_linear_probing(simple_neural_tree_v2):
    """Test backbone freezing for linear probing."""
    # Enable linear probing
    simple_neural_tree_v2.fit_cfg = DictConfig({"linear_probing": True})

    # Check initial gradient state
    root_param = next(simple_neural_tree_v2.root_nodes.parameters())
    assert root_param.requires_grad is True

    # Freeze backbone
    simple_neural_tree_v2._freeze_backbone()

    # Verify root parameters are frozen
    for param in simple_neural_tree_v2.root_nodes.parameters():
        assert param.requires_grad is False


def test_weight_averaging_callback_integration():
    """Test integration with weight averaging callback."""
    callback = WeightAveragingCallback(decay=0.999, start_step=0)

    # Create simple module
    module = NeuralTreeLightningV2()
    module.linear = nn.Linear(10, 1)  # Add a simple parameter for testing

    # Simulate training start
    trainer = Mock()
    callback.on_train_start(trainer, module)

    assert callback.averaged_parameters is not None
    assert "linear.weight" in callback.averaged_parameters
    assert "linear.bias" in callback.averaged_parameters

    # Simulate parameter update
    original_weight = module.linear.weight.data.clone()
    module.linear.weight.data += 0.1  # Simulate gradient update

    # Update averaged parameters
    callback.on_train_batch_end(trainer, module, None, None, 0)

    # Verify averaging occurred
    expected_avg = 0.999 * original_weight + 0.001 * module.linear.weight.data
    torch.testing.assert_close(
        callback.averaged_parameters["linear.weight"],
        expected_avg,
        rtol=1e-6,
        atol=1e-6,
    )


def test_v3_root_compatibility(neural_tree_with_v3_root, mock_task):
    """Test compatibility with TransformerRootV3 and torch.compile."""
    # Setup task
    neural_tree_with_v3_root.task_dict = {"test_task": mock_task}

    # Test that v3 root works in training
    batch = {
        "test_task_leaf": {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "batch_size": 2,
        }
    }

    with patch.object(neural_tree_with_v3_root, "optimizers") as mock_opt:
        mock_optimizer = Mock()
        mock_opt.return_value = mock_optimizer

        with patch.object(neural_tree_with_v3_root, "manual_backward"):
            metrics = neural_tree_with_v3_root.training_step(batch, 0)

    assert isinstance(metrics, dict)
    assert "test_task/train_loss" in metrics


def test_torch_compile_compatibility(neural_tree_with_v3_root):
    """Test torch.compile compatibility with v3 root."""
    # Create sample inputs for v3 root (use correct parameter name)
    input_ids = torch.randint(0, 100, (2, 10))

    # Test compilation (should not raise errors)
    try:
        compiled_forward = torch.compile(neural_tree_with_v3_root.root_nodes["transformer"])
        output = compiled_forward(tgt_tok_idxs=input_ids)
        assert output is not None
    except Exception as e:
        pytest.fail(f"torch.compile failed: {e}")


def test_build_tree_compatibility(simple_neural_tree_v2):
    """Test build_tree method for compatibility with existing training scripts."""
    # Mock configuration
    cfg = DictConfig(
        {
            "tasks": {},
            "data": {},
        }
    )

    # Mock the parent build_tree method
    with patch.object(simple_neural_tree_v2.__class__.__bases__[0], "build_tree") as mock_build:
        mock_build.return_value = {"test_task": Mock()}

        result = simple_neural_tree_v2.build_tree(cfg, skip_task_setup=False)

    assert result is not None
    assert simple_neural_tree_v2.task_dict == result
    mock_build.assert_called_once_with(cfg, skip_task_setup=False)


def test_get_dataloader_compatibility(simple_neural_tree_v2):
    """Test get_dataloader method for compatibility."""
    # Test without task_dict
    dataloader = simple_neural_tree_v2.get_dataloader("train")
    assert dataloader is None

    # Test with mock tasks
    mock_task = Mock()
    mock_task.get_dataloader.return_value = Mock()
    simple_neural_tree_v2.task_dict = {"test_task": mock_task}

    dataloader = simple_neural_tree_v2.get_dataloader("train")
    assert dataloader is not None


def test_missing_task_warning(simple_neural_tree_v2):
    """Test warning when task is missing from task_dict."""
    # Create batch with unknown task
    batch = {
        "unknown_task_leaf": {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "batch_size": 2,
        }
    }

    with patch.object(simple_neural_tree_v2, "optimizers") as mock_opt:
        mock_optimizer = Mock()
        mock_opt.return_value = mock_optimizer

        with pytest.warns(UserWarning, match="Task unknown_task not found"):
            metrics = simple_neural_tree_v2.training_step(batch, 0)

    # Should return empty metrics for unknown tasks
    assert len(metrics) == 0


def test_hyperparameter_saving(simple_neural_tree_v2):
    """Test that hyperparameters are saved correctly."""
    # Check that hyperparameters are saved (excluding module dicts)
    assert hasattr(simple_neural_tree_v2, "hparams")

    # Should not contain module dicts in hparams
    for exclude_key in ["root_nodes", "trunk_node", "branch_nodes", "leaf_nodes"]:
        assert exclude_key not in simple_neural_tree_v2.hparams
