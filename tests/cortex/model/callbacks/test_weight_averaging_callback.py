"""
Tests for weight averaging callback.

Tests the modernized weight averaging implementation using Lightning callbacks.
"""

from unittest.mock import Mock

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from cortex.model.callbacks import ModelCheckpointWithAveraging, WeightAveragingCallback


class SimpleModule(nn.Module):
    """Simple module for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        self.frozen_param = nn.Parameter(torch.randn(5))
        self.frozen_param.requires_grad = False


@pytest.fixture
def weight_averaging_callback():
    """Create weight averaging callback for testing."""
    return WeightAveragingCallback(
        decay=0.9,
        start_step=2,
        update_frequency=1,
        apply_averaging_at_end=True,
    )


@pytest.fixture
def simple_module():
    """Create simple module for testing."""
    return SimpleModule()


def test_weight_averaging_callback_initialization():
    """Test callback initialization with different configurations."""
    # Test default initialization
    callback = WeightAveragingCallback()
    assert callback.decay == 0.999
    assert callback.start_step == 0
    assert callback.update_frequency == 1
    assert callback.apply_averaging_at_end is True

    # Test with legacy config
    config = DictConfig(
        {
            "decay": 0.95,
            "start_step": 10,
            "update_frequency": 5,
        }
    )
    callback = WeightAveragingCallback(averaging_config=config)
    assert callback.decay == 0.95
    assert callback.start_step == 10
    assert callback.update_frequency == 5


def test_on_train_start(weight_averaging_callback, simple_module):
    """Test initialization of averaged parameters."""
    trainer = Mock()

    # Initially no averaged parameters
    assert weight_averaging_callback.averaged_parameters is None

    # Call on_train_start
    weight_averaging_callback.on_train_start(trainer, simple_module)

    # Check averaged parameters are initialized
    assert weight_averaging_callback.averaged_parameters is not None
    assert "linear.weight" in weight_averaging_callback.averaged_parameters
    assert "linear.bias" in weight_averaging_callback.averaged_parameters
    # Frozen parameters should not be included
    assert "frozen_param" not in weight_averaging_callback.averaged_parameters

    # Check values are copied correctly
    torch.testing.assert_close(
        weight_averaging_callback.averaged_parameters["linear.weight"],
        simple_module.linear.weight.data,
    )


def test_update_averaged_parameters_before_start_step(weight_averaging_callback, simple_module):
    """Test that averaging doesn't happen before start_step."""
    trainer = Mock()

    # Initialize
    weight_averaging_callback.on_train_start(trainer, simple_module)
    original_weight = weight_averaging_callback.averaged_parameters["linear.weight"].clone()

    # Modify model parameters
    simple_module.linear.weight.data += 1.0

    # Call batch end before start_step
    weight_averaging_callback.step_count = 1  # Before start_step=2
    weight_averaging_callback.on_train_batch_end(trainer, simple_module, None, None, 0)

    # Averaged parameters should not change
    torch.testing.assert_close(
        weight_averaging_callback.averaged_parameters["linear.weight"],
        original_weight,
    )


def test_update_averaged_parameters_after_start_step(weight_averaging_callback, simple_module):
    """Test parameter averaging after start_step."""
    trainer = Mock()

    # Initialize
    weight_averaging_callback.on_train_start(trainer, simple_module)
    original_avg = weight_averaging_callback.averaged_parameters["linear.weight"].clone()

    # Modify model parameters
    delta = torch.ones_like(simple_module.linear.weight.data)
    simple_module.linear.weight.data += delta
    new_param = simple_module.linear.weight.data.clone()

    # Call batch end after start_step
    weight_averaging_callback.step_count = 2  # At start_step=2
    weight_averaging_callback.on_train_batch_end(trainer, simple_module, None, None, 0)

    # Check EMA update: averaged = 0.9 * original + 0.1 * new
    expected = 0.9 * original_avg + 0.1 * new_param
    torch.testing.assert_close(
        weight_averaging_callback.averaged_parameters["linear.weight"],
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_update_frequency(weight_averaging_callback, simple_module):
    """Test that updates respect update_frequency."""
    # Set update frequency to 3 and start step to 3
    weight_averaging_callback.update_frequency = 3
    weight_averaging_callback.start_step = 3
    trainer = Mock()

    # Initialize
    weight_averaging_callback.on_train_start(trainer, simple_module)
    original_avg = weight_averaging_callback.averaged_parameters["linear.weight"].clone()

    # Modify parameters
    simple_module.linear.weight.data += 1.0

    # First call at start_step but not divisible by frequency (step 3, 3%3==0, should update)
    weight_averaging_callback.step_count = 3
    weight_averaging_callback.on_train_batch_end(trainer, simple_module, None, None, 0)

    # Should update (step 3 >= 3 and 3 % 3 == 0)
    assert not torch.equal(
        weight_averaging_callback.averaged_parameters["linear.weight"],
        original_avg,
    )

    # Reset to test non-update case
    weight_averaging_callback.averaged_parameters["linear.weight"] = original_avg.clone()
    simple_module.linear.weight.data += 1.0

    # Call at step that doesn't match frequency (step 4, 4%3!=0, should not update)
    weight_averaging_callback.step_count = 4
    weight_averaging_callback.on_train_batch_end(trainer, simple_module, None, None, 0)

    # Should not update (step 5 % 3 != 0 after increment)
    torch.testing.assert_close(
        weight_averaging_callback.averaged_parameters["linear.weight"],
        original_avg,
    )


def test_apply_averaged_parameters_at_end(weight_averaging_callback, simple_module):
    """Test applying averaged parameters at training end."""
    trainer = Mock()

    # Initialize and modify parameters
    weight_averaging_callback.on_train_start(trainer, simple_module)
    original_param = simple_module.linear.weight.data.clone()

    # Simulate some averaging
    weight_averaging_callback.averaged_parameters["linear.weight"] = torch.zeros_like(original_param)

    # Apply at training end
    weight_averaging_callback.on_train_end(trainer, simple_module)

    # Check parameters are replaced
    torch.testing.assert_close(
        simple_module.linear.weight.data,
        weight_averaging_callback.averaged_parameters["linear.weight"],
    )


def test_get_averaged_model(weight_averaging_callback, simple_module):
    """Test getting averaged model copy."""
    trainer = Mock()

    # Initialize
    weight_averaging_callback.on_train_start(trainer, simple_module)

    # Modify averaged parameters
    averaged_weight = torch.zeros_like(simple_module.linear.weight.data)
    weight_averaging_callback.averaged_parameters["linear.weight"] = averaged_weight

    # Get averaged model
    averaged_model = weight_averaging_callback.get_averaged_model(simple_module)

    # Check it's a different object
    assert averaged_model is not simple_module

    # Check averaged parameters are applied
    torch.testing.assert_close(
        averaged_model.linear.weight.data,
        averaged_weight,
    )

    # Original model should be unchanged
    assert not torch.equal(
        simple_module.linear.weight.data,
        averaged_weight,
    )


def test_state_dict_and_load_state_dict(weight_averaging_callback, simple_module):
    """Test callback state saving and loading."""
    trainer = Mock()

    # Initialize and modify state
    weight_averaging_callback.on_train_start(trainer, simple_module)
    weight_averaging_callback.step_count = 42

    # Get state dict
    state_dict = weight_averaging_callback.state_dict()

    assert "averaged_parameters" in state_dict
    assert "step_count" in state_dict
    assert "decay" in state_dict
    assert state_dict["step_count"] == 42
    assert state_dict["decay"] == 0.9

    # Create new callback and load state
    new_callback = WeightAveragingCallback()
    new_callback.load_state_dict(state_dict)

    assert new_callback.step_count == 42
    assert new_callback.decay == 0.9
    assert new_callback.averaged_parameters is not None


def test_model_checkpoint_with_averaging():
    """Test enhanced checkpoint callback."""
    weight_callback = WeightAveragingCallback()
    checkpoint_callback = ModelCheckpointWithAveraging(
        weight_averaging_callback=weight_callback,
        save_averaged_checkpoint=True,
    )

    # Test initialization
    assert checkpoint_callback.weight_averaging_callback is weight_callback
    assert checkpoint_callback.save_averaged_checkpoint is True
    assert checkpoint_callback.averaged_checkpoint_suffix == "_averaged"


def test_model_checkpoint_save_averaged(simple_module):
    """Test saving averaged checkpoint."""
    weight_callback = WeightAveragingCallback()
    checkpoint_callback = ModelCheckpointWithAveraging(
        weight_averaging_callback=weight_callback,
        save_averaged_checkpoint=True,
    )

    # Mock trainer with checkpoint callback
    trainer = Mock()
    trainer.checkpoint_callback = Mock()
    trainer.checkpoint_callback.dirpath = "/tmp/checkpoints"
    trainer.save_checkpoint = Mock()

    # Initialize weight averaging
    weight_callback.on_train_start(trainer, simple_module)

    # Call on_train_end
    checkpoint_callback.on_train_end(trainer, simple_module)

    # Verify save_checkpoint was called
    trainer.save_checkpoint.assert_called_once()
    call_args = trainer.save_checkpoint.call_args[0]
    assert "/tmp/checkpoints/final_model_averaged.ckpt" in call_args[0]


def test_no_averaged_parameters_handling(simple_module):
    """Test behavior when no averaged parameters exist."""
    callback = WeightAveragingCallback()
    trainer = Mock()

    # Don't call on_train_start, so no averaged parameters
    assert callback.averaged_parameters is None

    # These should not crash
    callback.on_train_batch_end(trainer, simple_module, None, None, 0)
    callback.on_train_end(trainer, simple_module)

    # get_averaged_model should return a copy
    averaged_model = callback.get_averaged_model(simple_module)
    assert averaged_model is not simple_module


def test_disable_apply_averaging_at_end(simple_module):
    """Test disabling automatic application of averaged weights."""
    callback = WeightAveragingCallback(apply_averaging_at_end=False)
    trainer = Mock()

    # Initialize
    callback.on_train_start(trainer, simple_module)
    original_weight = simple_module.linear.weight.data.clone()

    # Modify averaged parameters
    callback.averaged_parameters["linear.weight"] = torch.zeros_like(original_weight)

    # Call on_train_end
    callback.on_train_end(trainer, simple_module)

    # Original parameters should be unchanged
    torch.testing.assert_close(
        simple_module.linear.weight.data,
        original_weight,
    )
