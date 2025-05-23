"""
Weight averaging callback for neural tree training.

Modernized weight averaging using Lightning callbacks instead of manual
implementation in the training step. Supports both exponential moving averages
and other weight averaging strategies.
"""

import copy
from typing import Any, Dict, Optional

from lightning import Callback, LightningModule, Trainer
from omegaconf import DictConfig


class WeightAveragingCallback(Callback):
    """
    Lightning callback for weight averaging during training.

    Implements exponential moving average (EMA) and other weight averaging
    strategies as a clean Lightning callback instead of manual implementation.
    """

    def __init__(
        self,
        averaging_config: Optional[DictConfig] = None,
        decay: float = 0.999,
        start_step: int = 0,
        update_frequency: int = 1,
        apply_averaging_at_end: bool = True,
    ):
        """
        Initialize weight averaging callback.

        Args:
            averaging_config: Weight averaging configuration (legacy compatibility)
            decay: EMA decay factor
            start_step: Step to start weight averaging
            update_frequency: How often to update averaged weights
            apply_averaging_at_end: Whether to replace model weights with averaged weights at training end
        """
        super().__init__()

        # Handle legacy configuration
        if averaging_config is not None:
            self.decay = averaging_config.get("decay", decay)
            self.start_step = averaging_config.get("start_step", start_step)
            self.update_frequency = averaging_config.get("update_frequency", update_frequency)
        else:
            self.decay = decay
            self.start_step = start_step
            self.update_frequency = update_frequency

        self.apply_averaging_at_end = apply_averaging_at_end

        # Internal state
        self.averaged_parameters = None
        self.step_count = 0

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize averaged parameters at training start."""
        # Create a copy of model parameters for averaging
        self.averaged_parameters = {}
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self.averaged_parameters[name] = param.data.clone()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update averaged parameters after each training batch."""
        # Check if we should update averaged weights
        if (
            self.step_count >= self.start_step
            and self.step_count % self.update_frequency == 0
            and self.averaged_parameters is not None
        ):
            self._update_averaged_parameters(pl_module)

        # Increment step count after the check
        self.step_count += 1

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Optionally apply averaged weights at training end."""
        if self.apply_averaging_at_end and self.averaged_parameters is not None:
            self._apply_averaged_parameters(pl_module)

    def _update_averaged_parameters(self, pl_module: LightningModule) -> None:
        """Update exponential moving average of parameters."""
        for name, param in pl_module.named_parameters():
            if param.requires_grad and name in self.averaged_parameters:
                # EMA update: averaged = decay * averaged + (1 - decay) * current
                self.averaged_parameters[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def _apply_averaged_parameters(self, pl_module: LightningModule) -> None:
        """Replace model parameters with averaged parameters."""
        for name, param in pl_module.named_parameters():
            if param.requires_grad and name in self.averaged_parameters:
                param.data.copy_(self.averaged_parameters[name])

    def get_averaged_model(self, pl_module: LightningModule) -> LightningModule:
        """
        Get a copy of the model with averaged parameters.

        Args:
            pl_module: The Lightning module to average

        Returns:
            Copy of the model with averaged parameters applied
        """
        if self.averaged_parameters is None:
            return copy.deepcopy(pl_module)

        # Create a deep copy of the model
        averaged_model = copy.deepcopy(pl_module)

        # Apply averaged parameters
        for name, param in averaged_model.named_parameters():
            if param.requires_grad and name in self.averaged_parameters:
                param.data.copy_(self.averaged_parameters[name])

        return averaged_model

    def state_dict(self) -> Dict[str, Any]:
        """Return callback state for checkpointing."""
        return {
            "averaged_parameters": self.averaged_parameters,
            "step_count": self.step_count,
            "decay": self.decay,
            "start_step": self.start_step,
            "update_frequency": self.update_frequency,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        self.averaged_parameters = state_dict.get("averaged_parameters")
        self.step_count = state_dict.get("step_count", 0)
        self.decay = state_dict.get("decay", self.decay)
        self.start_step = state_dict.get("start_step", self.start_step)
        self.update_frequency = state_dict.get("update_frequency", self.update_frequency)


class ModelCheckpointWithAveraging(Callback):
    """
    Enhanced model checkpoint callback that can save averaged weights.

    Extends Lightning's model checkpointing to optionally save weight-averaged
    models alongside regular checkpoints.
    """

    def __init__(
        self,
        weight_averaging_callback: Optional[WeightAveragingCallback] = None,
        save_averaged_checkpoint: bool = True,
        averaged_checkpoint_suffix: str = "_averaged",
    ):
        """
        Initialize enhanced checkpoint callback.

        Args:
            weight_averaging_callback: Weight averaging callback to use for averaged checkpoints
            save_averaged_checkpoint: Whether to save averaged model checkpoints
            averaged_checkpoint_suffix: Suffix for averaged checkpoint files
        """
        super().__init__()
        self.weight_averaging_callback = weight_averaging_callback
        self.save_averaged_checkpoint = save_averaged_checkpoint
        self.averaged_checkpoint_suffix = averaged_checkpoint_suffix

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save final checkpoint with averaged weights if enabled."""
        if (
            self.save_averaged_checkpoint
            and self.weight_averaging_callback is not None
            and self.weight_averaging_callback.averaged_parameters is not None
        ):
            # Save averaged checkpoint using callback's averaged weights
            if trainer.checkpoint_callback and hasattr(trainer.checkpoint_callback, "dirpath"):
                import os

                checkpoint_dir = trainer.checkpoint_callback.dirpath
                averaged_path = os.path.join(checkpoint_dir, f"final_model{self.averaged_checkpoint_suffix}.ckpt")

                trainer.save_checkpoint(averaged_path, weights_only=False)
                print(f"Saved averaged model checkpoint to {averaged_path}")
