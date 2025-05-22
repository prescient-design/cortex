"""
Lightning module v2 for neural tree architecture with HuggingFace integration.

This module modernizes the Lightning integration for v2/v3 infrastructure with:
- Callback-based weight averaging and model management
- Better integration with TransformerRootV2/V3 and HuggingFace models
- Cleaner separation between model architecture and training logic
- Improved multi-task training patterns
"""

import warnings
from typing import Any, Dict, Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch import nn

from cortex.model.tree import NeuralTree


class NeuralTreeLightningV2(NeuralTree, L.LightningModule):
    """
    Lightning module v2 for neural tree architecture.

    Modernized Lightning integration with:
    - Clean separation of model and training concerns
    - Callback-based weight averaging and checkpointing
    - Multi-task training with manual optimization
    - HuggingFace model integration support
    """

    def __init__(
        self,
        root_nodes: Optional[nn.ModuleDict] = None,
        trunk_node: Optional[nn.Module] = None,
        branch_nodes: Optional[nn.ModuleDict] = None,
        leaf_nodes: Optional[nn.ModuleDict] = None,
        fit_cfg: Optional[DictConfig] = None,
        optimizer_config: Optional[DictConfig] = None,
        scheduler_config: Optional[DictConfig] = None,
        **kwargs,
    ):
        """
        Initialize Lightning module v2.

        Args:
            root_nodes: Root nodes (v1/v2/v3 compatible)
            trunk_node: Trunk node for feature aggregation
            branch_nodes: Branch nodes for task-specific processing
            leaf_nodes: Leaf nodes for final task outputs
            fit_cfg: Training configuration (legacy compatibility)
            optimizer_config: Optimizer configuration
            scheduler_config: LR scheduler configuration
            **kwargs: Additional arguments
        """
        root_nodes = root_nodes or nn.ModuleDict()
        branch_nodes = branch_nodes or nn.ModuleDict()
        leaf_nodes = leaf_nodes or nn.ModuleDict()

        super().__init__(
            root_nodes=root_nodes,
            trunk_node=trunk_node,
            branch_nodes=branch_nodes,
            leaf_nodes=leaf_nodes,
        )

        # Store configuration
        self.fit_cfg = fit_cfg or DictConfig({})
        self.optimizer_config = optimizer_config or self.fit_cfg.get("optimizer", {})
        self.scheduler_config = scheduler_config or self.fit_cfg.get("lr_scheduler", {})

        # Multi-task training requires manual optimization
        self.automatic_optimization = False

        # Lightning 2.x step output accumulation
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # Task registry for batch formatting
        self.task_dict = {}

        # Save hyperparameters for Lightning callbacks
        self.save_hyperparameters(ignore=["root_nodes", "trunk_node", "branch_nodes", "leaf_nodes"])

    def build_tree(self, cfg: DictConfig, skip_task_setup: bool = False):
        """
        Build neural tree from configuration.

        This method maintains compatibility with existing training scripts
        while supporting both v1 and v2/v3 infrastructure.

        Args:
            cfg: Hydra configuration
            skip_task_setup: Whether to skip task setup
        """
        # Delegate to parent for tree construction
        task_dict = super().build_tree(cfg, skip_task_setup=skip_task_setup)
        self.task_dict = task_dict
        return task_dict

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns optimizer and scheduler configuration compatible with Lightning 2.x.
        """
        import hydra

        # Create optimizer
        if self.optimizer_config:
            optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        else:
            # Default to Adam
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Configure scheduler if provided
        if self.scheduler_config:
            scheduler = hydra.utils.instantiate(self.scheduler_config, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",  # Default metric to monitor
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, float]:
        """
        Training step with multi-task processing.

        Processes each task independently with manual optimization for
        better control over multi-task training dynamics.

        Args:
            batch: Multi-task batch dictionary
            batch_idx: Batch index

        Returns:
            Dictionary of training metrics
        """
        # Get leaf keys and shuffle for randomized task order
        leaf_keys = list(batch.keys())
        rng = np.random.default_rng()
        rng.shuffle(leaf_keys)

        optimizer = self.optimizers()

        # Enable training mode and gradients
        self.train()
        self.requires_grad_(True)

        # Linear probing mode (freeze backbone if configured)
        if self.fit_cfg.get("linear_probing", False):
            self._freeze_backbone()

        # Process each task
        step_metrics = {}
        batch_sizes = {}

        for leaf_key in leaf_keys:
            # Format task batch
            task_key = leaf_key.rsplit("_", 1)[0]  # Remove leaf suffix
            task = self.task_dict.get(task_key)

            if task is None:
                warnings.warn(f"Task {task_key} not found in task_dict, skipping", stacklevel=2)
                continue

            # Format batch for this task
            task_batch = task.format_batch(batch[leaf_key])
            root_inputs = task_batch["root_inputs"]
            leaf_targets = task_batch["leaf_targets"].get(task_key, {})

            # Forward pass through neural tree
            tree_outputs = self(root_inputs, leaf_keys=[leaf_key])
            leaf_node = self.leaf_nodes[leaf_key]

            # Get root outputs if needed (for MLM tasks)
            root_key = getattr(leaf_node, "root_key", None)
            root_outputs = tree_outputs.root_outputs.get(root_key) if root_key else None
            leaf_outputs = tree_outputs.leaf_outputs[leaf_key]

            # Compute loss and update weights
            optimizer.zero_grad()
            loss = leaf_node.loss(
                leaf_outputs=leaf_outputs,
                root_outputs=root_outputs,
                **leaf_targets,
            )
            self.manual_backward(loss)
            optimizer.step()

            # Record metrics
            step_metrics.setdefault(task_key, []).append(loss.item())
            batch_sizes.setdefault(task_key, []).append(batch[leaf_key]["batch_size"])

        # Aggregate metrics
        aggregated_metrics = {}
        for task_key, losses in step_metrics.items():
            aggregated_metrics[f"{task_key}/train_loss"] = np.mean(losses)
            aggregated_metrics[f"{task_key}/train_batch_size"] = np.mean(batch_sizes[task_key])

        # Store for epoch-end processing (Lightning 2.x)
        self.training_step_outputs.append(aggregated_metrics)

        return aggregated_metrics

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, float]:
        """
        Validation step with multi-task evaluation.

        Args:
            batch: Multi-task batch dictionary
            batch_idx: Batch index

        Returns:
            Dictionary of validation metrics
        """
        leaf_keys = list(batch.keys())

        step_metrics = {}
        batch_sizes = {}

        for leaf_key in leaf_keys:
            # Format task batch
            task_key = leaf_key.rsplit("_", 1)[0]
            task = self.task_dict.get(task_key)

            if task is None:
                continue

            task_batch = task.format_batch(batch[leaf_key])
            root_inputs = task_batch["root_inputs"]
            leaf_targets = task_batch["leaf_targets"].get(task_key, {})

            # Forward pass
            tree_outputs = self(root_inputs, leaf_keys=[leaf_key])
            leaf_node = self.leaf_nodes[leaf_key]

            root_key = getattr(leaf_node, "root_key", None)
            root_outputs = tree_outputs.root_outputs.get(root_key) if root_key else None
            leaf_outputs = tree_outputs.leaf_outputs[leaf_key]

            # Compute validation loss
            loss = leaf_node.loss(
                leaf_outputs=leaf_outputs,
                root_outputs=root_outputs,
                **leaf_targets,
            )

            # Record metrics
            step_metrics.setdefault(task_key, []).append(loss.item())
            batch_sizes.setdefault(task_key, []).append(batch[leaf_key]["batch_size"])

        # Aggregate metrics
        aggregated_metrics = {}
        for task_key, losses in step_metrics.items():
            aggregated_metrics[f"{task_key}/val_loss"] = np.mean(losses)
            aggregated_metrics[f"{task_key}/val_batch_size"] = np.mean(batch_sizes[task_key])

        # Store for epoch-end processing
        self.validation_step_outputs.append(aggregated_metrics)

        return aggregated_metrics

    def on_train_epoch_end(self) -> None:
        """Process accumulated training outputs at epoch end."""
        if not self.training_step_outputs:
            return

        # Aggregate metrics across all steps
        step_metrics = pd.DataFrame.from_records(self.training_step_outputs)
        epoch_metrics = step_metrics.mean().to_dict()

        # Log metrics by task
        self._log_task_metrics(epoch_metrics, prefix="train")

        # Clear accumulated outputs
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        """Process accumulated validation outputs at epoch end."""
        if not self.validation_step_outputs:
            return

        # Aggregate metrics across all steps
        step_metrics = pd.DataFrame.from_records(self.validation_step_outputs)
        epoch_metrics = step_metrics.mean().to_dict()

        # Log metrics by task
        self._log_task_metrics(epoch_metrics, prefix="val")

        # Clear accumulated outputs
        self.validation_step_outputs.clear()

    def _log_task_metrics(self, metrics: Dict[str, float], prefix: str) -> None:
        """
        Log metrics grouped by task.

        Args:
            metrics: Dictionary of metric name -> value
            prefix: Metric prefix (train/val)
        """
        # Group metrics by task
        task_groups = {}
        for metric_name, value in metrics.items():
            if f"/{prefix}_" in metric_name:
                task_key = metric_name.split("/")[0]
                if task_key not in task_groups:
                    task_groups[task_key] = {}

                # Get batch size for logging
                batch_size_key = f"{task_key}/{prefix}_batch_size"
                batch_size = metrics.get(batch_size_key, 1)
                task_groups[task_key]["batch_size"] = batch_size

                # Add metric (excluding batch size)
                if not metric_name.endswith("_batch_size"):
                    task_groups[task_key][metric_name] = value

        # Log each task's metrics
        for task_key, task_metrics in task_groups.items():
            batch_size = task_metrics.pop("batch_size", 1)
            self.log_dict(
                task_metrics,
                logger=True,
                prog_bar=True,
                batch_size=int(batch_size),
                sync_dist=True,
            )

    def _freeze_backbone(self) -> None:
        """Freeze root and trunk nodes for linear probing."""
        # Freeze all root nodes
        for root_node in self.root_nodes.values():
            for param in root_node.parameters():
                param.requires_grad = False

        # Freeze trunk node if present
        if self.trunk_node is not None:
            for param in self.trunk_node.parameters():
                param.requires_grad = False

    def get_dataloader(self, split: str):
        """
        Get dataloader for specified split.

        Maintains compatibility with existing training scripts.

        Args:
            split: Data split ("train", "val", "test")

        Returns:
            DataLoader for the specified split
        """
        # This method delegates to the task setup from build_tree
        # In practice, this is handled by the data module
        if hasattr(self, "task_dict") and self.task_dict:
            # Return combined dataloader from tasks
            from lightning.pytorch.utilities.combined_loader import CombinedLoader

            task_loaders = {}
            for task_key, task in self.task_dict.items():
                if hasattr(task, "get_dataloader"):
                    task_loaders[f"{task_key}_leaf"] = task.get_dataloader(split)

            if task_loaders:
                return CombinedLoader(task_loaders, mode="min_size")

        return None

    # Abstract method implementations for NeuralTree compatibility
    def _predict_batch(self, batch, leaf_keys=None):
        """Predict batch for inference (required by NeuralTree)."""
        if leaf_keys is None:
            leaf_keys = list(self.leaf_nodes.keys())

        # Format inputs if tasks are available
        if hasattr(self, "task_dict") and self.task_dict:
            for leaf_key in leaf_keys:
                task_key = leaf_key.rsplit("_", 1)[0]
                task = self.task_dict.get(task_key)
                if task:
                    batch = task.format_batch(batch)
                    break

        # Forward through neural tree
        return self(batch.get("root_inputs", batch), leaf_keys=leaf_keys)

    def evaluate(self, dataloader, leaf_keys=None):
        """Evaluate model on dataloader (required by NeuralTree)."""
        self.eval()
        with torch.no_grad():
            outputs = []
            for batch in dataloader:
                output = self._predict_batch(batch, leaf_keys=leaf_keys)
                outputs.append(output)
        return outputs

    def predict(self, dataloader, leaf_keys=None):
        """Predict on dataloader (required by NeuralTree)."""
        return self.evaluate(dataloader, leaf_keys=leaf_keys)

    def prediction_metrics(self, predictions, targets):
        """Compute prediction metrics (required by NeuralTree)."""
        # Basic implementation - can be extended by subclasses
        metrics = {}
        if hasattr(predictions, "leaf_outputs") and hasattr(targets, "leaf_targets"):
            for leaf_key in predictions.leaf_outputs:
                if leaf_key in targets.leaf_targets:
                    # Compute basic loss
                    leaf_node = self.leaf_nodes.get(leaf_key)
                    if leaf_node and hasattr(leaf_node, "loss"):
                        loss = leaf_node.loss(predictions.leaf_outputs[leaf_key], **targets.leaf_targets[leaf_key])
                        metrics[f"{leaf_key}_loss"] = loss.item()
        return metrics
