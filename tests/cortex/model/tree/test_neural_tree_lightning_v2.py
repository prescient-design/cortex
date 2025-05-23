"""Tests for NeuralTreeLightningV2."""

import os
import tempfile

import torch
from omegaconf import DictConfig
from torch import nn

from cortex.model.branch import Conv1dBranch
from cortex.model.leaf import ClassifierLeaf, RegressorLeaf
from cortex.model.root import HuggingFaceRoot
from cortex.model.tree import NeuralTreeLightningV2
from cortex.model.trunk import SumTrunk


class TestNeuralTreeLightningV2:
    """Test suite for NeuralTreeLightningV2."""

    def test_basic_initialization(self):
        """Test basic initialization with modules."""
        # Create components
        root_nodes = nn.ModuleDict(
            {
                "bert": HuggingFaceRoot(
                    model_name_or_path="prajjwal1/bert-tiny",
                    pooling_strategy="none",  # Return full sequence for Conv1dBranch
                )
            }
        )

        trunk_node = SumTrunk(
            in_dims=[128],  # bert-tiny hidden size
            out_dim=64,
            project_features=True,
        )

        branch_nodes = nn.ModuleDict({"task_branch": Conv1dBranch(in_dim=64, out_dim=32, hidden_dims=[48])})

        leaf_nodes = nn.ModuleDict({"regressor": RegressorLeaf(branch_key="task_branch", in_dim=32, out_dim=1)})

        # Create model
        model = NeuralTreeLightningV2(
            root_nodes=root_nodes, trunk_node=trunk_node, branch_nodes=branch_nodes, leaf_nodes=leaf_nodes
        )

        # Check structure
        assert "bert" in model.root_nodes
        assert isinstance(model.root_nodes["bert"], HuggingFaceRoot)
        assert hasattr(model, "trunk_node")
        assert "task_branch" in model.branch_nodes
        assert "regressor" in model.leaf_nodes

    def test_forward_pass(self):
        """Test forward pass with HuggingFace inputs."""
        # Create components
        root_nodes = nn.ModuleDict(
            {
                "bert": HuggingFaceRoot(
                    model_name_or_path="prajjwal1/bert-tiny",
                    pooling_strategy="none",  # Return full sequence for Conv1dBranch
                )
            }
        )

        trunk_node = SumTrunk(in_dims=[128], out_dim=64, project_features=True)

        branch_nodes = nn.ModuleDict({"task_branch": Conv1dBranch(in_dim=64, out_dim=32, hidden_dims=[48])})

        leaf_nodes = nn.ModuleDict({"regressor": RegressorLeaf(branch_key="task_branch", in_dim=32, out_dim=1)})

        model = NeuralTreeLightningV2(
            root_nodes=root_nodes, trunk_node=trunk_node, branch_nodes=branch_nodes, leaf_nodes=leaf_nodes
        )

        # Create inputs
        batch_size = 2
        seq_len = 10
        root_inputs = {
            "bert": {
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
            }
        }

        # Forward pass
        outputs = model(root_inputs, leaf_keys=["regressor"])

        # Check outputs
        assert hasattr(outputs, "root_outputs")
        assert hasattr(outputs, "trunk_outputs")
        assert hasattr(outputs, "branch_outputs")
        assert hasattr(outputs, "leaf_outputs")
        assert "bert" in outputs.root_outputs
        assert "task_branch" in outputs.branch_outputs
        assert "regressor" in outputs.leaf_outputs

    def test_multi_task_setup(self):
        """Test multi-task configuration."""
        # Create shared components
        root_nodes = nn.ModuleDict(
            {
                "shared_encoder": HuggingFaceRoot(
                    model_name_or_path="prajjwal1/bert-tiny",
                    pooling_strategy="none",  # Return full sequence for Conv1dBranch
                )
            }
        )

        trunk_node = SumTrunk(in_dims=[128], out_dim=64, project_features=True)

        # Multiple branches for different tasks
        branch_nodes = nn.ModuleDict(
            {
                "regression_branch": Conv1dBranch(in_dim=64, out_dim=32, hidden_dims=[48]),
                "classification_branch": Conv1dBranch(in_dim=64, out_dim=32, hidden_dims=[48]),
            }
        )

        # Multiple leaf nodes
        leaf_nodes = nn.ModuleDict(
            {
                "value_prediction": RegressorLeaf(branch_key="regression_branch", in_dim=32, out_dim=1),
                "class_prediction": ClassifierLeaf(
                    branch_key="classification_branch",
                    in_dim=32,
                    num_classes=5,  # 5 classes
                ),
            }
        )

        model = NeuralTreeLightningV2(
            root_nodes=root_nodes, trunk_node=trunk_node, branch_nodes=branch_nodes, leaf_nodes=leaf_nodes
        )

        # Create inputs
        batch_size = 2
        seq_len = 10
        root_inputs = {
            "shared_encoder": {
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
            }
        }

        # Forward pass for both tasks
        outputs = model(root_inputs, leaf_keys=["value_prediction", "class_prediction"])

        # Check all outputs are present
        assert "value_prediction" in outputs.leaf_outputs
        assert "class_prediction" in outputs.leaf_outputs
        # RegressorLeaf outputs loc and scale
        assert outputs.leaf_outputs["value_prediction"].loc.shape == (batch_size, 1)
        assert outputs.leaf_outputs["value_prediction"].scale.shape == (batch_size, 1)
        # ClassifierLeaf outputs logits
        assert hasattr(outputs.leaf_outputs["class_prediction"], "logits")
        assert outputs.leaf_outputs["class_prediction"].logits.shape == (batch_size, 5)

    def test_lightning_save_load(self):
        """Test Lightning checkpoint save/load."""
        # Create simple model
        root_nodes = nn.ModuleDict(
            {
                "encoder": HuggingFaceRoot(
                    model_name_or_path="prajjwal1/bert-tiny",
                    pooling_strategy="none",  # Return full sequence for Conv1dBranch
                )
            }
        )

        trunk_node = SumTrunk(in_dims=[128], out_dim=64)

        branch_nodes = nn.ModuleDict({"branch": Conv1dBranch(in_dim=64, out_dim=32)})

        leaf_nodes = nn.ModuleDict({"leaf": RegressorLeaf(branch_key="branch", in_dim=32, out_dim=1)})

        model = NeuralTreeLightningV2(
            root_nodes=root_nodes, trunk_node=trunk_node, branch_nodes=branch_nodes, leaf_nodes=leaf_nodes
        )

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "model.ckpt")

            # Save using Lightning's method
            trainer = torch.ones(1)  # Dummy trainer
            model.trainer = trainer
            torch.save(model.state_dict(), checkpoint_path)

            # Load checkpoint
            new_model = NeuralTreeLightningV2(
                root_nodes=root_nodes, trunk_node=trunk_node, branch_nodes=branch_nodes, leaf_nodes=leaf_nodes
            )
            new_model.load_state_dict(torch.load(checkpoint_path))

            # Verify weights are the same
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), new_model.named_parameters()):
                assert n1 == n2
                assert torch.allclose(p1, p2)

    def test_optimizer_configuration(self):
        """Test optimizer configuration."""
        # Create model with optimizer config
        root_nodes = nn.ModuleDict(
            {
                "encoder": HuggingFaceRoot(
                    model_name_or_path="prajjwal1/bert-tiny",
                    pooling_strategy="none",  # Return full sequence for Conv1dBranch
                )
            }
        )

        trunk_node = SumTrunk(in_dims=[128], out_dim=64)
        branch_nodes = nn.ModuleDict({"branch": Conv1dBranch(in_dim=64, out_dim=32)})
        leaf_nodes = nn.ModuleDict()

        optimizer_config = DictConfig({"_target_": "torch.optim.Adam", "lr": 0.001, "weight_decay": 0.01})

        scheduler_config = DictConfig({"_target_": "torch.optim.lr_scheduler.CosineAnnealingLR", "T_max": 100})

        model = NeuralTreeLightningV2(
            root_nodes=root_nodes,
            trunk_node=trunk_node,
            branch_nodes=branch_nodes,
            leaf_nodes=leaf_nodes,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
        )

        # Configure optimizers
        optimizer_dict = model.configure_optimizers()

        # Check optimizer
        if isinstance(optimizer_dict, dict):
            optimizer = optimizer_dict["optimizer"]
            assert isinstance(optimizer, torch.optim.Adam)
            assert optimizer.param_groups[0]["lr"] == 0.001
            assert optimizer.param_groups[0]["weight_decay"] == 0.01

            # Check scheduler
            assert "lr_scheduler" in optimizer_dict
            scheduler_info = optimizer_dict["lr_scheduler"]
            assert isinstance(scheduler_info["scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR)
        else:
            # Just optimizer returned (no scheduler)
            assert isinstance(optimizer_dict, torch.optim.Adam)
