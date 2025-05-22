"""HuggingFace-compatible NeuralTree model implementation."""

import warnings
from typing import Any, Dict, Optional, Union

import hydra
import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel

from cortex.config import NeuralTreeConfig
from cortex.model.tree import NeuralTree, NeuralTreeOutput


class NeuralTreeModel(PreTrainedModel):
    """
    HuggingFace-compatible wrapper for NeuralTree architecture.

    This class preserves all existing cortex functionality while enabling:
    - HuggingFace ecosystem integration (save/load, Hub integration)
    - Mixed HF pretrained + custom root nodes
    - Standard configuration management
    - torch.compile compatibility (when properly configured)
    """

    config_class = NeuralTreeConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock", "ConvResidBlock"]

    def __init__(self, config: NeuralTreeConfig):
        super().__init__(config)
        self.config = config

        # Build root nodes (mixed HF + custom)
        self.root_nodes = nn.ModuleDict()
        for root_name, root_config in config.roots.items():
            if root_config.use_hf_model:
                # Load HuggingFace pretrained model
                hf_config = root_config.hf_config
                if isinstance(hf_config, dict):
                    from transformers import BertConfig

                    # For now, just use BertConfig as default for testing
                    # In practice, this would be determined by model_type
                    hf_config = BertConfig(**hf_config)

                self.root_nodes[root_name] = AutoModel.from_config(hf_config)
            else:
                # Use traditional cortex root node
                self.root_nodes[root_name] = hydra.utils.instantiate(root_config.cortex_config)

        # Build trunk node using existing Hydra instantiation
        if config.trunk:
            self.trunk_node = hydra.utils.instantiate(config.trunk)
        else:
            raise ValueError("trunk configuration is required")

        # Build branch nodes
        self.branch_nodes = nn.ModuleDict()
        for branch_name, branch_config in config.branches.items():
            self.branch_nodes[branch_name] = hydra.utils.instantiate(branch_config)

        # Build leaf nodes - these will be created by tasks later
        self.leaf_nodes = nn.ModuleDict()

        # Store task configurations for later instantiation
        self._task_configs = config.tasks

        # Initialize corruption handling for torch.compile compatibility
        self._corruption_layer = None
        if hasattr(self.config, "enable_torch_compile") and self.config.enable_torch_compile:
            self._init_compilation_friendly_corruption()

    def _init_compilation_friendly_corruption(self):
        """Initialize compilation-friendly corruption layer if needed."""
        # This will be implemented when we get to the torch.compile milestone
        # For now, we preserve existing corruption behavior
        pass

    def forward(
        self,
        root_inputs: Dict[str, Any],
        corruption_params: Optional[Dict[str, Any]] = None,
        trunk_outputs: Optional[Any] = None,
        branch_outputs: Optional[Dict[str, torch.Tensor]] = None,
        leaf_keys: Optional[list[str]] = None,
        return_dict: bool = True,
    ) -> Union[NeuralTreeOutput, tuple]:
        """
        Forward pass through the neural tree.

        Args:
            root_inputs: Dictionary mapping root names to input tensors/dicts
            corruption_params: Optional corruption parameters for guided generation
            trunk_outputs: Optional pre-computed trunk outputs
            branch_outputs: Optional pre-computed branch outputs
            leaf_keys: Optional subset of leaf nodes to compute
            return_dict: Whether to return NeuralTreeOutput or tuple

        Returns:
            NeuralTreeOutput containing all node outputs, or tuple if return_dict=False
        """
        # Process root inputs
        root_outputs = {}
        if root_inputs is not None:
            for root_name, root_input in root_inputs.items():
                if root_name not in self.root_nodes:
                    raise KeyError(f"Root key {root_name} not found in root nodes")

                root_node = self.root_nodes[root_name]

                # Handle both HF models and cortex models
                if hasattr(root_node, "config") and hasattr(root_node.config, "model_type"):
                    # This is likely a HF model
                    if isinstance(root_input, dict):
                        output = root_node(**root_input)
                        # Extract relevant features from HF model output
                        if hasattr(output, "last_hidden_state"):
                            # Standard transformer output
                            from cortex.model.root import RootNodeOutput

                            root_outputs[root_name] = RootNodeOutput(
                                root_features=output.last_hidden_state, padding_mask=root_input.get("attention_mask")
                            )
                        else:
                            # Use output directly
                            root_outputs[root_name] = output
                    else:
                        output = root_node(root_input)
                        root_outputs[root_name] = output
                else:
                    # Traditional cortex root node
                    if isinstance(root_input, dict):
                        root_outputs[root_name] = root_node(**root_input)
                    else:
                        root_outputs[root_name] = root_node(root_input)

            # Apply corruption if specified (for guided generation)
            if corruption_params is not None:
                root_outputs = self._apply_corruption(root_outputs, corruption_params)

            # Compute trunk outputs
            trunk_inputs = list(root_outputs.values())
            trunk_outputs = self.trunk_node(*trunk_inputs)

        # Compute branch outputs on demand
        if branch_outputs is None:
            branch_outputs = {}

        # Compute leaf outputs
        leaf_outputs = {}
        leaf_keys = leaf_keys or list(self.leaf_nodes.keys())

        for leaf_key in leaf_keys:
            if leaf_key not in self.leaf_nodes:
                warnings.warn(f"Leaf key {leaf_key} not found in leaf nodes, skipping")
                continue

            leaf_node = self.leaf_nodes[leaf_key]
            branch_key = leaf_node.branch_key

            if branch_key not in self.branch_nodes:
                raise KeyError(f"Branch key {branch_key} not found in branch nodes")

            # Compute branch output if not cached
            if branch_key not in branch_outputs:
                branch_outputs[branch_key] = self.branch_nodes[branch_key](trunk_outputs)

            leaf_outputs[leaf_key] = leaf_node(branch_outputs[branch_key])

        # Create output
        output = NeuralTreeOutput(
            root_outputs=root_outputs,
            trunk_outputs=trunk_outputs,
            branch_outputs=branch_outputs,
            leaf_outputs=leaf_outputs,
        )

        if return_dict:
            return output
        else:
            return (root_outputs, trunk_outputs, branch_outputs, leaf_outputs)

    def _apply_corruption(self, root_outputs: Dict[str, Any], corruption_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply corruption to root outputs for guided generation."""
        # For now, delegate to existing corruption processes in root nodes
        # This will be modernized in the torch.compile milestone
        corrupted_outputs = {}
        for root_name, root_output in root_outputs.items():
            if root_name in corruption_params:
                # If the root node has corruption capability, use it
                root_node = self.root_nodes[root_name]
                if hasattr(root_node, "corruption_process") and root_node.corruption_process is not None:
                    # Use existing corruption logic
                    corrupted_outputs[root_name] = root_node.corruption_process(
                        root_output, corruption_params[root_name]
                    )
                else:
                    corrupted_outputs[root_name] = root_output
            else:
                corrupted_outputs[root_name] = root_output
        return corrupted_outputs

    def guided_forward(
        self, sequences: torch.Tensor, corruption_params: Dict[str, Any], guidance_layer: str = "trunk", **kwargs
    ) -> NeuralTreeOutput:
        """
        Forward pass with guided generation support for LaMBO optimizer.

        This method provides a clean interface for the LaMBO optimizer
        to manipulate model internals during guided generation.
        """
        # This will be fully implemented in the LaMBO modernization milestone
        # For now, provide basic guided forward
        if guidance_layer == "trunk":
            # Process sequences through roots
            root_inputs = {"sequence": sequences}  # Simplified for now
            return self.forward(root_inputs, corruption_params=corruption_params, **kwargs)
        else:
            raise NotImplementedError(f"Guidance layer {guidance_layer} not yet implemented")

    def add_task(self, task_name: str, task_config: Dict[str, Any], leaf_configs: Dict[str, Dict[str, Any]]):
        """
        Add a task with its associated leaf nodes.

        This method allows dynamic task addition while preserving
        the existing cortex task management patterns.
        """
        # Store task config
        self._task_configs[task_name] = task_config

        # Instantiate leaf nodes for this task
        for leaf_name, leaf_config in leaf_configs.items():
            full_leaf_name = f"{task_name}_{leaf_name}"
            self.leaf_nodes[full_leaf_name] = hydra.utils.instantiate(leaf_config)

    def get_task_outputs(self, task_name: str, outputs: NeuralTreeOutput) -> Dict[str, Any]:
        """Extract outputs for a specific task from tree outputs."""
        return outputs.fetch_task_outputs(task_name)

    @classmethod
    def from_cortex_tree(cls, cortex_tree: NeuralTree, config: Optional[NeuralTreeConfig] = None) -> "NeuralTreeModel":
        """
        Create NeuralTreeModel from existing cortex SequenceModelTree.

        This enables migration from existing cortex models.
        """
        if config is None:
            # Create minimal config from existing tree
            config = NeuralTreeConfig()

        # Create new model
        model = cls(config)

        # Copy existing components
        model.root_nodes = cortex_tree.root_nodes
        model.trunk_node = cortex_tree.trunk_node
        model.branch_nodes = cortex_tree.branch_nodes
        model.leaf_nodes = cortex_tree.leaf_nodes

        return model

    def prepare_inputs_for_generation(self, **kwargs):
        """Prepare inputs for HuggingFace generation interface."""
        # This will be implemented when we add generation support
        return kwargs
