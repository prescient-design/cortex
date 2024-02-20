import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig
from torch import nn

from cortex.model.branch import BranchNodeOutput
from cortex.model.leaf import LeafNode, LeafNodeOutput
from cortex.model.root import RootNodeOutput
from cortex.model.trunk import TrunkNodeOutput


@dataclass
class NeuralTreeOutput:
    root_outputs: dict[str, RootNodeOutput]
    trunk_outputs: dict[str, TrunkNodeOutput]
    branch_outputs: dict[str, BranchNodeOutput]
    leaf_outputs: dict[str, LeafNodeOutput]


def fetch_task_outputs(tree_output: NeuralTreeOutput, task_key: str):
    outputs = []
    for leaf_key in tree_output.leaf_outputs:
        if leaf_key.startswith(task_key):
            outputs.append(tree_output.leaf_outputs[leaf_key])
    if len(outputs) == 0:
        raise ValueError(f"Task {task_key} not found in tree output")
    field_names = [f.name for f in fields(outputs[0])]
    outputs = {name: torch.stack([getattr(out, name) for out in outputs]) for name in field_names}
    return outputs


class NeuralTree(ABC, nn.Module):
    """
    Compute tree graph composed of root, trunk, branch, and leaf neural network nodes
    """

    def __init__(
        self,
        root_nodes: nn.ModuleDict,
        trunk_node: nn.Module,
        branch_nodes: nn.ModuleDict,
        leaf_nodes: nn.ModuleDict,
    ) -> None:
        super().__init__()
        self.root_nodes = root_nodes
        self.trunk_node = trunk_node
        self.branch_nodes = branch_nodes
        self.leaf_nodes = leaf_nodes

    @abstractmethod
    def build_tree(self, *args, **kwargs):
        pass

    def forward(
        self,
        root_inputs: dict,
        trunk_outputs: Optional[TrunkNodeOutput] = None,
        branch_outputs: Optional[dict[str, torch.Tensor]] = None,
        leaf_keys: Optional[list[str]] = None,
    ) -> NeuralTreeOutput:
        root_outputs = OrderedDict()
        if root_inputs is None:
            assert trunk_outputs is not None or branch_outputs is not None
        else:
            for r_key, r_input in root_inputs.items():
                if r_key not in self.root_nodes:
                    raise KeyError(f"root key {r_key} not found in root nodes")
                if isinstance(r_input, dict):
                    root_outputs[r_key] = self.root_nodes[r_key](**r_input)
                else:
                    root_outputs[r_key] = self.root_nodes[r_key](r_input)
            trunk_inputs = list(root_outputs.values())
            trunk_outputs = self.trunk_node(*trunk_inputs)

        if branch_outputs is None:
            branch_outputs = OrderedDict()

        leaf_outputs = OrderedDict()
        leaf_keys = self.leaf_nodes.keys() if leaf_keys is None else leaf_keys
        for l_key in leaf_keys:
            if l_key not in self.leaf_nodes:
                raise KeyError(f"leaf key {l_key} not found in leaf nodes")
            b_key = self.leaf_nodes[l_key].branch_key
            if b_key not in self.branch_nodes:
                raise KeyError(f"branch key {b_key} not found in branch nodes")
            if b_key not in branch_outputs:
                branch_outputs.setdefault(b_key, self.branch_nodes[b_key](trunk_outputs))
            leaf_outputs[l_key] = self.leaf_nodes[l_key](branch_outputs[b_key])

        outputs = NeuralTreeOutput(
            root_outputs=root_outputs,
            trunk_outputs=trunk_outputs,
            branch_outputs=branch_outputs,
            leaf_outputs=leaf_outputs,
        )

        return outputs

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def _predict_batch(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def prediction_metrics(self, *args, **kwargs):
        pass

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def freeze_roots(self) -> None:
        for r_node in self.root_nodes.values():
            r_node.requires_grad_(False)

    def freeze_trunk(self) -> None:
        self.trunk_node.requires_grad_(False)

    def freeze_branches(self) -> None:
        for b_node in self.branch_nodes.values():
            b_node.requires_grad_(False)

    def add_branch(
        self,
        branch_cfg: DictConfig,
        branch_key: str,
    ) -> None:
        if branch_key in self.branch_nodes:
            msg = f"Branch {branch_key} already exists, no new branch added."
            warnings.warn(msg, stacklevel=2)
        else:
            self.branch_nodes[branch_key] = hydra.utils.instantiate(branch_cfg)

    def add_leaf(
        self,
        leaf_node: LeafNode,
        leaf_key: str,
    ) -> None:
        if leaf_key in self.leaf_nodes:
            msg = f"Leaf {leaf_key} already exists, no new leaf added."
            warnings.warn(msg, stacklevel=2)
        else:
            self.leaf_nodes[leaf_key] = leaf_node

    def call_from_trunk_output(self, trunk_output, leaf_keys: Optional[list[str]] = None, **kwargs):
        return self(root_inputs=None, trunk_outputs=trunk_output, leaf_keys=leaf_keys, **kwargs)
