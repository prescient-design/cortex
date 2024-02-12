import numpy as np
import torch
from torch import nn

from cortex.constants import COMPLEX_SEP_TOKEN
from cortex.model.branch import Conv1dBranch
from cortex.model.leaf import ClassifierLeaf, RegressorLeaf
from cortex.model.root import Conv1dRoot
from cortex.model.tree import NeuralTreeOutput, SequenceModelTree
from cortex.model.trunk import SumTrunk
from cortex.tokenization import ProteinSequenceTokenizerFast
from cortex.transforms import HuggingFaceTokenizerTransform


def test_seq_model_tree():
    batch_size = 2
    num_roots = 3
    num_branches = 3
    embed_dim = 4
    out_dim = 5
    num_blocks = 7
    kernel_size = 11
    max_seq_len = 13
    num_classes = 23
    dropout_prob = 0.125
    layernorm = True
    pos_encoding = True
    tokenizer = ProteinSequenceTokenizerFast()

    root_nodes = [
        Conv1dRoot(
            tokenizer_transform=HuggingFaceTokenizerTransform(tokenizer),
            max_len=max_seq_len,
            out_dim=embed_dim,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout_prob=dropout_prob,
            layernorm=layernorm,
            pos_encoding=pos_encoding,
        )
        for _ in range(num_roots)
    ]
    root_nodes = nn.ModuleDict(
        {
            "root_0": root_nodes[0],
            "root_1": root_nodes[1],
            "root_2": root_nodes[2],
        }
    )

    trunk_node = SumTrunk(in_dims=[embed_dim] * num_roots, out_dim=embed_dim)

    branch_nodes = [
        Conv1dBranch(
            in_dim=embed_dim,
            out_dim=embed_dim,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout_prob=dropout_prob,
            layernorm=layernorm,
        )
        for _ in range(num_branches)
    ]
    branch_nodes = nn.ModuleDict(
        {
            "branch_0": branch_nodes[0],
            "branch_1": branch_nodes[1],
            "branch_2": branch_nodes[2],
        }
    )

    leaf_nodes = {}
    leaf_count = 0
    for b_key in branch_nodes:
        leaf_nodes[f"leaf_{leaf_count}"] = ClassifierLeaf(embed_dim, num_classes, b_key)
        leaf_count += 1
        leaf_nodes[f"leaf_{leaf_count}"] = RegressorLeaf(
            in_dim=embed_dim,
            out_dim=out_dim,
            branch_key=b_key,
        )
        leaf_count += 1
    leaf_nodes = nn.ModuleDict(leaf_nodes)

    tree = SequenceModelTree(root_nodes, trunk_node, branch_nodes, leaf_nodes)

    root_inputs = {}
    for r_key in root_nodes:
        root_inputs[r_key] = np.array(
            [
                f"{COMPLEX_SEP_TOKEN} A V {COMPLEX_SEP_TOKEN} A V {COMPLEX_SEP_TOKEN} A V C C",
                f"{COMPLEX_SEP_TOKEN} A V {COMPLEX_SEP_TOKEN} A V {COMPLEX_SEP_TOKEN} A V C C",
            ]
        )

    tree_outputs = tree(root_inputs)

    assert isinstance(tree_outputs, NeuralTreeOutput)

    for l_key, l_node in tree.leaf_nodes.items():
        l_out = tree_outputs.leaf_outputs[l_key]
        if isinstance(l_node, ClassifierLeaf):
            assert l_out.logits.size() == torch.Size((batch_size, num_classes))
        elif isinstance(l_node, RegressorLeaf):
            assert l_out.loc.size() == torch.Size((batch_size, out_dim))
            assert l_out.scale.size() == torch.Size((batch_size, out_dim))
            assert torch.all(l_out.scale > 0)
        else:
            pass
