import numpy as np
import torch

from cortex.constants import COMPLEX_SEP_TOKEN
from cortex.model.root import Conv1dRoot, Conv1dRootOutput
from cortex.tokenization import ProteinSequenceTokenizerFast
from cortex.transforms import HuggingFaceTokenizerTransform


def test_seqcnn_root():
    batch_size = 2
    out_dim = 3
    embed_dim = 4
    num_blocks = 7
    kernel_size = 11
    max_seq_len = 13
    dropout_prob = 0.125
    layernorm = True
    pos_encoding = True
    tokenizer = ProteinSequenceTokenizerFast()

    root_node = Conv1dRoot(
        tokenizer_transform=HuggingFaceTokenizerTransform(tokenizer),
        max_len=max_seq_len,
        out_dim=out_dim,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        dropout_prob=dropout_prob,
        layernorm=layernorm,
        pos_encoding=pos_encoding,
    )

    # src_tok_idxs = torch.randint(0, vocab_size, (batch_size, max_seq_len))
    seq_array = np.array(
        [
            f"{COMPLEX_SEP_TOKEN} A V {COMPLEX_SEP_TOKEN} A V {COMPLEX_SEP_TOKEN} A V C C",
            f"{COMPLEX_SEP_TOKEN} A V {COMPLEX_SEP_TOKEN} A V {COMPLEX_SEP_TOKEN} A V C C",
        ]
    )
    root_output = root_node(seq_array)
    assert isinstance(root_output, Conv1dRootOutput)
    root_features = root_output.root_features
    padding_mask = root_output.padding_mask

    assert torch.is_tensor(root_features)
    assert torch.is_tensor(padding_mask)

    assert root_features.size() == torch.Size((batch_size, max_seq_len, out_dim))
    assert padding_mask.size() == torch.Size((batch_size, max_seq_len))
