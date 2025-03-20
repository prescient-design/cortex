import numpy as np
import torch

from cortex.constants import COMPLEX_SEP_TOKEN
from cortex.corruption import MaskCorruptionProcess
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


def test_conv1d_root_with_per_element_corrupt_frac():
    """Test Conv1dRoot handles per-element corrupt_frac correctly."""
    batch_size = 4
    out_dim = 3
    embed_dim = 4
    max_seq_len = 13
    tokenizer = ProteinSequenceTokenizerFast()

    # Create a root node with corruption process
    corruption_process = MaskCorruptionProcess()
    root_node = Conv1dRoot(
        tokenizer_transform=HuggingFaceTokenizerTransform(tokenizer),
        max_len=max_seq_len,
        out_dim=out_dim,
        embed_dim=embed_dim,
        corruption_process=corruption_process,
    )

    # Create input sequences
    seq_array = np.array(
        [
            f"{COMPLEX_SEP_TOKEN} A V {COMPLEX_SEP_TOKEN} A V C C",
            f"{COMPLEX_SEP_TOKEN} A V {COMPLEX_SEP_TOKEN} A V C C",
            f"{COMPLEX_SEP_TOKEN} A V {COMPLEX_SEP_TOKEN} A V C C",
            f"{COMPLEX_SEP_TOKEN} A V {COMPLEX_SEP_TOKEN} A V C C",
        ]
    )

    # Test case 1: Scalar corrupt_frac
    scalar_corrupt_frac = 0.3
    root_output1 = root_node(seq_array, corrupt_frac=scalar_corrupt_frac)

    # Verify corrupt_frac is a tensor with batch dimension
    assert isinstance(root_output1.corrupt_frac, torch.Tensor)
    assert root_output1.corrupt_frac.shape[0] == batch_size
    assert torch.allclose(
        root_output1.corrupt_frac,
        torch.tensor([scalar_corrupt_frac] * batch_size, device=root_output1.corrupt_frac.device),
    )

    # Test case 2: Per-element corrupt_frac
    per_element_corrupt_frac = torch.tensor([0.1, 0.2, 0.3, 0.4])
    root_output2 = root_node(seq_array, corrupt_frac=per_element_corrupt_frac)

    # Verify corrupt_frac maintains per-element values
    assert isinstance(root_output2.corrupt_frac, torch.Tensor)
    assert root_output2.corrupt_frac.shape[0] == batch_size

    # Debug: Print the actual values
    print(f"Expected: {per_element_corrupt_frac}")
    print(f"Actual: {root_output2.corrupt_frac}")

    # Temporarily commenting out this assertion until we fix the issue
    assert torch.allclose(root_output2.corrupt_frac, per_element_corrupt_frac.to(root_output2.corrupt_frac.device))

    # Test case 3: None corrupt_frac (should sample from corruption process)
    root_output3 = root_node(seq_array, corrupt_frac=None)

    # Verify corrupt_frac is a tensor with batch dimension
    assert isinstance(root_output3.corrupt_frac, torch.Tensor)
    assert root_output3.corrupt_frac.shape[0] == batch_size
    # Values should be between 0 and 1
    assert torch.all(root_output3.corrupt_frac >= 0.0)
    assert torch.all(root_output3.corrupt_frac <= 1.0)
