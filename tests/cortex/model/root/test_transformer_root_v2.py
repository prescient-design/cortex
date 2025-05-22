import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from cortex.model.root import TransformerRootV2, TransformerRootOutput


class MockTokenizerTransform:
    """Mock tokenizer transform for testing."""

    def __init__(self):
        self.tokenizer = Mock()
        self.tokenizer.vocab = {"[PAD]": 0, "[MASK]": 1, "A": 2, "B": 3, "C": 4}
        self.tokenizer.padding_idx = 0
        self.tokenizer.masking_idx = 1
        self.tokenizer.get_corruptible_mask = Mock(return_value=torch.ones(2, 5, dtype=torch.bool))


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    return MockTokenizerTransform()


@pytest.fixture
def transformer_root_v2(mock_tokenizer):
    """Create TransformerRootV2 for testing."""
    return TransformerRootV2(
        tokenizer_transform=mock_tokenizer,
        max_len=10,
        out_dim=64,
        embed_dim=32,
        num_blocks=1,
        num_heads=2,
    )


@pytest.fixture
def pre_tokenized_inputs():
    """Pre-tokenized inputs from CortexDataset."""
    return {
        "tgt_tok_idxs": torch.tensor([[2, 3, 4, 0, 0], [3, 2, 4, 3, 0]], dtype=torch.long),
        "padding_mask": torch.tensor([[True, True, True, False, False], [True, True, True, True, False]]),
    }


def test_transformer_root_v2_initialization(transformer_root_v2):
    """Test TransformerRootV2 initializes correctly."""
    assert transformer_root_v2.max_len == 10
    assert transformer_root_v2.out_dim == 64
    assert transformer_root_v2.embed_dim == 32
    assert transformer_root_v2.pad_tok_idx == 0
    assert transformer_root_v2.tok_encoder is not None
    assert transformer_root_v2.encoder is not None


def test_forward_with_pre_tokenized_inputs(transformer_root_v2, pre_tokenized_inputs):
    """Test forward pass with pre-tokenized inputs from CortexDataset."""

    output = transformer_root_v2(
        tgt_tok_idxs=pre_tokenized_inputs["tgt_tok_idxs"],
        padding_mask=pre_tokenized_inputs["padding_mask"],
        corrupt_frac=0.0,
    )

    assert isinstance(output, TransformerRootOutput)
    assert output.root_features.shape == (2, 5, 64)  # batch_size, seq_len, out_dim
    assert output.padding_mask.shape == (2, 5)
    assert output.tgt_tok_idxs is not None
    assert output.src_tok_idxs is not None


def test_forward_with_corruption(transformer_root_v2, pre_tokenized_inputs):
    """Test forward pass with corruption."""

    output = transformer_root_v2(
        tgt_tok_idxs=pre_tokenized_inputs["tgt_tok_idxs"],
        padding_mask=pre_tokenized_inputs["padding_mask"],
        corrupt_frac=0.5,
    )

    assert isinstance(output, TransformerRootOutput)
    assert output.corrupt_frac is not None
    assert torch.all(output.corrupt_frac == 0.5)


def test_backward_compatibility_warning(transformer_root_v2):
    """Test backward compatibility with seq_array inputs."""

    seq_array = np.array(["ABC", "BCA"])

    with pytest.warns(DeprecationWarning, match="Using deprecated seq_array"):
        with patch("cortex.model.root._transformer_root.TransformerRoot.forward") as mock_forward:
            mock_forward.return_value = TransformerRootOutput(
                root_features=torch.randn(2, 3, 64),
                padding_mask=torch.ones(2, 3, dtype=torch.bool),
            )

            output = transformer_root_v2(seq_array=seq_array)
            mock_forward.assert_called_once()


def test_init_seq_with_corruption_process(mock_tokenizer):
    """Test init_seq with corruption process."""

    # Mock corruption process
    mock_corruption = Mock()
    mock_corruption.sample_corrupt_frac.return_value = torch.tensor([0.3, 0.7])

    root = TransformerRootV2(
        tokenizer_transform=mock_tokenizer,
        max_len=10,
        corruption_process=mock_corruption,
    )

    tgt_tok_idxs = torch.tensor([[2, 3, 4], [3, 2, 4]], dtype=torch.long)

    # When corruption_process is set and corrupt_frac is None, it should sample
    _, _, corrupt_frac = root.init_seq(tgt_tok_idxs=tgt_tok_idxs, corrupt_frac=None)

    assert torch.allclose(corrupt_frac, torch.tensor([0.3, 0.7]))
    mock_corruption.sample_corrupt_frac.assert_called_once_with(n=2)


def test_truncation_for_long_sequences(transformer_root_v2):
    """Test sequence truncation for inputs longer than max_len."""

    # Create sequence longer than max_len (10)
    long_sequence = torch.tensor([[2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3]], dtype=torch.long)  # 14 tokens
    padding_mask = torch.ones_like(long_sequence, dtype=torch.bool)

    output = transformer_root_v2(
        tgt_tok_idxs=long_sequence,
        padding_mask=padding_mask,
    )

    # Should be truncated to max_len (10), keeping last token
    assert output.tgt_tok_idxs.size(-1) == 10
    assert output.tgt_tok_idxs[0, -1] == 3  # Last token should be preserved


def test_embedding_normalization(transformer_root_v2):
    """Test token embedding normalization."""

    src_tok_idxs = torch.tensor([[2, 3, 4]], dtype=torch.long)

    embeddings, _ = transformer_root_v2.embed_seq(
        src_tok_idxs=src_tok_idxs,
        normalize_embeds=True,
    )

    # Check that embeddings are normalized
    norms = embeddings.norm(dim=-1)
    expected_norm = np.sqrt(transformer_root_v2.embed_dim)
    assert torch.allclose(norms, torch.full_like(norms, expected_norm), atol=1e-6)


def test_device_handling(transformer_root_v2):
    """Test proper device handling for tensors."""

    # Test with CPU tensors
    tgt_tok_idxs = torch.tensor([[2, 3, 4]], dtype=torch.long)
    padding_mask = torch.tensor([[True, True, True]])

    output = transformer_root_v2(
        tgt_tok_idxs=tgt_tok_idxs,
        padding_mask=padding_mask,
    )

    # All outputs should be on the same device as the model
    model_device = transformer_root_v2.device
    assert output.root_features.device == model_device
    assert output.padding_mask.device == model_device
    if output.corrupt_frac is not None:
        assert output.corrupt_frac.device == model_device
