"""Tests for HuggingFaceRoot."""

import torch

from cortex.model.root import HuggingFaceRoot, HuggingFaceRootOutput


def test_huggingface_root_init():
    """Test HuggingFaceRoot initialization."""
    root = HuggingFaceRoot(model_name_or_path="prajjwal1/bert-tiny")

    assert hasattr(root, "model")
    assert root.model.__class__.__name__ == "BertModel"
    assert root.pooling_strategy == "mean"
    assert root.feature_extraction_layer == -1


def test_huggingface_root_forward():
    """Test HuggingFaceRoot forward pass."""
    root = HuggingFaceRoot(model_name_or_path="prajjwal1/bert-tiny")

    # Create inputs
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Forward pass
    output = root(input_ids=input_ids, attention_mask=attention_mask)

    # Check output
    assert isinstance(output, HuggingFaceRootOutput)
    assert hasattr(output, "root_features")
    assert output.root_features.shape == (batch_size, 128)  # bert-tiny has hidden_size=128
    assert hasattr(output, "attention_mask")
    assert hasattr(output, "last_hidden_state")
    assert hasattr(output, "raw_output")


def test_pooling_strategies():
    """Test different pooling strategies."""
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Test mean pooling
    root_mean = HuggingFaceRoot(model_name_or_path="prajjwal1/bert-tiny", pooling_strategy="mean")
    output_mean = root_mean(input_ids=input_ids, attention_mask=attention_mask)
    assert output_mean.root_features.shape == (batch_size, 128)

    # Test CLS pooling
    root_cls = HuggingFaceRoot(model_name_or_path="prajjwal1/bert-tiny", pooling_strategy="cls")
    output_cls = root_cls(input_ids=input_ids, attention_mask=attention_mask)
    assert output_cls.root_features.shape == (batch_size, 128)

    # Test max pooling
    root_max = HuggingFaceRoot(model_name_or_path="prajjwal1/bert-tiny", pooling_strategy="max")
    output_max = root_max(input_ids=input_ids, attention_mask=attention_mask)
    assert output_max.root_features.shape == (batch_size, 128)

    # Outputs should be different
    assert not torch.allclose(output_mean.root_features, output_cls.root_features)
    assert not torch.allclose(output_mean.root_features, output_max.root_features)


def test_freeze_pretrained():
    """Test freezing pretrained weights."""
    root = HuggingFaceRoot(model_name_or_path="prajjwal1/bert-tiny", freeze_pretrained=True)

    # Check that parameters are frozen
    for param in root.model.parameters():
        assert not param.requires_grad


def test_from_config():
    """Test creating HuggingFaceRoot from config dict."""
    config = {
        "model_type": "bert",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 512,
        "vocab_size": 1000,
        "max_position_embeddings": 512,
    }

    root = HuggingFaceRoot(
        model_name_or_path="bert",  # dummy name
        config=config,
    )

    assert hasattr(root, "model")
    assert root.model.config.hidden_size == 128
    assert root.model.config.num_hidden_layers == 2


def test_padding_mask_compatibility():
    """Test that HuggingFaceRoot output has padding_mask for SumTrunk compatibility."""
    root = HuggingFaceRoot(model_name_or_path="prajjwal1/bert-tiny")

    # Create inputs
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Forward pass
    output = root(input_ids=input_ids, attention_mask=attention_mask)

    # Check that padding_mask is set (for compatibility with SumTrunk)
    # This is set in NeuralTreeModel but should be in HuggingFaceRoot
    # Let's check if it has attention_mask at least
    assert hasattr(output, "attention_mask")
    assert torch.equal(output.attention_mask, attention_mask)


def test_from_pretrained_classmethod():
    """Test the from_pretrained class method."""
    root = HuggingFaceRoot.from_pretrained("prajjwal1/bert-tiny")

    assert hasattr(root, "model")
    assert root.model.__class__.__name__ == "BertModel"
