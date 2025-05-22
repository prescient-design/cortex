"""Tests for NeuralTreeConfig and HuggingFace integration."""

import tempfile

import pytest

from cortex.config import NeuralTreeConfig, RootConfig


@pytest.fixture
def sample_cortex_config():
    """Sample cortex root configuration."""
    return {"_target_": "cortex.model.root.TransformerRoot", "max_len": 512, "out_dim": 64}


@pytest.fixture
def sample_hf_config():
    """Sample HuggingFace configuration."""
    return {"model_type": "bert", "hidden_size": 768}


@pytest.fixture
def sample_hydra_config():
    """Sample Hydra-style configuration."""
    return {
        "roots": {"protein_seq": {"_target_": "cortex.model.root.TransformerRoot", "max_len": 512, "out_dim": 64}},
        "trunk": {"_target_": "cortex.model.trunk.SumTrunk", "out_dim": 64},
        "branches": {"property_branch": {"_target_": "cortex.model.branch.Conv1dBranch", "out_dim": 32}},
        "tasks": {"fluorescence": {"_target_": "cortex.task.RegressionTask", "target_col": "log_fluorescence"}},
        "ensemble_size": 2,
        "channel_dim": 128,
        "dropout_prob": 0.1,
    }


def test_hf_root_config_validation():
    """Test that HF root config requires hf_config."""
    with pytest.raises(ValueError, match="hf_config must be provided"):
        RootConfig(use_hf_model=True, hf_config=None)


def test_cortex_root_config_validation():
    """Test that cortex root config requires cortex_config."""
    with pytest.raises(ValueError, match="cortex_config must be provided"):
        RootConfig(use_hf_model=False, cortex_config=None)


def test_valid_hf_root_config(sample_hf_config):
    """Test valid HF root config creation."""
    config = RootConfig(use_hf_model=True, hf_config=sample_hf_config, processor_name="bert-base-uncased")
    assert config.use_hf_model is True
    assert config.hf_config["model_type"] == "bert"
    assert config.processor_name == "bert-base-uncased"


def test_valid_cortex_root_config(sample_cortex_config):
    """Test valid cortex root config creation."""
    config = RootConfig(use_hf_model=False, cortex_config=sample_cortex_config)
    assert config.use_hf_model is False
    assert config.cortex_config["_target_"] == "cortex.model.root.TransformerRoot"


def test_default_neural_tree_config_creation():
    """Test creating default config."""
    config = NeuralTreeConfig()
    assert config.model_type == "neural_tree"
    assert isinstance(config.roots, dict)
    assert len(config.roots) == 0
    assert config.ensemble_size == 1
    assert config.channel_dim == 64
    assert config.dropout_prob == 0.0


def test_add_hf_root():
    """Test adding HuggingFace root."""
    config = NeuralTreeConfig()
    config.add_hf_root("bert_root", "bert-base-uncased", "bert-base-uncased")

    assert "bert_root" in config.roots
    assert config.roots["bert_root"].use_hf_model is True
    assert config.roots["bert_root"].processor_name == "bert-base-uncased"
    assert "bert_root" in config.processors
    assert config.processors["bert_root"] == "bert-base-uncased"


def test_add_cortex_root(sample_cortex_config):
    """Test adding cortex root."""
    config = NeuralTreeConfig()
    config.add_cortex_root("custom_root", sample_cortex_config, "custom-processor")

    assert "custom_root" in config.roots
    assert config.roots["custom_root"].use_hf_model is False
    assert config.roots["custom_root"].cortex_config == sample_cortex_config
    assert "custom_root" in config.processors
    assert config.processors["custom_root"] == "custom-processor"


def test_dict_to_root_config_conversion():
    """Test automatic conversion of dict to RootConfig."""
    config = NeuralTreeConfig(
        roots={"test_root": {"use_hf_model": False, "cortex_config": {"_target_": "test.Target"}}}
    )

    assert isinstance(config.roots["test_root"], RootConfig)
    assert config.roots["test_root"].use_hf_model is False


def test_config_serialization_round_trip():
    """Test saving and loading config."""
    # Create test config
    config = NeuralTreeConfig()
    config.add_cortex_root(
        "custom_root", {"_target_": "cortex.model.root.TransformerRoot", "max_len": 512, "out_dim": 64}
    )
    config.trunk = {"_target_": "cortex.model.trunk.SumTrunk", "out_dim": 64}
    config.ensemble_size = 3
    config.channel_dim = 128

    # Save to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        config.save_pretrained(temp_dir)

        # Load back
        loaded_config = NeuralTreeConfig.from_pretrained(temp_dir)

        # Verify equality
        assert loaded_config.model_type == config.model_type
        assert loaded_config.ensemble_size == config.ensemble_size
        assert loaded_config.channel_dim == config.channel_dim
        assert len(loaded_config.roots) == len(config.roots)
        assert "custom_root" in loaded_config.roots
        assert loaded_config.roots["custom_root"].use_hf_model is False


def test_from_hydra_config(sample_hydra_config):
    """Test creating NeuralTreeConfig from Hydra config."""
    config = NeuralTreeConfig.from_hydra_config(sample_hydra_config)

    assert len(config.roots) == 1
    assert "protein_seq" in config.roots
    assert config.roots["protein_seq"].use_hf_model is False
    assert config.ensemble_size == 2
    assert config.channel_dim == 128
    assert config.dropout_prob == 0.1
    assert config.trunk["_target_"] == "cortex.model.trunk.SumTrunk"
    assert "property_branch" in config.branches
    assert "fluorescence" in config.tasks


def test_to_hydra_config():
    """Test converting NeuralTreeConfig back to Hydra format."""
    config = NeuralTreeConfig()
    config.add_cortex_root(
        "protein_seq", {"_target_": "cortex.model.root.TransformerRoot", "max_len": 512, "out_dim": 64}
    )
    config.trunk = {"_target_": "cortex.model.trunk.SumTrunk", "out_dim": 64}
    config.ensemble_size = 2
    config.channel_dim = 128

    hydra_config = config.to_hydra_config()

    assert hydra_config["ensemble_size"] == 2
    assert hydra_config["channel_dim"] == 128
    assert "protein_seq" in hydra_config["roots"]
    assert hydra_config["roots"]["protein_seq"]["_target_"] == "cortex.model.root.TransformerRoot"
    assert hydra_config["trunk"]["_target_"] == "cortex.model.trunk.SumTrunk"


def test_round_trip_hydra_conversion():
    """Test converting to Hydra and back preserves config."""
    original_hydra = {
        "roots": {"test_root": {"_target_": "cortex.model.root.TransformerRoot", "max_len": 256, "out_dim": 32}},
        "trunk": {"_target_": "cortex.model.trunk.SumTrunk"},
        "ensemble_size": 3,
        "channel_dim": 64,
    }

    # Hydra -> NeuralTreeConfig -> Hydra
    config = NeuralTreeConfig.from_hydra_config(original_hydra)
    converted_back = config.to_hydra_config()

    assert converted_back["ensemble_size"] == 3
    assert converted_back["channel_dim"] == 64
    assert "test_root" in converted_back["roots"]
    assert converted_back["roots"]["test_root"]["max_len"] == 256


def test_hf_root_in_hydra_conversion():
    """Test HF root handling in Hydra conversion."""
    config = NeuralTreeConfig()

    # Add mock HF root (using dict instead of actual AutoConfig to avoid internet)
    config.roots["bert_root"] = RootConfig(
        use_hf_model=True, hf_config={"model_type": "bert", "hidden_size": 768}, processor_name="bert-base-uncased"
    )

    hydra_config = config.to_hydra_config()

    assert "bert_root" in hydra_config["roots"]
    hf_root_config = hydra_config["roots"]["bert_root"]
    assert hf_root_config["_target_"] == "cortex.model.root.HuggingFaceRoot"
    assert hf_root_config["model_name_or_path"] == "bert-base-uncased"
