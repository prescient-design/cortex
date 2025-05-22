"""Tests for NeuralTreeModel and HuggingFace integration."""

from unittest.mock import Mock, patch

import pytest
import torch

from cortex.config import NeuralTreeConfig, RootConfig
from cortex.model import NeuralTreeModel


@pytest.fixture
def minimal_config():
    """Create minimal config for testing."""
    config = NeuralTreeConfig()
    config.trunk = {"_target_": "cortex.model.trunk.SumTrunk", "out_dim": 64}
    return config


@pytest.fixture
def mock_model_components():
    """Create mocked model components that are proper torch.nn.Module subclasses."""

    # Create proper mock modules
    class MockRoot(torch.nn.Module):
        def forward(self, x):
            from cortex.model.root import RootNodeOutput

            return RootNodeOutput(root_features=torch.randn(2, 10, 64), corrupt_frac=None)

    class MockTrunk(torch.nn.Module):
        def forward(self, *args):
            return torch.randn(2, 64)

    class MockBranch(torch.nn.Module):
        def forward(self, x):
            return torch.randn(2, 32)

    class MockLeaf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.branch_key = "test_branch"

        def forward(self, x):
            mock_output = Mock()
            mock_output.predictions = torch.randn(2, 1)
            return mock_output

    mock_root = MockRoot()
    mock_trunk = MockTrunk()
    mock_branch = MockBranch()
    mock_leaf = MockLeaf()

    return {
        "root": mock_root,
        "trunk": mock_trunk,
        "branch": mock_branch,
        "leaf": mock_leaf,
    }


def test_config_class_attribute():
    """Test that config_class is properly set."""
    assert NeuralTreeModel.config_class == NeuralTreeConfig


@patch("cortex.model.neural_tree_model.hydra.utils.instantiate")
def test_model_initialization_with_cortex_roots(mock_instantiate, minimal_config):
    """Test model initialization with cortex roots."""

    # Mock the instantiation to return proper modules
    class MockTrunk(torch.nn.Module):
        def forward(self, *args):
            return torch.randn(2, 64)

    class MockRoot(torch.nn.Module):
        def forward(self, x):
            from cortex.model.root import RootNodeOutput

            return RootNodeOutput(root_features=torch.randn(2, 10, 64))

    def mock_instantiate_side_effect(config):
        if "trunk" in str(config.get("_target_", "")):
            return MockTrunk()
        else:
            return MockRoot()

    mock_instantiate.side_effect = mock_instantiate_side_effect

    # Add cortex root
    minimal_config.add_cortex_root("test_root", {"_target_": "cortex.model.root.TransformerRoot", "max_len": 128})

    model = NeuralTreeModel(minimal_config)

    assert isinstance(model.root_nodes, torch.nn.ModuleDict)
    assert "test_root" in model.root_nodes
    assert isinstance(model.trunk_node, MockTrunk)
    assert isinstance(model.branch_nodes, torch.nn.ModuleDict)
    assert isinstance(model.leaf_nodes, torch.nn.ModuleDict)


@patch("cortex.model.neural_tree_model.AutoModel")
@patch("cortex.model.neural_tree_model.hydra.utils.instantiate")
def test_model_initialization_with_hf_roots(mock_instantiate, mock_auto_model, minimal_config):
    """Test model initialization with HuggingFace roots."""

    # Mock HF model and trunk with proper Module subclasses
    class MockHFModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Mock()
            self.config.model_type = "bert"

        def forward(self, **kwargs):
            mock_output = Mock()
            mock_output.last_hidden_state = torch.randn(2, 10, 768)
            return mock_output

    class MockTrunk(torch.nn.Module):
        def forward(self, *args):
            return torch.randn(2, 64)

    mock_hf_model = MockHFModel()
    mock_auto_model.from_config.return_value = mock_hf_model
    mock_trunk = MockTrunk()
    mock_instantiate.return_value = mock_trunk

    # Add HF root
    minimal_config.roots["bert_root"] = RootConfig(
        use_hf_model=True, hf_config={"model_type": "bert", "hidden_size": 768}
    )

    model = NeuralTreeModel(minimal_config)

    assert "bert_root" in model.root_nodes
    assert isinstance(model.root_nodes["bert_root"], MockHFModel)
    mock_auto_model.from_config.assert_called_once()


def test_add_task(minimal_config):
    """Test adding tasks dynamically."""
    with patch("cortex.model.neural_tree_model.hydra.utils.instantiate") as mock_instantiate:

        class MockTrunk(torch.nn.Module):
            def forward(self, *args):
                return torch.randn(2, 64)

        class MockLeaf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.branch_key = "property_branch"

            def forward(self, x):
                return Mock()

        mock_trunk = MockTrunk()
        mock_leaf = MockLeaf()
        mock_instantiate.side_effect = [mock_trunk, mock_leaf]

        model = NeuralTreeModel(minimal_config)

        # Add task
        task_config = {"target_col": "fluorescence"}
        leaf_configs = {"regressor": {"_target_": "cortex.model.leaf.RegressorLeaf", "branch_key": "property_branch"}}

        model.add_task("test_task", task_config, leaf_configs)

        assert "test_task" in model._task_configs
        assert "test_task_regressor" in model.leaf_nodes
        assert isinstance(model.leaf_nodes["test_task_regressor"], MockLeaf)


def test_forward_with_cortex_roots(mock_model_components):
    """Test forward pass with cortex roots."""
    config = NeuralTreeConfig()
    config.trunk = {"_target_": "mock.Trunk"}

    with patch("cortex.model.neural_tree_model.hydra.utils.instantiate") as mock_instantiate:
        mock_instantiate.return_value = mock_model_components["trunk"]

        model = NeuralTreeModel(config)
        model.root_nodes["test_root"] = mock_model_components["root"]
        model.branch_nodes["test_branch"] = mock_model_components["branch"]
        model.leaf_nodes["test_leaf"] = mock_model_components["leaf"]

        # Test forward pass
        root_inputs = {"test_root": torch.randn(2, 10)}
        leaf_keys = ["test_leaf"]

        output = model.forward(root_inputs, leaf_keys=leaf_keys)

        # Verify calls
        mock_model_components["root"].assert_called_once_with(root_inputs["test_root"])
        mock_model_components["trunk"].assert_called_once()
        mock_model_components["branch"].assert_called_once()
        mock_model_components["leaf"].assert_called_once()

        # Verify output structure
        assert hasattr(output, "root_outputs")
        assert hasattr(output, "trunk_outputs")
        assert hasattr(output, "branch_outputs")
        assert hasattr(output, "leaf_outputs")
        assert "test_root" in output.root_outputs
        assert "test_leaf" in output.leaf_outputs


def test_forward_with_hf_roots(mock_model_components):
    """Test forward pass with HuggingFace roots."""
    config = NeuralTreeConfig()
    config.trunk = {"_target_": "mock.Trunk"}

    with patch("cortex.model.neural_tree_model.hydra.utils.instantiate") as mock_instantiate:
        mock_instantiate.return_value = mock_model_components["trunk"]

        model = NeuralTreeModel(config)

        # Mock HF model
        mock_hf_model = Mock()
        mock_hf_model.config.model_type = "bert"
        mock_hf_output = Mock()
        mock_hf_output.last_hidden_state = torch.randn(2, 10, 768)
        mock_hf_model.return_value = mock_hf_output

        model.root_nodes["bert_root"] = mock_hf_model
        model.branch_nodes["test_branch"] = mock_model_components["branch"]
        model.leaf_nodes["test_leaf"] = mock_model_components["leaf"]

        # Test forward pass with HF input format
        root_inputs = {"bert_root": {"input_ids": torch.randint(0, 1000, (2, 10)), "attention_mask": torch.ones(2, 10)}}
        leaf_keys = ["test_leaf"]

        output = model.forward(root_inputs, leaf_keys=leaf_keys)

        # Verify HF model was called correctly
        mock_hf_model.assert_called_once_with(
            input_ids=root_inputs["bert_root"]["input_ids"], attention_mask=root_inputs["bert_root"]["attention_mask"]
        )

        # Verify output structure
        assert "bert_root" in output.root_outputs
        assert "test_leaf" in output.leaf_outputs


def test_guided_forward(mock_model_components):
    """Test guided forward for LaMBO integration."""
    config = NeuralTreeConfig()
    config.trunk = {"_target_": "mock.Trunk"}

    with patch("cortex.model.neural_tree_model.hydra.utils.instantiate") as mock_instantiate:
        mock_instantiate.return_value = mock_model_components["trunk"]

        model = NeuralTreeModel(config)
        model.root_nodes["sequence"] = mock_model_components["root"]
        model.branch_nodes["test_branch"] = mock_model_components["branch"]
        model.leaf_nodes["test_leaf"] = mock_model_components["leaf"]

        # Test guided forward
        sequences = torch.randint(0, 20, (2, 10))
        corruption_params = {"sequence": {"noise_level": 0.1}}

        output = model.guided_forward(sequences=sequences, corruption_params=corruption_params, guidance_layer="trunk")

        # Verify it delegates to forward
        assert hasattr(output, "root_outputs")
        assert hasattr(output, "trunk_outputs")


def test_from_cortex_tree():
    """Test creating NeuralTreeModel from existing cortex tree."""

    # Create proper mock modules
    class MockModule(torch.nn.Module):
        def forward(self, x):
            return x

    # Mock existing cortex tree
    mock_cortex_tree = Mock()
    mock_cortex_tree.root_nodes = torch.nn.ModuleDict({"root1": MockModule()})
    mock_cortex_tree.trunk_node = MockModule()
    mock_cortex_tree.branch_nodes = torch.nn.ModuleDict({"branch1": MockModule()})
    mock_cortex_tree.leaf_nodes = torch.nn.ModuleDict({"leaf1": MockModule()})

    # Create config with trunk
    config = NeuralTreeConfig()
    config.trunk = {"_target_": "cortex.model.trunk.SumTrunk", "out_dim": 64}

    with patch("cortex.model.neural_tree_model.hydra.utils.instantiate") as mock_instantiate:
        mock_instantiate.return_value = MockModule()
        model = NeuralTreeModel.from_cortex_tree(mock_cortex_tree, config)

        assert len(model.root_nodes) == 1
        assert "root1" in model.root_nodes
        assert isinstance(model.trunk_node, MockModule)
        assert len(model.branch_nodes) == 1
        assert "branch1" in model.branch_nodes
        assert len(model.leaf_nodes) == 1
        assert "leaf1" in model.leaf_nodes


def test_get_task_outputs(mock_model_components):
    """Test extracting task outputs."""
    config = NeuralTreeConfig()
    config.trunk = {"_target_": "mock.Trunk"}

    with patch("cortex.model.neural_tree_model.hydra.utils.instantiate") as mock_instantiate:
        mock_instantiate.return_value = mock_model_components["trunk"]

        model = NeuralTreeModel(config)

        # Mock tree outputs
        from cortex.model.tree import NeuralTreeOutput

        tree_outputs = Mock(spec=NeuralTreeOutput)
        tree_outputs.fetch_task_outputs.return_value = {"predictions": torch.randn(2, 1)}

        task_outputs = model.get_task_outputs("test_task", tree_outputs)

        tree_outputs.fetch_task_outputs.assert_called_once_with("test_task")
        assert "predictions" in task_outputs
