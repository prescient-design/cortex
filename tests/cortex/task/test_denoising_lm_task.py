from unittest.mock import MagicMock

from cortex.corruption import SubstitutionCorruptionProcess
from cortex.model.leaf import DenoisingLanguageModelLeaf
from cortex.task import DenoisingLanguageModelTask


class MockTokenizer:
    def __init__(self):
        self.vocab = {str(i): i for i in range(100)}  # Mock vocabulary


def test_denoising_lm_task_create_leaf_with_corruption():
    """
    Test that DenoisingLanguageModelTask correctly passes corruption parameters to the leaf.
    """
    # Create mocks
    mock_data_module = MagicMock()
    mock_tokenizer = MockTokenizer()

    # Create corruption process
    vocab_size = len(mock_tokenizer.vocab)
    corruption_process = SubstitutionCorruptionProcess(vocab_size=vocab_size)
    corruption_rate = 0.05

    # Create task with corruption parameters
    task = DenoisingLanguageModelTask(
        data_module=mock_data_module,
        input_map={"seq": ["sequence"]},
        leaf_key="test_task",
        root_key="seq",
        tokenizer=mock_tokenizer,
        corruption_process=corruption_process,
        corruption_rate=corruption_rate,
    )

    # Create leaf node
    leaf = task.create_leaf(in_dim=128, branch_key="test_branch")

    # Verify the leaf has the correct parameters
    assert isinstance(leaf, DenoisingLanguageModelLeaf)
    assert leaf.corruption_process is corruption_process
    assert leaf.corruption_rate == corruption_rate
    assert leaf.in_dim == 128
    assert leaf.num_classes == vocab_size
    assert leaf.branch_key == "test_branch"
    assert leaf.root_key == "seq"


def test_denoising_lm_task_default_corruption_params():
    """
    Test that DenoisingLanguageModelTask works with default corruption parameters.
    """
    # Create mocks
    mock_data_module = MagicMock()
    mock_tokenizer = MockTokenizer()

    # Create task with default parameters
    task = DenoisingLanguageModelTask(
        data_module=mock_data_module,
        input_map={"seq": ["sequence"]},
        leaf_key="test_task",
        root_key="seq",
        tokenizer=mock_tokenizer,
    )

    # Create leaf node
    leaf = task.create_leaf(in_dim=128, branch_key="test_branch")

    # Verify the leaf has the default parameters
    assert isinstance(leaf, DenoisingLanguageModelLeaf)
    assert leaf.corruption_process is None
    assert leaf.corruption_rate == 0.01  # Default value
    assert leaf.in_dim == 128
    assert leaf.num_classes == len(mock_tokenizer.vocab)
    assert leaf.branch_key == "test_branch"
    assert leaf.root_key == "seq"
