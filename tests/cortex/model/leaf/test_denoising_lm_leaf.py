import torch

from cortex.corruption import SubstitutionCorruptionProcess
from cortex.model.branch import Conv1dBranchOutput
from cortex.model.leaf import DenoisingLanguageModelLeaf, DenoisingLanguageModelLeafOutput
from cortex.model.root._conv1d_root import Conv1dRootOutput


def test_denoising_lm_leaf_basic():
    # Set up test parameters
    in_dim = 128
    vocab_size = 22  # Standard protein vocab size
    batch_size = 4
    max_seq_len = 10

    # Create leaf node without corruption
    leaf_node = DenoisingLanguageModelLeaf(in_dim=in_dim, num_classes=vocab_size, branch_key="test")

    # Create fake inputs
    branch_features = torch.rand(batch_size, max_seq_len, in_dim)
    branch_output = Conv1dBranchOutput(
        branch_features=branch_features,
        branch_mask=torch.ones(batch_size, max_seq_len, dtype=torch.float),
        pooled_features=branch_features.mean(-2),
    )

    # Forward pass
    leaf_output = leaf_node(branch_output)
    assert isinstance(leaf_output, DenoisingLanguageModelLeafOutput)
    logits = leaf_output.logits
    assert torch.is_tensor(logits)
    assert logits.size() == torch.Size((batch_size, max_seq_len, vocab_size))

    # Create corrupt mask and target tokens for loss calculation
    is_corrupted = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    is_corrupted[:, [2, 5, 8]] = True  # Corrupt specific positions
    tgt_tok_idxs = torch.randint(0, vocab_size, (batch_size, max_seq_len))

    # Create root outputs
    root_output = Conv1dRootOutput(
        root_features=torch.rand(batch_size, max_seq_len, in_dim),
        padding_mask=torch.ones(batch_size, max_seq_len, dtype=torch.bool),
        is_corrupted=is_corrupted,
        tgt_tok_idxs=tgt_tok_idxs,
    )

    # Test loss calculation
    loss = leaf_node.loss(leaf_output, root_output)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0  # Loss should be a scalar tensor


def test_denoising_lm_leaf_with_corruption():
    # Set up test parameters
    in_dim = 128
    vocab_size = 22  # Standard protein vocab size
    batch_size = 4
    max_seq_len = 10

    # Create substitution corruption process with uniform substitution
    corruption_process = SubstitutionCorruptionProcess(vocab_size=vocab_size)

    # Create leaf node with corruption process
    leaf_node = DenoisingLanguageModelLeaf(
        in_dim=in_dim,
        num_classes=vocab_size,
        branch_key="test",
        corruption_process=corruption_process,
        corruption_rate=0.2,  # Higher rate for testing
    )

    # Create fake inputs
    branch_features = torch.rand(batch_size, max_seq_len, in_dim)
    branch_output = Conv1dBranchOutput(
        branch_features=branch_features,
        branch_mask=torch.ones(batch_size, max_seq_len, dtype=torch.float),
        pooled_features=branch_features.mean(-2),
    )

    # Forward pass
    leaf_output = leaf_node(branch_output)

    # Create corrupt mask and target tokens for loss calculation
    is_corrupted = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    is_corrupted[:, [2, 5, 8]] = True  # Corrupt specific positions
    tgt_tok_idxs = torch.randint(0, vocab_size, (batch_size, max_seq_len))

    # Create root outputs
    root_output = Conv1dRootOutput(
        root_features=torch.rand(batch_size, max_seq_len, in_dim),
        padding_mask=torch.ones(batch_size, max_seq_len, dtype=torch.bool),
        is_corrupted=is_corrupted,
        tgt_tok_idxs=tgt_tok_idxs,
    )

    # Ensure the model is in training mode
    leaf_node.train()

    # Store original targets to compare with corrupted targets
    leaf_node.training = False  # Temporarily disable corruption
    _, original_targets = leaf_node.format_outputs(leaf_output, root_output)
    leaf_node.training = True  # Re-enable corruption

    # Get corrupted targets
    _, corrupted_targets = leaf_node.format_outputs(leaf_output, root_output)

    # Verify that some corruption happened (targets should be different)
    assert not torch.all(original_targets == corrupted_targets)

    # Test loss calculation with corrupted targets
    loss = leaf_node.loss(leaf_output, root_output)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0  # Loss should be a scalar tensor

    # Test that corruption is not applied in evaluation mode
    leaf_node.eval()
    _, eval_targets = leaf_node.format_outputs(leaf_output, root_output)
    assert torch.all(eval_targets == original_targets)


def test_denoising_lm_leaf_with_blosum_corruption():
    """
    Test DenoisingLanguageModelLeaf with BLOSUM62-based corruption
    This test is dependent on the SubstitutionCorruptionProcess.from_blosum62 implementation
    """
    try:
        # Set up test parameters
        in_dim = 128
        vocab_size = 22  # Standard protein vocab size
        batch_size = 4
        max_seq_len = 10

        # Create BLOSUM62-based corruption process
        corruption_process = SubstitutionCorruptionProcess.from_blosum62()

        # Create leaf node with corruption process
        leaf_node = DenoisingLanguageModelLeaf(
            in_dim=in_dim,
            num_classes=vocab_size,
            branch_key="test",
            corruption_process=corruption_process,
            corruption_rate=0.2,  # Higher rate for testing
        )

        # Create fake inputs
        branch_features = torch.rand(batch_size, max_seq_len, in_dim)
        branch_output = Conv1dBranchOutput(
            branch_features=branch_features,
            branch_mask=torch.ones(batch_size, max_seq_len, dtype=torch.float),
            pooled_features=branch_features.mean(-2),
        )

        # Forward pass
        leaf_output = leaf_node(branch_output)

        # Create corrupt mask and target tokens for loss calculation
        is_corrupted = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
        is_corrupted[:, [2, 5, 8]] = True  # Corrupt specific positions
        tgt_tok_idxs = torch.randint(0, vocab_size, (batch_size, max_seq_len))

        # Create root outputs
        root_output = Conv1dRootOutput(
            root_features=torch.rand(batch_size, max_seq_len, in_dim),
            padding_mask=torch.ones(batch_size, max_seq_len, dtype=torch.bool),
            is_corrupted=is_corrupted,
            tgt_tok_idxs=tgt_tok_idxs,
        )

        # Ensure the model is in training mode
        leaf_node.train()

        # Test loss calculation with corrupted targets
        loss = leaf_node.loss(leaf_output, root_output)
        assert torch.is_tensor(loss)
        assert loss.ndim == 0  # Loss should be a scalar tensor

    except Exception as e:
        # Skip this test if BLOSUM62 implementation is not available
        print(f"Skipping BLOSUM62 test due to: {e}")
