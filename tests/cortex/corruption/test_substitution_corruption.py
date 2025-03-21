import os
import tempfile

import torch

from cortex.constants import CANON_AMINO_ACIDS
from cortex.corruption import SubstitutionCorruptionProcess


class MockTokenizer:
    """Mock tokenizer class for testing SubstitutionCorruptionProcess."""

    def __init__(self):
        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<eos>": 2,
            "<cls>": 3,
            "<mask>": 4,
            ".": 5,  # complex separator
        }

        # Add amino acids to vocabulary
        for i, aa in enumerate(CANON_AMINO_ACIDS):
            self.vocab[aa] = i + 6

        # Add some ambiguous tokens
        self.vocab["B"] = len(self.vocab)
        self.vocab["O"] = len(self.vocab)

        # Special tokens that shouldn't be corrupted
        self.corruption_vocab_excluded = {"<pad>", "<unk>", "<eos>", "<cls>", "<mask>", "."}


def test_substitution_corruption_uniform():
    """Test uniform substitution corruption process."""
    tokenizer = MockTokenizer()
    vocab_size = len(tokenizer.vocab)

    # Create substitution corruption process with uniform substitution
    corruption_process = SubstitutionCorruptionProcess.from_tokenizer(tokenizer)

    # Check that the substitution matrix is properly initialized
    assert corruption_process.substitution_matrix.shape == (vocab_size, vocab_size)

    # Create a simple input tensor with amino acid IDs
    # We'll use A=6, C=7, D=8, E=9, G=10
    x_start = torch.tensor([[6, 7, 8, 9, 10]])

    # Test with zero corruption
    x_corrupt, is_corrupted = corruption_process(x_start, corrupt_frac=0.0)
    assert torch.allclose(x_start, x_corrupt)
    assert not torch.any(is_corrupted)

    # Test with complete corruption, using a fixed random seed for reproducibility
    torch.manual_seed(42)
    x_corrupt, is_corrupted = corruption_process(x_start, corrupt_frac=1.0)
    assert torch.all(is_corrupted)
    assert not torch.allclose(x_start, x_corrupt)

    # Check that excluded tokens are not corrupted
    x_with_special = torch.tensor([[0, 6, 7, 8, 3]])  # <pad>, A, C, D, <cls>
    x_corrupt, is_corrupted = corruption_process(x_with_special, corrupt_frac=1.0)
    # Special tokens should remain unchanged
    assert x_corrupt[0, 0] == 0  # <pad> token
    assert x_corrupt[0, 4] == 3  # <cls> token
    # Amino acids should be corrupted
    assert is_corrupted[0, 1]
    assert is_corrupted[0, 2]
    assert is_corrupted[0, 3]


def test_substitution_corruption_from_blosum62():
    """Test BLOSUM62-based substitution corruption process."""
    # Create a temporary vocab file for testing
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        vocab_file = f.name
        # Add canonical amino acids plus some special tokens
        f.write("<pad>\n")  # 0
        f.write("<unk>\n")  # 1
        f.write("<eos>\n")  # 2
        f.write("<cls>\n")  # 3
        f.write("<mask>\n")  # 4
        f.write(".\n")  # 5 (complex separator)

        # Add canonical amino acids
        for aa in CANON_AMINO_ACIDS:
            f.write(f"{aa}\n")

        # Add some ambiguous tokens
        f.write("B\n")
        f.write("O\n")

    try:
        tokenizer = MockTokenizer()

        # Create BLOSUM62-based substitution corruption process
        corruption_process = SubstitutionCorruptionProcess.from_blosum62(vocab_file_path=vocab_file)

        # Basic checks on the substitution matrix
        vocab_size = len(tokenizer.vocab)
        assert corruption_process.substitution_matrix.shape == (vocab_size, vocab_size)

        # Create a simple input tensor with amino acid IDs
        # A=6, C=7, D=8, E=9, G=10
        x_start = torch.tensor([[6, 7, 8, 9, 10]])

        # Test with complete corruption, using a fixed random seed for reproducibility
        torch.manual_seed(42)
        x_corrupt, is_corrupted = corruption_process(x_start, corrupt_frac=1.0)
        assert torch.all(is_corrupted)
        assert not torch.allclose(x_start, x_corrupt)

        # Verify that the BLOSUM substitutions respect amino acid similarity
        # Run many simulations to verify statistical properties
        torch.manual_seed(42)
        num_simulations = 1000
        a_to_g_count = 0
        a_to_c_count = 0

        for _ in range(num_simulations):
            # I (12) should be more likely to be substituted with L (13) than D (8)
            # because I and L are more similar than I and D in BLOSUM62
            x_test = torch.tensor([[12]])  # Just test I (Isoleucine)
            x_corrupt, _ = corruption_process(x_test, corrupt_frac=1.0)

            if x_corrupt.item() == 13:  # L (Leucine)
                a_to_g_count += 1
            elif x_corrupt.item() == 8:  # D (Aspartic acid)
                a_to_c_count += 1

        # I->L should be more frequent than I->D due to BLOSUM62 scores
        # (hydrophobic pairs vs. hydrophobic-acidic pairs)
        assert a_to_g_count > a_to_c_count, f"I->L: {a_to_g_count}, I->D: {a_to_c_count}"

    finally:
        # Clean up the temporary file
        os.unlink(vocab_file)
