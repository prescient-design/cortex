"""
Tests for the BLOSUM substitution matrix functionality.
"""

import pytest
import torch

from cortex.metrics._blosum import (
    batch_lookup_blosum62,
    create_blosum62_matrix,
    create_blosum62_transition_matrix,
    lookup_blosum62_score,
)


def test_create_blosum62_matrix():
    """Test that the BLOSUM62 matrix is created correctly."""
    blosum62, aa_to_idx = create_blosum62_matrix()

    # Check matrix dimensions
    assert isinstance(blosum62, torch.Tensor)
    assert blosum62.shape == (len(aa_to_idx), len(aa_to_idx))
    assert blosum62.dtype == torch.int8

    # Check that diagonal values are positive (same amino acid substitutions)
    for aa, idx in aa_to_idx.items():
        assert blosum62[idx, idx] > 0

    # Check symmetry
    assert torch.all(blosum62 == blosum62.T)

    # Check specific known values
    # A-A should have higher score than A-P
    a_idx = aa_to_idx["A"]
    p_idx = aa_to_idx["P"]
    assert blosum62[a_idx, a_idx] > blosum62[a_idx, p_idx]

    # C-C should have high conservation score
    c_idx = aa_to_idx["C"]
    assert blosum62[c_idx, c_idx] >= 9

    # Check that hydrophobic amino acids have positive scores with each other
    hydrophobic = ["I", "L", "M", "V"]
    for aa1 in hydrophobic:
        for aa2 in hydrophobic:
            if aa1 in aa_to_idx and aa2 in aa_to_idx:
                assert blosum62[aa_to_idx[aa1], aa_to_idx[aa2]] >= 0


def test_lookup_blosum62_score():
    """Test looking up BLOSUM62 scores for amino acid pairs."""
    blosum62, aa_to_idx = create_blosum62_matrix()

    # Test with identical sequences
    seq1 = "ACDEFGHIKLMNPQRSTVWY"
    scores = lookup_blosum62_score(seq1, seq1, blosum62, aa_to_idx)

    # Check output shape
    assert scores.shape == (len(seq1), len(seq1))

    # Diagonal should contain self-substitution scores
    for i, aa in enumerate(seq1):
        assert scores[i, i] == blosum62[aa_to_idx[aa], aa_to_idx[aa]]

    # Test with different sequences
    seq2 = "ACDKLMNPQRSTVWYFGHE"  # Reordered to ensure different
    scores = lookup_blosum62_score(seq1, seq2, blosum62, aa_to_idx)

    # Check output shape
    assert scores.shape == (len(seq1), len(seq2))

    # Verify a few specific locations
    assert scores[0, 0] == blosum62[aa_to_idx["A"], aa_to_idx["A"]]  # A-A
    assert scores[1, 1] == blosum62[aa_to_idx["C"], aa_to_idx["C"]]  # C-C
    assert scores[0, 1] == blosum62[aa_to_idx["A"], aa_to_idx["C"]]  # A-C

    # Test with invalid amino acid
    with pytest.raises(ValueError):
        lookup_blosum62_score("ACDEFGHIKLMNPQRSTVWYX", seq1, blosum62, aa_to_idx)


def test_create_blosum62_transition_matrix():
    """Test the creation of a BLOSUM62 transition matrix."""
    transition_matrix, aa_to_idx = create_blosum62_transition_matrix()

    # Check that the matrix is a proper tensor
    assert isinstance(transition_matrix, torch.Tensor)

    # Check dimensions (should be n_amino_acids x n_amino_acids)
    n_amino_acids = len(transition_matrix)
    assert transition_matrix.shape == (n_amino_acids, n_amino_acids)

    # Check that aa_to_idx contains mappings for all amino acids
    assert len(aa_to_idx) == n_amino_acids

    # Ensure all values are non-negative
    assert torch.all(transition_matrix >= 0)

    # Check probability properties
    assert torch.all(transition_matrix <= 1.0)

    assert torch.allclose(transition_matrix.sum(dim=1), torch.ones(n_amino_acids))

    # Get the original BLOSUM matrix for comparison
    blosum62, _ = create_blosum62_matrix()

    # Check that the transition probabilities align with BLOSUM scores
    # Higher BLOSUM scores should correspond to higher transition probabilities
    blosum_val_pairs = []
    for i in range(n_amino_acids):
        for j in range(n_amino_acids):
            if i != j:  # Skip diagonal
                blosum_val_pairs.append((blosum62[i, j].item(), transition_matrix[i, j].item()))

    # Sort pairs by BLOSUM score
    blosum_val_pairs.sort()
    blosum_scores, prob_values = zip(*blosum_val_pairs)

    # Verify general trend: higher BLOSUM score -> higher transition probability
    # We can't check strict monotonicity due to background frequencies,
    # but trends should be observable with binned averages
    bin_size = len(blosum_scores) // 5  # 5 bins
    binned_probs = []

    for i in range(0, len(blosum_scores), bin_size):
        if i + bin_size <= len(blosum_scores):
            bin_avg = sum(prob_values[i : i + bin_size]) / bin_size
            binned_probs.append(bin_avg)

    # Bins with higher BLOSUM scores should have higher average probabilities
    for i in range(1, len(binned_probs)):
        assert binned_probs[i] >= binned_probs[i - 1], f"Bin {i} avg prob not >= bin {i-1}"

    # Check that similar amino acids have higher transition probabilities
    # using Dayhoff's classification
    from cortex.constants import AMINO_ACID_GROUPS

    # Test for one group: hydrophobic amino acids
    hydrophobic_indices = [aa_to_idx[aa] for aa in AMINO_ACID_GROUPS["hydrophobic"] if aa in aa_to_idx]

    # Pick a hydrophobic amino acid and check its transition probabilities
    if hydrophobic_indices:
        i = hydrophobic_indices[0]
        # Calculate average transition probability to other hydrophobic AAs vs. non-hydrophobic
        hydrophobic_probs = []
        non_hydrophobic_probs = []

        for j in range(n_amino_acids):
            if j != i:  # Skip self
                if j in hydrophobic_indices:
                    hydrophobic_probs.append(transition_matrix[i, j].item())
                else:
                    non_hydrophobic_probs.append(transition_matrix[i, j].item())

        if hydrophobic_probs and non_hydrophobic_probs:
            avg_hydrophobic = sum(hydrophobic_probs) / len(hydrophobic_probs)
            avg_non_hydrophobic = sum(non_hydrophobic_probs) / len(non_hydrophobic_probs)

            # Hydrophobic amino acids should be more likely to transition to other hydrophobic AAs
            assert avg_hydrophobic > avg_non_hydrophobic


def test_batch_lookup_blosum62():
    """Test batch lookup of BLOSUM62 scores."""
    blosum62, aa_to_idx = create_blosum62_matrix()

    # Create a batch of sequences
    batch_seq1 = ["ACDEF", "GHI", "KLMNP"]
    batch_seq2 = ["ACDEF", "GHI", "KLMNP"]

    # Get batch scores
    batch_scores = batch_lookup_blosum62(batch_seq1, batch_seq2, blosum62, aa_to_idx)

    # Check that we get the right number of score matrices
    assert len(batch_scores) == len(batch_seq1)

    # Check each score matrix
    for i, (seq1, seq2, scores) in enumerate(zip(batch_seq1, batch_seq2, batch_scores)):
        assert scores.shape == (len(seq1), len(seq2))

        # Check diagonal values for identical sequences
        if seq1 == seq2:
            for j, aa in enumerate(seq1):
                assert scores[j, j] == blosum62[aa_to_idx[aa], aa_to_idx[aa]]

    # Test with different length sequences
    batch_seq1 = ["ACDEF", "GHI", "KLMNP"]
    batch_seq2 = ["ACD", "GHIKLM", "P"]

    batch_scores = batch_lookup_blosum62(batch_seq1, batch_seq2, blosum62, aa_to_idx)

    # Check shapes
    for i, (seq1, seq2, scores) in enumerate(zip(batch_seq1, batch_seq2, batch_scores)):
        assert scores.shape == (len(seq1), len(seq2))
