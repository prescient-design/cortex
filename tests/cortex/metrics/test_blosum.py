"""
Tests for the BLOSUM substitution matrix functionality.
"""

import os
import tempfile

import pytest
import torch

from cortex.metrics._blosum import (
    batch_blosum62_distance,
    batch_lookup_blosum62,
    blosum62_distance,
    create_blosum62_matrix,
    create_blosum62_transition_matrix,
    create_tokenizer_compatible_transition_matrix,
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
        assert binned_probs[i] >= binned_probs[i - 1], f"Bin {i} avg prob not >= bin {i - 1}"

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


def test_blosum62_distance():
    """Test BLOSUM62 distance function."""
    blosum62, aa_to_idx = create_blosum62_matrix()
    gap_token = "-"

    # Test with identical sequences
    seq1 = "ACDEFGHIKLMNPQRSTVWY"
    seq2 = "ACDEFGHIKLMNPQRSTVWY"
    distance = blosum62_distance(seq1, seq2, blosum62, aa_to_idx)

    # Identical sequences should have minimal distance
    assert isinstance(distance, torch.Tensor)
    assert distance.item() < 0.1

    # Test with aligned sequences and some gaps
    seq3 = "ACDEF-GHIKLM"
    seq4 = "ACDEF-GHIXYZ"  # X, Y, Z are not valid amino acids
    distance = blosum62_distance(seq3, seq4, blosum62, aa_to_idx, gap_token=gap_token)

    # Should only compare the valid positions (ACDEFGHI) and ignore gaps + invalid chars
    assert 0 <= distance.item() <= 1.0

    # Test with mostly invalid characters but some valid matches
    seq5 = "ACXXX-XXXX"
    seq6 = "ACYYY-YYYY"
    distance = blosum62_distance(seq5, seq6, blosum62, aa_to_idx, gap_token=gap_token)

    # Should only compare the valid positions (AC) and have low distance for these
    assert 0 <= distance.item() <= 0.5  # Good match on the valid positions

    # Test with completely non-matching but valid sequences
    seq7 = "ACDKLM"
    seq8 = "VWYSPF"  # All valid but totally different amino acids
    distance = blosum62_distance(seq7, seq8, blosum62, aa_to_idx)

    # Should have high distance
    assert distance.item() > 0.7

    # Test with sequences that have no valid comparable positions
    seq9 = "----XXX"
    seq10 = "----YYY"
    distance = blosum62_distance(seq9, seq10, blosum62, aa_to_idx, gap_token=gap_token)

    # No valid comparison positions should result in maximum distance
    assert distance.item() == 1.0

    # Test length validation
    with pytest.raises(ValueError):
        blosum62_distance("ABC", "ABCD", blosum62, aa_to_idx)

    # Test metric properties

    # 1. Reflexivity: d(x,x) = 0 (or near zero)
    for aa_seq in ["ACDEFG", "KLMNPQ", "RSTVWY"]:
        distance = blosum62_distance(aa_seq, aa_seq, blosum62, aa_to_idx)
        assert distance.item() < 0.001, f"Reflexivity failed for {aa_seq}: {distance.item()}"

    # 2. Symmetry: d(x,y) = d(y,x)
    seq_a = "ACDEFG"
    seq_b = "ACKLMN"
    d_ab = blosum62_distance(seq_a, seq_b, blosum62, aa_to_idx)
    d_ba = blosum62_distance(seq_b, seq_a, blosum62, aa_to_idx)
    assert torch.isclose(d_ab, d_ba), f"Symmetry failed: d(a,b)={d_ab.item()}, d(b,a)={d_ba.item()}"

    # 3. Triangle inequality: d(x,z) ≤ d(x,y) + d(y,z)
    seq_x = "ACDEFG"
    seq_y = "ACKLMN"
    seq_z = "RSTVWY"
    d_xy = blosum62_distance(seq_x, seq_y, blosum62, aa_to_idx)
    d_yz = blosum62_distance(seq_y, seq_z, blosum62, aa_to_idx)
    d_xz = blosum62_distance(seq_x, seq_z, blosum62, aa_to_idx)
    assert d_xz <= d_xy + d_yz, f"Triangle inequality failed: {d_xz} > {d_xy} + {d_yz}"


def test_batch_blosum62_distance():
    """Test batch BLOSUM62 distance function."""
    blosum62, aa_to_idx = create_blosum62_matrix()
    gap_token = "-"

    # Create batch of aligned sequence pairs (all the same length within each pair)
    batch_seq1 = ["ACDEF", "GHI--", "KLMNP", "AAAAAA", "A----"]
    batch_seq2 = ["ACDEF", "XYZ--", "KLMNP", "VWY---", "C----"]

    # Calculate batch distances
    distances = batch_blosum62_distance(batch_seq1, batch_seq2, blosum62, aa_to_idx, gap_token=gap_token)

    # Check output shape
    assert isinstance(distances, torch.Tensor)
    assert distances.shape == (len(batch_seq1),)

    # Check distance values
    for i, (seq1, seq2) in enumerate(zip(batch_seq1, batch_seq2)):
        # Calculate individual distance
        single_distance = blosum62_distance(seq1, seq2, blosum62, aa_to_idx, gap_token=gap_token)

        # Batch result should match individual calculation
        assert torch.isclose(distances[i], single_distance)

        # Distance should be in [0, 1]
        assert 0 <= distances[i].item() <= 1.0

    # Check specific semantic relationships:
    # 1. Identical valid sequences should have low distance
    assert distances[0] < 0.1  # "ACDEF" vs "ACDEF"

    # 2. Completely matching sequences should have low distance
    assert distances[2] < 0.1  # "KLMNP" vs "KLMNP"

    # 3. Sequences with only invalid tokens should have maximum distance
    batch_invalids1 = ["----", "XXXX"]
    batch_invalids2 = ["----", "YYYY"]
    invalid_distances = batch_blosum62_distance(
        batch_invalids1, batch_invalids2, blosum62, aa_to_idx, gap_token=gap_token
    )
    assert torch.all(invalid_distances == 1.0)

    # 4. With gap tokens
    batch_with_gaps1 = ["A-CD-E", "G-H-I-"]
    batch_with_gaps2 = ["A-CD-E", "G-X-Y-"]
    gap_distances = batch_blosum62_distance(
        batch_with_gaps1, batch_with_gaps2, blosum62, aa_to_idx, gap_token=gap_token
    )
    assert gap_distances[0] < 0.1  # Identical
    assert 0 <= gap_distances[1] <= 1.0  # Some valid comparisons (G-H vs G-X)

    # Test batch triangle inequality with multiple examples
    def check_triangle_inequality(seqs):
        n = len(seqs)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    d_ij = blosum62_distance(seqs[i], seqs[j], blosum62, aa_to_idx)
                    d_jk = blosum62_distance(seqs[j], seqs[k], blosum62, aa_to_idx)
                    d_ik = blosum62_distance(seqs[i], seqs[k], blosum62, aa_to_idx)
                    # Allow small numerical error
                    assert d_ik <= d_ij + d_jk + 1e-5, f"Triangle inequality failed: {d_ik} > {d_ij} + {d_jk}"

    # Check with several diverse sequences
    test_sequences = [
        "ACDEFG",  # Hydrophilic
        "KLMNPQ",  # Mixed
        "RSTVWY",  # Hydrophobic
        "AAAAAA",  # Homopolymer
        "ACACAC",  # Alternating
    ]
    check_triangle_inequality(test_sequences)


def test_create_tokenizer_compatible_transition_matrix():
    """Test the creation of a tokenizer-compatible transition matrix."""
    # Create a temporary vocab file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        vocab_file = f.name
        # Add canonical amino acids plus some special tokens
        f.write("<pad>\n")  # 0
        f.write("<unk>\n")  # 1
        f.write("<eos>\n")  # 2
        f.write("<cls>\n")  # 3
        f.write("<mask>\n")  # 4
        f.write(".\n")  # 5 (complex separator)
        f.write("A\n")  # 6
        f.write("C\n")  # 7
        f.write("D\n")  # 8
        f.write("E\n")  # 9
        f.write("G\n")  # 10
        f.write("F\n")  # 11
        f.write("I\n")  # 12
        f.write("H\n")  # 13
        f.write("K\n")  # 14
        f.write("B\n")  # 15 (ambiguous)

    try:
        # Use the temporary file as vocab file
        transition_matrix = create_tokenizer_compatible_transition_matrix(vocab_file)

        # Basic checks
        assert isinstance(transition_matrix, torch.Tensor)

        # File should have 16 tokens
        assert transition_matrix.shape == (16, 16)

        # Check that matrix is well-formed
        assert torch.all(transition_matrix >= 0)
        assert torch.all(transition_matrix <= 1.0)

        # Special tokens should be identity mapped
        for i in range(6):  # First 6 tokens
            assert transition_matrix[i, i] == 1.0
            row_sum = transition_matrix[i].sum().item()
            assert abs(row_sum - 1.0) < 1e-6

        # Amino acids should have rows that sum to 1
        for i in range(6, 15):  # Amino acid tokens
            row_sum = transition_matrix[i].sum().item()
            assert abs(row_sum - 1.0) < 1e-6

            # Amino acids shouldn't map to themselves
            assert transition_matrix[i, i] == 0.0

        # Check amino acid relationships make sense
        # E.g., A(6) -> C(7) should have a lower probability than A(6) -> G(10)
        # because A and G are more similar than A and C
        assert transition_matrix[6, 10] > transition_matrix[6, 7]

        # Hydrophobic amino acids should have higher transition probability
        # I(12) -> F(11) should be higher than I(12) -> D(8)
        assert transition_matrix[12, 11] > transition_matrix[12, 8]

    finally:
        # Clean up the temporary file
        os.unlink(vocab_file)
