import importlib.resources
from typing import Dict, List, Tuple

import torch

from cortex.constants import CANON_AMINO_ACIDS, STANDARD_AA_FREQS


def create_blosum62_matrix() -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Creates the BLOSUM62 substitution matrix as a PyTorch tensor.

    The matrix is organized according to the Dayhoff classification of amino acids,
    with amino acids grouped by their physicochemical properties.

    Returns:
        Tuple[torch.Tensor, Dict[str, int]]: The BLOSUM62 matrix as a torch.Tensor
        and a dictionary mapping amino acid characters to indices.
    """
    # Create a dictionary mapping amino acids to their indices
    aa_to_idx = {aa: idx for idx, aa in enumerate(CANON_AMINO_ACIDS)}

    # Initialize the BLOSUM62 matrix with zeros
    n_amino_acids = len(CANON_AMINO_ACIDS)
    blosum62 = torch.zeros((n_amino_acids, n_amino_acids), dtype=torch.int8)

    # Define the BLOSUM62 values
    # Values from the standard BLOSUM62 matrix
    blosum_values = {
        # Sulfur polymerization group
        ("C", "C"): 9,
        # Small group
        ("A", "A"): 4,
        ("A", "G"): 0,
        ("A", "P"): -1,
        ("A", "S"): 1,
        ("A", "T"): 0,
        ("G", "G"): 6,
        ("G", "P"): -2,
        ("G", "S"): 0,
        ("G", "T"): -2,
        ("P", "P"): 7,
        ("P", "S"): -1,
        ("P", "T"): -1,
        ("S", "S"): 4,
        ("S", "T"): 1,
        ("T", "T"): 5,
        # Acid and amide group
        ("D", "D"): 6,
        ("D", "E"): 2,
        ("D", "N"): 1,
        ("D", "Q"): 0,
        ("E", "E"): 5,
        ("E", "N"): 0,
        ("E", "Q"): 2,
        ("N", "N"): 6,
        ("N", "Q"): 0,
        ("Q", "Q"): 5,
        # Basic group
        ("H", "H"): 8,
        ("H", "K"): -1,
        ("H", "R"): 0,
        ("K", "K"): 5,
        ("K", "R"): 2,
        ("R", "R"): 5,
        # Hydrophobic group
        ("I", "I"): 4,
        ("I", "L"): 2,
        ("I", "M"): 1,
        ("I", "V"): 3,
        ("L", "L"): 4,
        ("L", "M"): 2,
        ("L", "V"): 1,
        ("M", "M"): 5,
        ("M", "V"): 1,
        ("V", "V"): 4,
        # Aromatic group
        ("F", "F"): 6,
        ("F", "W"): 1,
        ("F", "Y"): 3,
        ("W", "W"): 11,
        ("W", "Y"): 2,
        ("Y", "Y"): 7,
        # Cross-group interactions
        # Sulfur - Small
        ("C", "A"): 0,
        ("C", "G"): -3,
        ("C", "P"): -3,
        ("C", "S"): -1,
        ("C", "T"): -1,
        # Sulfur - Acid/Amide
        ("C", "D"): -3,
        ("C", "E"): -4,
        ("C", "N"): -3,
        ("C", "Q"): -3,
        # Sulfur - Basic
        ("C", "H"): -3,
        ("C", "K"): -3,
        ("C", "R"): -3,
        # Sulfur - Hydrophobic
        ("C", "I"): -1,
        ("C", "L"): -1,
        ("C", "M"): -1,
        ("C", "V"): -1,
        # Sulfur - Aromatic
        ("C", "F"): -2,
        ("C", "W"): -2,
        ("C", "Y"): -2,
        # Small - Acid/Amide
        ("A", "D"): -2,
        ("A", "E"): -1,
        ("A", "N"): -2,
        ("A", "Q"): -1,
        ("G", "D"): -1,
        ("G", "E"): -2,
        ("G", "N"): 0,
        ("G", "Q"): -2,
        ("P", "D"): -1,
        ("P", "E"): -1,
        ("P", "N"): -2,
        ("P", "Q"): -1,
        ("S", "D"): 0,
        ("S", "E"): 0,
        ("S", "N"): 1,
        ("S", "Q"): 0,
        ("T", "D"): -1,
        ("T", "E"): -1,
        ("T", "N"): 0,
        ("T", "Q"): -1,
        # Small - Basic
        ("A", "H"): -2,
        ("A", "K"): -1,
        ("A", "R"): -1,
        ("G", "H"): -2,
        ("G", "K"): -2,
        ("G", "R"): -2,
        ("P", "H"): -2,
        ("P", "K"): -1,
        ("P", "R"): -2,
        ("S", "H"): -1,
        ("S", "K"): 0,
        ("S", "R"): -1,
        ("T", "H"): -2,
        ("T", "K"): -1,
        ("T", "R"): -1,
        # Small - Hydrophobic
        ("A", "I"): -1,
        ("A", "L"): -1,
        ("A", "M"): -1,
        ("A", "V"): 0,
        ("G", "I"): -4,
        ("G", "L"): -4,
        ("G", "M"): -3,
        ("G", "V"): -3,
        ("P", "I"): -3,
        ("P", "L"): -3,
        ("P", "M"): -2,
        ("P", "V"): -2,
        ("S", "I"): -2,
        ("S", "L"): -2,
        ("S", "M"): -1,
        ("S", "V"): -2,
        ("T", "I"): -1,
        ("T", "L"): -1,
        ("T", "M"): -1,
        ("T", "V"): 0,
        # Small - Aromatic
        ("A", "F"): -2,
        ("A", "W"): -3,
        ("A", "Y"): -2,
        ("G", "F"): -3,
        ("G", "W"): -2,
        ("G", "Y"): -3,
        ("P", "F"): -4,
        ("P", "W"): -4,
        ("P", "Y"): -3,
        ("S", "F"): -2,
        ("S", "W"): -3,
        ("S", "Y"): -2,
        ("T", "F"): -2,
        ("T", "W"): -2,
        ("T", "Y"): -2,
        # Acid/Amide - Basic
        ("D", "H"): -1,
        ("D", "K"): -1,
        ("D", "R"): -2,
        ("E", "H"): 0,
        ("E", "K"): 1,
        ("E", "R"): 0,
        ("N", "H"): 1,
        ("N", "K"): 0,
        ("N", "R"): 0,
        ("Q", "H"): 0,
        ("Q", "K"): 1,
        ("Q", "R"): 1,
        # Acid/Amide - Hydrophobic
        ("D", "I"): -3,
        ("D", "L"): -4,
        ("D", "M"): -3,
        ("D", "V"): -3,
        ("E", "I"): -3,
        ("E", "L"): -3,
        ("E", "M"): -2,
        ("E", "V"): -2,
        ("N", "I"): -3,
        ("N", "L"): -3,
        ("N", "M"): -2,
        ("N", "V"): -3,
        ("Q", "I"): -3,
        ("Q", "L"): -2,
        ("Q", "M"): 0,
        ("Q", "V"): -2,
        # Acid/Amide - Aromatic
        ("D", "F"): -3,
        ("D", "W"): -4,
        ("D", "Y"): -3,
        ("E", "F"): -3,
        ("E", "W"): -3,
        ("E", "Y"): -2,
        ("N", "F"): -3,
        ("N", "W"): -4,
        ("N", "Y"): -2,
        ("Q", "F"): -3,
        ("Q", "W"): -2,
        ("Q", "Y"): -1,
        # Basic - Hydrophobic
        ("H", "I"): -3,
        ("H", "L"): -3,
        ("H", "M"): -2,
        ("H", "V"): -3,
        ("K", "I"): -3,
        ("K", "L"): -2,
        ("K", "M"): -1,
        ("K", "V"): -2,
        ("R", "I"): -3,
        ("R", "L"): -2,
        ("R", "M"): -1,
        ("R", "V"): -3,
        # Basic - Aromatic
        ("H", "F"): -1,
        ("H", "W"): -2,
        ("H", "Y"): 2,
        ("K", "F"): -3,
        ("K", "W"): -3,
        ("K", "Y"): -2,
        ("R", "F"): -3,
        ("R", "W"): -3,
        ("R", "Y"): -2,
        # Hydrophobic - Aromatic
        ("I", "F"): 0,
        ("I", "W"): -3,
        ("I", "Y"): -1,
        ("L", "F"): 0,
        ("L", "W"): -2,
        ("L", "Y"): -1,
        ("M", "F"): 0,
        ("M", "W"): -1,
        ("M", "Y"): -1,
        ("V", "F"): -1,
        ("V", "W"): -3,
        ("V", "Y"): -1,
    }

    # Fill the BLOSUM62 matrix
    for (aa1, aa2), value in blosum_values.items():
        i, j = aa_to_idx[aa1], aa_to_idx[aa2]
        blosum62[i, j] = value
        # Fill the symmetric counterpart
        if i != j:  # Skip diagonal elements
            blosum62[j, i] = value

    return blosum62, aa_to_idx


def create_blosum62_transition_matrix() -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Convert BLOSUM matrix to transition probability matrix.

    The transition matrix follows discrete Markov process conventions:
    - Row index i represents the current state (amino acid)
    - Column index j represents the next state (amino acid)
    - Each entry [i,j] is the probability of transitioning from amino acid i to j

    BLOSUM scores are log-odds scores: 2 * log2(p(a,b)/(p(a)*p(b)))
    To convert back to substitution probabilities:
    1. Convert score to odds ratio: 2^(score/2)
    2. Multiply by background frequency: odds_ratio * p(b)

    The resulting substitution probabilities reflect the underlying evolutionary
    model captured by the BLOSUM matrix.

    Returns:
        Tuple[torch.Tensor, Dict[str, int]]:
            - A transition probability matrix based on BLOSUM substitution rates
            - Dictionary mapping amino acids to indices
    """
    blosum62, aa_to_idx = create_blosum62_matrix()
    marginal_freqs = torch.tensor([STANDARD_AA_FREQS[aa] for aa in CANON_AMINO_ACIDS], dtype=torch.float32)

    # We use 2^(score/2) as per standard BLOSUM interpretation
    odds_ratios = torch.exp2(blosum62.to(torch.float32) / 2)
    # Zero out the diagonal to ensure we don't self-substitute
    odds_ratios.fill_diagonal_(0.0)

    # Calculate transition probabilities
    # For each row i, multiply odds_ratios[i,j] by background frequency of j
    unnormalized_probs = odds_ratios * marginal_freqs.unsqueeze(0)

    # Normalize rows to get proper transition probabilities
    row_sums = unnormalized_probs.sum(dim=1, keepdim=True)
    transition_probs = unnormalized_probs / row_sums

    return transition_probs, aa_to_idx


def lookup_blosum62_score(seq1: str, seq2: str, blosum62: torch.Tensor, aa_to_idx: Dict[str, int]) -> torch.Tensor:
    """
    Compute BLOSUM62 alignment scores between two amino acid sequences.

    Args:
        seq1: First amino acid sequence
        seq2: Second amino acid sequence
        blosum62: BLOSUM62 substitution matrix
        aa_to_idx: Dictionary mapping amino acids to indices in the BLOSUM62 matrix

    Returns:
        torch.Tensor: A tensor of scores with shape (len(seq1), len(seq2))
    """
    # Convert sequences to index tensors
    try:
        seq1_indices = torch.tensor([aa_to_idx[aa] for aa in seq1], dtype=torch.long)
        seq2_indices = torch.tensor([aa_to_idx[aa] for aa in seq2], dtype=torch.long)
    except KeyError as e:
        raise ValueError(f"Unknown amino acid in sequence: {e}") from e

    scores = blosum62[seq1_indices.unsqueeze(1), seq2_indices.unsqueeze(0)]

    return scores


def blosum62_distance(
    seq1: str,
    seq2: str,
    blosum62: torch.Tensor,
    aa_to_idx: Dict[str, int],
    lambda_param: float = 0.267,
    gap_token: str = "-",
) -> torch.Tensor:
    """
    Compute a proper distance metric between two aligned protein sequences using BLOSUM62.

    This function assumes the sequences are pre-aligned and of the same length.
    Gap tokens are treated as neutral and don't contribute to the distance calculation.

    Args:
        seq1: First protein sequence (aligned)
        seq2: Second protein sequence (aligned)
        blosum62: BLOSUM62 matrix as torch.Tensor
        aa_to_idx: Dictionary mapping amino acids to indices in the BLOSUM62 matrix
        lambda_param: Karlin-Altschul parameter for the scoring system (default=0.267)
        gap_token: Character representing gaps in the alignment (default="-")

    Returns:
        torch.Tensor: A distance value between 0 and 1
    """
    if len(seq1) != len(seq2):
        raise ValueError(f"Sequences must be of the same length (pre-aligned): {len(seq1)} != {len(seq2)}")

    # Filter positions where both sequences have valid amino acids (not gaps)
    valid_positions = []
    for i, (aa1, aa2) in enumerate(zip(seq1, seq2)):
        if aa1 != gap_token and aa1 in aa_to_idx and aa2 != gap_token and aa2 in aa_to_idx:
            valid_positions.append(i)

    # If no valid positions, return maximum distance
    if not valid_positions:
        return torch.tensor(1.0, dtype=torch.float32)

    # Extract the valid characters
    seq1_valid = [seq1[i] for i in valid_positions]
    seq2_valid = [seq2[i] for i in valid_positions]

    # Convert to tensors
    seq1_indices = torch.tensor([aa_to_idx[aa] for aa in seq1_valid], dtype=torch.long)
    seq2_indices = torch.tensor([aa_to_idx[aa] for aa in seq2_valid], dtype=torch.long)

    # Calculate alignment score (sum of BLOSUM scores for aligned positions)
    alignment_score = blosum62[seq1_indices, seq2_indices].sum()

    # Get diagonal (self-substitution) scores for maximum possible
    diag_indices = torch.arange(blosum62.size(0))
    self_scores = blosum62[diag_indices, diag_indices]

    # Maximum possible score is the sum of self-substitution scores for each amino acid
    max_score1 = torch.sum(self_scores[seq1_indices])
    max_score2 = torch.sum(self_scores[seq2_indices])
    max_possible = torch.min(max_score1, max_score2)

    # Convert similarity to distance
    raw_distance = max_possible - alignment_score

    # Transform to ensure triangle inequality using PyTorch exp
    proper_distance = 1.0 - torch.exp(torch.tensor(-float(raw_distance) / lambda_param))

    return proper_distance


def batch_lookup_blosum62(
    batch_seq1: List[str], batch_seq2: List[str], blosum62: torch.Tensor, aa_to_idx: Dict[str, int]
) -> List[torch.Tensor]:
    """
    Compute BLOSUM62 alignment scores for batches of sequences.

    Args:
        batch_seq1: List of first amino acid sequences
        batch_seq2: List of second amino acid sequences
        blosum62: BLOSUM62 substitution matrix
        aa_to_idx: Dictionary mapping amino acids to indices in the BLOSUM62 matrix

    Returns:
        List[torch.Tensor]: List of score tensors, each with shape (len(seq1_i), len(seq2_i))
    """
    return [lookup_blosum62_score(seq1, seq2, blosum62, aa_to_idx) for seq1, seq2 in zip(batch_seq1, batch_seq2)]


def batch_blosum62_distance(
    batch_seq1: List[str],
    batch_seq2: List[str],
    blosum62: torch.Tensor,
    aa_to_idx: Dict[str, int],
    lambda_param: float = 0.267,
    gap_token: str = "-",
) -> torch.Tensor:
    """
    Compute BLOSUM62 distances for batches of aligned protein sequence pairs.

    This function assumes the sequences in each pair are pre-aligned and of the same length.
    Gap tokens are treated as neutral and don't contribute to the distance calculation.

    Args:
        batch_seq1: List of first protein sequences (aligned)
        batch_seq2: List of second protein sequences (aligned)
        blosum62: BLOSUM62 substitution matrix
        aa_to_idx: Dictionary mapping amino acids to indices in the BLOSUM62 matrix
        lambda_param: Karlin-Altschul parameter for the scoring system (default=0.267)
        gap_token: Character representing gaps in the alignment (default="-")

    Returns:
        torch.Tensor: A tensor of distances with shape (len(batch_seq1),)
    """
    # Check batch sizes match
    if len(batch_seq1) != len(batch_seq2):
        raise ValueError("Batch sizes must match: len(batch_seq1) != len(batch_seq2)")

    # Handle empty batch
    if len(batch_seq1) == 0:
        return torch.tensor([], dtype=torch.float32)

    # Compute distances for each pair
    distances = torch.zeros(len(batch_seq1), dtype=torch.float32)
    for i, (seq1, seq2) in enumerate(zip(batch_seq1, batch_seq2)):
        distances[i] = blosum62_distance(seq1, seq2, blosum62, aa_to_idx, lambda_param, gap_token)

    return distances


def create_tokenizer_compatible_transition_matrix(vocab_file_path=None) -> torch.Tensor:
    """
    Create a transition probability matrix compatible with a protein sequence tokenizer.

    This function creates a transition probability matrix where:
    1. Rows and columns are ordered according to vocabulary indices from the vocab file
    2. Non-canonical amino acid tokens have identity transitions (1.0 on diagonal)
    3. Canonical amino acid transitions follow BLOSUM62 probabilities

    Args:
        vocab_file_path: Path to the vocabulary file. If None, uses the default path:
                       cortex/assets/protein_seq_tokenizer_32/vocab.txt

    Returns:
        torch.Tensor: A vocab_size x vocab_size transition probability matrix
    """

    # Get the transition matrix based on BLOSUM62
    transition_probs, aa_to_idx = create_blosum62_transition_matrix()

    # Use default path if none provided
    if vocab_file_path is None:
        vocab_file_path = (
            importlib.resources.files("cortex") / "assets" / "protein_seq_tokenizer_32" / "vocab.txt"
        ).as_posix()

    # Load vocabulary from file
    vocab = {}
    with open(vocab_file_path, "r") as f:
        for i, line in enumerate(f):
            token = line.strip()
            vocab[token] = i

    # Get vocabulary size and initialize full matrix with identity matrix
    # (defaulting to no substitution for non-amino acid tokens)
    vocab_size = len(vocab)
    full_matrix = torch.eye(vocab_size, dtype=torch.float32)

    # Create mapping from amino acid to token ID in vocabulary
    amino_to_token_id = {}
    for aa in CANON_AMINO_ACIDS:
        if aa in vocab:
            amino_to_token_id[aa] = vocab[aa]

    # Map BLOSUM transition probabilities to token IDs in the vocabulary
    for aa_i, i in aa_to_idx.items():
        if aa_i in amino_to_token_id:
            token_i = amino_to_token_id[aa_i]
            # Zero out diagonal for amino acids (no self-substitution)
            full_matrix[token_i, token_i] = 0.0

            for aa_j, j in aa_to_idx.items():
                if aa_j in amino_to_token_id:
                    token_j = amino_to_token_id[aa_j]
                    if token_i != token_j:
                        full_matrix[token_i, token_j] = transition_probs[i, j]

    # Normalize rows to get proper probability distributions
    for i in range(vocab_size):
        # Only normalize rows for amino acids that we want to substitute
        if i in amino_to_token_id.values():
            row_sum = full_matrix[i].sum()
            if row_sum > 0:
                full_matrix[i] = full_matrix[i] / row_sum

    return full_matrix
