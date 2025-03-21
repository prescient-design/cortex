from typing import Optional, Set, Union

import torch

from cortex.metrics._blosum import create_tokenizer_compatible_transition_matrix

from ._abstract_corruption import CorruptionProcess


class SubstitutionCorruptionProcess(CorruptionProcess):
    """
    Corrupt input tensor by substituting values according to a substitution probability matrix.
    Each tensor element is corrupted independently with probability `corrupt_frac`.

    If no substitution_matrix is provided, uniform random substitution is used.
    If a substitution_matrix is provided, it defines the probability of substituting
    token i with token j.

    Args:
        vocab_size: Size of the vocabulary (number of possible tokens).
        excluded_token_ids: Set of token IDs that should not be corrupted or used as substitutes.
        substitution_matrix: Optional substitution probability matrix of shape (vocab_size, vocab_size).
            Each row i contains the probability distribution for substituting token i with any other token.
            If None, uniform random substitution is used.
        schedule: Noise schedule type ("linear", "cosine", etc.).
        max_steps: Maximum number of diffusion steps.
    """

    def __init__(
        self,
        vocab_size: int,
        excluded_token_ids: Optional[Set[int]] = None,
        substitution_matrix: Optional[torch.Tensor] = None,
        schedule: str = "cosine",
        max_steps: int = 1000,
        *args,
        **kwargs,
    ):
        super().__init__(schedule, max_steps, *args, **kwargs)
        self.vocab_size = vocab_size
        self.excluded_token_ids = excluded_token_ids or set()

        # Initialize the substitution matrix
        if substitution_matrix is None:
            # Create matrix with zeros for excluded tokens
            substitution_matrix = torch.zeros(vocab_size, vocab_size)

            # For each non-excluded token, set uniform probability to all other non-excluded tokens
            valid_token_ids = [i for i in range(vocab_size) if i not in self.excluded_token_ids]

            if valid_token_ids:
                # Calculate probability for each valid substitution
                # (excluding self-substitution)
                prob = 1.0 / (len(valid_token_ids) - 1) if len(valid_token_ids) > 1 else 0.0

                for i in valid_token_ids:
                    for j in valid_token_ids:
                        if i != j:  # Don't substitute with same token
                            substitution_matrix[i, j] = prob
        else:
            # Validate the substitution matrix
            if not substitution_matrix.shape == (vocab_size, vocab_size):
                raise ValueError("Substitution matrix must have shape (vocab_size, vocab_size)")

            if not torch.all(substitution_matrix >= 0) and not torch.all(substitution_matrix <= 1):
                raise ValueError("Substitution matrix must have values in [0, 1]")

            if not torch.allclose(substitution_matrix.sum(dim=-1), torch.ones(vocab_size)):
                raise ValueError("Substitution matrix rows must sum to 1")

        # Store substitution matrix as an attribute
        self.substitution_matrix = substitution_matrix

    def _corrupt(
        self, x_start: torch.Tensor, corrupt_frac: Union[float, torch.Tensor], *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Corrupt the input tensor by substituting tokens according to the substitution matrix.

        Args:
            x_start: Input tensor to corrupt.
            corrupt_frac: Fraction of tokens to corrupt, either a scalar or per-example tensor.

        Returns:
            Tuple of (corrupted tensor, corruption mask).
        """
        # Handle per-example corrupt_frac for broadcasting
        if isinstance(corrupt_frac, torch.Tensor) and corrupt_frac.dim() > 0:
            # Reshape to enable broadcasting: [batch_size] -> [batch_size, 1, ...]
            corrupt_frac = corrupt_frac.view(*corrupt_frac.shape, *([1] * (x_start.dim() - corrupt_frac.dim())))

        # Generate corruption mask (avoiding excluded tokens)
        corrupted_allowed = torch.ones_like(x_start, dtype=torch.bool)
        for token_id in self.excluded_token_ids:
            corrupted_allowed &= x_start != token_id

        # Only corrupt allowed tokens with probability corrupt_frac
        is_corrupted = torch.rand_like(x_start, dtype=torch.float64) < corrupt_frac
        is_corrupted &= corrupted_allowed

        # Only proceed if there are tokens to corrupt
        if not torch.any(is_corrupted):
            return x_start, is_corrupted

        # Create new tensor for corrupted tokens
        x_corrupt = x_start.clone()

        # Flatten the tensor for easier indexing
        flat_shape = x_start.shape
        flat_x = x_start.reshape(-1)
        flat_is_corrupted = is_corrupted.reshape(-1)
        flat_x_corrupt = x_corrupt.reshape(-1)

        # Get indices of tokens to corrupt
        corrupt_indices = torch.nonzero(flat_is_corrupted).squeeze(1)

        if len(corrupt_indices) > 0:
            # Get original token values
            original_tokens = flat_x[corrupt_indices]

            # Generate substitutions based on the substitution matrix
            # For each token to corrupt, sample from its row in the substitution matrix
            sub_matrix = self.substitution_matrix.to(original_tokens.device)
            substitution_probs = sub_matrix[original_tokens]

            # Sample new tokens according to substitution probabilities
            new_tokens = torch.multinomial(substitution_probs, num_samples=1).squeeze(1)

            # Apply substitutions
            flat_x_corrupt[corrupt_indices] = new_tokens

        # Reshape back to original shape
        x_corrupt = flat_x_corrupt.reshape(flat_shape)

        return x_corrupt, is_corrupted

    @classmethod
    def from_tokenizer(cls, tokenizer, **kwargs):
        """
        Create a SubstitutionCorruptionProcess using a tokenizer's vocabulary.

        Args:
            tokenizer: A tokenizer with vocab and corruption_vocab_excluded attributes.

        Returns:
            SubstitutionCorruptionProcess with uniform substitution matrix respecting tokenizer constraints.
        """
        vocab_size = len(tokenizer.vocab)
        excluded_token_ids = set()

        # Convert excluded token strings to token IDs
        for token in tokenizer.corruption_vocab_excluded:
            if token in tokenizer.vocab:
                excluded_token_ids.add(tokenizer.vocab[token])

        return cls(vocab_size=vocab_size, excluded_token_ids=excluded_token_ids, **kwargs)

    @classmethod
    def from_blosum62(cls, vocab_file_path=None, **kwargs):
        """
        Create a SubstitutionCorruptionProcess using BLOSUM62 substitution probabilities.

        Args:
            tokenizer: A tokenizer with vocabulary mapping functions (used for excluded tokens).
            vocab_file_path: Optional path to vocab file. If None, uses default in ProteinSequenceTokenizer.

        Returns:
            SubstitutionCorruptionProcess with BLOSUM62-based substitution matrix.
        """
        # Create transition matrix based on BLOSUM62 and the vocab file
        transition_matrix = create_tokenizer_compatible_transition_matrix(vocab_file_path)

        return cls(
            vocab_size=transition_matrix.size(0),
            excluded_token_ids=None,
            substitution_matrix=transition_matrix,
            **kwargs,
        )
