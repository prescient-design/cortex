import importlib.resources
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from cortex.constants import ALIGNMENT_GAP_TOKEN, CANON_AMINO_ACIDS, COMPLEX_SEP_TOKEN, NULL_TOKENS
from cortex.tokenization._cached_bert_tokenizer import CachedBertTokenizerFast


@dataclass
class ProteinComplex:
    """
    Dataclass for protein complex.
    Args:
        chains: dict[str, str]: an ordered dict of chain_id: chain_sequence pairs (e.g. {"VH": "AVAVAV", "VL": "ACVACA"})
        species: Optional[str]: species of the complex (e.g. <human>, <mouse>, etc.)
        format: Optional[str]: format of the complex  (e.g. <igg>, <igm>, etc.)
    """

    chains: OrderedDict[str, str]
    species: Optional[str] = None
    format: Optional[str] = None


def tokenize_protein_complex(
    complex: ProteinComplex,
    sep_with_chain_ids: bool = False,
    include_species: bool = False,
    include_format: bool = False,
):
    """
    Tokenize a protein complex.
    Args:
        complex: ProteinComplex: a protein complex dataclass
        seq_with_chain_ids: bool: whether to include chain ids in the tokenized sequence
    Returns:
        str: tokenized protein complex

    Example:
    >>> complex = ProteinComplex(
    ...     chains={
    ...         "VH": "A V A V A V",
    ...         "VL": "A C V A C A",
    ...     },
    ... )
    >>> tokens = tokenize_protein_complex(complex)
    >>> tokens
    "A V A V A V . A C V A C A"
    """
    tokens = []
    if include_species:
        tokens.append(complex.species)
    if include_format:
        tokens.append(complex.format)
    for chain_count, (chain_id, chain_seq) in enumerate(complex.chains.items()):
        if sep_with_chain_ids:
            tokens.append(f"[{chain_id}]")
        elif chain_count > 0:
            tokens.append(COMPLEX_SEP_TOKEN)
        tokens.extend(list(chain_seq.replace(" ", "")))
    return " ".join(tokens)


class ProteinSequenceTokenizerFast(CachedBertTokenizerFast):
    """
    Subclass of CachedBertTokenizerFast with vocabulary for protein complexes.
    """

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        custom_tokens: Optional[list[str]] = None,
        ambiguous_tokens: Optional[list[str]] = None,
        do_lower_case: bool = False,
        unk_token: str = "<unk>",
        sep_token: str = "<eos>",
        pad_token: str = "<pad>",
        cls_token: str = "<cls>",
        mask_token: str = "<mask>",
        **kwargs,
    ):
        if vocab_file is None:
            vocab_file = (
                importlib.resources.files("cortex") / "assets" / "protein_seq_tokenizer_32" / "vocab.txt"
            ).as_posix()

            custom_tokens = NULL_TOKENS
            ambiguous_tokens = ["B", "O", "U", "Z"]

        # tokens to exclude from sampling can be specified as custom or ambiguous
        if custom_tokens is None:
            custom_tokens = []
        if ambiguous_tokens is None:
            ambiguous_tokens = []

        print(f"using vocab from {vocab_file}")

        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        # ban utility tokens from sample output
        exclude_tokens_from_samples = [
            unk_token,
            sep_token,
            pad_token,
            cls_token,
            mask_token,
            COMPLEX_SEP_TOKEN,
        ]
        exclude_tokens_from_samples.extend(custom_tokens)
        exclude_tokens_from_samples.extend(ambiguous_tokens)
        self.sampling_vocab_excluded = set(exclude_tokens_from_samples)

        # prevent utility token input corruption
        exclude_tokens_from_corruption = [
            unk_token,
            sep_token,
            pad_token,
            cls_token,
            mask_token,
            COMPLEX_SEP_TOKEN,
        ]
        exclude_tokens_from_corruption.extend(custom_tokens)
        self.corruption_vocab_excluded = set(exclude_tokens_from_corruption)

        self.chain_tokens = CANON_AMINO_ACIDS + [ALIGNMENT_GAP_TOKEN]
