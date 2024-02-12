import importlib.resources
from typing import Optional

from cortex.constants import (ALIGNMENT_GAP_TOKEN, CANON_AMINO_ACIDS,
                              COMPLEX_SEP_TOKEN, NULL_TOKENS)
from cortex.tokenization._cached_bert_tokenizer import CachedBertTokenizerFast


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
