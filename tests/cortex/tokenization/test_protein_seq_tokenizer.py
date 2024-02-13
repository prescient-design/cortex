import pandas as pd
import torch

from cortex.constants import ANTIGEN_COL, COMPLEX_SEP_TOKEN, VARIABLE_HEAVY_COL, VARIABLE_LIGHT_COL
from cortex.tokenization import ProteinComplex, ProteinSequenceTokenizerFast, tokenize_protein_complex
from cortex.transforms.functional import tokenize_igg_ag_df


def test_ab_ag_tokenizer_fast():
    tokenizer = ProteinSequenceTokenizerFast()
    unk_token_id = tokenizer.vocab["<unk>"]

    seq = "A V A V A V . A C V A C A . Q W E R T Y"
    tokens = tokenizer.cached_encode(seq)
    assert not torch.any(torch.tensor(tokens) == unk_token_id)


def test_tokenize_complex():
    complex = ProteinComplex(
        chains={
            "VH": "AVAVAV",
            "VL": "ACVACA",
        },
    )
    tokens = tokenize_protein_complex(complex)
    assert tokens == f"A V A V A V {COMPLEX_SEP_TOKEN} A C V A C A"


def test_tokenize_igg_ag_complex_df():
    data = {
        VARIABLE_HEAVY_COL: ["AVAVAV", "AVAVAV "],
        VARIABLE_LIGHT_COL: ["ACVACA", " ACVACA"],
        ANTIGEN_COL: ["QWERTY", "QWERTY"],
    }

    df = pd.DataFrame(data)

    df = tokenize_igg_ag_df(
        df,
        randomize_chain_order=False,
        use_custom_chain_tokens=False,
        use_custom_format_tokens=False,
        inplace=False,
    )
    print(df.tokenized_ab_ag_complex[0])
    assert df.tokenized_ab_ag_complex[0] == "A V A V A V . A C V A C A . Q W E R T Y"
    assert df.tokenized_ab_ag_complex[1] == "A V A V A V . A C V A C A . Q W E R T Y"
