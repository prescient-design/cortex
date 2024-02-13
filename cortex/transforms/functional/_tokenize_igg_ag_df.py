import random

import pandas as pd

from cortex.constants import (
    AB_AG_COMPLEX_COL,
    ANTIGEN_COL,
    ANTIGEN_COMPLEX_TOKEN,
    VARIABLE_HEAVY_CHAIN_TOKEN,
    VARIABLE_HEAVY_COL,
    VARIABLE_LIGHT_CHAIN_TOKEN,
    VARIABLE_LIGHT_COL,
)
from cortex.tokenization import ProteinComplex, tokenize_protein_complex


def tokenize_igg_ag_df(
    data: pd.DataFrame,
    randomize_chain_order: bool = False,
    use_custom_chain_tokens: bool = False,
    use_custom_format_tokens: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    if not inplace:
        data = data.copy()

    tokenized_input = []
    for _, row in data.iterrows():
        igg_chains = [
            (VARIABLE_HEAVY_CHAIN_TOKEN, row[VARIABLE_HEAVY_COL]),
            (VARIABLE_LIGHT_CHAIN_TOKEN, row[VARIABLE_LIGHT_COL]),
        ]
        ag_chains = [
            (ANTIGEN_COMPLEX_TOKEN, row[ANTIGEN_COL]),
        ]

        if randomize_chain_order:
            random.shuffle(igg_chains)

        complex = ProteinComplex(
            chains=dict(igg_chains + ag_chains),
        )
        tokenized_input.append(
            tokenize_protein_complex(
                complex,
                sep_with_chain_ids=use_custom_chain_tokens,
                include_species=False,
                include_format=use_custom_format_tokens,
            )
        )

    data[AB_AG_COMPLEX_COL] = tokenized_input
    if not inplace:
        return data
