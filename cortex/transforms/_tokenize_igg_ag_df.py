import pandas as pd

from cortex.constants import AB_AG_COMPLEX_COL, ANTIGEN_COL, VARIABLE_HEAVY_COL, VARIABLE_LIGHT_COL
from cortex.transforms._transform import Transform
from cortex.transforms.functional import tokenize_igg_ag_df


class TokenizeIggAgComplex(Transform):
    _output_column = AB_AG_COMPLEX_COL
    _required_columns = [
        ANTIGEN_COL,
        VARIABLE_HEAVY_COL,
        VARIABLE_LIGHT_COL,
    ]

    def __init__(
        self,
        randomize_chain_order: bool = False,
        use_custom_chain_tokens: bool = True,
        use_custom_format_tokens: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._randomize_chain_order = randomize_chain_order
        self._use_custom_chain_tokens = use_custom_chain_tokens
        self._use_custom_format_tokens = use_custom_format_tokens

    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        return tokenize_igg_ag_df(
            data,
            randomize_chain_order=self._randomize_chain_order,
            use_custom_chain_tokens=self._use_custom_chain_tokens,
            use_custom_format_tokens=self._use_custom_format_tokens,
            inplace=False,
        )
