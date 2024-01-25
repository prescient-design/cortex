from typing import Any

import numpy as np

from cortex.tokenization import CachedBertTokenizerFast
from cortex.transforms._transform import Transform


class HuggingFaceTokenizerTransform(Transform):
    """
    Wraps a CachedBertTokenizerFast object in a pytorch transform module for use in
    a sequence of transforms.
    """

    def __init__(
        self,
        tokenizer: CachedBertTokenizerFast,
    ):
        super().__init__()
        self.tokenizer = tokenizer

    def validate(self, flat_inputs: list[Any]) -> None:
        pass

    def transform(self, data: np.ndarray) -> list[list[int]]:
        """
        Args:
            data: a numpy array of strings.
        Returns:
            A list of lists of integers, where each integer is the index of a token in the tokenizer's
            vocabulary.
        """
        res = [self.tokenizer.cached_encode(text.item()) for text in data.reshape(-1, 1)]
        return res

    def forward(self, data: Any) -> list[list[int]]:
        return self.transform(data)
