# copied from https://github.com/pytorch/text/blob/main/torchtext/transforms.py
# torchtext is no longer maintained and is incompatible with torch >= 2.4
from typing import Any, Optional

import torch
from torch import Tensor
from torch.nn import Module

from cortex.transforms.functional import to_tensor


class ToTensor(Module):
    r"""Convert input to torch tensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    """

    def __init__(self, padding_value: Optional[int] = None, dtype: torch.dtype = torch.long) -> None:
        super().__init__()
        self.padding_value = padding_value
        self.dtype = dtype

    def forward(self, input: Any) -> Tensor:
        """
        :param input: Sequence or batch of token ids
        :type input: Union[List[int], List[List[int]]]
        :rtype: Tensor
        """
        return to_tensor(input, padding_value=self.padding_value, dtype=self.dtype)
