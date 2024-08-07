from typing import Any, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def to_tensor(input: Any, padding_value: Optional[int] = None, dtype: torch.dtype = torch.long) -> Tensor:
    r"""Convert input to torch tensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    :param input: Sequence or batch of token ids
    :type input: Union[list[int], list[list[int]]]
    :rtype: Tensor
    """
    if torch.jit.isinstance(input, list[int]):
        return torch.tensor(input, dtype=torch.long)
    elif torch.jit.isinstance(input, list[list[int]]):
        if padding_value is None:
            output = torch.tensor(input, dtype=dtype)
            return output
        else:
            output = pad_sequence(
                [torch.tensor(ids, dtype=dtype) for ids in input], batch_first=True, padding_value=float(padding_value)
            )
            return output
    else:
        raise TypeError("Input type not supported")
