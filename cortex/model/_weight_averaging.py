from typing import Iterable, Optional

import torch


def online_weight_update_(
    src_state_dict: dict[str, torch.Tensor],
    tgt_state_dict: dict[str, torch.Tensor],
    decay_rate: float,
    param_prefixes: Optional[Iterable[str]] = None,
):
    if param_prefixes is None:
        param_keys = src_state_dict.keys()
    else:
        param_keys = [k for k in src_state_dict.keys() if any(k.startswith(prefix) for prefix in param_prefixes)]

    for param_key in param_keys:
        param_src = src_state_dict[param_key]
        param_tgt = tgt_state_dict[param_key]
        if torch.is_tensor(param_tgt) and param_tgt.dtype is not torch.bool and param_tgt.dtype is not torch.long:
            param_tgt.mul_(decay_rate)
            param_tgt.data.add_(param_src.data * (1.0 - decay_rate))
        elif torch.is_tensor(param_tgt):
            param_tgt.copy_(param_src)
        else:
            raise RuntimeError("Parameter {} is not a tensor.".format(param_key))

    return None
