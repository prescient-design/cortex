import warnings
from typing import Any, Mapping, Optional

import torch
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.utils.transforms import normalize_indices
from torch import Tensor, nn


class DDPStandardize(Standardize):
    r"""Standardize outcomes (zero mean, unit variance).

    This module is stateful: If in train mode, calling forward updates the
    module state (i.e. the mean/std normalizing constants). If in eval mode,
    calling forward simply applies the standardization using the current module
    state.
    """

    def __init__(
        self,
        m: int,
        outputs: Optional[list[int]] = None,
        batch_shape: torch.Size = torch.Size(),  # noqa: B008
        min_stdv: float = 1e-8,
    ) -> None:
        r"""Standardize outcomes (zero mean, unit variance).

        Args:
            m: The output dimension.
            outputs: Which of the outputs to standardize. If omitted, all
                outputs will be standardized.
            batch_shape: The batch_shape of the training targets.
            min_stddv: The minimum standard deviation for which to perform
                standardization (if lower, only de-mean the data).
        """
        OutcomeTransform.__init__(self)
        self.register_parameter("means", nn.Parameter(torch.zeros(*batch_shape, 1, m), requires_grad=False))
        self.register_parameter("stdvs", nn.Parameter(torch.ones(*batch_shape, 1, m), requires_grad=False))
        self.register_parameter("_stdvs_sq", nn.Parameter(torch.ones(*batch_shape, 1, m), requires_grad=False))
        self.register_parameter("_is_trained", nn.Parameter(torch.tensor(0.0), requires_grad=False))
        self._outputs = normalize_indices(outputs, d=m)
        self._m = m
        self._batch_shape = batch_shape
        self._min_stdv = min_stdv

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:
        r"""Custom logic for loading the state dict."""
        if "_is_trained" not in state_dict:
            warnings.warn(
                "Key '_is_trained' not found in state_dict. Setting to True. "
                "In a future release, this will result in an error.",
                DeprecationWarning,
                stacklevel=2,
            )
            state_dict = {**state_dict, "_is_trained": torch.tensor(1.0)}
        super().load_state_dict(state_dict, strict=strict)

    def forward(self, Y: Tensor, Yvar: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        r"""Standardize outcomes.

        If the module is in train mode, this updates the module state (i.e. the
        mean/std normalizing constants). If the module is in eval mode, simply
        applies the normalization using the module state.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        if self.training:
            if Y.shape[:-2] != self._batch_shape:
                raise RuntimeError(
                    f"Expected Y.shape[:-2] to be {self._batch_shape}, matching "
                    "the `batch_shape` argument to `Standardize`, but got "
                    f"Y.shape[:-2]={Y.shape[:-2]}."
                )
            if Y.size(-1) != self._m:
                raise RuntimeError(f"Wrong output dimension. Y.size(-1) is {Y.size(-1)}; expected " f"{self._m}.")
            stdvs = Y.std(dim=-2, keepdim=True)
            stdvs = stdvs.where(stdvs >= self._min_stdv, torch.full_like(stdvs, 1.0))
            means = Y.mean(dim=-2, keepdim=True)
            if self._outputs is not None:
                unused = [i for i in range(self._m) if i not in self._outputs]
                means[..., unused] = 0.0
                stdvs[..., unused] = 1.0
            self.means.data = means
            self.stdvs.data = stdvs
            self._stdvs_sq.data = stdvs.pow(2)
            self._is_trained.data = torch.tensor(1.0)

        Y_tf = (Y - self.means) / self.stdvs
        Yvar_tf = Yvar / self._stdvs_sq if Yvar is not None else None
        return Y_tf, Yvar_tf
