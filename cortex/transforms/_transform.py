# TODO replace with prescient.Transform
from enum import Enum
from typing import Any, Callable, Union

from torch import Tensor
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_unflatten


def _check_type(
    obj: Any,
    types_or_checks: tuple[Union[type, Callable[[Any], bool]], ...],
) -> bool:
    return any(
        isinstance(obj, type_or_check) if isinstance(type_or_check, type) else type_or_check(obj)
        for type_or_check in types_or_checks
    )


class Transform(Module):
    _transformed_types: tuple[
        Union[type, Callable[[Any], bool]],
        ...,
    ] = (Tensor, str)

    def __init__(self) -> None:
        super().__init__()

    def extra_repr(self) -> str:
        extra = []

        for name, value in self.__dict__.items():
            if name.startswith("_") or name == "training":
                continue

            if not isinstance(value, (bool, int, float, str, tuple, list, Enum)):
                continue

            extra.append(f"{name}={value}")

        return ", ".join(extra)

    def forward(self, *data: Any) -> Any:
        if len(data) == 1:
            data = data[0]

        flattened_data, spec = tree_flatten(data)

        self.validate(flattened_data)

        params = self.parameters_dict(flattened_data)

        flattened_transformed_data = []

        for inpt in flattened_data:
            if _check_type(inpt, self._transformed_types):
                flattened_transformed_data = [
                    *flattened_transformed_data,
                    self.transform(inpt, params),
                ]
            else:
                flattened_transformed_data = [
                    *flattened_transformed_data,
                    inpt,
                ]

        return tree_unflatten(flattened_transformed_data, spec)

    def parameters_dict(self, flattened_data: list[Any]) -> dict[str, Any]:  # noqa: ARG002
        return {}

    def transform(self, data: Any, parameters: dict[str, Any]) -> Any:
        raise NotImplementedError

    def validate(self, flat_inputs: list[Any]) -> None:
        raise NotImplementedError
