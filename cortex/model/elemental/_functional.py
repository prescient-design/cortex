from torch import Tensor


def identity(x: Tensor) -> Tensor:
    return x


def permute_spatial_channel_dims(x: Tensor) -> Tensor:
    return x.permute(0, 2, 1)


def swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()
