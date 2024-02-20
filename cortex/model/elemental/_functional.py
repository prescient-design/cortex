from torch import Tensor


def identity(x: Tensor) -> Tensor:
    """
    This function returns its input.
    """
    return x


def permute_spatial_channel_dims(x: Tensor) -> Tensor:
    """
    Permute the last two dimensions of a 3D tensor.
    """
    return x.permute(0, 2, 1)


def swish(x: Tensor) -> Tensor:
    """
    Swish activation function.
    """
    return x * x.sigmoid()
