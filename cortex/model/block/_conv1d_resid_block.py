from torch import Tensor, nn

from cortex.model.elemental import MaskLayerNorm1d, swish


class Conv1dResidBlock(nn.Module):
    """
    1D explicit convolution pre-norm residual block with optional layer-norm and swish activation.
    Each block has two convolution layers.
    The order of operations is norm -> act -> conv -> norm -> act -> conv -> add.

    This module expects inputs to be a tuple of (features, padding_mask).

    Dropout is applied to the output of the block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        layernorm: bool = True,
        dropout_p: float = 0.0,
        act_fn: str = "swish",
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
            stride=stride,
            bias=False,
            dilation=dilation,
        )
        self.conv_2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding="same",
            stride=stride,
            bias=False,
            dilation=dilation,
        )
        if layernorm:
            self.norm_1 = MaskLayerNorm1d(normalized_shape=[in_channels, 1])
            self.norm_2 = MaskLayerNorm1d(normalized_shape=[out_channels, 1])
        else:
            self.norm_1 = lambda x: x
            self.norm_2 = lambda x: x

        if act_fn == "swish":
            self.act_fn = swish
        else:
            self.act_fn = nn.ReLU(inplace=True)

        if not in_channels == out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding="same", stride=1)
        else:
            self.proj = None

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """
        assumes inputs are already properly masked
        assumes norm and act are applied pointwise in spatial dimension
        """

        resid, mask = inputs
        x, _ = self.norm_1((resid, mask))
        x = self.act_fn(x)
        x = self.conv_1(x)
        x = mask[:, None] * x

        x, _ = self.norm_2((x, mask))
        x = self.act_fn(x)
        x = self.conv_2(x)
        x = mask[:, None] * x

        if self.proj is not None:
            resid = self.proj(resid)

        x = x + resid

        return self.dropout(x), mask
