from torch import nn


class MLP(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        bias: bool = False,
        dropout_p: float = 0.0,
    ):
        out_channels = out_channels if out_channels else in_channels
        super().__init__(
            nn.Linear(in_channels, 4 * in_channels, bias=bias),
            nn.GELU(),
            nn.Linear(4 * in_channels, out_channels, bias=bias),
            nn.Dropout(dropout_p),
        )
