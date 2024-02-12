from torch import Tensor, nn


# TODO: replace with regular nn.LayerNorm
class MaskLayerNorm1d(nn.LayerNorm):
    """
    Transformer-style layer-norm layer
    """

    def forward(self, inp: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x, mask = inp

        xmean = x.mean(dim=-2, keepdim=True)
        xxmean = x.pow(2).mean(dim=-2, keepdim=True)
        var = xxmean - xmean.pow(2)

        std = var.clamp(self.eps) ** 0.5
        ratio = self.weight / std

        output = (x - xmean) * ratio + self.bias

        return output, mask
