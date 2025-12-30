import torch
from torch import nn


def skip_connection_2d(
    in_channels: int,
    out_channels: int,
    skip_type: str = "soft-gating",
    bias: bool = False,
):
    """
    Skip connection factory for FNO-2D.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    skip_type : {"identity", "linear", "soft-gating"}
        Type of skip connection
    bias : bool
        Whether to use bias (only for linear)

    Returns
    -------
    nn.Module
        Skip connection module
    """
    skip_type = skip_type.lower()

    if skip_type == "identity":
        return nn.Identity()

    elif skip_type == "linear":
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    elif skip_type == "soft-gating":
        if in_channels != out_channels:
            raise ValueError(
                "Soft-gating requires in_channels == out_channels, "
                f"got {in_channels} and {out_channels}"
            )
        return SoftGating2D(in_channels)

    else:
        raise ValueError(
            f"Unknown skip_type='{skip_type}'. "
            "Expected one of ['identity', 'linear', 'soft-gating']"
        )


class SoftGating2D(nn.Module):
    """
    Channel-wise soft gating for 2D inputs.

    Applies:
        y = x * w (+ b)

    where w has shape (1, C, 1, 1)
    """

    def __init__(self, channels: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return x * self.weight + self.bias
        return x * self.weight
