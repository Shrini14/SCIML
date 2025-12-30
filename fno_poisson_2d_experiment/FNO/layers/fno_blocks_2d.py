import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_conv_2d import SpectralConv2D
from .channel_mlp_2d import ChannelMLP2D
from .skip_connections_2d import skip_connection_2d
from .normalization_layers_2d import InstanceNorm2D


class FNOBlock2D(nn.Module):
    """
    Single Fourier Neural Operator block for 2D problems.

    Structure:
        x -> SpectralConv2D -> + skip -> (norm) -> GELU
          -> ChannelMLP2D   -> + skip -> (norm) -> GELU

    Input
    -----
    x : torch.Tensor
        Shape (B, C, H, W)

    Output
    ------
    y : torch.Tensor
        Shape (B, C, H, W)
    """

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        fno_skip: str = "identity",
        mlp_skip: str = "identity",
        use_norm: bool = False,
    ):
        super().__init__()

        self.residual_scale = residual_scale
        self.use_norm = use_norm

        # Spectral convolution
        self.spectral_conv = SpectralConv2D(
            in_channels=width,
            out_channels=width,
            modes1=modes1,
            modes2=modes2,
        )

        # Channel MLP
        self.channel_mlp = ChannelMLP2D(
            in_channels=width,
            hidden_channels=int(width * mlp_ratio),
            dropout=dropout,
        )

        # Skip connections
        self.fno_skip = skip_connection_2d(
            in_channels=width,
            out_channels=width,
            skip_type=fno_skip,
        )

        self.mlp_skip = skip_connection_2d(
            in_channels=width,
            out_channels=width,
            skip_type=mlp_skip,
        )

        # Optional normalization
        if use_norm:
            self.norm1 = InstanceNorm2D()
            self.norm2 = InstanceNorm2D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----- Spectral block -----
        x_skip = self.fno_skip(x)
        y = self.spectral_conv(x)

        if self.use_norm:
            y = self.norm1(y)

        x = x_skip + self.residual_scale * y
        x = F.gelu(x)

        # ----- Channel MLP block -----
        x_skip = self.mlp_skip(x)
        y = self.channel_mlp(x)

        if self.use_norm:
            y = self.norm2(y)

        x = x_skip + self.residual_scale * y
        x = F.gelu(x)

        return x
