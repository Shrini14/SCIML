import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChannelMLP2D(nn.Module):
    """
    Channel-wise MLP for 2D Fourier Neural Operator.

    Applies pointwise (1x1) convolutions over channels at each spatial location.

    Input
    -----
    x : torch.Tensor
        Shape (B, C_in, H, W)

    Output
    ------
    y : torch.Tensor
        Shape (B, C_out, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        n_layers: int = 2,
        non_linearity=F.gelu,
        dropout: float = 0.0,
    ):
        super().__init__()

        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels

        self.n_layers = n_layers
        self.non_linearity = non_linearity

        self.layers = nn.ModuleList()
        self.dropouts = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers - 1)])
            if dropout > 0.0
            else None
        )

        for i in range(n_layers):
            if i == 0 and n_layers == 1:
                self.layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1)
                )
            elif i == 0:
                self.layers.append(
                    nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
                )
            elif i == n_layers - 1:
                self.layers.append(
                    nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
                )
            else:
                self.layers.append(
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C_in, H, W)

        Returns
        -------
        torch.Tensor
            Shape (B, C_out, H, W)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
                if self.dropouts is not None:
                    x = self.dropouts[i](x)
        return x
