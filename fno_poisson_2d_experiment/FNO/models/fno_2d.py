from typing import Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from ..layers.fno_blocks_2d import FNOBlock2D
from ..layers.channel_mlp_2d import ChannelMLP2D
from ..layers.embeddings import GridEmbedding2D


Number = Union[int, float]


class FNO2D(BaseModel, name="FNO2D"):
    """
    2D Fourier Neural Operator.

    Pipeline:
        input
          -> (optional) positional embedding
          -> lifting (ChannelMLP)
          -> stacked FNOBlock2D
          -> projection (ChannelMLP)
          -> output

    Input shape:
        (B, in_channels, H, W)

    Output shape:
        (B, out_channels, H, W)
    """

    def __init__(
        self,
        modes1: int,
        modes2: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_ratio: Number = 2.0,
        projection_ratio: Number = 2.0,
        positional_embedding: Union[str, nn.Module] = "grid",
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        fno_skip: Literal["identity", "linear", "soft-gating"] = "identity",
        mlp_skip: Literal["identity", "linear", "soft-gating"] = "identity",
        use_norm: bool = False,
    ):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        # -------------------------
        # Positional embedding
        # -------------------------
        if positional_embedding == "grid":
            self.positional_embedding = GridEmbedding2D(
                in_channels=in_channels,
                grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
            )
            lifting_in_channels = in_channels + 2
        elif positional_embedding is None:
            self.positional_embedding = None
            lifting_in_channels = in_channels
        elif isinstance(positional_embedding, nn.Module):
            self.positional_embedding = positional_embedding
            lifting_in_channels = in_channels + 2
        else:
            raise ValueError(
                "positional_embedding must be 'grid', nn.Module, or None"
            )

        # -------------------------
        # Lifting layer
        # -------------------------
        lifting_channels = int(lifting_ratio * hidden_channels)
        self.lifting = ChannelMLP2D(
            in_channels=lifting_in_channels,
            out_channels=hidden_channels,
            hidden_channels=lifting_channels,
            n_layers=2,
        )

        # -------------------------
        # FNO blocks
        # -------------------------
        self.blocks = nn.ModuleList(
            [
                FNOBlock2D(
                    width=hidden_channels,
                    modes1=modes1,
                    modes2=modes2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    fno_skip=fno_skip,
                    mlp_skip=mlp_skip,
                    use_norm=use_norm,
                )
                for _ in range(n_layers)
            ]
        )

        # -------------------------
        # Projection layer
        # -------------------------
        projection_channels = int(projection_ratio * hidden_channels)
        self.projection = ChannelMLP2D(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, in_channels, H, W)

        Returns
        -------
        torch.Tensor
            Shape (B, out_channels, H, W)
        """
        # Positional embedding
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        # Lifting
        x = self.lifting(x)

        # FNO blocks
        for block in self.blocks:
            x = block(x)

        # Projection
        x = self.projection(x)

        return x
