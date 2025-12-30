from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn


# ============================================================
# Base Embedding Interface
# ============================================================

class Embedding(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def out_channels(self):
        pass


# ============================================================
# Grid Embedding (2D)
# ============================================================

class GridEmbedding2D(Embedding):
    """
    2D grid positional embedding.

    Input:
        (B, C, H, W)

    Output:
        (B, C + 2, H, W)
    """

    def __init__(self, in_channels: int, grid_boundaries=[[0.0, 1.0], [0.0, 1.0]]):
        super().__init__()
        self.in_channels = in_channels
        self.grid_boundaries = grid_boundaries

        self._grid = None
        self._resolution = None

    @property
    def out_channels(self):
        return self.in_channels + 2

    def _build_grid(self, spatial_dims, device, dtype):
        H, W = spatial_dims
        x0, x1 = self.grid_boundaries[0]
        y0, y1 = self.grid_boundaries[1]

        gx = torch.linspace(x0, x1, H, device=device, dtype=dtype)
        gy = torch.linspace(y0, y1, W, device=device, dtype=dtype)

        grid_x, grid_y = torch.meshgrid(gx, gy, indexing="ij")

        grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        return grid_x, grid_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Shape (B, C+2, H, W)
        """
        B, _, H, W = x.shape
        spatial_dims = (H, W)

        if self._grid is None or self._resolution != spatial_dims:
            self._grid = self._build_grid(
                spatial_dims, x.device, x.dtype
            )
            self._resolution = spatial_dims

        grid_x, grid_y = self._grid
        grid_x = grid_x.expand(B, -1, -1, -1)
        grid_y = grid_y.expand(B, -1, -1, -1)

        return torch.cat([x, grid_x, grid_y], dim=1)


class SinusoidalEmbedding(Embedding):
    def __init__(
        self,
        in_channels: int,
        num_frequencies: int,
        embedding_type: str = "transformer",
        max_positions: int = 10000,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_frequencies = num_frequencies
        self.embedding_type = embedding_type
        self.max_positions = max_positions

    @property
    def out_channels(self):
        return 2 * self.num_frequencies * self.in_channels

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        if self.embedding_type == "nerf":
            freqs = 2 ** torch.arange(
                self.num_frequencies, device=x.device
            ) * torch.pi
        else:
            freqs = torch.arange(
                self.num_frequencies, device=x.device
            ) / self.num_frequencies
            freqs = (1 / self.max_positions) ** freqs

        x = torch.einsum("bij,k->bijk", x, freqs)
        x = torch.stack([x.sin(), x.cos()], dim=-1)
        x = x.view(x.shape[0], x.shape[1], -1)

        return x.squeeze(0) if squeeze else x




def rotate_half(x):
    x = x.reshape(*x.shape[:-1], 2, -1)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim, min_freq=1 / 64, scale=1.0):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.min_freq = min_freq
        self.scale = scale

    def forward(self, coordinates):
        coordinates = coordinates * (self.scale / self.min_freq)
        freqs = torch.einsum("...i,j->...ij", coordinates, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)
