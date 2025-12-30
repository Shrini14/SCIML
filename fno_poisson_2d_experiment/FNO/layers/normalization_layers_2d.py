import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceNorm2D(nn.Module):
    """
    Instance Normalization for 2D FNO.

    Normalizes each sample independently across spatial dimensions.

    Input:
        (B, C, H, W)
    Output:
        (B, C, H, W)
    """

    def __init__(self, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.norm = nn.InstanceNorm2d(
            num_features=None,  # set dynamically
            eps=eps,
            affine=affine,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create norm lazily to know channel count
        if self.norm.num_features is None:
            self.norm.num_features = x.shape[1]
            self.norm.weight = None
            self.norm.bias = None
        return self.norm(x)


class BatchNorm2D(nn.Module):
    """
    Batch Normalization for 2D FNO.

    Normalizes across batch and spatial dimensions.

    Input:
        (B, C, H, W)
    Output:
        (B, C, H, W)
    """

    def __init__(self, num_channels: int, **kwargs):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class AdaIN2D(nn.Module):
    """
    Adaptive Instance Normalization for 2D FNO.

    Conditioning is done using an external embedding vector.

    Input:
        x : (B, C, H, W)
        embedding : (embed_dim,)
    """

    def __init__(
        self,
        embed_dim: int,
        num_channels: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, 2 * num_channels),
        )

        self.embedding = None

    def set_embedding(self, embedding: torch.Tensor):
        self.embedding = embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding is None:
            raise RuntimeError("AdaIN2D: embedding must be set before forward()")

        # Instance norm without affine
        x = F.instance_norm(x, eps=self.eps)

        # Generate scale and bias
        scale, shift = torch.chunk(self.mlp(self.embedding), 2, dim=0)
        scale = scale.view(1, -1, 1, 1)
        shift = shift.view(1, -1, 1, 1)

        return scale * x + shift
