import torch
import torch.nn as nn


class SpectralConv2D(nn.Module):
    """
    2D Fourier Spectral Convolution layer for FNO.

    Input:
        x: (B, C_in, H, W)

    Output:
        y: (B, C_out, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Learnable complex weights
        self.weight = nn.Parameter(
            torch.randn(
                in_channels,
                out_channels,
                modes1,
                modes2,
                dtype=torch.cfloat,
            ) * (1.0 / (in_channels * out_channels))
        )

        self.bias = (
            nn.Parameter(torch.zeros(out_channels, 1, 1)) if bias else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # FFT (real-to-complex)
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Clamp modes to available resolution
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)

        # Allocate output Fourier tensor
        out_ft = torch.zeros(
            B,
            self.out_channels,
            H,
            W // 2 + 1,
            device=x.device,
            dtype=x_ft.dtype,
        )

        # Low-frequency spectral multiplication
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :m1, :m2],
            self.weight[:, :, :m1, :m2],
        )

        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")

        if self.bias is not None:
            x = x + self.bias

        return x
