import numpy as np
import torch


def generate_poisson_sample(H=64, W=64):
    """
    Generate one Poisson-2D sample:
    -Δu = f on [0,1]² with zero BC
    """

    x = np.linspace(0, 1, H)
    y = np.linspace(0, 1, W)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Smooth forcing
    f = (
        np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        + 0.3 * np.sin(4 * np.pi * X) * np.sin(3 * np.pi * Y)
    )

    # Analytical solution (for this f)
    u = (
        (1 / (8 * np.pi**2)) * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        + (0.3 / (25 * np.pi**2)) * np.sin(4 * np.pi * X) * np.sin(3 * np.pi * Y)
    )

    f = torch.tensor(f, dtype=torch.float32).unsqueeze(0)
    u = torch.tensor(u, dtype=torch.float32).unsqueeze(0)

    return f, u


def create_dataset(N=8, H=64, W=64):
    f_list, u_list = [], []

    for _ in range(N):
        f, u = generate_poisson_sample(H, W)
        f_list.append(f)
        u_list.append(u)

    f = torch.stack(f_list)  # (N,1,H,W)
    u = torch.stack(u_list)

    return f, u
