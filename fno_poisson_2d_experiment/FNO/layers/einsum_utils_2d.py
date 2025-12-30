import torch

def einsum_complexhalf(eq, *args):
    """
    Stub for FNO-2D.

    Complex-half precision is NOT used in FNO-2D.
    This function exists only for API compatibility.
    """
    return torch.einsum(eq, *args)
