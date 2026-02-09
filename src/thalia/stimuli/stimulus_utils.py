"""
Task Stimulus Utilities - Common patterns for stimulus generation.

This module consolidates repeated stimulus generation patterns across
task modules, reducing code duplication and ensuring consistency.
"""

from __future__ import annotations

import torch

from thalia.components.synapses import WeightInitializer


def create_random_stimulus(
    dim: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """
    Create random stimulus pattern with Gaussian noise.

    Args:
        dim: Dimensionality of stimulus
        device: Device to create tensor on
        mean: Mean of distribution (default: 0.0)
        std: Standard deviation of distribution (default: 1.0)

    Returns:
        Random stimulus tensor [dim]
    """
    return WeightInitializer.gaussian(
        n_output=dim, n_input=1, mean=mean, std=std, device=str(device)
    ).squeeze()


def create_zero_stimulus(
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create zero-valued stimulus (silence/baseline).

    Args:
        dim: Dimensionality of stimulus
        device: Device to create tensor on
        dtype: Data type (default: float32)

    Returns:
        Zero tensor [dim]
    """
    return torch.zeros(dim, dtype=dtype, device=device)


def create_random_position(
    workspace_size: float,
    device: torch.device,
    n_dims: int = 2,
) -> torch.Tensor:
    """
    Create random position within workspace.

    Args:
        workspace_size: Size of workspace (assumes square/cubic space)
        device: Device to create tensor on
        n_dims: Number of spatial dimensions (default: 2)

    Returns:
        Random position tensor [n_dims] in range [0, workspace_size]
    """
    return torch.rand(n_dims, device=device) * workspace_size


def add_proprioceptive_noise(
    proprioception: torch.Tensor,
    noise_scale: float = 0.1,
) -> torch.Tensor:
    """
    Add noise to proprioceptive feedback.

    Simulates sensory noise in joint angle/position sensing.

    Args:
        proprioception: Clean proprioceptive signal
        noise_scale: Scale of noise (default: 0.1)

    Returns:
        Noisy proprioceptive signal
    """
    return proprioception + torch.randn_like(proprioception) * noise_scale
