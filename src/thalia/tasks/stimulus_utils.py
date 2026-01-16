"""
Task Stimulus Utilities - Common patterns for stimulus generation.

This module consolidates repeated stimulus generation patterns across
task modules, reducing code duplication and ensuring consistency.

Author: Thalia Project
Date: December 12, 2025
"""

from __future__ import annotations

import torch

from thalia.constants.task import (
    PROPRIOCEPTION_NOISE_SCALE,
    SPIKE_PROBABILITY_LOW,
    STIMULUS_NOISE_SCALE,
)


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

    Example:
        >>> stimulus = create_random_stimulus(100, device)
        >>> stimulus.shape
        torch.Size([100])
    """
    return torch.randn(dim, device=device) * std + mean


def add_noise(
    stimulus: torch.Tensor,
    noise_scale: float = STIMULUS_NOISE_SCALE,
) -> torch.Tensor:
    """
    Add Gaussian noise to stimulus.

    Args:
        stimulus: Input stimulus tensor
        noise_scale: Scale of noise to add (default: 0.05)

    Returns:
        Stimulus with added noise

    Example:
        >>> clean = torch.ones(100, device=device)
        >>> noisy = add_noise(clean, noise_scale=0.1)
    """
    return stimulus + torch.randn_like(stimulus) * noise_scale


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

    Example:
        >>> baseline = create_zero_stimulus(100, device)
        >>> (baseline == 0).all()
        True
    """
    return torch.zeros(dim, dtype=dtype, device=device)


def create_motor_spikes(
    n_motor: int,
    device: torch.device,
    spike_probability: float = SPIKE_PROBABILITY_LOW,
) -> torch.Tensor:
    """
    Generate random motor spikes at given probability.

    Uses Bernoulli sampling to create binary spike patterns.

    Args:
        n_motor: Number of motor neurons
        device: Device to create tensor on
        spike_probability: Probability of spike (default: 0.05)

    Returns:
        Binary spike tensor [n_motor], dtype=bool

    Example:
        >>> spikes = create_motor_spikes(50, device, spike_probability=0.1)
        >>> spikes.dtype
        torch.bool
        >>> 0.05 < spikes.float().mean() < 0.15  # ~10% spikes
        True
    """
    return torch.rand(n_motor, device=device) < spike_probability


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

    Example:
        >>> pos = create_random_position(workspace_size=10.0, device=device)
        >>> (pos >= 0).all() and (pos <= 10.0).all()
        True
    """
    return torch.rand(n_dims, device=device) * workspace_size


def add_proprioceptive_noise(
    proprioception: torch.Tensor,
    noise_scale: float = PROPRIOCEPTION_NOISE_SCALE,
) -> torch.Tensor:
    """
    Add noise to proprioceptive feedback.

    Simulates sensory noise in joint angle/position sensing.

    Args:
        proprioception: Clean proprioceptive signal
        noise_scale: Scale of noise (default: 0.1)

    Returns:
        Noisy proprioceptive signal

    Example:
        >>> clean_proprio = torch.tensor([0.5, 0.3], device=device)
        >>> noisy_proprio = add_proprioceptive_noise(clean_proprio)
    """
    return proprioception + torch.randn_like(proprioception) * noise_scale


def create_partial_stimulus(
    dim: int,
    device: torch.device,
    active_fraction: float = 0.5,
    strength: float = 1.0,
) -> torch.Tensor:
    """
    Create stimulus with only fraction of dimensions active.

    Useful for creating sparse patterns or partial cues.

    Args:
        dim: Total dimensionality
        device: Device to create tensor on
        active_fraction: Fraction of dimensions to activate (default: 0.5)
        strength: Strength of active dimensions (default: 1.0)

    Returns:
        Partially active stimulus [dim]

    Example:
        >>> # Half the dimensions active at strength 1.0
        >>> stim = create_partial_stimulus(100, device, active_fraction=0.5)
        >>> (stim != 0).float().mean()  # ~50% non-zero
        tensor(0.5000)
    """
    stimulus = torch.zeros(dim, device=device)
    n_active = int(dim * active_fraction)
    stimulus[:n_active] = strength
    return stimulus
