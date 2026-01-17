"""
Spike Utility Functions.

This module provides common utility functions for working with spike tensors,
centralizing frequently-used operations to eliminate code duplication.

Key Functions:
    - compute_firing_rate: Calculate population firing rate from binary spikes
    - compute_spike_count: Count total spikes in a tensor
    - compute_isi: Calculate inter-spike intervals

These utilities work with binary spike tensors (0 or 1) and handle edge cases
like empty tensors or all-zero spike trains.
"""

from __future__ import annotations

import torch


def compute_firing_rate(spikes: torch.Tensor) -> float:
    """Compute population firing rate from binary spike tensor.

    Calculates the fraction of neurons firing in the population. This is
    commonly used for diagnostics, homeostatic regulation, and health monitoring.

    Args:
        spikes: Binary spike tensor (any shape: [n_neurons], [batch, n_neurons], etc.)
               Values should be 0 (no spike) or 1 (spike)

    Returns:
        Firing rate as a fraction (0.0 to 1.0), where:
        - 0.0 = no neurons firing
        - 1.0 = all neurons firing
        - 0.05 = 5% of neurons firing (typical sparse rate)

    Example:
        >>> spikes = torch.tensor([1, 0, 0, 1, 0])  # 2 out of 5 neurons firing
        >>> compute_firing_rate(spikes)
        0.4

        >>> empty_spikes = torch.tensor([])
        >>> compute_firing_rate(empty_spikes)
        0.0

    Notes:
        - Returns 0.0 for empty tensors (numel() == 0)
        - Works with any shape tensor (flattens internally)
        - Result is always a Python float for easy logging/comparison
    """
    if spikes.numel() == 0:
        return 0.0
    return spikes.float().mean().item()


def compute_spike_count(spikes: torch.Tensor) -> int:
    """Count total number of spikes in tensor.

    Args:
        spikes: Binary spike tensor (any shape)

    Returns:
        Total number of spikes (sum of all 1s)

    Example:
        >>> spikes = torch.tensor([1, 0, 1, 1, 0])
        >>> compute_spike_count(spikes)
        3
    """
    if spikes.numel() == 0:
        return 0
    return int(spikes.sum().item())


def compute_spike_density(spikes: torch.Tensor, window_size: int) -> torch.Tensor:
    """Compute local spike density using sliding window.

    Args:
        spikes: Binary spike tensor [time, neurons]
        window_size: Size of sliding window in timesteps

    Returns:
        Spike density tensor [time, neurons] with same shape as input

    Example:
        >>> spikes = torch.tensor([[1, 0], [1, 1], [0, 1]])
        >>> density = compute_spike_density(spikes, window_size=2)
    """
    if spikes.numel() == 0 or window_size < 1:
        return torch.zeros_like(spikes)

    # Use 1D convolution for efficient sliding window
    if spikes.dim() == 1:
        spikes = spikes.unsqueeze(0).unsqueeze(0)  # [1, 1, time]
        kernel = torch.ones(1, 1, window_size, device=spikes.device) / window_size
        density = torch.nn.functional.conv1d(spikes.float(), kernel, padding=window_size // 2)
        return density.squeeze()
    elif spikes.dim() == 2:
        # [time, neurons] → [neurons, 1, time]
        spikes_t = spikes.t().unsqueeze(1)
        kernel = torch.ones(1, 1, window_size, device=spikes.device) / window_size
        density = torch.nn.functional.conv1d(spikes_t.float(), kernel, padding=window_size // 2)
        return density.squeeze(1).t()
    else:
        raise ValueError(f"Expected 1D or 2D spike tensor, got {spikes.dim()}D")


def is_silent(spikes: torch.Tensor, threshold: float = 0.001) -> bool:
    """Check if spike tensor is effectively silent.

    Args:
        spikes: Binary spike tensor (any shape)
        threshold: Firing rate threshold below which population is considered silent
                  Default 0.001 = 0.1% firing rate

    Returns:
        True if firing rate is below threshold (population is silent)

    Example:
        >>> sparse_spikes = torch.zeros(1000)
        >>> sparse_spikes[0] = 1  # Only 1 out of 1000 neurons firing
        >>> is_silent(sparse_spikes, threshold=0.01)  # 0.1% < 1% threshold
        True
    """
    return compute_firing_rate(spikes) < threshold


def is_saturated(spikes: torch.Tensor, threshold: float = 0.95) -> bool:
    """Check if spike tensor is saturated (too many neurons firing).

    Args:
        spikes: Binary spike tensor (any shape)
        threshold: Firing rate threshold above which population is considered saturated
                  Default 0.95 = 95% firing rate

    Returns:
        True if firing rate exceeds threshold (population is saturated)

    Example:
        >>> dense_spikes = torch.ones(1000)
        >>> dense_spikes[0:10] = 0  # 99% firing
        >>> is_saturated(dense_spikes)
        True
    """
    return compute_firing_rate(spikes) > threshold
