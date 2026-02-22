"""
Spike Utility Functions.

This module provides common utility functions for working with spike tensors,
centralizing frequently-used operations to eliminate code duplication.

These utilities work with binary spike tensors (0 or 1) and handle edge cases
like empty tensors or all-zero spike trains.
"""

from __future__ import annotations

import torch


def validate_spike_tensor(spikes: torch.Tensor, tensor_name: str = "<unspecified>") -> None:
    """Validate that a tensor is a binary spike tensor (0s and 1s) .

    Args:
        spikes: Tensor to validate
        tensor_name: Name of the tensor for error messages

    Raises:
        ValueError: If the tensor is not binary (contains values other than 0 or 1)
    """
    if not spikes.dim() == 1:
        raise ValueError(f"{tensor_name} must be a 1D tensor (ADR-005), got shape {spikes.shape}.")

    if not torch.all((spikes == 0) | (spikes == 1)):
        raise ValueError(f"{tensor_name} must be a binary spike tensor (ADR-005) containing only 0s and 1s.")


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

    Notes:
        - Returns 0.0 for empty tensors (numel() == 0)
        - Works with any shape tensor (flattens internally)
        - Result is always a Python float for easy logging/comparison
    """
    if spikes is None or spikes.numel() == 0:
        return 0.0
    return float(spikes.float().mean().item())


def compute_spike_count(spikes: torch.Tensor) -> int:
    """Count total number of spikes in tensor.

    Args:
        spikes: Binary spike tensor (any shape)

    Returns:
        Total number of spikes (sum of all 1s)
    """
    if spikes.numel() == 0:
        return 0
    return int(spikes.sum().item())


def cosine_similarity_safe(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-8,
    dim: int = -1,
) -> torch.Tensor:
    """Compute cosine similarity with safe epsilon handling.

    Provides consistent epsilon handling for numerical stability.
    Works with both 1D vectors and batched tensors.

    Args:
        a: First tensor
        b: Second tensor
        eps: Small constant for numerical stability (default: 1e-8)
        dim: Dimension along which to compute similarity (default: -1)

    Returns:
        Cosine similarity value(s)
    """
    # Convert bool tensors to float (binary spikes → 0.0/1.0)
    if a.dtype == torch.bool:
        a = a.float()
    if b.dtype == torch.bool:
        b = b.float()

    # Handle 1D vectors
    if a.dim() == 1 and b.dim() == 1:
        norm_a: torch.Tensor = a.norm() + eps
        norm_b: torch.Tensor = b.norm() + eps
        return torch.tensor((a @ b) / (norm_a * norm_b))

    # Handle higher dimensional tensors
    a_norm: torch.Tensor = a / (a.norm(dim=dim, keepdim=True) + eps)
    b_norm: torch.Tensor = b / (b.norm(dim=dim, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=dim)
