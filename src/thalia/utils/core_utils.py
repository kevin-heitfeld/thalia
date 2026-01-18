"""
Core Utilities for THALIA.

This module provides common utility functions used across the codebase
to reduce code duplication and ensure consistency.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from typing import Union

import torch


def ensure_1d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 1D by squeezing or averaging batch dimension.

    For 2D tensors with batch_size=1, squeezes to 1D.
    For 2D tensors with batch_size>1, averages across batch dimension.
    Already 1D tensors are returned unchanged.

    This is useful for operations like torch.outer() that require 1D inputs.

    Args:
        tensor: Input tensor (1D or 2D)

    Returns:
        1D tensor

    Example:
        >>> x = torch.randn(1, 100)  # Shape: [1, 100]
        >>> x = ensure_1d(x)  # Shape: [100]

        >>> y = torch.randn(32, 100)  # Shape: [32, 100]
        >>> y = ensure_1d(y)  # Shape: [100] (averaged across batch)
    """
    if tensor.dim() == 2:
        return tensor.squeeze(0) if tensor.shape[0] == 1 else tensor.mean(dim=0)
    return tensor


def clamp_weights(
    weights: torch.Tensor,
    w_min: float = 0.0,
    w_max: float = 1.0,
    inplace: bool = True,
) -> torch.Tensor:
    """Clamp weight tensor to valid range.

    Standard pattern for enforcing weight bounds after learning updates.
    Operates in-place by default for efficiency.

    Args:
        weights: Weight tensor to clamp
        w_min: Minimum weight value (default: 0.0)
        w_max: Maximum weight value (default: 1.0)
        inplace: If True, modify weights in place (default: True)

    Returns:
        Clamped weight tensor

    Example:
        >>> clamp_weights(self.weights, cfg.w_min, cfg.w_max)
        >>> # Or with .data for nn.Parameter:
        >>> clamp_weights(self.weights.data, cfg.w_min, cfg.w_max)
    """
    if inplace:
        return weights.clamp_(w_min, w_max)
    return weights.clamp(w_min, w_max)


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

    Example:
        >>> sim = cosine_similarity_safe(pattern1, pattern2)
        >>> # For batched: sim has shape [batch_size]
    """
    # Handle 1D vectors
    if a.dim() == 1 and b.dim() == 1:
        norm_a: torch.Tensor = a.norm() + eps
        norm_b: torch.Tensor = b.norm() + eps
        return torch.tensor((a @ b) / (norm_a * norm_b))

    # Handle higher dimensional tensors
    a_norm: torch.Tensor = a / (a.norm(dim=dim, keepdim=True) + eps)
    b_norm: torch.Tensor = b / (b.norm(dim=dim, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=dim)


def zeros_like_config(
    *dims: int,
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a zero tensor with dimensions and device from config.

    Args:
        *dims: Tensor dimensions
        device: Device for the tensor
        dtype: Data type (default: float32)

    Returns:
        Zero tensor with specified shape on specified device

    Example:
        >>> trace = zeros_like_config(n_output, n_input, device=self.device)
    """
    return torch.zeros(*dims, device=torch.device(device), dtype=dtype)


def ones_like_config(
    *dims: int,
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a ones tensor with dimensions and device from config.

    Args:
        *dims: Tensor dimensions
        device: Device for the tensor
        dtype: Data type (default: float32)

    Returns:
        Ones tensor with specified shape on specified device
    """
    return torch.ones(*dims, device=torch.device(device), dtype=dtype)


def assert_single_instance(batch_size: int, context: str = "This component") -> None:
    """Assert that batch_size is 1, enforcing THALIA's single-instance architecture.

    THALIA models a single continuous brain state processing a temporal stream.
    Unlike ML training frameworks that batch independent samples for efficiency,
    THALIA maintains continuous temporal dynamics (membrane potentials, synaptic
    traces, adaptation state, working memory) that cannot be meaningfully batched.

    For parallel evaluation (e.g., in RL training), instantiate multiple instances
    rather than using batch_size > 1.

    Args:
        batch_size: The batch size to validate
        context: Description of where this check occurs (for error message)

    Raises:
        ValueError: If batch_size != 1

    Example:
        >>> def reset_state(self, batch_size: int = 1) -> None:
        ...     assert_single_instance(batch_size, "LayeredCortex")
        ...     # Continue with state initialization
    """
    if batch_size != 1:
        raise ValueError(
            f"{context} only supports batch_size=1, got {batch_size}. "
            "THALIA models a single continuous brain with temporal dynamics "
            "(membrane potentials, synaptic traces, adaptation state, working memory). "
            "For parallel simulations, create multiple instances."
        )


def initialize_phase_preferences(
    n_neurons: int,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Initialize random phase preferences for oscillator-coupled neurons.

    Many cortical neurons have preferred oscillator phases at which they are
    most excitable. This function initializes uniformly distributed random
    phase preferences in the range [0, 2π).

    This consolidates the pattern `torch.rand(n_neurons, device=device) * 2 * torch.pi`
    that appeared in multiple regions (cortex, thalamus, etc.).

    Args:
        n_neurons: Number of neurons requiring phase preferences
        device: Device for tensor creation (CPU or CUDA)

    Returns:
        Phase preference tensor [n_neurons] with values in [0, 2π)

    Example:
        >>> # Instead of:
        >>> phase_prefs = torch.rand(100, device=device) * 2 * torch.pi
        >>>
        >>> # Use:
        >>> from thalia.utils.core_utils import initialize_phase_preferences
        >>> phase_prefs = initialize_phase_preferences(100, device=device)
    """
    device = torch.device(device) if isinstance(device, str) else device
    return torch.rand(n_neurons, device=device) * (2 * math.pi)


__all__ = [
    "ensure_1d",
    "clamp_weights",
    "cosine_similarity_safe",
    "zeros_like_config",
    "ones_like_config",
    "assert_single_instance",
    "initialize_phase_preferences",
]
