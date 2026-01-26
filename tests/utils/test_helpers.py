"""Shared test utilities for Thalia tests.

This module provides common helper functions for test setup and data generation,
eliminating duplication across test files.

Author: Thalia Project
Date: December 22, 2025
Updated: January 17, 2026 (Extract Common Testing Patterns)
"""

import torch

from thalia.config import BrainConfig
from thalia.core.brain_builder import BrainBuilder
from thalia.core.dynamic_brain import DynamicBrain


def generate_sparse_spikes(
    n_neurons: int, firing_rate: float = 0.2, device: str = "cpu", dtype: torch.dtype = torch.bool
) -> torch.Tensor:
    """Generate binary spike vector with specified firing rate.

    Creates a binary tensor where the fraction of True values equals
    the specified firing rate. Useful for generating sparse spike inputs
    in tests.

    Args:
        n_neurons: Number of neurons
        firing_rate: Fraction of neurons spiking (0.0-1.0)
        device: Device for tensor ("cpu" or "cuda")
        dtype: Data type for output (default: torch.bool for spikes)

    Returns:
        Binary spike tensor [n_neurons] with specified firing rate
    """
    threshold = 1.0 - firing_rate
    spikes = torch.rand(n_neurons, device=device) > threshold
    if dtype != torch.bool:
        spikes = spikes.to(dtype)
    return spikes


def generate_random_weights(
    n_output: int,
    n_input: int,
    scale: float = 0.5,
    sparsity: float = 0.0,
    device: str = "cpu",
    positive_only: bool = False,
) -> torch.Tensor:
    """Generate random weight matrix with optional sparsity.

    Creates a weight matrix with Gaussian random values, optionally
    masked for sparsity. Useful for initializing test weight matrices.

    Args:
        n_output: Output dimension (rows)
        n_input: Input dimension (columns)
        scale: Weight scale factor (multiplies randn output)
        sparsity: Fraction of zero connections (0.0-1.0)
            0.0 = fully connected, 0.3 = 30% zeros, 1.0 = all zeros
        device: Device for tensor ("cpu" or "cuda")
        positive_only: If True, use rand() instead of randn() for positive weights

    Returns:
        Weight matrix [n_output, n_input]
    """
    if positive_only:
        weights = torch.rand(n_output, n_input, device=device) * scale
    else:
        weights = torch.randn(n_output, n_input, device=device) * scale

    if sparsity > 0:
        mask = torch.rand(n_output, n_input, device=device) > sparsity
        weights = weights * mask

    return weights


def create_test_brain(device: str = "cpu", **config_overrides) -> "DynamicBrain":
    """Create minimal DynamicBrain for testing.

    Useful for integration tests that need a functioning brain without custom setup.

    Args:
        device: Device for computations
        **config_overrides: Size parameters passed to BrainBuilder.preset()

    Returns:
        Initialized DynamicBrain instance
    """
    brain_config=BrainConfig(device=device, dt_ms=1.0)
    brain = BrainBuilder.preset("default", brain_config=brain_config, **config_overrides)

    return brain


__all__ = [
    "generate_sparse_spikes",
    "generate_random_weights",
    "create_test_brain",
]
