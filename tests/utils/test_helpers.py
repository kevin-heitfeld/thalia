"""Shared test utilities for Thalia tests.

This module provides common helper functions for test setup and data generation,
eliminating duplication across test files.

Usage:
======
    from tests.utils.test_helpers import generate_sparse_spikes, generate_random_weights

    # Generate binary spike vector with 20% firing rate
    spikes = generate_sparse_spikes(100, firing_rate=0.2, device="cpu")

    # Generate random weight matrix with optional sparsity
    weights = generate_random_weights(64, 128, scale=0.3, sparsity=0.2)

Author: Thalia Project
Date: December 22, 2025
"""

import torch


def generate_sparse_spikes(
    n_neurons: int,
    firing_rate: float = 0.2,
    device: str = "cpu",
    dtype: torch.dtype = torch.bool
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

    Example:
        >>> spikes = generate_sparse_spikes(100, firing_rate=0.2)
        >>> spikes.sum().item()  # Approximately 20 spikes
        20
        >>> spikes.dtype
        torch.bool
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
    positive_only: bool = False
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

    Example:
        >>> weights = generate_random_weights(64, 128, scale=0.3, sparsity=0.2)
        >>> weights.shape
        torch.Size([64, 128])
        >>> (weights == 0).float().mean().item()  # Approximately 0.2
        0.203125
    """
    if positive_only:
        weights = torch.rand(n_output, n_input, device=device) * scale
    else:
        weights = torch.randn(n_output, n_input, device=device) * scale

    if sparsity > 0:
        mask = torch.rand(n_output, n_input, device=device) > sparsity
        weights = weights * mask

    return weights


def generate_batch_spikes(
    batch_size: int,
    n_neurons: int,
    firing_rate: float = 0.2,
    device: str = "cpu"
) -> torch.Tensor:
    """Generate batch of spike vectors.

    Convenience wrapper for generating multiple spike vectors at once.
    Each row in the batch has independent random spikes.

    Args:
        batch_size: Number of spike vectors
        n_neurons: Number of neurons per vector
        firing_rate: Fraction of neurons spiking (0.0-1.0)
        device: Device for tensor

    Returns:
        Batch of spike vectors [batch_size, n_neurons]

    Example:
        >>> spikes = generate_batch_spikes(32, 100, firing_rate=0.2)
        >>> spikes.shape
        torch.Size([32, 100])
    """
    return torch.stack([
        generate_sparse_spikes(n_neurons, firing_rate, device)
        for _ in range(batch_size)
    ])


def create_test_region_config(**overrides):
    """Create a minimal test region configuration.

    Provides sensible defaults for testing, with ability to override
    specific parameters.

    NOTE: This helper maintains backward compatibility with n_input/n_output,
    but new tests should use semantic config patterns directly:
    - Thalamus: ThaliamusConfig(input_size=X, relay_size=Y)
    - Cortex: CortexConfig(input_size=X, layer_sizes=[L4, L23, L5])
    - Hippocampus: HippocampusConfig(input_size=X, ca3_size=Y, ca1_size=Z)
    - Prefrontal: PrefrontalConfig(input_size=X, n_neurons=Y)
    - Striatum: StriatumConfig(n_actions=X, neurons_per_action=Y, input_sources={...})
    - Cerebellum: CerebellumConfig(input_size=X, purkinje_size=Y)

    Args:
        **overrides: Configuration parameters to override

    Returns:
        Configuration dictionary suitable for region initialization

    Example:
        >>> config = create_test_region_config(n_input=128, n_output=64)
        >>> config["device"]
        'cpu'
    """
    defaults = {
        "n_input": 100,
        "n_output": 64,
        "device": "cpu",
        "dt_ms": 1.0,
        "learning_rate": 0.01,
        "enable_learning": True,
    }
    defaults.update(overrides)
    return defaults


__all__ = [
    "generate_sparse_spikes",
    "generate_random_weights",
    "generate_batch_spikes",
    "create_test_region_config",
]
