"""Shared test utilities for Thalia tests.

This module provides common helper functions for test setup and data generation,
eliminating duplication across test files.

Usage:
======
    from tests.utils.test_helpers import (
        generate_sparse_spikes,
        generate_random_weights,
        create_test_brain,
        create_minimal_thalia_config,
    )

    # Generate binary spike vector with 20% firing rate
    spikes = generate_sparse_spikes(100, firing_rate=0.2, device="cpu")

    # Generate random weight matrix with optional sparsity
    weights = generate_random_weights(64, 128, scale=0.3, sparsity=0.2)

    # Create minimal test brain
    brain = create_test_brain(regions=["thalamus", "cortex", "hippocampus"])

    # Create custom config
    config = create_minimal_thalia_config(input_size=64, cortex_size=128)

Author: Thalia Project
Date: December 22, 2025
Updated: January 17, 2026 (Task 2.4 - Extract Common Testing Patterns)
"""

from typing import List, Optional

import torch

from thalia.config import BrainConfig, GlobalConfig, RegionSizes, ThaliaConfig
from thalia.core.brain_builder import BrainBuilder


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
    batch_size: int, n_neurons: int, firing_rate: float = 0.2, device: str = "cpu"
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
    return torch.stack(
        [generate_sparse_spikes(n_neurons, firing_rate, device) for _ in range(batch_size)]
    )


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


def create_minimal_thalia_config(
    device: str = "cpu",
    dt_ms: float = 1.0,
    input_size: int = 10,
    thalamus_size: int = 20,
    cortex_size: int = 30,
    hippocampus_size: int = 40,
    pfc_size: int = 20,
    n_actions: int = 5,
    **overrides,
) -> "ThaliaConfig":
    """Create minimal ThaliaConfig for testing.

    Provides sensible defaults for integration tests that need a full brain.
    All size parameters can be overridden.

    Args:
        device: Device for computations ("cpu" or "cuda")
        dt_ms: Timestep in milliseconds
        input_size: Input dimension
        thalamus_size: Thalamus relay neurons
        cortex_size: Cortex neurons (distributed across L4/L2-3/L5)
        hippocampus_size: Hippocampus neurons (distributed across DG/CA3/CA1)
        pfc_size: Prefrontal cortex neurons
        n_actions: Number of actions for striatum
        **overrides: Additional parameters to override

    Returns:
        ThaliaConfig instance with minimal settings

    Example:
        >>> config = create_minimal_thalia_config(input_size=64, cortex_size=128)
        >>> brain = BrainBuilder.preset("default", config.global_)
    """
    config = ThaliaConfig(
        global_=GlobalConfig(device=device, dt_ms=dt_ms),
        brain=BrainConfig(
            sizes=RegionSizes(
                input_size=input_size,
                thalamus_size=thalamus_size,
                cortex_size=cortex_size,
                hippocampus_size=hippocampus_size,
                pfc_size=pfc_size,
                n_actions=n_actions,
            ),
        ),
    )

    # Apply any additional overrides to brain config
    for key, value in overrides.items():
        if hasattr(config.brain.sizes, key):
            setattr(config.brain.sizes, key, value)

    return config


def create_test_brain(
    regions: Optional[List[str]] = None, device: str = "cpu", **config_overrides
) -> "DynamicBrain":
    """Create minimal DynamicBrain for testing.

    Convenience wrapper that creates a ThaliaConfig and DynamicBrain in one call.
    Useful for integration tests that need a functioning brain without custom setup.

    Args:
        regions: List of region names to include (None = all regions)
        device: Device for computations
        **config_overrides: Parameters passed to create_minimal_thalia_config()

    Returns:
        Initialized DynamicBrain instance

    Example:
        >>> brain = create_test_brain(regions=["thalamus", "cortex"])
        >>> brain = create_test_brain(input_size=64, cortex_size=128)
        >>> brain = create_test_brain(device="cuda" if torch.cuda.is_available() else "cpu")
    """
    config = create_minimal_thalia_config(device=device, **config_overrides)
    brain = BrainBuilder.preset(
        "default",
        global_config=config.global_,
        thalamus_relay_size=config.brain.sizes.thalamus_size,
        cortex_size=config.brain.sizes.cortex_size,
        pfc_n_neurons=config.brain.sizes.pfc_size,
        striatum_actions=config.brain.sizes.n_actions,
    )

    # Note: region filtering would require surgery module, kept simple for now
    # If users need specific regions, they can use BrainBuilder or surgery tools
    return brain


def create_test_spike_input(
    n_neurons: int, n_timesteps: int = 10, firing_rate: float = 0.2, device: str = "cpu"
) -> torch.Tensor:
    """Create temporal spike sequence for testing.

    Generates a sequence of spike vectors over time, useful for testing
    temporal dynamics and learning.

    Args:
        n_neurons: Number of neurons
        n_timesteps: Length of sequence
        firing_rate: Fraction of neurons spiking per timestep
        device: Device for tensor

    Returns:
        Spike sequence [n_timesteps, n_neurons] with binary spikes

    Example:
        >>> spikes = create_test_spike_input(100, n_timesteps=20, firing_rate=0.15)
        >>> spikes.shape
        torch.Size([20, 100])
        >>> spikes.dtype
        torch.bool
    """
    return torch.stack(
        [generate_sparse_spikes(n_neurons, firing_rate, device) for _ in range(n_timesteps)]
    )


def create_test_checkpoint_path(tmp_path: "pathlib.Path", name: str = "test_checkpoint") -> str:
    """Create temporary checkpoint file path for testing.

    Helper for tests that need to save/load checkpoints. Uses pytest's tmp_path
    fixture to ensure cleanup.

    Args:
        tmp_path: pytest tmp_path fixture
        name: Checkpoint file name (without extension)

    Returns:
        Full path string for checkpoint file

    Example:
        >>> def test_checkpoint_save(tmp_path):
        ...     ckpt_path = create_test_checkpoint_path(tmp_path, "my_test")
        ...     region.save_checkpoint(ckpt_path)
        ...     assert Path(ckpt_path).exists()
    """
    checkpoint_path = tmp_path / f"{name}.pt"
    return str(checkpoint_path)


__all__ = [
    "generate_sparse_spikes",
    "generate_random_weights",
    "generate_batch_spikes",
    "create_test_region_config",
    "create_minimal_thalia_config",
    "create_test_brain",
    "create_test_spike_input",
    "create_test_checkpoint_path",
]
