# pyright: strict
"""Weight Initialization Registry - Biologically-Motivated Initialization Strategies.

This module provides a unified registry of weight initialization methods used
across brain regions and pathways, ensuring consistent and biologically-plausible
initial synaptic weights.

**Biological Motivation**:
=========================
Initial synaptic weights critically influence network dynamics and learning:

1. **Symmetry Breaking**: Random initialization prevents identical neurons
   (avoids "dead" or redundant units)
2. **Proper Scale**: Prevents saturation (too large) or silence (too small)
3. **Sparse Connectivity**: Mimics cortical connectivity (~5-15% connection probability)
4. **Spatial Structure**: Topographic maps reflect biological organization
5. **Balanced E/I**: Initial balance prevents runaway excitation

Common Patterns in Thalia:
===========================

1. **Gaussian** - torch.randn() * scale + mean
   - Used by: LayeredCortex, Cerebellum, most regions
   - Good for: General purpose, symmetric initialization

2. **Uniform** - torch.rand() * scale
   - Used by: Hippocampus, Striatum
   - Good for: Positive-only weights, bounded initialization

3. **Xavier/Glorot** - Scale by 1/sqrt(fan_in)
   - Used by: Prefrontal, Striatum (fan-in scaling)
   - Good for: Deep networks, maintaining gradient scale

4. **Kaiming/He** - Scale by sqrt(2/fan_in)
   - Used by: Less common in Thalia (ReLU-focused)
   - Good for: ReLU activations

5. **Sparse Random** - Random weights with connectivity mask
   - Used by: Hippocampus trisynaptic circuit
   - Good for: Biological realism, pattern separation

6. **Topographic** - Gaussian connectivity based on spatial distance
   - Used by: Custom pathways with topographic=True
   - Good for: Spatial maps (visual, auditory)

7. **Orthogonal** - nn.init.orthogonal_()
   - Used by: Attention mechanisms
   - Good for: Preserving norms, avoiding correlations

Usage Example:
==============
    from thalia.core.weight_init import WeightInitializer, InitStrategy

    # Using registry
    initializer = WeightInitializer.get(InitStrategy.XAVIER)
    weights = initializer(n_output=64, n_input=128, device='cpu')

    # Direct initialization
    weights = WeightInitializer.xavier(n_output=64, n_input=128)

    # Custom parameters
    weights = WeightInitializer.sparse_random(
        n_output=100, n_input=200, sparsity=0.3, normalize_rows=True
    )

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import Callable, Dict

import torch
import torch.nn as nn

from thalia.core.errors import ConfigurationError


class InitStrategy(Enum):
    """Weight initialization strategies."""

    GAUSSIAN = auto()  # Gaussian (normal) distribution
    UNIFORM = auto()  # Uniform distribution
    XAVIER = auto()  # Xavier/Glorot initialization
    KAIMING = auto()  # Kaiming/He initialization
    SPARSE_RANDOM = auto()  # Sparse random connectivity
    TOPOGRAPHIC = auto()  # Topographic (spatial) connectivity
    ORTHOGONAL = auto()  # Orthogonal initialization
    ZEROS = auto()  # All zeros
    ONES = auto()  # All ones
    IDENTITY = auto()  # Identity matrix
    CONSTANT = auto()  # Constant value


class WeightInitializer:
    """
    Centralized weight initialization registry.

    Provides standardized initialization methods used across Thalia.
    All methods return torch.Tensor, not nn.Parameter.
    """

    # Registry of initialization functions
    _registry: Dict[InitStrategy, Callable] = {}

    @classmethod
    def register(cls, strategy: InitStrategy):
        """Decorator to register an initialization function."""

        def decorator(func: Callable) -> Callable:
            cls._registry[strategy] = func
            return func

        return decorator

    @classmethod
    def get(cls, strategy: InitStrategy) -> Callable:
        """Get initialization function by strategy."""
        if strategy not in cls._registry:
            available = ", ".join([s.name for s in InitStrategy])
            raise ConfigurationError(
                f"Unknown initialization strategy: '{strategy}'. "
                f"Available strategies: {available}. "
                f"Use InitStrategy enum (e.g., InitStrategy.XAVIER) or register a custom strategy."
            )
        return cls._registry[strategy]

    @staticmethod
    def gaussian(
        n_output: int,
        n_input: int,
        mean: float = 0.0,
        std: float = 0.1,
        device: str = "cpu",
        **kwargs,
    ) -> torch.Tensor:
        """
        Gaussian (normal) distribution initialization.

        weights ~ N(mean, std²)

        Args:
            n_output: Number of output neurons
            n_input: Number of input neurons
            mean: Mean of distribution
            std: Standard deviation
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input]
        """
        return torch.randn(n_output, n_input, device=device, requires_grad=False) * std + mean

    @staticmethod
    def uniform(
        n_output: int,
        n_input: int,
        low: float = 0.0,
        high: float = 1.0,
        device: str = "cpu",
        **kwargs,
    ) -> torch.Tensor:
        """
        Uniform distribution initialization.

        weights ~ U(low, high)

        Args:
            n_output: Number of output neurons
            n_input: Number of input neurons
            low: Lower bound
            high: Upper bound
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input]
        """
        return (
            torch.rand(n_output, n_input, device=device, requires_grad=False) * (high - low) + low
        )

    @staticmethod
    def xavier(
        n_output: int, n_input: int, gain: float = 1.0, device: str = "cpu", **kwargs
    ) -> torch.Tensor:
        """
        Xavier/Glorot initialization.

        Maintains variance across layers in deep networks.
        Scale = gain * sqrt(2 / (fan_in + fan_out))

        Good for: tanh, sigmoid activations

        Args:
            n_output: Number of output neurons
            n_input: Number of input neurons
            gain: Scaling factor (1.0 for tanh, 5/3 for sigmoid)
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input]
        """
        std = gain * math.sqrt(2.0 / (n_input + n_output))
        return torch.randn(n_output, n_input, device=device, requires_grad=False) * std

    @staticmethod
    def kaiming(
        n_output: int,
        n_input: int,
        mode: str = "fan_in",
        nonlinearity: str = "relu",
        device: str = "cpu",
        **kwargs,
    ) -> torch.Tensor:
        """
        Kaiming/He initialization.

        Maintains variance for ReLU-like activations.
        Scale = sqrt(2 / fan) where fan = fan_in or fan_out

        Good for: ReLU, LeakyReLU activations

        Args:
            n_output: Number of output neurons
            n_input: Number of input neurons
            mode: 'fan_in' or 'fan_out'
            nonlinearity: 'relu' or 'leaky_relu'
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input]
        """
        fan = n_input if mode == "fan_in" else n_output
        gain = math.sqrt(2.0) if nonlinearity == "relu" else math.sqrt(2.0 / (1 + 0.01**2))
        std = gain / math.sqrt(fan)
        return torch.randn(n_output, n_input, device=device, requires_grad=False) * std

    @staticmethod
    def sparse_random(
        n_output: int,
        n_input: int,
        sparsity: float = 0.3,
        weight_scale: float = 0.1,
        normalize_rows: bool = False,
        device: str = "cpu",
        **kwargs,
    ) -> torch.Tensor:
        """
        Sparse random connectivity initialization.

        Creates biological sparse connectivity patterns.
        Each output neuron connects to a random subset of inputs.

        Args:
            n_output: Number of output neurons
            n_input: Number of input neurons
            sparsity: Fraction of connections to keep (0-1)
            weight_scale: Scale of random weights
            normalize_rows: If True, normalize each row to consistent sum
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity
        """
        # Create random connectivity mask
        mask = torch.rand(n_output, n_input, device=device, requires_grad=False) < sparsity

        # Random weights where connected
        weights = torch.rand(n_output, n_input, device=device, requires_grad=False) * weight_scale
        weights = weights * mask.float()

        if normalize_rows:
            # Normalize rows for consistent input strength
            row_sums = weights.sum(dim=1, keepdim=True) + 1e-6
            target_sum = n_input * sparsity * weight_scale * 0.5
            weights = weights / row_sums * target_sum

        return weights

    @staticmethod
    def topographic(
        n_output: int,
        n_input: int,
        base_weight: float = 0.1,
        sigma_factor: float = 4.0,
        boost_strength: float = 0.3,
        device: str = "cpu",
        **kwargs,
    ) -> torch.Tensor:
        """
        Topographic (spatial) connectivity initialization.

        Creates Gaussian connectivity based on spatial proximity.
        Neurons close in index space have stronger connections.

        Mimics retinotopic maps in visual cortex, tonotopic maps
        in auditory cortex, etc.

        Args:
            n_output: Number of output neurons
            n_input: Number of input neurons
            base_weight: Base weight value
            sigma_factor: Controls width of Gaussian (larger = wider)
            boost_strength: Strength of topographic boost
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with topographic structure
        """
        weights = torch.ones(n_output, n_input, device=device, requires_grad=False) * base_weight

        sigma = n_output / sigma_factor

        for src_idx in range(n_input):
            # Map input to output space
            center = int((src_idx / n_input) * n_output)

            for tgt_idx in range(n_output):
                # Distance in output space (with wraparound)
                dist = abs(tgt_idx - center)
                dist = min(dist, n_output - dist)

                # Gaussian boost
                boost = boost_strength * torch.exp(torch.tensor(-(dist**2) / (2 * sigma**2)))
                weights[tgt_idx, src_idx] += boost

        return weights

    @staticmethod
    def orthogonal(
        n_output: int, n_input: int, gain: float = 1.0, device: str = "cpu", **kwargs
    ) -> torch.Tensor:
        """
        Orthogonal initialization.

        Creates orthogonal matrix (columns/rows are orthonormal).
        Preserves norms during forward/backward passes.

        Good for: Avoiding gradient vanishing, RNNs, transformers

        Args:
            n_output: Number of output neurons
            n_input: Number of input neurons
            gain: Scaling factor
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input]
        """
        weights = torch.empty(n_output, n_input, device=device, requires_grad=False)
        nn.init.orthogonal_(weights, gain=gain)
        return weights

    @staticmethod
    def zeros(n_output: int, n_input: int, device: str = "cpu", **kwargs) -> torch.Tensor:
        """All zeros initialization."""
        return torch.zeros(n_output, n_input, device=device, requires_grad=False)

    @staticmethod
    def ones(n_output: int, n_input: int, device: str = "cpu", **kwargs) -> torch.Tensor:
        """All ones initialization."""
        return torch.ones(n_output, n_input, device=device, requires_grad=False)

    @staticmethod
    def identity(n_output: int, n_input: int, device: str = "cpu", **kwargs) -> torch.Tensor:
        """
        Identity matrix initialization.

        Creates identity matrix (padded/cropped if not square).
        """
        size = min(n_output, n_input)
        weights = torch.zeros(n_output, n_input, device=device, requires_grad=False)
        weights[:size, :size] = torch.eye(size, device=device, requires_grad=False)
        return weights

    @staticmethod
    def constant(
        n_output: int, n_input: int, value: float = 0.1, device: str = "cpu", **kwargs
    ) -> torch.Tensor:
        """Constant value initialization."""
        return torch.full((n_output, n_input), value, device=device, requires_grad=False)


# Register all strategies
WeightInitializer.register(InitStrategy.GAUSSIAN)(WeightInitializer.gaussian)
WeightInitializer.register(InitStrategy.UNIFORM)(WeightInitializer.uniform)
WeightInitializer.register(InitStrategy.XAVIER)(WeightInitializer.xavier)
WeightInitializer.register(InitStrategy.KAIMING)(WeightInitializer.kaiming)
WeightInitializer.register(InitStrategy.SPARSE_RANDOM)(WeightInitializer.sparse_random)
WeightInitializer.register(InitStrategy.TOPOGRAPHIC)(WeightInitializer.topographic)
WeightInitializer.register(InitStrategy.ORTHOGONAL)(WeightInitializer.orthogonal)
WeightInitializer.register(InitStrategy.ZEROS)(WeightInitializer.zeros)
WeightInitializer.register(InitStrategy.ONES)(WeightInitializer.ones)
WeightInitializer.register(InitStrategy.IDENTITY)(WeightInitializer.identity)
WeightInitializer.register(InitStrategy.CONSTANT)(WeightInitializer.constant)


__all__ = [
    "InitStrategy",
    "WeightInitializer",
]
