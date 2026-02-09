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
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn


class InitStrategy(Enum):
    """Weight initialization strategies."""

    CONSTANT = auto()  # Constant value
    GAUSSIAN = auto()  # Gaussian (normal) distribution
    IDENTITY = auto()  # Identity matrix
    KAIMING = auto()  # Kaiming/He initialization
    ONES = auto()  # All ones
    ORTHOGONAL = auto()  # Orthogonal initialization
    SPARSE_GAUSSIAN = auto()  # Sparse connectivity with Gaussian weights
    SPARSE_RANDOM = auto()  # Sparse random connectivity
    SPARSE_UNIFORM = auto()  # Sparse connectivity with uniform weights
    TOPOGRAPHIC = auto()  # Topographic (spatial) connectivity
    UNIFORM = auto()  # Uniform distribution
    XAVIER = auto()  # Xavier/Glorot initialization
    ZEROS = auto()  # All zeros


class WeightInitializer:
    """
    Centralized weight initialization registry.

    Provides standardized initialization methods used across Thalia.
    All methods return torch.Tensor, not nn.Parameter.
    """

    # Registry of initialization functions
    _registry: Dict[InitStrategy, Callable[..., torch.Tensor]] = {}

    @classmethod
    def register(cls, strategy: InitStrategy) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
        """Decorator to register an initialization function."""

        def decorator(func: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
            cls._registry[strategy] = func
            return func

        return decorator

    @classmethod
    def get(cls, strategy: InitStrategy) -> Callable[..., torch.Tensor]:
        """Get initialization function by strategy."""
        if strategy not in cls._registry:
            available = ", ".join([s.name for s in InitStrategy])
            raise ValueError(
                f"Unknown initialization strategy: '{strategy}'. "
                f"Available strategies: {available}. "
                f"Use InitStrategy enum (e.g., InitStrategy.XAVIER) or register a custom strategy."
            )
        return cls._registry[strategy]

    @classmethod
    def list_initializers(cls) -> list[str]:
        """List all registered initialization strategies."""
        return [strategy.name for strategy in cls._registry.keys()]

    @staticmethod
    def constant(
        n_output: int,
        n_input: int,
        value: float = 0.1,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Constant value initialization."""
        return torch.full((n_output, n_input), value, device=device, requires_grad=False)

    @staticmethod
    def gaussian(
        n_output: int,
        n_input: int,
        mean: float = 0.0,
        std: float = 0.1,
        device: Union[str, torch.device] = "cpu",
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
    def identity(
        n_output: int,
        n_input: int,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Identity matrix initialization.

        Creates identity matrix (padded/cropped if not square).
        """
        size = min(n_output, n_input)
        weights = torch.zeros(n_output, n_input, device=device, requires_grad=False)
        weights[:size, :size] = torch.eye(size, device=device, requires_grad=False)
        return weights

    @staticmethod
    def kaiming(
        n_output: int,
        n_input: int,
        mode: str = "fan_in",
        nonlinearity: str = "relu",
        device: Union[str, torch.device] = "cpu",
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
    def ones(
        n_output: int,
        n_input: int,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """All ones initialization."""
        return torch.ones(n_output, n_input, device=device, requires_grad=False)

    @staticmethod
    def orthogonal(
        n_output: int,
        n_input: int,
        gain: float = 1.0,
        device: Union[str, torch.device] = "cpu",
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
    def sparse_gaussian(
        n_output: int,
        n_input: int,
        sparsity: float = 0.3,
        mean: float = 0.0,
        std: float = 0.1,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Sparse Gaussian connectivity with Gaussian weight distribution.

        Creates biological sparse connectivity with Gaussian distributed weights.

        Args:
            n_output: Number of output neurons
            n_input: Number of input neurons
            sparsity: Fraction of connections to keep (0-1)
            mean: Mean of Gaussian distribution
            std: Standard deviation of Gaussian distribution
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity
        """
        mask = torch.rand(n_output, n_input, device=device, requires_grad=False) < sparsity
        weights = torch.randn(n_output, n_input, device=device, requires_grad=False) * std + mean
        return weights * mask.float()

    @staticmethod
    def sparse_random(
        n_output: int,
        n_input: int,
        sparsity: float = 0.3,
        weight_scale: float = 0.1,
        normalize_rows: bool = False,
        device: Union[str, torch.device] = "cpu",
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
    def sparse_uniform(
        n_output: Optional[int] = None,
        n_input: Optional[int] = None,
        n_pre: Optional[int] = None,
        n_post: Optional[int] = None,
        sparsity: Optional[float] = None,
        prob: Optional[float] = None,
        weight_range: Optional[tuple[float, float]] = None,
        w_min: Optional[float] = None,
        w_max: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Sparse uniform connectivity with uniform weight distribution.

        Creates biological sparse connectivity with uniformly distributed weights.

        Args:
            n_output: Number of output (post-synaptic) neurons (or use n_post)
            n_input: Number of input (pre-synaptic) neurons (or use n_pre)
            n_pre: Alias for n_input (pre-synaptic neurons)
            n_post: Alias for n_output (post-synaptic neurons)
            sparsity: Fraction of connections to keep (0-1) (or use prob)
            prob: Alias for sparsity (connection probability)
            weight_range: (min, max) for uniform weight distribution (or use w_min, w_max)
            w_min: Minimum weight value
            w_max: Maximum weight value
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity
        """
        # Handle parameter aliases
        n_out = n_post if n_post is not None else n_output
        n_in = n_pre if n_pre is not None else n_input
        sparsity_val = prob if prob is not None else sparsity

        if w_min is not None and w_max is not None:
            weight_range_val = (w_min, w_max)
        elif weight_range is not None:
            weight_range_val = weight_range
        else:
            weight_range_val = (0.0, 0.1)

        if n_out is None or n_in is None:
            raise ValueError("Must provide either (n_output, n_input) or (n_pre, n_post)")
        if sparsity_val is None:
            sparsity_val = 0.3

        mask = torch.rand(n_out, n_in, device=device, requires_grad=False) < sparsity_val
        weights = torch.rand(n_out, n_in, device=device, requires_grad=False)
        weights = weights * (weight_range_val[1] - weight_range_val[0]) + weight_range_val[0]
        return weights * mask.float()

    @staticmethod
    def topographic(
        n_output: int,
        n_input: int,
        base_weight: float = 0.1,
        sigma_factor: float = 4.0,
        boost_strength: float = 0.3,
        device: Union[str, torch.device] = "cpu",
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
    def uniform(
        n_output: int,
        n_input: int,
        low: float = 0.0,
        high: float = 1.0,
        device: Union[str, torch.device] = "cpu",
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
        n_output: int,
        n_input: int,
        gain: float = 1.0,
        device: Union[str, torch.device] = "cpu",
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
    def zeros(
        n_output: int,
        n_input: int,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """All zeros initialization."""
        return torch.zeros(n_output, n_input, device=device, requires_grad=False)


# Register all strategies
WeightInitializer.register(InitStrategy.CONSTANT)(WeightInitializer.constant)
WeightInitializer.register(InitStrategy.GAUSSIAN)(WeightInitializer.gaussian)
WeightInitializer.register(InitStrategy.IDENTITY)(WeightInitializer.identity)
WeightInitializer.register(InitStrategy.KAIMING)(WeightInitializer.kaiming)
WeightInitializer.register(InitStrategy.ONES)(WeightInitializer.ones)
WeightInitializer.register(InitStrategy.ORTHOGONAL)(WeightInitializer.orthogonal)
WeightInitializer.register(InitStrategy.SPARSE_GAUSSIAN)(WeightInitializer.sparse_gaussian)
WeightInitializer.register(InitStrategy.SPARSE_RANDOM)(WeightInitializer.sparse_random)
WeightInitializer.register(InitStrategy.SPARSE_UNIFORM)(WeightInitializer.sparse_uniform)
WeightInitializer.register(InitStrategy.TOPOGRAPHIC)(WeightInitializer.topographic)
WeightInitializer.register(InitStrategy.UNIFORM)(WeightInitializer.uniform)
WeightInitializer.register(InitStrategy.XAVIER)(WeightInitializer.xavier)
WeightInitializer.register(InitStrategy.ZEROS)(WeightInitializer.zeros)
