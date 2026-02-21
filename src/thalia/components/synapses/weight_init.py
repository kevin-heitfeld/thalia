# pyright: strict
"""Weight Initialization Registry - Biologically-Motivated Initialization Strategies."""

from __future__ import annotations

from typing import Union

import torch


class WeightInitializer:
    """
    Centralized weight initialization registry.

    Provides standardized initialization methods used across Thalia.
    All methods return torch.Tensor, not nn.Parameter.
    """

    # Global weight scale factor for conductance-based synapses (normalized by g_L)
    GLOBAL_WEIGHT_SCALE: float = 0.0  # Set to 0.0 to disable synaptic conductances and test intrinsic excitability alone

    # Global learning/plasticity enable flag
    GLOBAL_LEARNING_ENABLED: bool = False  # Set to False to disable all synaptic plasticity

    @staticmethod
    def sparse_gaussian(
        n_input: int,
        n_output: int,
        connectivity: float,
        mean: float,
        std: float,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Sparse Gaussian connectivity with Gaussian weight distribution.

        Creates biological sparse connectivity with Gaussian distributed weights.

        Args:
            n_input: Number of input neurons
            n_output: Number of output neurons
            connectivity: Connection probability (fraction of connections present, 0-1)
            mean: Mean of Gaussian distribution
            std: Standard deviation of Gaussian distribution
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity
        """
        mask = torch.rand(n_output, n_input, device=device, requires_grad=False) < connectivity
        weights = torch.randn(n_output, n_input, device=device, requires_grad=False)
        weights = weights * std + mean
        weights = weights * mask.float() * WeightInitializer.GLOBAL_WEIGHT_SCALE
        return weights.abs()  # Ensure positive conductances for biological realism

    @staticmethod
    def sparse_random(
        n_input: int,
        n_output: int,
        connectivity: float,
        weight_scale: float,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Sparse random connectivity initialization.

        Creates biological sparse connectivity patterns.
        Each output neuron connects to a random subset of inputs.

        Args:
            n_input: Number of input neurons
            n_output: Number of output neurons
            connectivity: Connection probability (fraction of connections present, 0-1)
            weight_scale: Scale of random weights (CONDUCTANCE units, normalized by g_L)
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity
        """
        mask = torch.rand(n_output, n_input, device=device, requires_grad=False) < connectivity
        weights = torch.rand(n_output, n_input, device=device, requires_grad=False)
        weights = weights * weight_scale
        weights = weights * mask.float() * WeightInitializer.GLOBAL_WEIGHT_SCALE
        return weights.abs()  # Ensure positive conductances for biological realism

    @staticmethod
    def sparse_uniform(
        n_input: int,
        n_output: int,
        connectivity: float,
        w_min: float,
        w_max: float,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Sparse uniform connectivity with uniform weight distribution.

        Creates biological sparse connectivity with uniformly distributed weights.

        Args:
            n_input: Number of input (pre-synaptic) neurons
            n_output: Number of output (post-synaptic) neurons
            connectivity: Connection probability (fraction of connections present, 0-1)
            w_min: Minimum weight value
            w_max: Maximum weight value
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity
        """
        mask = torch.rand(n_output, n_input, device=device, requires_grad=False) < connectivity
        weights = torch.rand(n_output, n_input, device=device, requires_grad=False)
        weights = weights * (w_max - w_min) + w_min
        weights = weights * mask.float() * WeightInitializer.GLOBAL_WEIGHT_SCALE
        return weights.abs()  # Ensure positive conductances for biological realism

    @staticmethod
    def add_phase_diversity(weights: torch.Tensor, phase_diversity: float) -> torch.Tensor:
        """
        Add phase diversity to recurrent weights to promote rich dynamics.

        Args:
            weights: Original weight matrix [n_output, n_input]
            phase_diversity: Degree of phase diversity (0 = no diversity, higher = more diversity)

        Returns:
            Weight matrix with added phase diversity
        """
        if phase_diversity <= 0.0:
            raise ValueError("Phase diversity must be greater than 0.")
        noise = torch.randn_like(weights) * (1.0 + phase_diversity) * WeightInitializer.GLOBAL_WEIGHT_SCALE
        return (weights + noise).abs()  # Ensure positive conductances for biological realism
