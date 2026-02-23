"""Weight Initialization Registry - Biologically-Motivated Initialization Strategies."""

from __future__ import annotations

from typing import Union

import torch

from thalia import GlobalConfig


class WeightInitializer:
    """
    Centralized weight initialization registry.

    Provides standardized initialization methods used across Thalia.
    All methods return torch.Tensor, not nn.Parameter.
    """

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
        if not (0.0 <= connectivity <= 1.0):
            raise ValueError("Connectivity must be between 0 and 1.")
        if mean < 0.0 or std < 0.0:
            raise ValueError("Mean and std must be non-negative for conductance-based synapses.")

        mask = torch.rand(n_output, n_input, device=device, requires_grad=False) < connectivity
        weights = torch.randn(n_output, n_input, device=device, requires_grad=False)
        weights = weights * std + mean
        weights = weights * mask.float() * GlobalConfig.SYNAPTIC_WEIGHT_SCALE
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
        if not (0.0 <= connectivity <= 1.0):
            raise ValueError("Connectivity must be between 0 and 1.")
        if weight_scale < 0.0:
            raise ValueError("Weight scale must be non-negative for conductance-based synapses.")

        mask = torch.rand(n_output, n_input, device=device, requires_grad=False) < connectivity
        weights = torch.rand(n_output, n_input, device=device, requires_grad=False)
        weights = weights * weight_scale
        weights = weights * mask.float() * GlobalConfig.SYNAPTIC_WEIGHT_SCALE
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
        if not (0.0 <= connectivity <= 1.0):
            raise ValueError("Connectivity must be between 0 and 1.")
        if w_min < 0.0 or w_max < 0.0:
            raise ValueError("Weight values must be non-negative for conductance-based synapses.")
        if w_min > w_max:
            raise ValueError("Minimum weight cannot be greater than maximum weight.")

        mask = torch.rand(n_output, n_input, device=device, requires_grad=False) < connectivity
        weights = torch.rand(n_output, n_input, device=device, requires_grad=False)
        weights = weights * (w_max - w_min) + w_min
        weights = weights * mask.float() * GlobalConfig.SYNAPTIC_WEIGHT_SCALE
        return weights.abs()  # Ensure positive conductances for biological realism

    @staticmethod
    def sparse_gaussian_no_autapses(
        n_input: int,
        n_output: int,
        connectivity: float,
        mean: float,
        std: float,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Sparse Gaussian connectivity with diagonal zeroed (no autapses).

        Identical to :meth:`sparse_gaussian` but guarantees the diagonal is zero.
        Only valid for square weight matrices (``n_input == n_output``).

        Args:
            n_input: Number of input neurons (must equal ``n_output``)
            n_output: Number of output neurons (must equal ``n_input``)
            connectivity: Connection probability (fraction of connections present, 0-1)
            mean: Mean of Gaussian distribution
            std: Standard deviation of Gaussian distribution
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity and zero diagonal

        Raises:
            ValueError: If ``n_input != n_output`` (non-square matrices cannot have autapses)
        """
        if n_input != n_output:
            raise ValueError(
                f"sparse_gaussian_no_autapses requires a square matrix "
                f"(n_input={n_input} != n_output={n_output}). "
                "Use sparse_gaussian for non-square weight matrices."
            )
        weights = WeightInitializer.sparse_gaussian(
            n_input=n_input,
            n_output=n_output,
            connectivity=connectivity,
            mean=mean,
            std=std,
            device=device,
        )
        weights.fill_diagonal_(0.0)  # Eliminate autapses (biologically absent)
        return weights

    @staticmethod
    def sparse_random_no_autapses(
        n_input: int,
        n_output: int,
        connectivity: float,
        weight_scale: float,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Sparse random connectivity with diagonal zeroed (no autapses).

        Identical to :meth:`sparse_random` but guarantees the diagonal is zero.
        Only valid for square weight matrices (``n_input == n_output``).

        Args:
            n_input: Number of input neurons (must equal ``n_output``)
            n_output: Number of output neurons (must equal ``n_input``)
            connectivity: Connection probability (fraction of connections present, 0-1)
            weight_scale: Scale of random weights (CONDUCTANCE units, normalised by g_L)
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity and zero diagonal

        Raises:
            ValueError: If ``n_input != n_output`` (non-square matrices cannot have autapses)
        """
        if n_input != n_output:
            raise ValueError(
                f"sparse_random_no_autapses requires a square matrix "
                f"(n_input={n_input} != n_output={n_output}). "
                "Use sparse_random for non-square weight matrices."
            )
        weights = WeightInitializer.sparse_random(
            n_input=n_input,
            n_output=n_output,
            connectivity=connectivity,
            weight_scale=weight_scale,
            device=device,
        )
        weights.fill_diagonal_(0.0)  # Eliminate autapses (biologically absent)
        return weights

    @staticmethod
    def sparse_uniform_no_autapses(
        n_input: int,
        n_output: int,
        connectivity: float,
        w_min: float,
        w_max: float,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Sparse uniform connectivity with diagonal zeroed (no autapses).

        Identical to :meth:`sparse_uniform` but guarantees the diagonal is zero.
        Only valid for square weight matrices (``n_input == n_output``).

        Args:
            n_input: Number of input neurons (must equal ``n_output``)
            n_output: Number of output neurons (must equal ``n_input``)
            connectivity: Connection probability (fraction of connections present, 0-1)
            w_min: Minimum weight value
            w_max: Maximum weight value
            device: Device for tensor

        Returns:
            Weight matrix [n_output, n_input] with sparse connectivity and zero diagonal

        Raises:
            ValueError: If ``n_input != n_output`` (non-square matrices cannot have autapses)
        """
        if n_input != n_output:
            raise ValueError(
                f"sparse_uniform_no_autapses requires a square matrix "
                f"(n_input={n_input} != n_output={n_output}). "
                "Use sparse_uniform for non-square weight matrices."
            )
        weights = WeightInitializer.sparse_uniform(
            n_input=n_input,
            n_output=n_output,
            connectivity=connectivity,
            w_min=w_min,
            w_max=w_max,
            device=device,
        )
        weights.fill_diagonal_(0.0)  # Eliminate autapses (biologically absent)
        return weights
