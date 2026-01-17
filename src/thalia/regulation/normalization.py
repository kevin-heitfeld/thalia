"""
Divisive Normalization: Automatic Gain Control for Neural Networks.

Divisive normalization is a canonical neural computation found throughout
the brain. It normalizes each neuron's activity by the activity of a pool
of neurons, providing:

1. AUTOMATIC GAIN CONTROL: Responses scale appropriately regardless of
   input intensity (weak or strong inputs produce similar relative patterns)

2. CONTRAST ENHANCEMENT: Differences between neurons are amplified

3. INPUT INVARIANCE: The same pattern at different intensities produces
   similar normalized outputs

4. NOISE SUPPRESSION: Common-mode noise is divided out

Mathematical Form:
==================
    output_i = input_i / (σ² + Σ_j pool_ij × input_j)

Where:
- input_i is the activity of neuron i
- pool_ij defines which neurons contribute to the normalization pool
- σ² is a semi-saturation constant (prevents division by zero and sets
  the point at which normalization "kicks in")

Biological Basis:
=================
- Retinal gain control adapts to ambient light levels
- V1 normalization explains cross-orientation suppression
- Cortical normalization provides attention-like effects
- Found in virtually every sensory system

Key References:
- Carandini & Heeger (2012): Normalization as a canonical neural computation
- Schwartz & Simoncelli (2001): Natural signal statistics and sensory gain control
- Reynolds & Heeger (2009): The normalization model of attention

Usage:
======
    from thalia.regulation.normalization import DivisiveNormalization, DivisiveNormConfig

    norm = DivisiveNormalization(DivisiveNormConfig(sigma=1.0))

    # Normalize activations
    normalized = norm(activations)

    # With custom pool (e.g., local neighborhood)
    normalized = norm(activations, pool=local_activity)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DivisiveNormConfig:
    """Configuration for divisive normalization.

    Attributes:
        sigma: Semi-saturation constant (default: 1.0)
            Controls when normalization "kicks in":
            - Small σ: Normalization is strong even for weak inputs
            - Large σ: Normalization only affects strong inputs
            Biologically, this corresponds to the spontaneous activity level.

        pool_type: Type of normalization pool (default: "global")
            - "global": Normalize by sum of all activities (simplest)
            - "local": Normalize by local neighborhood (requires pool_size)
            - "learned": Use learned pooling weights (advanced)
            - "self": Normalize by own activity only (divisive gain control)

        pool_size: Size of local pool for "local" pool_type
            Only used when pool_type="local".

        epsilon: Small constant for numerical stability (default: 1e-8)

        power: Power for the normalization (default: 1.0)
            - 1.0: Standard divisive normalization
            - 2.0: Squared normalization (L2-like)

        learnable_sigma: Whether sigma is a learnable parameter (default: False)
    """

    sigma: float = 1.0
    pool_type: str = "global"  # "global", "local", "learned", "self"
    pool_size: Optional[int] = None
    epsilon: float = 1e-8
    power: float = 1.0
    learnable_sigma: bool = False


class DivisiveNormalization(nn.Module):
    """Divisive normalization layer.

    Normalizes input by the sum of a pool of activities, providing
    automatic gain control and contrast enhancement.

    Can be applied to:
    - Membrane potentials (before spike generation)
    - Input currents (as they arrive)
    - Spike counts (for rate coding)
    - Any activity tensor
    """

    def __init__(
        self,
        config: Optional[DivisiveNormConfig] = None,
        n_features: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.config = config or DivisiveNormConfig()
        self.n_features = n_features
        self._device = device

        # Sigma can be learnable
        if self.config.learnable_sigma:
            self.sigma = nn.Parameter(torch.tensor(self.config.sigma, device=device))
        else:
            self.register_buffer("sigma", torch.tensor(self.config.sigma, device=device))

        # For learned pooling
        if self.config.pool_type == "learned" and n_features is not None:
            self.pool_weights: Optional[nn.Parameter] = nn.Parameter(
                torch.ones(n_features, n_features, device=device) / n_features
            )
        else:
            self.pool_weights = None

        # For local pooling, create a fixed kernel
        if self.config.pool_type == "local" and self.config.pool_size is not None:
            kernel_size = self.config.pool_size
            # Simple box filter for local pooling
            kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
            self.register_buffer("local_kernel", kernel)
        else:
            self.local_kernel = None

    @property
    def device(self) -> torch.device:
        """Get device."""
        return self._device or self.sigma.device

    def compute_pool(
        self,
        x: torch.Tensor,
        custom_pool: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the normalization pool.

        Args:
            x: Input tensor [..., features]
            custom_pool: Optional pre-computed pool

        Returns:
            Pool tensor for normalization
        """
        if custom_pool is not None:
            return custom_pool

        cfg = self.config

        if cfg.pool_type == "global":
            # Sum over all features
            return x.sum(dim=-1, keepdim=True)

        elif cfg.pool_type == "self":
            # Each neuron normalizes by itself only
            return x

        elif cfg.pool_type == "local":
            # Local neighborhood pooling
            if self.local_kernel is None:
                raise ValueError("Local pooling requires pool_size in config")
            # Reshape for conv1d: [batch, 1, features]
            x_conv = x.unsqueeze(-2) if x.dim() == 2 else x
            pool = F.conv1d(x_conv.transpose(-1, -2), self.local_kernel, padding="same").transpose(
                -1, -2
            )
            return pool.squeeze(-2) if x.dim() == 2 else pool

        elif cfg.pool_type == "learned":
            # Learned pooling weights
            if self.pool_weights is None:
                raise ValueError("Learned pooling requires n_features in init")
            return torch.matmul(x, self.pool_weights.t())

        else:
            raise ValueError(f"Unknown pool_type: {cfg.pool_type}")

    def forward(
        self,
        x: torch.Tensor,
        pool: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply divisive normalization.

        Args:
            x: Input tensor of any shape [..., features]
            pool: Optional custom pool (if None, computed from x)

        Returns:
            Normalized tensor of same shape as x
        """
        cfg = self.config

        # Compute normalization pool
        norm_pool = self.compute_pool(x, pool)

        # Apply power if not 1.0
        if cfg.power != 1.0:
            norm_pool = torch.pow(torch.abs(norm_pool) + cfg.epsilon, cfg.power)

        # Divisive normalization
        sigma_sq = self.sigma**2
        denominator = sigma_sq + norm_pool + cfg.epsilon

        return x / denominator

    def normalize_with_gain(
        self,
        x: torch.Tensor,
        pool: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize and return both output and gain factor.

        Useful for diagnostics or when you need to know how much
        normalization was applied.

        Args:
            x: Input tensor
            pool: Optional custom pool

        Returns:
            Tuple of (normalized_x, gain_factor)
            where normalized_x = x * gain_factor
        """
        cfg = self.config
        norm_pool = self.compute_pool(x, pool)

        if cfg.power != 1.0:
            norm_pool = torch.pow(torch.abs(norm_pool) + cfg.epsilon, cfg.power)

        sigma_sq = self.sigma**2
        denominator = sigma_sq + norm_pool + cfg.epsilon
        gain = 1.0 / denominator

        return x * gain, gain


class ContrastNormalization(nn.Module):
    """Contrast normalization: subtractive + divisive.

    First subtracts the mean (centering), then divides by the norm.
    This is similar to batch normalization but computed locally.

    output = (input - mean) / (σ² + std)
    """

    def __init__(
        self,
        config: Optional[DivisiveNormConfig] = None,
    ):
        super().__init__()
        self.config = config or DivisiveNormConfig()
        self.divisive = DivisiveNormalization(self.config)

    def forward(
        self,
        x: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        """Apply contrast normalization.

        Args:
            x: Input tensor
            dim: Dimension to normalize over

        Returns:
            Contrast-normalized tensor
        """
        # Subtractive: remove mean
        mean = x.mean(dim=dim, keepdim=True)
        centered = x - mean

        # Divisive: normalize by standard deviation
        std = centered.std(dim=dim, keepdim=True)
        pool = std.expand_as(centered)

        result: torch.Tensor = self.divisive(centered, pool=pool)
        return result


class SpatialDivisiveNorm(nn.Module):
    """Spatial divisive normalization for 2D feature maps.

    Commonly used in visual processing where local contrast
    normalization is important (e.g., after convolutional layers).
    """

    def __init__(
        self,
        kernel_size: int = 5,
        sigma: float = 1.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.kernel_size: int = kernel_size
        self.sigma: float = sigma
        self.epsilon: float = epsilon

        # Create Gaussian kernel for local pooling
        x = torch.arange(kernel_size) - kernel_size // 2
        gauss_1d = torch.exp(-(x**2) / (2 * (kernel_size / 4) ** 2))
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d / gauss_2d.sum()

        self.register_buffer("kernel", gauss_2d.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial divisive normalization.

        Args:
            x: Input tensor [batch, channels, height, width]

        Returns:
            Normalized tensor of same shape
        """
        _batch, channels, _h, _w = x.shape

        # Expand kernel for all channels
        kernel: torch.Tensor = self.kernel.expand(channels, 1, -1, -1)  # type: ignore[operator]

        # Compute local energy
        x_sq = x**2
        local_energy = F.conv2d(
            x_sq,
            kernel,
            padding=self.kernel_size // 2,
            groups=channels,
        )

        # Divisive normalization
        denominator = self.sigma**2 + local_energy + self.epsilon

        return x / torch.sqrt(denominator)
