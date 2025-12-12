"""
Global Configuration - Universal parameters shared across all THALIA modules.

These parameters are truly global - they affect everything and should
be defined once in a single place.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import torch

from .base import BaseConfig


@dataclass
class GlobalConfig(BaseConfig):
    """Universal parameters shared across all THALIA modules.

    Inherits device, dtype, and seed from BaseConfig.

    These are parameters that:
    1. Affect multiple modules (device, timing)
    2. Must be consistent everywhere (vocab_size)
    3. Define fundamental simulation properties (dt_ms, theta)

    Example:
        config = GlobalConfig(
            device="cuda",
            vocab_size=10000,
            theta_frequency_hz=6.0,  # Slower theta for longer sequences
        )
    """

    # =========================================================================
    # TIMING
    # =========================================================================
    dt_ms: float = 1.0
    """Simulation timestep in milliseconds. Smaller = more precise but slower."""

    # =========================================================================
    # OSCILLATOR FREQUENCIES (brain rhythms)
    # =========================================================================
    delta_frequency_hz: float = 2.0
    """Delta oscillation frequency. Range: 0.5-4 Hz (biological).
    Deep sleep, slow-wave sleep, memory consolidation."""

    theta_frequency_hz: float = 8.0
    """Theta oscillation frequency. Range: 4-10 Hz (biological).
    Memory encoding/retrieval rhythm, spatial navigation, phase coding."""

    alpha_frequency_hz: float = 10.0
    """Alpha oscillation frequency. Range: 8-13 Hz (biological).
    Attention gating, inhibitory control, sensory suppression."""

    beta_frequency_hz: float = 20.0
    """Beta oscillation frequency. Range: 13-30 Hz (biological).
    Motor control, active cognitive processing, decision-making."""

    gamma_frequency_hz: float = 40.0
    """Gamma oscillation frequency. Range: 30-100 Hz (biological).
    Feature binding, local processing, attention, consciousness."""

    ripple_frequency_hz: float = 150.0
    """Sharp-wave ripple frequency. Range: 100-200 Hz (biological).
    Memory replay during rest/sleep, hippocampal consolidation."""

    # =========================================================================
    # VOCABULARY (for language processing)
    # =========================================================================
    vocab_size: int = 50257
    """Token vocabulary size. Default is GPT-2 size.
    Must be consistent across encoder, decoder, and data pipeline."""

    # =========================================================================
    # SPARSITY
    # =========================================================================
    default_sparsity: float = 0.05
    """Default target sparsity (fraction of neurons active).
    Individual regions can override this."""

    # =========================================================================
    # WEIGHT BOUNDS
    # =========================================================================
    w_min: float = 0.0
    """Minimum weight value. Usually 0 (no negative weights) or small negative."""

    w_max: float = 1.0
    """Maximum weight value. Prevents runaway potentiation."""

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values."""
        if self.dt_ms <= 0:
            raise ValueError(f"dt_ms must be positive, got {self.dt_ms}")

        # Validate oscillator frequencies (biological ranges)
        if not (0.5 <= self.delta_frequency_hz <= 4.0):
            raise ValueError(
                f"delta_frequency_hz should be 0.5-4 Hz, got {self.delta_frequency_hz}"
            )
        if not (4.0 <= self.theta_frequency_hz <= 10.0):
            raise ValueError(
                f"theta_frequency_hz should be 4-10 Hz, got {self.theta_frequency_hz}"
            )
        if not (8.0 <= self.alpha_frequency_hz <= 13.0):
            raise ValueError(
                f"alpha_frequency_hz should be 8-13 Hz, got {self.alpha_frequency_hz}"
            )
        if not (13.0 <= self.beta_frequency_hz <= 30.0):
            raise ValueError(
                f"beta_frequency_hz should be 13-30 Hz, got {self.beta_frequency_hz}"
            )
        if not (30.0 <= self.gamma_frequency_hz <= 100.0):
            raise ValueError(
                f"gamma_frequency_hz should be 30-100 Hz, got {self.gamma_frequency_hz}"
            )
        if not (100.0 <= self.ripple_frequency_hz <= 200.0):
            raise ValueError(
                f"ripple_frequency_hz should be 100-200 Hz, got {self.ripple_frequency_hz}"
            )

        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

        if self.default_sparsity <= 0 or self.default_sparsity >= 1:
            raise ValueError(
                f"default_sparsity must be in (0, 1), got {self.default_sparsity}"
            )

        if self.w_min > self.w_max:
            raise ValueError(f"w_min ({self.w_min}) > w_max ({self.w_max})")

    @property
    def torch_device(self) -> torch.device:
        """Get torch.device object."""
        return torch.device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get torch.dtype object."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.float32)

    @property
    def delta_period_ms(self) -> float:
        """Period of delta oscillation in milliseconds."""
        return 1000.0 / self.delta_frequency_hz

    @property
    def theta_period_ms(self) -> float:
        """Period of theta oscillation in milliseconds."""
        return 1000.0 / self.theta_frequency_hz

    @property
    def alpha_period_ms(self) -> float:
        """Period of alpha oscillation in milliseconds."""
        return 1000.0 / self.alpha_frequency_hz

    @property
    def beta_period_ms(self) -> float:
        """Period of beta oscillation in milliseconds."""
        return 1000.0 / self.beta_frequency_hz

    @property
    def gamma_period_ms(self) -> float:
        """Period of gamma oscillation in milliseconds."""
        return 1000.0 / self.gamma_frequency_hz

    @property
    def ripple_period_ms(self) -> float:
        """Period of ripple oscillation in milliseconds."""
        return 1000.0 / self.ripple_frequency_hz

    def summary(self) -> str:
        """Return a formatted summary of global configuration."""
        lines = [
            "=== Global Configuration ===",
            f"  Device: {self.device}",
            f"  Data type: {self.dtype}",
            f"  Timestep: {self.dt_ms} ms",
            "",
            "  Oscillator Frequencies:",
            f"    Delta:  {self.delta_frequency_hz:>5.1f} Hz ({self.delta_period_ms:>6.1f} ms period)",
            f"    Theta:  {self.theta_frequency_hz:>5.1f} Hz ({self.theta_period_ms:>6.1f} ms period)",
            f"    Alpha:  {self.alpha_frequency_hz:>5.1f} Hz ({self.alpha_period_ms:>6.1f} ms period)",
            f"    Beta:   {self.beta_frequency_hz:>5.1f} Hz ({self.beta_period_ms:>6.1f} ms period)",
            f"    Gamma:  {self.gamma_frequency_hz:>5.1f} Hz ({self.gamma_period_ms:>6.1f} ms period)",
            f"    Ripple: {self.ripple_frequency_hz:>5.1f} Hz ({self.ripple_period_ms:>6.1f} ms period)",
            "",
            f"  Vocabulary: {self.vocab_size} tokens",
            f"  Sparsity: {self.default_sparsity:.1%}",
            f"  Weights: [{self.w_min}, {self.w_max}]",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "device": self.device,
            "dtype": self.dtype,
            "dt_ms": self.dt_ms,
            "delta_frequency_hz": self.delta_frequency_hz,
            "theta_frequency_hz": self.theta_frequency_hz,
            "alpha_frequency_hz": self.alpha_frequency_hz,
            "beta_frequency_hz": self.beta_frequency_hz,
            "gamma_frequency_hz": self.gamma_frequency_hz,
            "ripple_frequency_hz": self.ripple_frequency_hz,
            "vocab_size": self.vocab_size,
            "default_sparsity": self.default_sparsity,
            "w_min": self.w_min,
            "w_max": self.w_max,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GlobalConfig":
        """Create from dictionary."""
        return cls(**d)
