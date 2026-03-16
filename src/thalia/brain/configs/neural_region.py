"""Base configuration classes for neural regions in Thalia."""

from __future__ import annotations

from dataclasses import dataclass

from thalia import GlobalConfig
from thalia.errors import ConfigurationError


@dataclass
class NeuralRegionConfig:
    """Base config for neural regions.

    Specific regions extend this with their own parameters.
    """

    # =========================================================================
    # GENERAL PARAMETERS
    # =========================================================================
    dt_ms: float = GlobalConfig.DEFAULT_DT_MS  # Simulation timestep in milliseconds.

    learning_rate: float = 0.001  # Base learning rate for plasticity.

    # =========================================================================
    # SYNAPTIC WEIGHT BOUNDS
    # =========================================================================
    w_min: float = 0.0  # Minimum synaptic weight (usually 0.0 for excitatory).
    w_max: float = 1.0  # Maximum synaptic weight (prevents runaway potentiation).

    # =========================================================================
    # GAP JUNCTIONS
    # =========================================================================
    gap_junction_strength:  float = 0.0  # Gap junction conductance strength.
    gap_junction_threshold: float = 0.3  # Connectivity threshold for gap junction coupling.
    gap_junction_max_neighbors: int = 6  # Maximum gap junction neighbors per neuron.

    # =========================================================================
    # HOMEOSTATIC INTRINSIC PLASTICITY
    # =========================================================================
    # ADAPTIVE GAIN CONTROL to maintain target firing rates
    # Biological basis: Intrinsic plasticity (ion channel remodeling)
    gain_learning_rate: float = 0.02  # Learning rate for gain adaptation (0 = disabled)
    gain_tau_ms: float = 2000.0  # Time constant for firing rate averaging

    # ADAPTIVE THRESHOLD PLASTICITY (complementary to gain adaptation)
    threshold_learning_rate: float = 0.05  # Learning rate for threshold adaptation
    threshold_min: float = 0.1  # Minimum adaptive threshold
    threshold_max: float = 1.0  # Maximum adaptive threshold

    # SYNAPTIC SCALING (complementary to gain adaptation)
    # Biology: Chronically underactive neurons scale up ALL input synapses globally
    # This is distinct from gain adaptation (intrinsic excitability) and works together
    # with it to maintain network stability.
    synaptic_scaling_lr: float = 0.001  # Learning rate for weight scaling (slow)
    synaptic_scaling_min_activity: float = 0.005  # Minimum activity (0.5%) to trigger scaling
    synaptic_scaling_max_factor: float = 2.0  # Maximum scaling factor (prevent explosion)

    # =========================================================================
    # VALIDATION
    # =========================================================================
    def __post_init__(self) -> None:
        """Validate config field values."""
        if self.dt_ms <= 0:
            raise ConfigurationError(f"dt_ms must be > 0, got {self.dt_ms}")
        if self.learning_rate < 0:
            raise ConfigurationError(f"learning_rate must be >= 0, got {self.learning_rate}")
        if self.w_min < 0:
            raise ConfigurationError(f"w_min must be >= 0, got {self.w_min}")
        if self.w_max <= 0:
            raise ConfigurationError(f"w_max must be > 0, got {self.w_max}")
        if self.w_min > self.w_max:
            raise ConfigurationError(f"w_min ({self.w_min}) must be <= w_max ({self.w_max})")
        if self.gap_junction_strength < 0:
            raise ConfigurationError(f"gap_junction_strength must be >= 0, got {self.gap_junction_strength}")
        if self.gap_junction_max_neighbors < 0:
            raise ConfigurationError(f"gap_junction_max_neighbors must be >= 0, got {self.gap_junction_max_neighbors}")
        if self.gain_learning_rate < 0:
            raise ConfigurationError(f"gain_learning_rate must be >= 0, got {self.gain_learning_rate}")
        if self.gain_tau_ms <= 0:
            raise ConfigurationError(f"gain_tau_ms must be > 0, got {self.gain_tau_ms}")
        if self.threshold_learning_rate < 0:
            raise ConfigurationError(f"threshold_learning_rate must be >= 0, got {self.threshold_learning_rate}")
        if self.threshold_min < 0:
            raise ConfigurationError(f"threshold_min must be >= 0, got {self.threshold_min}")
        if self.threshold_min > self.threshold_max:
            raise ConfigurationError(f"threshold_min ({self.threshold_min}) must be <= threshold_max ({self.threshold_max})")
        if self.synaptic_scaling_lr < 0:
            raise ConfigurationError(f"synaptic_scaling_lr must be >= 0, got {self.synaptic_scaling_lr}")
        if self.synaptic_scaling_max_factor < 1.0:
            raise ConfigurationError(f"synaptic_scaling_max_factor must be >= 1.0, got {self.synaptic_scaling_max_factor}")
