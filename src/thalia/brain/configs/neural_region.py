"""Base configuration classes for neural regions in Thalia."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    device: str = "cpu"  # Device to run on: 'cpu', 'cuda', 'cuda:0', etc.
    seed: Optional[int] = None  # Random seed for reproducibility. None = no seeding.
    dt_ms: float = GlobalConfig.DEFAULT_DT_MS  # Simulation timestep in milliseconds.

    learning_rate: float = 0.001
    """Base learning rate for plasticity."""

    # =========================================================================
    # SYNAPTIC WEIGHT BOUNDS
    # =========================================================================
    w_min: float = 0.0
    """Minimum synaptic weight (usually 0.0 for excitatory)."""

    w_max: float = 1.0
    """Maximum synaptic weight (prevents runaway potentiation)."""

    # =========================================================================
    # ADAPTIVE GAIN CONTROL (HOMEOSTATIC INTRINSIC PLASTICITY)
    # =========================================================================
    # Adaptive gain control to maintain target firing rates (Turrigiano 2008)
    # Biological basis: Intrinsic plasticity (ion channel remodeling)
    gain_learning_rate: float = 0.02  # INCREASED from 0.005 to respond faster to activity collapse
    target_firing_rate: float = 0.05  # Target firing rate for homeostatic plasticity
    gain_tau_ms: float = 2000.0  # Time constant for firing rate averaging
    baseline_noise_conductance_enabled: bool = True  # Stochastic synaptic background (miniature EPSPs)

    # =========================================================================
    # ADAPTIVE THRESHOLD PLASTICITY (complementary to gain adaptation)
    # =========================================================================
    threshold_learning_rate: float = 0.05  # Learning rate for threshold adaptation
    threshold_min: float = 0.1  # Minimum adaptive threshold
    threshold_max: float = 1.0  # Maximum adaptive threshold

    # =========================================================================
    # SYNAPTIC SCALING (complementary to gain adaptation)
    # =========================================================================
    # Biology: Chronically underactive neurons scale up ALL input synapses globally
    # This is distinct from gain adaptation (intrinsic excitability) and works together
    # with it to maintain network stability.
    synaptic_scaling_lr: float = 0.001  # Learning rate for weight scaling (slow)
    synaptic_scaling_min_activity: float = 0.005  # Minimum activity (0.5%) to trigger scaling
    synaptic_scaling_max_factor: float = 2.0  # Maximum scaling factor (prevent explosion)

    # =========================================================================
    # ELIGIBILITY TRACES
    # =========================================================================
    eligibility_tau_ms: float = 1000.0
    """Time constant for extended eligibility traces in milliseconds.

    For DELAYED modulation (100-1000ms after spike correlation).
    Fast STDP uses tau_plus_ms/tau_minus_ms (~20ms) for coincidence detection.
    """

    # =========================================================================
    # GAP JUNCTIONS
    # =========================================================================
    gap_junction_strength: float = 0.0
    """Gap junction conductance strength."""

    gap_junction_threshold: float = 0.3
    """Connectivity threshold for gap junction coupling."""

    gap_junction_max_neighbors: int = 6
    """Maximum gap junction neighbors per neuron."""

    # =========================================================================
    # HETEROSYNAPTIC COMPETITION
    # =========================================================================
    heterosynaptic_ratio: float = 0.3
    """Fraction of LTD applied to non-active synapses during learning (0-1)."""

    # =========================================================================
    # HOMEOSTATIC PLASTICITY
    # =========================================================================
    activity_target: float = 0.1
    """Target fraction of neurons active per timestep."""

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION (SFA)
    # =========================================================================
    adapt_increment: float = 0.0
    """Adaptation current increase per spike (0 = disabled)."""

    adapt_tau: float = 100.0
    """Adaptation decay time constant in milliseconds."""

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
        if self.gain_learning_rate < 0:
            raise ConfigurationError(f"gain_learning_rate must be >= 0, got {self.gain_learning_rate}")
        if self.gain_tau_ms <= 0:
            raise ConfigurationError(f"gain_tau_ms must be > 0, got {self.gain_tau_ms}")
        if self.target_firing_rate < 0:
            raise ConfigurationError(f"target_firing_rate must be >= 0, got {self.target_firing_rate}")
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
        if self.eligibility_tau_ms <= 0:
            raise ConfigurationError(f"eligibility_tau_ms must be > 0, got {self.eligibility_tau_ms}")
        if self.gap_junction_strength < 0:
            raise ConfigurationError(f"gap_junction_strength must be >= 0, got {self.gap_junction_strength}")
        if self.gap_junction_max_neighbors < 0:
            raise ConfigurationError(f"gap_junction_max_neighbors must be >= 0, got {self.gap_junction_max_neighbors}")
        if not (0.0 <= self.heterosynaptic_ratio <= 1.0):
            raise ConfigurationError(f"heterosynaptic_ratio must be in [0, 1], got {self.heterosynaptic_ratio}")
        if not (0.0 <= self.activity_target <= 1.0):
            raise ConfigurationError(f"activity_target must be in [0, 1], got {self.activity_target}")
        if self.adapt_tau <= 0:
            raise ConfigurationError(f"adapt_tau must be > 0, got {self.adapt_tau}")
