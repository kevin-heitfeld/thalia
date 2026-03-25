"""Base configuration classes for neural regions in Thalia."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from thalia import GlobalConfig
from thalia.brain.gap_junctions import GapJunctionConfig
from thalia.brain.synapses import NMReceptorType
from thalia.errors import ConfigurationError
from thalia.typing import NeuromodulatorChannel


@dataclass(frozen=True)
class NMReceptorConfig:
    """Config-level neuromodulator receptor specification.

    Defines which receptor type, broadcast channel, buffer name, and target
    populations a region needs.  The base class resolves *n_receptors* at
    runtime by summing the neuron counts of the listed populations.

    If *populations* is empty the receptor is created with ``n_receptors=1``
    (a scalar global modulator signal).
    """

    receptor_type: NMReceptorType
    channel: NeuromodulatorChannel
    buffer_name: str
    populations: tuple[str, ...] = ()
    amplitude_scale: float = 1.0
    initial_value: float = 0.0


@dataclass
class NeuralPopulationConfig:
    tau_mem_ms: float
    v_threshold: float
    v_reset: float
    adapt_increment: float
    tau_adapt_ms: float
    noise_std: float


# =========================================================================
# COMPOSABLE SUB-CONFIGS FOR HOMEOSTATIC PLASTICITY
# =========================================================================


@dataclass
class HomeostaticGainConfig:
    """Adaptive gain control to maintain target firing rates.

    Biological basis: Intrinsic plasticity via ion channel remodeling
    (Turrigiano & Nelson 2004).  Modulates leak conductance (g_L_scale):
    lower g_L = higher input resistance = more excitable.
    """

    lr_per_ms: float = 0.001
    """Learning rate for gain adaptation per ms (0 = disabled)."""

    tau_ms: float = 2000.0
    """Time constant for firing rate averaging."""

    def __post_init__(self) -> None:
        if self.lr_per_ms < 0:
            raise ConfigurationError(f"HomeostaticGainConfig.lr_per_ms must be >= 0, got {self.lr_per_ms}")
        if self.tau_ms <= 0:
            raise ConfigurationError(f"HomeostaticGainConfig.tau_ms must be > 0, got {self.tau_ms}")


@dataclass
class HomeostaticThresholdConfig:
    """Adaptive threshold plasticity (complementary to gain adaptation).

    Lowers spike threshold when neurons are underactive; raises when overactive.
    Faster time constant than gain adaptation.
    """

    lr_per_ms: float = 0.001
    """Learning rate for threshold adaptation per ms."""

    threshold_min: float = 0.1
    """Minimum adaptive threshold."""

    threshold_max: float = 1.0
    """Maximum adaptive threshold."""

    def __post_init__(self) -> None:
        if self.lr_per_ms < 0:
            raise ConfigurationError(f"HomeostaticThresholdConfig.lr_per_ms must be >= 0, got {self.lr_per_ms}")
        if self.threshold_min < 0:
            raise ConfigurationError(f"HomeostaticThresholdConfig.threshold_min must be >= 0, got {self.threshold_min}")
        if self.threshold_min > self.threshold_max:
            raise ConfigurationError(
                f"HomeostaticThresholdConfig.threshold_min ({self.threshold_min}) "
                f"must be <= threshold_max ({self.threshold_max})"
            )


@dataclass
class SynapticScalingConfig:
    """Synaptic scaling (Turrigiano 2008 — complementary to gain adaptation).

    Per-neuron bidirectional multiplicative scaling of ALL incoming synaptic weights.
    Underactive neurons scale up; overactive neurons scale down.
    Preserves relative weight differences while stabilizing total drive.
    """

    lr_per_ms: float = 0.001
    """Learning rate for weight scaling per ms (slow)."""

    w_min: float = 0.0
    """Minimum synaptic weight (usually 0.0 for excitatory)."""

    w_max: float = 1.0
    """Maximum synaptic weight (prevents runaway potentiation)."""

    interval_steps: int = 10
    """Apply synaptic scaling every N simulation steps.

    Biological synaptic scaling operates on hours-to-days timescales
    (Turrigiano 2008), so applying it every single ms-timestep is both
    biologically unrealistic and computationally wasteful.  The effective
    learning rate is automatically compensated: ``lr_effective = lr_per_ms * interval_steps``.
    Set to 1 for legacy per-step behaviour.
    """

    def __post_init__(self) -> None:
        if self.lr_per_ms < 0:
            raise ConfigurationError(f"SynapticScalingConfig.lr_per_ms must be >= 0, got {self.lr_per_ms}")
        if self.w_min < 0:
            raise ConfigurationError(f"SynapticScalingConfig.w_min must be >= 0, got {self.w_min}")
        if self.w_max <= 0:
            raise ConfigurationError(f"SynapticScalingConfig.w_max must be > 0, got {self.w_max}")
        if self.w_min > self.w_max:
            raise ConfigurationError(f"SynapticScalingConfig.w_min ({self.w_min}) must be <= w_max ({self.w_max})")
        if self.interval_steps < 1:
            raise ConfigurationError(f"SynapticScalingConfig.interval_steps must be >= 1, got {self.interval_steps}")


def _default_gap_junctions() -> GapJunctionConfig:
    """Default gap junction config for NeuralRegionConfig (disabled)."""
    return GapJunctionConfig(
        coupling_strength=0.0,
        connectivity_threshold=0.3,
        max_neighbors=6,
    )


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

    learning_disabled: bool = False  # Set to True to disable all synaptic plasticity for this region.
    homeostasis_disabled: bool = False  # Set to True to disable homeostatic plasticity for this region.
    neuromodulation_disabled: bool = False  # Set to True to disable neuromodulator effects for this region.

    # =========================================================================
    # INHIBITORY STDP (Vogels et al. 2011)
    # =========================================================================
    # Default parameters for homeostatic tuning of I→E synapses.
    # Region-specific configs can override these.
    istdp_learning_rate: float = 0.001  # iSTDP learning rate (η)
    istdp_alpha: float = 0.12  # Target-rate offset (sets E/I balance set-point)
    istdp_tau_ms: float = 20.0  # Trace time constant (ms)

    # =========================================================================
    # COMPOSABLE SUB-CONFIGS
    # =========================================================================
    homeostatic_gain: HomeostaticGainConfig = field(default_factory=HomeostaticGainConfig)
    homeostatic_threshold: HomeostaticThresholdConfig = field(default_factory=HomeostaticThresholdConfig)
    synaptic_scaling: SynapticScalingConfig = field(default_factory=SynapticScalingConfig)
    gap_junctions: GapJunctionConfig = field(default_factory=_default_gap_junctions)

    homeostatic_target_rates: dict[str, float] = field(default_factory=dict)
    """Per-population homeostatic target firing rates (spikes/ms).

    Keys are population names (e.g. ``"l23_pyr"``, ``"dg"``).
    Region-specific config subclasses set biologically appropriate defaults.
    """

    neuromodulator_receptors: list[NMReceptorConfig] = field(default_factory=list)
    """Declarative neuromodulator receptor specifications.

    Each entry creates one receptor + concentration buffer.  Population sizes
    are resolved automatically by the base class.
    """

    # =========================================================================
    # VALIDATION
    # =========================================================================
    def __post_init__(self) -> None:
        """Validate config field values."""
        if self.dt_ms <= 0:
            raise ConfigurationError(f"dt_ms must be > 0, got {self.dt_ms}")
        if self.learning_rate < 0:
            raise ConfigurationError(f"learning_rate must be >= 0, got {self.learning_rate}")

        # Warn when any homeostasis rate implies an effective time constant under 1 second.
        # Effective tau = dt_ms / lr_per_ms.  Values below 1000 ms are biologically implausible
        # (homeostasis operates on minutes-to-hours timescales in vivo).
        _MIN_TAU_MS = 1000.0
        for name, lr in (
            ("homeostatic_gain.lr_per_ms", self.homeostatic_gain.lr_per_ms),
            ("homeostatic_threshold.lr_per_ms", self.homeostatic_threshold.lr_per_ms),
            ("synaptic_scaling.lr_per_ms", self.synaptic_scaling.lr_per_ms),
        ):
            if lr > 0:
                tau_ms = self.dt_ms / lr
                if tau_ms < _MIN_TAU_MS:
                    warnings.warn(
                        f"{self.__class__.__name__}.{name}={lr} implies an effective "
                        f"homeostasis time constant of {tau_ms:.1f} ms "
                        f"(= dt_ms / lr = {self.dt_ms} / {lr}), which is below the "
                        f"biological minimum of {_MIN_TAU_MS:.0f} ms (1 s). "
                        f"Consider reducing {name}.",
                        stacklevel=3,
                    )
