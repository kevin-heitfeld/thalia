"""
Brain Configuration - Settings for brain regions and architecture.

This module defines configuration for DynamicBrain and its
constituent regions.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

import torch

from thalia.diagnostics.criticality import CriticalityConfig

if TYPE_CHECKING:
    from thalia.config.region_configs import (
        CerebellumConfig,
        HippocampusConfig,
        PredictiveCortexConfig,
        PrefrontalConfig,
        StriatumConfig,
    )
    from thalia.coordination.oscillator import OscillatorCoupling


class RegionType(Enum):
    """Types of brain regions."""

    CORTEX = "cortex"
    HIPPOCAMPUS = "hippocampus"
    PFC = "pfc"
    STRIATUM = "striatum"
    CEREBELLUM = "cerebellum"


class CortexType(Enum):
    """Types of cortex implementation.

    LAYERED: Standard feedforward layered cortex (L4 → L2/3 → L5)
    PREDICTIVE: Layered cortex with predictive coding (local error signals)
    """

    LAYERED = "layered"
    PREDICTIVE = "predictive"


@dataclass
class NeuromodulationConfig:
    """Configuration for neuromodulatory systems (VTA, LC, NB).

    Neuromodulators gate learning and modulate neural processing:
    - Dopamine (VTA): Reward prediction error, gates striatal learning
    - Norepinephrine (LC): Arousal, attention, gain modulation
    - Acetylcholine (NB): Encoding vs retrieval mode in hippocampus

    Example:
        config = NeuromodulationConfig(
            dopamine_baseline=0.1,
            dopamine_learning_threshold=0.05,
            use_norepinephrine=True,
        )
    """

    # Dopamine (VTA - ventral tegmental area)
    dopamine_baseline: float = 0.0
    """Baseline dopamine level (tonic). Range: -0.5 to 0.5."""

    dopamine_learning_threshold: float = 0.01
    """Minimum |dopamine| to trigger learning. Filters noise."""

    dopamine_decay_tau_ms: float = 100.0
    """Time constant for dopamine decay back to baseline (milliseconds)."""

    # Norepinephrine (LC - locus coeruleus)
    use_norepinephrine: bool = False
    """Enable norepinephrine modulation (arousal, attention)."""

    norepinephrine_baseline: float = 0.5
    """Baseline norepinephrine (arousal level). Range: 0.0 to 1.0."""

    norepinephrine_gain_scale: float = 1.5
    """How much NE scales neural gain. 1.0 = no effect, >1.0 = amplification."""

    # Acetylcholine (NB - nucleus basalis)
    use_acetylcholine: bool = True
    """Enable acetylcholine modulation (encoding/retrieval)."""

    acetylcholine_encoding_level: float = 0.8
    """ACh level during encoding (high = strengthen new memories)."""

    acetylcholine_retrieval_level: float = 0.2
    """ACh level during retrieval (low = strengthen recall pathways)."""

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "=== Neuromodulation ===",
            "--- Dopamine (VTA) ---",
            f"  Baseline: {self.dopamine_baseline}",
            f"  Learning threshold: {self.dopamine_learning_threshold}",
            f"  Decay tau: {self.dopamine_decay_tau_ms} ms",
            "",
            "--- Norepinephrine (LC) ---",
            f"  Enabled: {self.use_norepinephrine}",
        ]
        if self.use_norepinephrine:
            lines.extend(
                [
                    f"  Baseline: {self.norepinephrine_baseline}",
                    f"  Gain scale: {self.norepinephrine_gain_scale}",
                ]
            )
        lines.extend(
            [
                "",
                "--- Acetylcholine (NB) ---",
                f"  Enabled: {self.use_acetylcholine}",
            ]
        )
        if self.use_acetylcholine:
            lines.extend(
                [
                    f"  Encoding level: {self.acetylcholine_encoding_level}",
                    f"  Retrieval level: {self.acetylcholine_retrieval_level}",
                ]
            )
        return "\n".join(lines)


# NOTE: Region-specific configs are now imported from canonical locations.
# Each region module (hippocampus, striatum, prefrontal, cerebellum) defines
# its own complete configuration. This eliminates duplication and ensures
# consistency between config definitions and actual usage.


# Lazy factory functions to avoid circular imports
def _default_cerebellum_config():
    from thalia.config.region_configs import CerebellumConfig

    return CerebellumConfig()


def _default_cortex_config():
    from thalia.config.region_configs import PredictiveCortexConfig

    return PredictiveCortexConfig()


def _default_hippocampus_config():
    from thalia.config.region_configs import HippocampusConfig

    return HippocampusConfig()


def _default_pfc_config():
    from thalia.config.region_configs import PrefrontalConfig

    return PrefrontalConfig()


def _default_striatum_config():
    from thalia.config.region_configs import StriatumConfig

    return StriatumConfig()


@dataclass
class BrainConfig:
    """Complete brain configuration for a single brain instance.

    Each brain instance is fully self-contained with
    its own device, dt, oscillator frequencies, etc.

    This enables:
    - Multiple independent brains with different devices (GPU vs CPU)
    - Different temporal resolutions per brain (dt_ms)
    - Different oscillator frequencies per brain
    - Per-brain learning modes (gradients on/off)

    Example:
        ```python
        # High-res brain on GPU
        brain1 = DynamicBrain(BrainConfig(
            device="cuda",
            dt_ms=0.1,
            theta_frequency_hz=10.0,
        ))

        # Standard brain on CPU
        brain2 = DynamicBrain(BrainConfig(
            device="cpu",
            dt_ms=1.0,
            theta_frequency_hz=8.0,
        ))
        ```
    """

    # =========================================================================
    # COMPUTATION
    # =========================================================================
    device: str = "cpu"
    """Device to run on: 'cpu', 'cuda', 'cuda:0', etc.

    Each brain can run on a different device for parallel processing or
    resource management.
    """

    dtype: str = "float32"
    """Data type for tensors: 'float32', 'float64', 'float16', 'bfloat16'.

    Different precision per brain allows speed vs accuracy tradeoffs.
    """

    seed: Optional[int] = None
    """Random seed for reproducibility. None = no seeding.

    Each brain can have its own seed for independent experiments.
    """

    enable_gradients: bool = False
    """Enable gradient computation for backpropagation.

    Default: False (disabled) for biologically-plausible local learning rules.

    Thalia uses local learning rules (STDP, BCM, Hebbian, three-factor) that
    do NOT require backpropagation. Disabling gradients provides:
    - 50% memory savings (no backward graph storage)
    - 20-40% faster forward passes
    - Explicit biological constraint enforcement

    Set to True for:
    - Experimental comparison with backprop
    - Metacognitive calibration modules
    - Hybrid bio/non-bio learning
    """

    # =========================================================================
    # TIMING
    # =========================================================================
    dt_ms: float = 1.0
    """Simulation timestep in milliseconds. Smaller = more precise but slower.

    **CRITICAL**: This is the single source of truth for temporal resolution.
    All decay factors, delays, and oscillators derive from this value.

    Can be changed adaptively during simulation via brain.set_timestep(new_dt).
    Typical values:
    - 1.0ms: Standard biological timescale (Brian2, most research)
    - 0.1ms: High-resolution for detailed temporal dynamics
    - 10ms: Fast replay for memory consolidation
    """

    # =========================================================================
    # OSCILLATOR FREQUENCIES
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

    # Oscillator configuration
    oscillator_couplings: Optional[List[OscillatorCoupling]] = None
    """Custom cross-frequency couplings (e.g., delta-theta, alpha-gamma).

    If None, uses default theta-gamma coupling (coupling_strength=0.8).
    If empty list [], disables all coupling.
    If provided, replaces defaults with custom couplings.

    Example:
        ```python
        config = BrainConfig(
            oscillator_couplings=[
                OscillatorCoupling('theta', 'gamma', coupling_strength=0.8),
                OscillatorCoupling('delta', 'theta', coupling_strength=0.6),
                OscillatorCoupling('alpha', 'gamma', coupling_strength=0.7),
            ]
        )
        ```

    See thalia.core.oscillator.OscillatorCoupling for full parameters.
    """

    # Timing (trial phases)
    encoding_timesteps: int = 15
    delay_timesteps: int = 10
    test_timesteps: int = 15

    # =========================================================================
    # NEURAL PROPERTIES
    # =========================================================================
    vocab_size: int = 50257
    """Token vocabulary size. Default is GPT-2 size.

    Each brain can have its own vocabulary. If brains communicate,
    an interface layer handles translation between vocabularies.
    """

    default_sparsity: float = 0.05
    """Default target sparsity (fraction of neurons active).
    Individual regions can override this."""

    w_min: float = 0.0
    """Minimum weight value. Usually 0 (no negative weights) or small negative."""

    w_max: float = 1.0
    """Maximum weight value. Prevents runaway potentiation."""

    # =========================================================================
    # ARCHITECTURE
    # =========================================================================

    # Region-specific configs (BEHAVIORAL PARAMS ONLY - no sizes)
    # Sizes are passed directly to BrainBuilder during construction
    cortex: "PredictiveCortexConfig" = field(default_factory=_default_cortex_config)
    hippocampus: "HippocampusConfig" = field(default_factory=_default_hippocampus_config)
    striatum: "StriatumConfig" = field(default_factory=_default_striatum_config)
    pfc: "PrefrontalConfig" = field(default_factory=_default_pfc_config)
    cerebellum: "CerebellumConfig" = field(default_factory=_default_cerebellum_config)

    # Region type selection (allows swapping implementations)
    cortex_type: CortexType = CortexType.PREDICTIVE
    """Which cortex implementation to use. PREDICTIVE (default) enables local error learning, LAYERED for simpler feedforward."""

    # Neuromodulation (dopamine, norepinephrine, acetylcholine)
    neuromodulation: NeuromodulationConfig = field(default_factory=NeuromodulationConfig)
    """Dopamine, norepinephrine, acetylcholine configuration."""

    # =========================================================================
    # Goal-conditioned behavior (PFC → Striatum modulation)
    # =========================================================================
    use_goal_conditioning: bool = False
    """Enable PFC goal context modulation of striatum (goal-directed behavior).

    When True:
        - Striatum creates pfc_modulation_d1/d2 weights
        - PFC working memory modulates action selection
        - CRITICAL: striatum_pfc_size MUST match sizes.pfc_size
    """

    use_population_coding: bool = True
    """Use population coding in striatum (multiple neurons per action)."""

    neurons_per_action: int = 10
    """Number of neurons per action when use_population_coding=True."""

    # =========================================================================
    # Auto-growth (optional capacity expansion)
    # =========================================================================
    auto_growth_enabled: bool = False
    """Enable automatic region growth based on capacity metrics.

    When True:
        - Brain checks capacity every auto_growth_check_interval timesteps
        - Regions exceeding auto_growth_threshold are grown automatically
        - Useful for open-ended learning and experiments

    When False (default):
        - Growth is managed explicitly via GrowthManager (curriculum training)
        - Provides full control over when and how regions grow
    """

    auto_growth_threshold: float = 0.8
    """Capacity utilization threshold for triggering auto-growth (0.0-1.0).

    When a region's capacity exceeds this threshold, it will be grown.
    Default 0.8 = grow when 80% of neurons are utilized.
    """

    auto_growth_check_interval: int = 1000
    """Check for growth needs every N timesteps.

    Lower values = more frequent checks (more overhead)
    Higher values = less frequent checks (may delay growth)
    Default 1000 = check every 1000 timesteps (~1 second at 1ms timesteps)
    """

    # =========================================================================
    # Brain-wide diagnostics
    # =========================================================================
    enable_criticality_monitor: bool = False
    """Enable brain-wide criticality monitoring (branching ratio tracking).

    When True:
        - CriticalityMonitor tracks branching ratio across all regions
        - Diagnostics include criticality metrics
        - Useful for research and debugging network dynamics

    When False (default):
        - No criticality monitoring (saves computation)
        - Standard for production training

    Note: This is brain-wide, not region-specific. It monitors spikes
    across all regions to estimate the network's criticality state.
    """

    criticality_config: CriticalityConfig = field(default_factory=CriticalityConfig)
    """Configuration for brain-wide criticality monitoring.

    Only used when enable_criticality_monitor=True.
    Tracks branching ratio and can provide weight scaling corrections.
    """

    def summary(self) -> str:
        """Return formatted summary of brain configuration."""
        lines = [
            "=== Brain Configuration ===",
            f"  Device: {self.device}",
            f"  Data type: {self.dtype}",
            f"  Timestep: {self.dt_ms} ms",
            f"  Gradients: {'enabled' if self.enable_gradients else 'disabled'}",
            "",
            "  Oscillator Frequencies:",
            f"    Delta:  {self.delta_frequency_hz:>5.1f} Hz",
            f"    Theta:  {self.theta_frequency_hz:>5.1f} Hz",
            f"    Alpha:  {self.alpha_frequency_hz:>5.1f} Hz",
            f"    Beta:   {self.beta_frequency_hz:>5.1f} Hz",
            f"    Gamma:  {self.gamma_frequency_hz:>5.1f} Hz",
            f"    Ripple: {self.ripple_frequency_hz:>5.1f} Hz",
            "",
            f"  Vocabulary: {self.vocab_size} tokens",
            f"  Sparsity: {self.default_sparsity:.1%}",
            f"  Weights: [{self.w_min}, {self.w_max}]",
            "",
            "--- Region Types ---",
            f"  Cortex: {self.cortex_type.value}",
            "",
            "--- Trial Timing ---",
            f"  Encoding: {self.encoding_timesteps} timesteps",
            f"  Delay: {self.delay_timesteps} timesteps",
            f"  Test: {self.test_timesteps} timesteps",
            "",
            self.neuromodulation.summary(),
        ]
        return "\n".join(lines)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values."""
        # Timing validation
        if self.dt_ms <= 0:
            raise ValueError(f"dt_ms must be positive, got {self.dt_ms}")

        # Oscillator frequency validation (biological ranges)
        if not (0.5 <= self.delta_frequency_hz <= 4.0):
            raise ValueError(
                f"delta_frequency_hz should be 0.5-4 Hz, got {self.delta_frequency_hz}"
            )
        if not (4.0 <= self.theta_frequency_hz <= 10.0):
            raise ValueError(f"theta_frequency_hz should be 4-10 Hz, got {self.theta_frequency_hz}")
        if not (8.0 <= self.alpha_frequency_hz <= 13.0):
            raise ValueError(f"alpha_frequency_hz should be 8-13 Hz, got {self.alpha_frequency_hz}")
        if not (13.0 <= self.beta_frequency_hz <= 30.0):
            raise ValueError(f"beta_frequency_hz should be 13-30 Hz, got {self.beta_frequency_hz}")
        if not (30.0 <= self.gamma_frequency_hz <= 100.0):
            raise ValueError(
                f"gamma_frequency_hz should be 30-100 Hz, got {self.gamma_frequency_hz}"
            )
        if not (100.0 <= self.ripple_frequency_hz <= 200.0):
            raise ValueError(
                f"ripple_frequency_hz should be 100-200 Hz, got {self.ripple_frequency_hz}"
            )

        # Neural properties validation
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

        if self.default_sparsity <= 0 or self.default_sparsity >= 1:
            raise ValueError(f"default_sparsity must be in (0, 1), got {self.default_sparsity}")

        if self.w_min > self.w_max:
            raise ValueError(f"w_min ({self.w_min}) > w_max ({self.w_max})")

    def get_torch_device(self) -> "torch.device":
        """Get PyTorch device object."""
        return torch.device(self.device)

    def get_torch_dtype(self) -> "torch.dtype":
        """Get PyTorch dtype object."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
        }
        return dtype_map.get(self.dtype, torch.float32)

    @property
    def theta_period_ms(self) -> float:
        """Get theta period in milliseconds (computed from frequency)."""
        return 1000.0 / self.theta_frequency_hz
