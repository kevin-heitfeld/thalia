"""
Brain Configuration - Settings for brain regions and architecture.

This module defines configuration for DynamicBrain and its
constituent regions, with clear inheritance from GlobalConfig.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING
from enum import Enum

# Import region configs from canonical locations
from thalia.regions.cortex.predictive_cortex import PredictiveCortexConfig
from thalia.regions.hippocampus.config import HippocampusConfig
from thalia.regions.striatum.config import StriatumConfig
from thalia.regions.prefrontal import PrefrontalConfig
from thalia.regions.cerebellum import CerebellumConfig
from thalia.diagnostics.criticality import CriticalityConfig

if TYPE_CHECKING:
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
            lines.extend([
                f"  Baseline: {self.norepinephrine_baseline}",
                f"  Gain scale: {self.norepinephrine_gain_scale}",
            ])
        lines.extend([
            "",
            "--- Acetylcholine (NB) ---",
            f"  Enabled: {self.use_acetylcholine}",
        ])
        if self.use_acetylcholine:
            lines.extend([
                f"  Encoding level: {self.acetylcholine_encoding_level}",
                f"  Retrieval level: {self.acetylcholine_retrieval_level}",
            ])
        return "\n".join(lines)


@dataclass
class RegionSizes:
    """Size configuration for brain regions.

    All sizes are in terms of number of neurons (or output size).
    Ratios and internal sizes are computed automatically.

    For cortex, you can either:
    1. Specify cortex_size only (uses default 1.0:1.5:1.0 ratios)
    2. Specify cortex_size AND explicit layer sizes (cortex_l4_size, cortex_l23_size, cortex_l5_size)
    """

    input_size: int = 256
    """Size of input to the brain (e.g., from sensory encoding)."""

    thalamus_size: int = 256
    """Size of thalamus relay output (typically matches input for 1:1 relay)."""

    cortex_size: int = 128
    """Output size of cortex. L2/3 and L5 layers will be sized relative to this."""

    # Explicit cortex layer sizes (optional, overrides ratio-based calculation)
    _cortex_l4_size: Optional[int] = None
    """Explicit L4 size (if None, computed from cortex_size * 1.0)."""

    _cortex_l23_size: Optional[int] = None
    """Explicit L2/3 size (if None, computed from cortex_size * 1.5)."""

    _cortex_l5_size: Optional[int] = None
    """Explicit L5 size (if None, computed from cortex_size * 1.0)."""

    hippocampus_size: int = 64
    """Output size of hippocampus (CA1 output)."""

    pfc_size: int = 32
    """Size of prefrontal cortex working memory."""

    n_actions: int = 2
    """Number of possible actions (for striatum output)."""

    # =========================================================================
    # DERIVED SIZES (computed properties)
    # =========================================================================

    @property
    def cortex_l4_size(self) -> int:
        """L4 (input layer) size - explicit or computed from cortex_size."""
        return self._cortex_l4_size if self._cortex_l4_size is not None else self.cortex_size

    @property
    def cortex_l23_size(self) -> int:
        """L2/3 (processing layer) size - explicit or computed as 1.5x cortex."""
        return self._cortex_l23_size if self._cortex_l23_size is not None else int(self.cortex_size * 1.5)

    @property
    def cortex_l5_size(self) -> int:
        """L5 (output layer) size - explicit or computed from cortex_size."""
        return self._cortex_l5_size if self._cortex_l5_size is not None else self.cortex_size

    @property
    def hippocampus_dg_size(self) -> int:
        """Dentate gyrus size - 5x expansion for pattern separation."""
        # DG expands from cortex L2/3 input
        return int(self.cortex_l23_size * 5)

    @property
    def hippocampus_ca3_size(self) -> int:
        """CA3 size - 50% of DG for pattern completion."""
        return int(self.hippocampus_dg_size * 0.5)

    @property
    def hippocampus_ca1_size(self) -> int:
        """CA1 size - matches configured hippocampus output."""
        return self.hippocampus_size

    def summary(self) -> str:
        """Return formatted summary of region sizes."""
        lines = [
            "=== Region Sizes ===",
            f"  Input: {self.input_size}",
            f"  Cortex: {self.cortex_size}",
            f"    L4: {self.cortex_l4_size}",
            f"    L2/3: {self.cortex_l23_size}",
            f"    L5: {self.cortex_l5_size}",
            f"  Hippocampus: {self.hippocampus_size}",
            f"    DG: {self.hippocampus_dg_size}",
            f"    CA3: {self.hippocampus_ca3_size}",
            f"    CA1: {self.hippocampus_ca1_size}",
            f"  PFC: {self.pfc_size}",
            f"  Actions: {self.n_actions}",
        ]
        return "\n".join(lines)


# NOTE: Region-specific configs are now imported from canonical locations.
# Each region module (hippocampus, striatum, prefrontal, cerebellum) defines
# its own complete configuration. This eliminates duplication and ensures
# consistency between config definitions and actual usage.


@dataclass
class BrainConfig:
    """Complete brain configuration.

    Combines region sizes with region-specific parameters and neuromodulation.
    Global parameters come from GlobalConfig.

    Note: cortex uses LayeredCortexConfig from thalia.regions.cortex.
    The n_input/n_output in cortex config are ignored - sizes come from
    RegionSizes.input_size and RegionSizes.cortex_size instead.
    """

    # Region sizes
    sizes: RegionSizes = field(default_factory=RegionSizes)

    # Region-specific configs (BEHAVIORAL PARAMS ONLY - no sizes)
    # Sizes come from RegionSizes and are passed separately during construction
    cortex: PredictiveCortexConfig = field(default_factory=PredictiveCortexConfig)
    hippocampus: HippocampusConfig = field(default_factory=HippocampusConfig)
    striatum: StriatumConfig = field(default_factory=StriatumConfig)
    pfc: PrefrontalConfig = field(default_factory=PrefrontalConfig)
    cerebellum: CerebellumConfig = field(default_factory=CerebellumConfig)

    # Region type selection (allows swapping implementations)
    cortex_type: CortexType = CortexType.PREDICTIVE
    """Which cortex implementation to use. PREDICTIVE (default) enables local error learning, LAYERED for simpler feedforward."""

    # Neuromodulation (dopamine, norepinephrine, acetylcholine)
    neuromodulation: NeuromodulationConfig = field(default_factory=NeuromodulationConfig)
    """Dopamine, norepinephrine, acetylcholine configuration."""

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

    # Execution mode
    parallel: bool = False

    # Device (should inherit from GlobalConfig, but provided for convenience)
    device: str = "cpu"

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

    # Phase 2: Model-based planning
    use_model_based_planning: bool = True
    """Enable mental simulation for action selection (Phase 2).

    When True:
        - select_action() uses MentalSimulationCoordinator for tree search
        - deliver_reward() triggers DynaPlanner background planning
        - Requires: PFC.predict_next_state, Hippocampus.retrieve_similar, Striatum.evaluate_state
    """

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
            self.sizes.summary(),
            "",
            "--- Region Types ---",
            f"  Cortex: {self.cortex_type.value}",
            "",
            "--- Trial Timing ---",
            f"  Encoding: {self.encoding_timesteps} timesteps",
            f"  Delay: {self.delay_timesteps} timesteps",
            f"  Test: {self.test_timesteps} timesteps",
            f"  Parallel: {self.parallel}",
            "",
            self.neuromodulation.summary(),
        ]
        return "\n".join(lines)

    @property
    def total_striatum_neurons(self) -> int:
        """Total neurons in striatum with population coding."""
        if self.use_population_coding:
            return self.sizes.n_actions * self.neurons_per_action
        return self.sizes.n_actions

    @property
    def striatum_pfc_size(self) -> int:
        """PFC size that striatum should use for goal conditioning.

        This MUST match sizes.pfc_size when use_goal_conditioning=True.
        Returns sizes.pfc_size by default for consistency.
        """
        return self.sizes.pfc_size
