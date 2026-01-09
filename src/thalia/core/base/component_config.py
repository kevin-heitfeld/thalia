"""
Component Configuration Base Classes.

This module provides base configuration classes for neural components
(regions, pathways, learning components). These were moved from config/base.py
to break the CONFIG ↔ REGIONS circular import dependency.

Rationale:
- NeuralComponentConfig is infrastructure (used by all regions/pathways)
- Regions import it, and BrainConfig aggregates region configs
- Moving to core/ breaks circular: REGIONS → CORE (not REGIONS → CONFIG → REGIONS)
- config/base.py now contains only pure system configs (device, dtype, seed)

Breaking Change (January 2026):
- Removed n_input, n_output, n_neurons from NeuralComponentConfig
- Regions now specify semantic dimensions (n_actions, layer sizes, etc.)
- Physical dimensions (neuron counts) computed from semantic specs
- Multi-source inputs always explicit (Dict[str, int])

Author: Thalia Project
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from thalia.config.base import BaseConfig

if TYPE_CHECKING:
    from thalia.components.synapses.stp import STPConfig
    from thalia.learning.rules.bcm import BCMConfig


@dataclass
class NeuralComponentConfig(BaseConfig):
    """Base config for neural components (regions, pathways, layers).

    **Semantic-First Design** (January 2026 Refactoring):
    Provides common parameters for all neural components WITHOUT specifying
    neuron counts. Each region defines its own semantic dimensions:
    - Striatum: n_actions, neurons_per_action, input_sources
    - Hippocampus: input_size, layer sizes (dg, ca3, ca2, ca1)
    - Cortex: input_size, layer sizes (l4, l23, l5, l6a, l6b)

    This base provides only universal parameters:
    - Temporal dynamics: dt_ms, axonal_delay_ms
    - Learning: learn, learning_rate, weight bounds
    - STDP: shared plasticity parameters
    - Homeostasis: activity regulation
    - Adaptation, competition, spillover mechanisms

    Growth operations use semantic units (actions, layers, concepts),
    not raw neuron counts.
    """

    # =========================================================================
    # TEMPORAL DYNAMICS
    # =========================================================================
    dt_ms: float = 1.0
    """Simulation timestep in milliseconds. Set from GlobalConfig.dt_ms by Brain."""

    axonal_delay_ms: float = 1.0
    """Axonal conduction delay in milliseconds.

    Biological ranges:
    - Within-region (local): 0.5-2ms
    - Inter-region (long-range): 1-10ms
    - Thalamo-cortical: 8-15ms
    - Striato-cortical: 10-20ms

    ALL neural connections have conduction delays - this is not optional.
    Regions and pathways differ only in typical delay values (configuration),
    not in whether delays exist (architectural difference).
    """

    # =========================================================================
    # LEARNING
    # =========================================================================
    learn: bool = True
    """Whether learning is enabled in this component."""

    learning_rate: float = 0.001
    """Base learning rate for plasticity."""

    # Weight bounds (shared across regions and pathways)
    w_min: float = 0.0
    """Minimum synaptic weight (usually 0.0 for excitatory)."""

    w_max: float = 1.0
    """Maximum synaptic weight (prevents runaway potentiation)."""

    # =========================================================================
    # STDP PARAMETERS (shared by most regions)
    # =========================================================================
    learning_rule: str = "STDP"
    """Which plasticity rule to use (STDP, PHASE_STDP, TRIPLET_STDP, etc.)."""

    stdp_lr: float = 0.01
    """STDP-specific learning rate."""

    tau_plus_ms: float = 20.0
    """LTP time constant in milliseconds."""

    tau_minus_ms: float = 20.0
    """LTD time constant in milliseconds."""

    a_plus: float = 1.0
    """LTP amplitude."""

    a_minus: float = 1.0
    """LTD amplitude."""

    max_trace: float = 10.0
    """Maximum trace value to prevent runaway accumulation."""

    # =========================================================================
    # EXTENDED ELIGIBILITY (for delayed modulation)
    # =========================================================================
    eligibility_tau_ms: float = 1000.0
    """Time constant for extended eligibility traces in milliseconds.

    For DELAYED modulation (100-1000ms after spike correlation).
    Fast STDP uses tau_plus_ms/tau_minus_ms (~20ms) for coincidence detection.
    """

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION
    # =========================================================================
    adapt_increment: float = 0.0
    """Adaptation current increase per spike (0 = disabled)."""

    adapt_tau: float = 100.0
    """Adaptation decay time constant in milliseconds."""

    # =========================================================================
    # HETEROSYNAPTIC COMPETITION
    # =========================================================================
    heterosynaptic_competition: bool = True
    """Enable heterosynaptic competition."""

    heterosynaptic_ratio: float = 0.3
    """Fraction of LTD applied to non-active synapses during learning (0-1)."""

    # =========================================================================
    # HOMEOSTATIC PLASTICITY
    # =========================================================================
    homeostasis_enabled: bool = True
    """Enable homeostatic regulation."""

    weight_budget: float = 1.0
    """Target sum of weights per neuron (row normalization constraint)."""

    activity_target: float = 0.1
    """Target fraction of neurons active per timestep."""

    activity_min: float = 0.01
    """Minimum activity level (prevents dead neurons)."""

    activity_max: float = 0.5
    """Maximum activity level (prevents seizure-like states)."""

    normalize_rows: bool = True
    """Normalize each neuron's input weights."""

    normalize_cols: bool = False
    """Normalize each input's output weights."""

    neuron_type: str = "lif"
    """Type of neuron model: 'lif', 'conductance', 'dendritic'."""

    target_firing_rate_hz: float = 5.0
    """Target firing rate in Hz for homeostatic regulation."""

    homeostatic_tau_ms: float = 10000.0
    """Time constant for homeostatic adaptation in milliseconds."""

    soft_normalization: bool = True
    """Use soft (multiplicative) normalization instead of hard constraint."""

    normalization_rate: float = 0.1
    """Rate of convergence toward target (soft normalization only)."""

    activity_tau_ms: float = 1000.0
    """Time constant for activity rate estimation."""

    enable_competition: bool = True
    """Enable competitive weight adjustment."""

    competition_strength: float = 0.1
    """Strength of competitive suppression between neurons."""

    # =========================================================================
    # SPILLOVER TRANSMISSION
    # =========================================================================
    enable_spillover: bool = False
    """Enable spillover (volume) transmission."""

    spillover_strength: float = 0.15
    """Spillover weight strength relative to direct synapses."""

    spillover_mode: str = "connectivity"
    """Spillover neighborhood definition method."""

    spillover_lateral_radius: int = 3
    """Neighborhood radius for lateral spillover mode."""

    spillover_similarity_threshold: float = 0.5
    """Minimum similarity for spillover in similarity mode."""

    spillover_normalize: bool = True
    """Normalize spillover weights to prevent runaway excitation."""


@dataclass
class LearningComponentConfig(BaseConfig):
    """Base config for learning components.

    Extends BaseConfig with learning-specific parameters:
    - learning_rate: Base learning rate
    - enabled: Whether learning is enabled
    """

    learning_rate: float = 0.01
    """Base learning rate."""

    enabled: bool = True
    """Whether this learning component is enabled."""


@dataclass
class PathwayConfig(NeuralComponentConfig):
    """Configuration for neural pathways (spiking connections between regions).

    **Pathways ARE simple feedforward connectors** - they don't need semantic dimensions.
    They just route spikes from source (n_input) to target (n_output) regions.

    Unlike regions (which specify semantic dimensions like n_actions), pathways
    use physical dimensions directly:
    - n_input: Source region size
    - n_output: Target region size
    - n_neurons: Intermediate population (usually = n_output)

    Inherits common parameters from NeuralComponentConfig:
    - dt_ms, device, dtype, seed
    - w_min, w_max: Weight bounds
    - learning_rate, stdp_lr: Learning parameters
    - axonal_delay_ms: Conduction delay (OVERRIDDEN to 5.0ms for long-range)

    Pathway-Specific Defaults:
    - axonal_delay_ms: 5.0ms (inter-region typical, vs 1.0ms for local)
    - adapt_increment: 0.0 (pathways are relay neurons, not pyramidal)
    - learning_rule: "STDP" (spike-timing dependent plasticity)

    Example:
        config = PathwayConfig(
            n_input=128,   # Source region size
            n_output=64,   # Target region size
            stdp_lr=0.01,
        )
    """

    # =========================================================================
    # PATHWAY DIMENSIONS (simple feedforward sizing)
    # =========================================================================
    n_input: int = 128
    """Input dimension (source region size)."""

    n_output: int = 64
    """Output dimension (target region size)."""

    n_neurons: int = 0
    """Number of intermediate neurons (computed = n_output if not specified)."""

    # =========================================================================
    # PATHWAY-SPECIFIC OVERRIDES
    # =========================================================================
    # Override axonal delay for long-range inter-region connections
    axonal_delay_ms: float = 5.0
    """Axonal conduction delay in milliseconds (inter-region default).

    Biological ranges for inter-region pathways:
    - Cortico-cortical: 5-10ms
    - Thalamo-cortical: 8-15ms
    - Striato-cortical: 10-20ms
    - Hippocampo-cortical: 10-15ms

    Default of 5.0ms is appropriate for typical cortico-cortical projections.
    Specific pathways can override for longer delays (e.g., thalamus: 10ms).
    """

    def __post_init__(self):
        """Synchronize n_neurons with n_output for pathway consistency."""
        # For pathways, n_neurons should match n_output (target size)
        if self.n_neurons == 0:
            object.__setattr__(self, 'n_neurons', self.n_output)

        # Synchronize learning_rate with stdp_lr if stdp_lr was explicitly set
        if hasattr(self, 'stdp_lr') and self.stdp_lr != 0.01:
            self.learning_rate = self.stdp_lr

    # =========================================================================
    # CONNECTIVITY
    # =========================================================================
    sparsity: float = 0.1
    """Target sparsity for pathway connections (fraction of non-zero weights)."""

    topographic: bool = False
    """Use topographic connectivity pattern."""

    delay_variability: float = 0.2
    """Variability in axonal delays (fraction of mean delay)."""

    # =========================================================================
    # NEURON MODEL PARAMETERS
    # =========================================================================
    tau_mem_ms: float = 20.0  # TAU_MEM_STANDARD
    """Membrane time constant in milliseconds."""

    tau_syn_ms: float = 5.0  # TAU_SYN_EXCITATORY
    """Synaptic time constant in milliseconds."""

    v_thresh: float = -50.0  # V_THRESHOLD_STANDARD
    """Spike threshold voltage in mV."""

    v_reset: float = -65.0  # V_RESET_STANDARD
    """Reset voltage after spike in mV."""

    v_rest: float = -70.0  # V_REST_STANDARD
    """Resting membrane potential in mV."""

    refractory_ms: float = 2.0  # TAU_REF_STANDARD
    """Refractory period in milliseconds."""

    # =========================================================================
    # WEIGHT INITIALIZATION
    # =========================================================================
    init_mean: float = 0.3
    """Initial weight mean."""

    init_std: float = 0.1
    """Initial weight standard deviation."""

    # =========================================================================
    # TEMPORAL CODING
    # =========================================================================
    temporal_coding: str = "RATE"  # TemporalCoding enum value
    """Which temporal coding scheme (RATE, LATENCY, PHASE, SYNCHRONY, BURST)."""

    oscillation_freq_hz: float = 8.0
    """Oscillation frequency for phase coding (Hz)."""

    phase_precision: float = 0.5
    """How tightly spikes lock to phase (0-1)."""

    # =========================================================================
    # SHORT-TERM PLASTICITY (STP)
    # =========================================================================
    stp_enabled: bool = False
    """Enable short-term plasticity."""

    stp_type: str = "DEPRESSING"  # STPType enum value
    """Preset synapse type (DEPRESSING, FACILITATING, DUAL)."""

    stp_config: Optional["STPConfig"] = None
    """Custom STP parameters (overrides stp_type)."""

    # =========================================================================
    # BCM SLIDING THRESHOLD (Metaplasticity)
    # =========================================================================
    bcm_enabled: bool = False
    """Enable BCM sliding threshold rule."""

    bcm_config: Optional["BCMConfig"] = None
    """Custom BCM parameters."""


__all__ = [
    "NeuralComponentConfig",
    "LearningComponentConfig",
    "PathwayConfig",
]
