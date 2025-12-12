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

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from thalia.config.base import BaseConfig

if TYPE_CHECKING:
    from thalia.components.synapses.stp import STPConfig
    from thalia.learning.bcm import BCMConfig


@dataclass
class NeuralComponentConfig(BaseConfig):
    """Base config for neural components (regions, pathways, layers).

    Provides common parameters for all neural components:
    - Dimensionality: n_input, n_output, n_neurons
    - Temporal dynamics: dt_ms, axonal_delay_ms
    - Learning: learn, learning_rate
    - Weight bounds: w_min, w_max
    - Device/dtype: From BaseConfig

    Both regions and pathways inherit from this. Regions typically have
    n_input = n_output = n_neurons (recurrent), while pathways have
    n_input (source) and n_output (target) that differ.
    """

    # Dimensionality
    n_neurons: int = 100
    """Number of neurons in the component."""

    n_input: int = 128
    """Input dimension (for regions: input size, for pathways: source region size)."""

    n_output: int = 64
    """Output dimension (for regions: output size, for pathways: target region size)."""

    # Temporal dynamics
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

    # Learning
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
    # STDP LEARNING PARAMETERS (Component Parity: applies to regions AND pathways)
    # =========================================================================
    # Spike-timing dependent plasticity parameters are needed by all components
    # that perform online learning based on spike correlations.

    learning_rule: str = "STDP"  # SpikingLearningRule enum value
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
    # EXTENDED ELIGIBILITY TRACES (Component Parity: applies to regions AND pathways)
    # =========================================================================
    # Extended eligibility traces for learning rules with delayed modulation
    # (rewards, errors, neuromodulation). This is SEPARATE from fast STDP traces
    # (tau_plus_ms/tau_minus_ms) which handle spike coincidence detection (<50ms).
    #
    # Extended eligibility handles temporal credit assignment over longer timescales
    # (100-1000ms) needed for:
    # - Three-factor rule: eligibility × dopamine (delayed reward)
    # - Error-corrective: eligibility × error (delayed feedback)
    # - Neuromodulated plasticity: eligibility × ACh/NE (delayed context)
    #
    # Biological basis: Calcium transients, PKA/CaMKII cascades, and synaptic tags
    # can persist for seconds, allowing correlation of pre/post activity with
    # delayed modulatory signals.

    eligibility_tau_ms: float = 1000.0
    """Time constant for extended eligibility traces in milliseconds.

    This is for DELAYED modulation (100-1000ms after spike correlation).
    Fast STDP uses tau_plus_ms/tau_minus_ms (~20ms) for coincidence detection.

    Typical values:
    - Striatum (RL): 1000ms (dopamine arrives 100-500ms after action)
    - Cerebellum (error): 1000ms (climbing fiber error delayed)
    - Cortex (neuromod): 500-1000ms (ACh/NE context signals)
    - Pathways: Usually matches connected regions

    Set to 0 to disable extended eligibility (pure STDP only).
    """

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION (Component Parity: applies to regions AND pathways)
    # =========================================================================
    # Spike-frequency adaptation (SFA) is universal in pyramidal neurons via
    # Ca²⁺-activated K⁺ channels (AHP currents). Adaptation prevents runaway
    # activity, enables temporal decorrelation, and implements gain control.
    #
    # Mechanism: Each spike increases an adaptation current that hyperpolarizes
    # the neuron, making subsequent spikes harder to generate. The adaptation
    # current decays exponentially with time constant adapt_tau.
    #
    # Biological basis: Present in ALL cortical pyramidal neurons, hippocampal
    # pyramidal cells, and other excitatory neurons. Only magnitude and timescale
    # vary across cell types.
    #
    # References:
    # - Madison & Nicoll (1984): Norepinephrine blocks accommodation in hippocampal pyramids
    # - Sanchez-Vives & McCormick (2000): Cellular mechanisms of long-lasting adaptation

    adapt_increment: float = 0.0
    """Adaptation current increase per spike (0 = disabled).

    Universal mechanism in pyramidal neurons via Ca²⁺-activated K⁺ channels.
    Each spike increases adaptation current by this amount, making subsequent
    spikes harder to generate (gain control, decorrelation).

    Typical values:
    - 0.0: Disabled (relay neurons, some interneurons)
    - 0.2: Moderate adaptation (PFC, maintains working memory while adapting)
    - 0.3: Strong adaptation (cortex L2/3, prevents dominance, decorrelates)
    - 0.5: Very strong (hippocampus CA3, prevents seizure-like activity)

    Biology: AHP magnitude varies across cell types but mechanism is universal.
    """

    adapt_tau: float = 100.0
    """Adaptation decay time constant in milliseconds.

    Controls how fast adaptation current decays. Biological range: 50-200ms.
    Longer tau = more persistent adaptation = longer decorrelation timescale.
    """

    # =========================================================================
    # HETEROSYNAPTIC COMPETITION (Component Parity: applies to regions AND pathways)
    # =========================================================================
    # Heterosynaptic competition implements competitive dynamics where strengthening
    # active synapses weakens inactive ones. This is a universal mechanism for
    # synaptic competition and resource allocation.
    #
    # Mechanism: When learning strengthens active synapses, inactive synapses on
    # the same postsynaptic neuron receive LTD. This creates zero-sum competition
    # for limited plasticity resources (protein synthesis, trafficking).
    #
    # Biological basis: Synaptic consolidation requires local protein synthesis
    # which is limited. Active synapses capture these resources, leaving less for
    # inactive synapses. This implements competitive allocation.
    #
    # References:
    # - Chistiakova & Volgushev (2009): Heterosynaptic plasticity in neocortex
    # - Fonseca et al. (2004): Competing for memory via synaptic tagging

    heterosynaptic_competition: bool = True
    """Enable heterosynaptic competition (universal competitive mechanism).

    When True, strengthening active synapses causes LTD in inactive synapses
    on the same neuron. Implements competitive resource allocation.
    """

    heterosynaptic_ratio: float = 0.3
    """Fraction of LTD applied to non-active synapses during learning (0-1).

    When learning causes LTP in active synapses, inactive synapses receive:
        Δw_inactive = -heterosynaptic_ratio × |Δw_active|

    Typical values:
    - 0.0: Disabled (pure Hebbian, no competition)
    - 0.2: Weak competition (cerebellum, needs fast convergence)
    - 0.3: Standard competition (cortex, striatum, PFC, pathways)
    - 0.5: Strong competition (sparse coding, winner-take-all)

    Biology: Models limited protein synthesis creating zero-sum competition.
    """

    # =========================================================================
    # HOMEOSTATIC PLASTICITY (Component Parity: applies to regions AND pathways)
    # =========================================================================
    # Unified constraint-based homeostasis maintains stable dynamics and prevents
    # pathological states. Both regions and pathways need homeostasis because they
    # both have:
    # - Synaptic weights that can grow unbounded during learning
    # - Activity patterns that can become pathological (silent or saturated)
    #
    # Biological basis: All neural tissue has homeostatic mechanisms to maintain
    # stable firing rates and prevent epilepsy/silence. This is universal.
    #
    # Implementation: Uses UnifiedHomeostasis for constraint-based regulation:
    # - Weight normalization: Each neuron's total input is bounded
    # - Activity regulation: Population activity is constrained
    # - Competitive adjustment: Strong weights suppress weak ones
    #
    # References:
    # - Turrigiano & Nelson (2004): Homeostatic plasticity in the developing nervous system
    # - Turrigiano (2008): The self-tuning neuron: synaptic scaling of excitatory synapses
    # - Desai et al. (1999): Plasticity in the intrinsic excitability of cortical pyramidal neurons

    # Enable/disable
    homeostasis_enabled: bool = True
    """Enable homeostatic regulation (weight normalization and activity control)."""

    # Weight constraints
    weight_budget: float = 1.0
    """Target sum of weights per neuron (row normalization constraint).

    NOTE: This is typically scaled by n_input when creating UnifiedHomeostasisConfig.
    For example, if you want each synapse to average 0.3, set weight_budget=0.3
    and then use weight_budget * n_input when initializing UnifiedHomeostasis.

    Default of 1.0 assumes you'll scale it appropriately for your architecture.
    """

    # Activity constraints
    activity_target: float = 0.1
    """Target fraction of neurons active per timestep."""

    activity_min: float = 0.01
    """Minimum activity level (prevents dead neurons)."""

    activity_max: float = 0.5
    """Maximum activity level (prevents seizure-like states)."""

    # Normalization settings
    normalize_rows: bool = True
    """Normalize each neuron's input weights (standard for feedforward/recurrent)."""

    normalize_cols: bool = False
    """Normalize each input's output weights (useful for sensory normalization)."""

    # Neuron model
    neuron_type: str = "lif"
    """Type of neuron model: 'lif', 'conductance', 'dendritic'."""

    target_firing_rate_hz: float = 5.0
    """Target firing rate in Hz for homeostatic regulation.

    Typical biological values:
    - Cortex: 1-10 Hz
    - Hippocampus: 0.1-5 Hz
    - Striatum: 0.5-5 Hz (MSNs)
    """

    homeostatic_tau_ms: float = 10000.0
    """Time constant for homeostatic adaptation in milliseconds.

    Controls how quickly neurons adjust their excitability.
    Typical value: 10000ms = 10s (slow adaptation).
    """

    soft_normalization: bool = True
    """Use soft (multiplicative) normalization instead of hard constraint enforcement."""

    normalization_rate: float = 0.1
    """Rate of convergence toward target (soft normalization only)."""

    activity_tau_ms: float = 1000.0
    """Time constant for activity rate estimation (exponential moving average)."""

    # Competition settings
    enable_competition: bool = True
    """Enable competitive weight adjustment (winner-take-all dynamics)."""

    competition_strength: float = 0.1
    """Strength of competitive suppression between neurons."""


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

    Inherits common parameters from NeuralComponentConfig:
    - n_input, n_output: Source and target region sizes
    - n_neurons: Intermediate neuron population (set to n_output)
    - dt_ms, device, dtype, seed: From NeuralComponentConfig
    - w_min, w_max: Weight bounds
    - learning_rate: Base learning rate (0.001)
    - stdp_lr: STDP learning rate (0.01)
    - axonal_delay_ms: Conduction delay (OVERRIDDEN to 5.0ms for long-range)

    All inter-region pathways in Thalia are spike-based, implementing:
    - Leaky integrate-and-fire neurons
    - STDP learning
    - Temporal coding schemes
    - Axonal delays and synaptic filtering

    Pathway-Specific Defaults:
    - axonal_delay_ms: 5.0ms (inter-region typical, vs 1.0ms for local)
    - adapt_increment: 0.0 (pathways are relay neurons, not pyramidal)
    - learning_rule: "STDP" (spike-timing dependent plasticity)

    Example:
        config = PathwayConfig(
            n_input=128,   # Source region size
            n_output=64,   # Target region size
            stdp_lr=0.01,
            temporal_coding=TemporalCoding.PHASE,
        )
    """

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
        self.n_neurons = self.n_output

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
