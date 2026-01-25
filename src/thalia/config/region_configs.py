"""
Centralized Configuration for All Neural Regions.

This module consolidates ALL region configuration dataclasses from the Thalia
architecture. Previously scattered across region subdirectories and inline in
implementation files, they are now centralized for:
- Easy discovery and comparison
- Consistent validation patterns
- Simplified imports
- Better documentation

**Architectural Decision (Tier 2.1)**:
Configurations define BEHAVIORAL parameters only (learning rates, time constants,
strengths, etc.). Sizes (n_neurons, input_size, etc.) are passed separately at
instantiation via the sizes dictionary pattern.

**Usage**:
```python
from thalia.config.region_configs import HippocampusConfig, StriatumConfig

# Create config with behavioral parameters
config = HippocampusConfig(learning_rate_stdp=0.01, theta_freq_hz=8.0)

# Pass sizes separately at instantiation
sizes = {"dg_size": 1000, "ca3_size": 500, "ca1_size": 400}
hippocampus = TrisynapticHippocampus(config=config, sizes=sizes, device="cpu")
```

**Organization**:
Configs are ordered by functional system:
1. Memory & Learning: Hippocampus, Cerebellum
2. Action Selection: Striatum, Prefrontal
3. Sensory Processing: Thalamus, Cortex, Multisensory

Each config includes:
- Docstring with biological context
- Inheritance from base classes (NeuralComponentConfig + learning mixins)
- Companion State class (where applicable)
- Default values from biological literature

**Companion State Classes**:
Region State classes are co-located with their configs for cohesion.
All state classes inherit from BaseRegionState to ensure protocol compliance.

Author: Thalia Project
Date: January 2026 (Tier 2.1 Consolidation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

from thalia.components.synapses.stp import STPType
from thalia.config.learning_config import (
    ErrorCorrectiveLearningConfig,
    HebbianLearningConfig,
    ModulatedLearningConfig,
    STDPLearningConfig,
)
from thalia.constants import (
    ADAPT_INCREMENT_CORTEX_L23,
    DEFAULT_EPSILON_EXPLORATION,
    EMA_DECAY_FAST,
    HIPPOCAMPUS_SPARSITY_TARGET,
    LEARNING_RATE_HEBBIAN_SLOW,
    LEARNING_RATE_ONE_SHOT,
    LEARNING_RATE_PRECISION,
    LEARNING_RATE_STDP,
    STDP_A_MINUS_CORTEX,
    STDP_A_PLUS_CORTEX,
    THALAMUS_ALPHA_GATE_THRESHOLD,
    THALAMUS_ALPHA_SUPPRESSION,
    THALAMUS_BURST_GAIN,
    THALAMUS_BURST_SPIKE_COUNT,
    THALAMUS_BURST_THRESHOLD,
    THALAMUS_CENTER_EXCITATION,
    THALAMUS_RELAY_STRENGTH,
    THALAMUS_SPATIAL_FILTER_WIDTH,
    THALAMUS_SURROUND_INHIBITION,
    THALAMUS_TONIC_THRESHOLD,
    THALAMUS_TRN_INHIBITION,
    THALAMUS_TRN_RECURRENT,
)
from thalia.core.base.component_config import NeuralComponentConfig
from thalia.learning.rules.bcm import BCMConfig

if TYPE_CHECKING:
    from thalia.diagnostics.criticality import CriticalityConfig
    from thalia.learning.ei_balance import EIBalanceConfig
    from thalia.learning.homeostasis.metabolic import MetabolicConfig


# ============================================================================
# MEMORY & LEARNING SYSTEMS
# ============================================================================


@dataclass
class HippocampusConfig(NeuralComponentConfig, STDPLearningConfig):
    """Configuration for hippocampus (trisynaptic circuit).

    Inherits STDP learning parameters from STDPLearningConfig:
    - learning_rate: Base learning rate (overridden with pathway-specific rates below)
    - learning_enabled: Global learning enable/disable
    - weight_min, weight_max: Weight bounds
    - tau_plus_ms, tau_minus_ms: STDP timing window parameters
    - a_plus, a_minus: LTP/LTD amplitudes
    - use_symmetric: Whether to use symmetric STDP

    The hippocampus has ~5x expansion from EC to DG, then compression back.

    **Size Specification**:
    Pure behavioral configuration. Sizes (input_size, dg_size, ca3_size, ca2_size, ca1_size)
    are passed separately via the `sizes` dict parameter to __init__().
    """

    # Override default learning rate with CA3-specific fast learning
    learning_rate: float = LEARNING_RATE_ONE_SHOT  # Fast one-shot learning for CA3 recurrent

    # DG sparsity (VERY sparse for pattern separation)
    dg_sparsity: float = HIPPOCAMPUS_SPARSITY_TARGET
    dg_inhibition: float = 5.0  # Strong lateral inhibition

    # CA3 recurrent dynamics
    ca3_recurrent_strength: float = 0.4  # Strength of recurrent connections
    ca3_sparsity: float = 0.10  # 10% active

    # CA2 dynamics (social memory and temporal context)
    ca2_sparsity: float = 0.12  # 12% active (slightly higher than CA3)
    ca2_plasticity_resistance: float = 0.1  # CA3→CA2 has 10x weaker plasticity (stability hub)

    # CA1 output
    ca1_sparsity: float = 0.15  # 15% active

    # Coincidence detection for comparison
    coincidence_window: float = 5.0  # ms window for spike coincidence

    # Spillover transmission (volume transmission)
    # Enable in hippocampus CA1 and CA3 where experimentally documented
    # (Vizi et al. 1999, Agnati et al. 2010, Sykova 2004)
    # Hippocampal spillover supports pattern completion and memory integration
    enable_spillover: bool = True  # Override base config (disabled by default)
    spillover_mode: str = "connectivity"  # Use shared inputs for neighborhood
    spillover_strength: float = 0.18  # 18% for CA regions (slightly higher than cortex)
    match_threshold: float = 0.3  # Fraction of coincident spikes for match

    # NMDA receptor parameters for CA1 coincidence detection
    # The threshold must be set high enough that only CA1 neurons with STRONG
    # CA3 input get their Mg²⁺ block removed. With ~48 CA3 spikes and 15%
    # connectivity, each CA1 receives 0.2-2.2 weighted input (mean ~1.0).
    # With tau=50ms and 15 test timesteps, the trace reaches ~40% of equilibrium,
    # so threshold=0.4 ensures only neurons with above-average CA3 input participate.
    nmda_tau: float = 50.0  # NMDA time constant (ms) - slow kinetics
    nmda_threshold: float = 0.4  # Threshold tuned for typical test duration
    nmda_steepness: float = 12.0  # Sharp discrimination above threshold
    ampa_ratio: float = 0.05  # Minimal ungated response (discrimination comes from NMDA)

    # Pathway-specific learning rates
    # Note: learning_rate (inherited from STDPLearningConfig) is used for CA3 recurrent
    ca3_ca2_learning_rate: float = 0.001  # Very weak CA3→CA2 (stability mechanism)
    ec_ca2_learning_rate: float = 0.01  # Strong EC→CA2 direct (temporal encoding)
    ca2_ca1_learning_rate: float = 0.005  # Moderate CA2→CA1 (social context to output)
    ec_ca1_learning_rate: float = 0.5  # Strong learning for EC→CA1 alignment

    # Feedforward inhibition parameters
    ffi_threshold: float = 0.3  # Input change threshold to trigger FFI
    ffi_strength: float = 0.8  # How much FFI suppresses activity
    ffi_tau: float = 5.0  # FFI decay time constant (ms)

    # =========================================================================
    # INTER-LAYER AXONAL DELAYS
    # =========================================================================
    # Biological signal propagation times within hippocampal circuit:
    # - DG→CA3 (mossy fibers): ~3ms
    # - CA3→CA2: ~2ms (shorter due to proximity)
    # - CA2→CA1: ~2ms
    # - CA3→CA1 (Schaffer collaterals): ~3ms (direct bypass)
    # Total circuit latency: ~7ms (slightly longer with CA2)
    #
    # Set to 0.0 for instant processing (current behavior, backward compatible)
    # Set to biological values for realistic temporal dynamics and STDP timing
    dg_to_ca3_delay_ms: float = 0.0  # DG→CA3 axonal delay (0=instant)
    ca3_to_ca2_delay_ms: float = 0.0  # CA3→CA2 axonal delay (0=instant)
    ca2_to_ca1_delay_ms: float = 0.0  # CA2→CA1 axonal delay (0=instant)
    ca3_to_ca1_delay_ms: float = 0.0  # CA3→CA1 axonal delay (0=instant, direct bypass)

    # CA3 Bistable Neuron Parameters
    # Real CA3 pyramidal neurons have intrinsic bistability via I_NaP (persistent
    # sodium) and I_CAN (calcium-activated nonspecific cation) currents. These
    # allow neurons to maintain firing without continuous external input.
    #
    # We model this with a "persistent activity" trace that:
    # 1. Accumulates when a neuron fires (like Ca²⁺ buildup activating I_CAN)
    # 2. Decays slowly (τ ~100-200ms, like Ca²⁺ clearance)
    # 3. Provides additional input current (positive feedback)
    #
    # This creates bistability: once a neuron starts firing, the persistent
    # activity helps keep it firing, stabilizing attractor states.
    ca3_persistent_tau: float = 300.0  # Decay time constant (ms) - very slow decay
    ca3_persistent_gain: float = 3.0  # Strong persistent contribution

    # EC Layer III input size (for direct EC→CA1 pathway)
    # If 0, uses the same input as EC layer II (n_input)
    # If >0, expects separate raw sensory input for the temporoammonic path
    ec_l3_input_size: int = 0

    # =========================================================================
    # THETA-GAMMA COUPLING
    # =========================================================================
    # Enable theta-gamma coupling from centralized oscillator manager
    theta_gamma_enabled: bool = True  # Use centralized oscillators for sequence encoding

    # =========================================================================
    # SHORT-TERM PLASTICITY (STP)
    # =========================================================================
    # Biologically, different hippocampal pathways have distinct STP properties:
    # - Mossy Fibers (DG→CA3): STRONGLY FACILITATING - repeated DG activity
    #   causes progressively stronger CA3 activation (U~0.03, τ_f~500ms)
    # - CA3→CA2: DEPRESSING - stability mechanism, prevents runaway activity
    # - CA2→CA1: FACILITATING - temporal sequences benefit from facilitation
    # - Schaffer Collaterals (CA3→CA1): MIXED/DEPRESSING - high-frequency
    #   activity causes depression, enabling novelty detection
    # - EC→CA1 direct: DEPRESSING - initial stimulus is strongest
    # - EC→CA2 direct: DEPRESSING - similar to EC→CA1
    #
    # References:
    # - Salin et al. (1996): Mossy fiber facilitation (U=0.03!)
    # - Dobrunz & Stevens (1997): Schaffer collateral STP
    # - Chevaleyre & Siegelbaum (2010): CA2 plasticity properties
    stp_enabled: bool = True
    stp_mossy_type: STPType = STPType.FACILITATING_STRONG  # DG→CA3 (MF)
    stp_ca3_ca2_type: STPType = STPType.DEPRESSING  # CA3→CA2 (stability)
    stp_ca2_ca1_type: STPType = STPType.FACILITATING  # CA2→CA1 (sequences)
    stp_ec_ca2_type: STPType = STPType.DEPRESSING  # EC→CA2 direct
    stp_schaffer_type: STPType = STPType.DEPRESSING  # CA3→CA1 (SC)
    stp_ec_ca1_type: STPType = STPType.DEPRESSING  # EC→CA1 direct
    # CA3→CA3 recurrent: DEPRESSING - prevents frozen attractors
    # Without STD, the same neurons fire every timestep because recurrent
    # connections reinforce active neurons. With STD, frequently-firing
    # synapses get temporarily weaker, allowing pattern transitions.
    stp_ca3_recurrent_type: STPType = STPType.DEPRESSING_FAST

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION (SFA)
    # =========================================================================
    # Real CA3 pyramidal neurons show strong adaptation: Ca²⁺ influx during
    # spikes activates K⁺ channels (I_AHP) that hyperpolarize the neuron.
    # This prevents the same neurons from dominating activity.
    # Inherited from base with hippocampus-specific override:
    adapt_increment: float = 0.5  # Very strong (prevents CA3 seizure-like activity)
    # adapt_tau: 100.0 (use base default)

    # =========================================================================
    # ACTIVITY-DEPENDENT INHIBITION
    # =========================================================================
    # Feedback inhibition from interneurons scales with total CA3 activity.
    # When many CA3 neurons fire, inhibition increases, making it harder
    # for the same neurons to fire again.
    ca3_feedback_inhibition: float = 0.3  # Inhibition per total activity

    # =========================================================================
    # GAP JUNCTIONS (Electrical Synapses)
    # =========================================================================
    # Gap junctions between CA1 interneurons (basket cells, bistratified cells)
    # provide fast electrical coupling for theta-gamma synchronization.
    # Critical for precise spike timing in episodic memory encoding/retrieval.
    #
    # Biological evidence:
    # - Fukuda & Kosaka (2000): Gap junctions in hippocampal GABAergic networks
    # - Traub et al. (2003): Electrical coupling essential for gamma in CA1
    # - Hormuzdi et al. (2001): Connexin36-mediated interneuron coupling
    #
    # CA1 interneurons (~10-15% of CA1 population) have dense gap junction
    # networks that synchronize inhibition during theta-gamma nested oscillations.
    gap_junctions_enabled: bool = True  # Enable gap junctions in CA1 interneurons
    gap_junction_strength: float = 0.12  # Coupling strength (biological: 0.05-0.2)
    gap_junction_threshold: float = 0.25  # Neighborhood connectivity threshold
    gap_junction_max_neighbors: int = 8  # Max neighbors per interneuron (biological: 4-12)

    # =========================================================================
    # HETEROSYNAPTIC PLASTICITY
    # =========================================================================
    # Synapses to inactive postsynaptic neurons weaken when nearby neurons
    # fire strongly. This prevents winner-take-all dynamics from freezing.
    heterosynaptic_ratio: float = 0.1  # LTD for inactive synapses

    # =========================================================================
    # THETA-PHASE RESETS
    # =========================================================================
    # Reset persistent activity at the start of each theta cycle to prevent
    # stale attractors from dominating. In real brains, theta troughs
    # (encoding phase) partially reset the network.
    theta_reset_persistent: bool = True  # Reset ca3_persistent at theta trough
    theta_reset_fraction: float = 0.5  # How much to decay (0=none, 1=full)

    # =========================================================================
    # THETA-GAMMA COUPLING (Phase Coding - EMERGENT)
    # =========================================================================
    # Note: Theta-gamma coupling (frequency, strength) is handled by the
    # centralized OscillatorManager. Phase preferences EMERGE from:
    # 1. Synaptic delays (different neurons receive inputs at different times)
    # 2. STDP (neurons strengthen connections at their preferred phase)
    # 3. Dendritic integration (~15ms window naturally filters by timing)
    #
    # Working memory capacity = gamma_freq / theta_freq (~40Hz / 8Hz ≈ 5-7 slots)
    # This emerges automatically - no hardcoded slots needed!

    # Phase diversity initialization: adds timing jitter to initial weights
    # This seeds the emergence of phase preferences (otherwise all neurons identical)
    phase_diversity_init: bool = True  # Initialize weights with timing diversity
    phase_jitter_std_ms: float = 5.0  # Std dev of timing jitter (0-10ms)

    # =========================================================================
    # MULTI-TIMESCALE CONSOLIDATION (Phase 1A Enhancement)
    # =========================================================================
    # Biological reality: Memory consolidation operates over multiple timescales
    # - Fast trace (synaptic tagging): Minutes, ~60s tau
    # - Slow trace (systems consolidation): Hours, ~3600s tau
    # - Consolidation: Gradual transfer from fast (episodic) to slow (semantic)
    #
    # This implements "systems consolidation theory" (McClelland et al., 1995):
    # - Fast learning in hippocampus captures episodic details
    # - Slow consolidation transfers statistical regularities to neocortex
    # - Gradual interleaving prevents catastrophic forgetting
    #
    # References:
    # - McClelland et al. (1995): Why there are complementary learning systems
    # - Dudai et al. (2015): The consolidation and transformation of memory
    # - Frankland & Bontempi (2005): Organization of recent and remote memories
    use_multiscale_consolidation: bool = False  # Enable multi-timescale traces
    fast_trace_tau_ms: float = 60_000.0  # Fast trace decay (1 minute = 60,000ms)
    slow_trace_tau_ms: float = 3_600_000.0  # Slow trace decay (1 hour = 3,600,000ms)
    consolidation_rate: float = 0.001  # Transfer rate from fast to slow (0.1% per timestep)
    slow_trace_contribution: float = 0.1  # Weight of slow trace in learning (10%)


@dataclass
class CerebellumConfig(NeuralComponentConfig, ErrorCorrectiveLearningConfig):
    """Configuration specific to cerebellar regions.

    The cerebellum implements ERROR-CORRECTIVE learning through:
    1. Parallel fiber → Purkinje cell connections (learned)
    2. Climbing fiber error signals from inferior olive
    3. LTD when climbing fiber active with parallel fiber

    Key biological features:
    - Error signal triggers immediate learning (not delayed like RL)
    - Can learn arbitrary input-output mappings quickly
    - Uses eligibility traces for temporal credit assignment

    Inherits from NeuralComponentConfig (structural) then ErrorCorrectiveLearningConfig (behavioral):
    - learning_rate_ltp: LTP rate (default 0.01)
    - learning_rate_ltd: LTD rate (default 0.01)
    - error_threshold: Minimum error (default 0.01)
    - use_eligibility_traces: Enable traces (default True)
    - eligibility_tau_ms: Trace decay (default 20.0)

    **Size Specification** (Semantic-First):
    - Sizes passed via sizes dict: input_size, granule_size, purkinje_size
    - Computed at instantiation: output_size (= purkinje_size), total_neurons (granule + purkinje or just purkinje)
    - Config contains only behavioral parameters (learning rates, circuit flags, etc.)
    """

    # Temporal processing
    temporal_window_ms: float = 10.0  # Window for coincidence detection

    # Cerebellum uses weaker heterosynaptic competition for faster convergence:
    heterosynaptic_ratio: float = 0.2  # Override base (0.3) - weaker competition

    # Input trace parameters
    input_trace_tau_ms: float = 20.0  # Input trace decay

    # =========================================================================
    # ENHANCED MICROCIRCUIT (optional, for increased biological detail)
    # =========================================================================
    # When enabled, uses granule→Purkinje→DCN circuit instead of direct
    # parallel fiber→Purkinje mapping. Provides:
    # - 4× sparse expansion in granule layer (pattern separation)
    # - Dendritic computation in Purkinje cells (complex/simple spikes)
    # - DCN integration (Purkinje sculpts tonic output)
    use_enhanced_microcircuit: bool = True

    granule_sparsity: float = 0.03  # Fraction of granule cells active (3%)
    purkinje_n_dendrites: int = 100  # Simplified dendritic compartments

    # =========================================================================
    # SHORT-TERM PLASTICITY (STP) - CRITICAL FOR CEREBELLAR TIMING
    # =========================================================================
    # Biologically, cerebellar synapses show distinct STP properties that are
    # CRITICAL for temporal processing and motor timing:
    #
    # 1. PARALLEL FIBERS→PURKINJE: DEPRESSING (U=0.5-0.7)
    #    - Implements temporal high-pass filter
    #    - Fresh inputs signal new patterns
    #    - Sustained inputs fade → cerebellum detects CHANGES, not steady-state
    #    - Enables sub-millisecond timing discrimination
    #    - WITHOUT THIS: Cerebellar timing precision COLLAPSES
    #
    # 2. MOSSY FIBERS→GRANULE CELLS: FACILITATING (U=0.15-0.25)
    #    - Burst detection for sparse coding
    #    - Amplifies repeated mossy fiber activity
    #    - Enhances pattern separation in granule layer
    #
    # 3. CLIMBING FIBERS→PURKINJE: RELIABLE (U≈0.9, minimal STP)
    #    - Error signal must be unambiguous
    #    - No adaptation - every climbing fiber spike matters
    #
    # References:
    # - Dittman et al. (2000): Nature 403:530-534 - Classic PF→Purkinje STP paper
    # - Atluri & Regehr (1996): Delayed release at granule cell synapses
    # - Isope & Barbour (2002): Facilitation at mossy fiber synapses
    #
    # BIOLOGICAL IMPORTANCE: This is perhaps the MOST important STP in the brain
    # for motor learning and timing. The cerebellar cortex is the brain's master
    # clock, and STP is essential for its temporal precision.
    stp_enabled: bool = True
    stp_pf_purkinje_type: STPType = STPType.DEPRESSING  # Parallel fiber depression
    stp_mf_granule_type: STPType = STPType.FACILITATING  # Mossy fiber facilitation

    # =========================================================================
    # GAP JUNCTIONS (Inferior Olive Synchronization)
    # =========================================================================
    # Inferior olive (IO) neurons are electrically coupled via gap junctions,
    # creating synchronized complex spikes across multiple Purkinje cells.
    # This coordination is critical for motor learning and timing precision.
    #
    # Biology:
    # - IO neurons form one of the densest gap junction networks in the brain
    # - Synchronization time: <1ms (ultra-fast electrical coupling)
    # - Functional role: Coordinates learning across multiple cerebellar modules
    # - Complex spikes arrive synchronously at related Purkinje cells
    #
    # References:
    # - Llinás & Yarom (1981): Electrophysiology of IO gap junctions
    # - De Zeeuw et al. (1998): Gap junctions in IO create synchronous climbing fiber activity
    # - Leznik & Llinás (2005): Role of gap junctions in IO oscillations
    # - Schweighofer et al. (1999): Computational role of IO synchronization
    gap_junctions_enabled: bool = True
    """Enable gap junction coupling in inferior olive neurons."""

    gap_junction_strength: float = 0.18
    """Gap junction conductance for IO neurons (biological: 0.1-0.3, IO has stronger coupling)."""

    gap_junction_threshold: float = 0.20
    """Connectivity threshold for gap junction coupling (shared error patterns)."""

    gap_junction_max_neighbors: int = 12
    """Maximum gap junction neighbors per IO neuron (biological: 6-15, IO is densely coupled)."""

    # =========================================================================
    # COMPLEX SPIKE DYNAMICS (Phase 2B Enhancement)
    # =========================================================================
    # Climbing fibers trigger complex spikes in Purkinje cells - bursts of 2-7
    # spikes with very short inter-spike intervals (1-2ms). These bursts encode
    # ERROR MAGNITUDE: larger errors → longer bursts → more calcium influx.
    #
    # This provides GRADED ERROR SIGNALING instead of binary (error/no-error),
    # enabling more nuanced learning: small errors trigger small corrections,
    # large errors trigger large corrections.
    #
    # Biological Evidence:
    # - Mathy et al. (2009): Complex spikes have 2-7 spikelets per burst
    # - Davie et al. (2008): Number of spikelets correlates with error magnitude
    # - Najafi & Medina (2013): Graded complex spikes enable graded learning
    # - Yang & Lisberger (2014): Complex spike amplitude predicts learning rate
    #
    # Mechanism:
    # 1. Climbing fiber error → complex spike burst (not single spike)
    # 2. Each spikelet triggers dendritic calcium influx (~0.2 units)
    # 3. Total calcium = n_spikes × ca2_per_spike
    # 4. LTD magnitude ∝ total calcium (graded learning)
    #
    # Example:
    # - Small error (0.2): 3 spikes → Ca²⁺ = 0.6 → small LTD
    # - Large error (0.8): 6 spikes → Ca²⁺ = 1.2 → large LTD
    #
    # References:
    # - Mathy et al. (2009): Nature Neuroscience 12:1378-1387
    # - Najafi & Medina (2013): J. Neuroscience 33:15825-15840
    # - Yang & Lisberger (2014): Neuron 82:1389-1401
    # - Davie et al. (2008): Nature Neuroscience 11:838-848
    use_complex_spike_bursts: bool = False
    """Enable complex spike burst dynamics (graded error signaling)."""

    min_complex_spike_count: int = 2
    """Minimum spikelets per complex spike burst (biological: 2-3)."""

    max_complex_spike_count: int = 7
    """Maximum spikelets per complex spike burst (biological: 5-8)."""

    complex_spike_isi_ms: float = 1.5
    """Inter-spike interval within burst (biological: 1-2ms, very fast)."""

    ca2_per_spikelet: float = 0.2
    """Calcium influx per spikelet (arbitrary units, scaled for learning)."""


# ============================================================================
# ACTION SELECTION & EXECUTIVE CONTROL
# ============================================================================


@dataclass
class StriatumConfig(ModulatedLearningConfig, NeuralComponentConfig):  # type: ignore[misc]
    """Configuration specific to striatal regions (behavior only, no sizes).

    **Size-Free Config Pattern** (January 2026 Refactoring):
    Config contains ONLY behavioral parameters. Sizes passed separately to __init__.

    Size computation handled by:
    - LayerSizeCalculator.striatum_from_actions() for size dicts
    - Striatum.__init__(config, sizes, device) receives computed sizes

    Behavioral parameters:
    - Learning rates (learning_rate, d1_lr_scale, d2_lr_scale)
    - Action selection (lateral_inhibition, softmax_temperature)
    - Exploration (ucb_coefficient, adaptive_exploration)
    - Neuromodulation sensitivity (d1_da_sensitivity, d2_da_sensitivity)

    Inherits dopamine-gated learning parameters from ModulatedLearningConfig:
    - learning_rate: Base learning rate for synaptic updates
    - learning_enabled: Global learning enable/disable
    - weight_min, weight_max: Weight bounds
    - modulator_threshold: Minimum dopamine level to enable learning
    - modulator_sensitivity: Scaling factor for dopamine influence
    - use_dopamine_gating: Whether to gate learning by dopamine levels

    Key Features:
    =============
    1. THREE-FACTOR LEARNING: Δw = eligibility × dopamine
    2. D1/D2 OPPONENT PATHWAYS: Go/No-Go balance
    3. POPULATION CODING: Multiple neurons per action
    4. ADAPTIVE EXPLORATION: UCB + uncertainty-driven

    Note: Dopamine computation has been centralized at the Brain level
    (Brain acts as VTA). Striatum receives dopamine via set_dopamine().
    """

    # =========================================================================
    # D1/D2 PATHWAY CONFIGURATION
    # =========================================================================
    # D1/D2 split ratio (behavioral parameter, not a size)
    d1_d2_ratio: float = 0.5
    """Fraction of neurons allocated to D1 pathway (0.5 = equal split)."""

    # Override default learning rate (5x base for faster RL updates)
    learning_rate: float = 0.005
    # Note: stdp_lr and tau_plus_ms/tau_minus_ms inherited from NeuralComponentConfig

    # Action selection
    lateral_inhibition: bool = True
    inhibition_strength: float = 2.0

    # =========================================================================
    # D1/D2 OPPONENT PATHWAYS
    # =========================================================================
    d1_lr_scale: float = 1.0
    d2_lr_scale: float = 1.0
    d1_da_sensitivity: float = 1.0
    d2_da_sensitivity: float = 1.0

    # =========================================================================
    # HOMEOSTATIC PLASTICITY
    # =========================================================================
    # NOTE: weight_budget is computed dynamically from initialized weights
    # to automatically adapt to any architecture (population_coding, n_input, etc.)
    homeostatic_soft: bool = True
    homeostatic_rate: float = 0.1
    activity_decay: float = EMA_DECAY_FAST  # EMA decay for activity tracking (~100 timestep window)

    # Note: heterosynaptic_competition and heterosynaptic_ratio inherited from base
    # Striatum-specific competition handled via baseline_pressure mechanism below

    # =========================================================================
    # BASELINE PRESSURE (drift towards balanced D1/D2)
    # =========================================================================
    baseline_pressure_enabled: bool = True
    baseline_pressure_rate: float = 0.015
    baseline_target_net: float = 0.0

    # =========================================================================
    # SOFTMAX ACTION SELECTION
    # =========================================================================
    softmax_action_selection: bool = True
    softmax_temperature: float = 2.0

    # =========================================================================
    # ADAPTIVE EXPLORATION (performance-based)
    # =========================================================================
    adaptive_exploration: bool = True
    performance_window: int = 10
    performance_exploration_scale: float = 0.3
    min_tonic_dopamine: float = 0.1
    max_tonic_dopamine: float = 0.5

    # =========================================================================
    # UCB EXPLORATION BONUS
    # =========================================================================
    ucb_exploration: bool = True
    ucb_coefficient: float = 2.0

    # =========================================================================
    # UNCERTAINTY-DRIVEN EXPLORATION
    # =========================================================================
    uncertainty_temperature: float = 0.05
    min_exploration_boost: float = 0.05

    # =========================================================================
    # TONIC vs PHASIC DOPAMINE
    # =========================================================================
    tonic_dopamine: float = 0.3
    tonic_modulates_d1_gain: bool = True
    tonic_d1_gain_scale: float = 0.5
    tonic_modulates_exploration: bool = True
    tonic_exploration_scale: float = 0.1

    # =========================================================================
    # BETA OSCILLATION MODULATION (Motor Control)
    # =========================================================================
    # Beta amplitude modulates D1/D2 balance for action maintenance vs switching
    # High beta → action persistence (D1 dominant, D2 suppressed)
    # Low beta → action flexibility (D2 effective, D1 reduced)
    beta_modulation_strength: float = 0.3  # [0, 1] - strength of beta influence

    @classmethod
    def from_n_actions(
        cls,
        n_actions: int,
        neurons_per_action: int = 10,
        input_sources: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> StriatumConfig:
        """Create config with pathway sizes computed from number of actions.

        Uses biological ratio: D1 and D2 are equal in size (50/50 split)

        Args:
            n_actions: Number of discrete actions
            neurons_per_action: Neurons per action in population coding
            input_sources: Multi-source inputs dict (required)
            **kwargs: Additional config parameters (d1_d2_ratio, learning_rate, etc.)

        Returns:
            StriatumConfig with computed D1/D2 pathway sizes

        Example:
            >>> config = StriatumConfig.from_n_actions(
            ...     n_actions=4,
            ...     input_sources={"cortex": 256, "thalamus": 128},
            ...     device="cpu"
            ... )
            >>> config.d1_size  # 20 (with default neurons_per_action=10, d1_d2_ratio=0.5)
            >>> config.d2_size  # 20
            >>> config.n_actions  # 4
        """
        if input_sources is None:
            raise ValueError(
                "input_sources is required for StriatumConfig. "
                "Example: input_sources={'cortex': 256, 'thalamus': 128}"
            )

        # Note: Size computation now handled by LayerSizeCalculator.
        # This method is deprecated - use standard config construction.
        return cls(**kwargs)

    # =========================================================================
    # FSI (FAST-SPIKING INTERNEURONS) - Parvalbumin+ Interneurons
    # =========================================================================
    # FSI are ~2% of striatal neurons (vs 95% MSNs) but critical for timing:
    # - Feedforward inhibition sharpens action selection
    # - Gap junction networks enable ultra-fast synchronization (<0.1ms)
    # - Synchronize MSN activity during beta oscillations (13-30 Hz)
    # - Critical for action initiation timing and motor control
    # Biology: Koós & Tepper (1999), Gittis et al. (2010)
    fsi_enabled: bool = True
    fsi_ratio: float = 0.02  # FSI as fraction of total striatal neurons (2%)

    # Gap junction configuration for FSI networks
    gap_junctions_enabled: bool = True
    gap_junction_strength: float = 0.15  # Biological: 0.05-0.3
    gap_junction_threshold: float = 0.25  # Neighborhood inference threshold
    gap_junction_max_neighbors: int = 10  # Biological: 4-12 neighbors

    # =========================================================================
    # GOAL-CONDITIONED VALUES
    # =========================================================================
    # Enable PFC goal context to modulate striatal action values
    # Biology: PFC → Striatum projections gate action selection by goal context
    use_goal_conditioning: bool = True  # Enable goal-conditioned value learning
    goal_modulation_strength: float = 0.5  # How strongly goals modulate values

    # =========================================================================
    # D1/D2 PATHWAY DELAYS (Temporal Competition)
    # =========================================================================
    # Biological timing for opponent pathways creates temporal competition:
    # - D1 "Go" pathway: Striatum → GPi/SNr → Thalamus (~15-20ms total)
    #   Direct inhibition of GPi/SNr → disinhibits thalamus → facilitates action
    # - D2 "No-Go" pathway: Striatum → GPe → STN → GPi/SNr (~23-28ms total)
    #   Indirect route via GPe and STN → inhibits thalamus → suppresses action
    # - Key insight: D1 pathway is ~8ms FASTER than D2 pathway
    #   Creates temporal competition window where D1 "vote" arrives first,
    #   D2 "veto" arrives later. Explains action selection timing and impulsivity.
    d1_to_output_delay_ms: float = 15.0  # D1 direct pathway delay
    d2_to_output_delay_ms: float = 25.0  # D2 indirect pathway delay (slower!)

    # =========================================================================
    # SHORT-TERM PLASTICITY (STP)
    # =========================================================================
    # Biologically, different striatal input pathways have distinct STP properties:
    # - Cortex→MSNs: DEPRESSING (U=0.4) - prevents sustained cortical input from
    #   saturating striatum, enables novelty detection (fresh inputs get stronger)
    # - Thalamus→MSNs: WEAK FACILITATION (U=0.25) - phasic input amplification,
    #   balances phasic (thalamus) and tonic (cortex) command signals
    #
    # References:
    # - Charpier et al. (1999): Corticostriatal EPSPs
    # - Partridge et al. (2000): Synaptic plasticity in striatum
    # - Ding et al. (2008): Thalamostriatal facilitation
    stp_enabled: bool = True  # Enable STP by default
    # NOTE: STP types use presets from stp_presets.py ("corticostriatal", "thalamostriatal")

    # Heterogeneous STP
    # Biological: Within same pathway, U varies 10-fold across synapses (Dobrunz & Stevens 1997)
    # Enables more realistic synaptic diversity and temporal dynamics
    heterogeneous_stp: bool = False  # Enable per-synapse STP parameter sampling
    stp_variability: float = 0.3  # Coefficient of variation (0.2-0.5 typical)
    stp_seed: Optional[int] = None  # Random seed for reproducibility

    # =========================================================================
    # MULTI-TIMESCALE ELIGIBILITY TRACES
    # =========================================================================
    # Biological: Synaptic tags (eligibility traces) have multiple timescales:
    # - Fast traces (~500ms): Immediate pre-post spike coincidence tagging
    # - Slow traces (~60s): Consolidated tags from fast traces, enables credit
    #   assignment over multiple seconds (e.g., delayed rewards in RL tasks)
    # Combined eligibility: fast_trace + α × slow_trace enables both rapid and
    # delayed credit assignment. Biology: Yagishita et al. (2014), Shindou et al. (2019)
    use_multiscale_eligibility: bool = False  # Enable fast + slow eligibility traces
    fast_eligibility_tau_ms: float = 500.0  # Fast trace decay (~500ms)
    slow_eligibility_tau_ms: float = 60000.0  # Slow trace decay (~60s)
    eligibility_consolidation_rate: float = (
        0.01  # Transfer rate from fast to slow (1% per timestep)
    )
    slow_trace_weight: float = 0.3  # Weight of slow trace in combined eligibility

    # =========================================================================
    # ELASTIC TENSOR CHECKPOINT FORMAT (Phase 1 - Growth Support)
    # =========================================================================
    # Enable elastic tensor format for checkpoint-growth compatibility.
    # Pre-allocates tensors with reserved capacity to enable fast growth.
    # Biology: Analogous to neural reserve capacity in brain development.
    growth_enabled: bool = True  # Enable elastic tensor format
    reserve_capacity: float = 0.5  # Fraction of extra capacity (0.5 = 50% headroom)
    # Example: 10 neurons with reserve_capacity=0.5 → allocate 15 neurons worth of memory
    # Growth within reserved space requires no reallocation (fast)
    # Growth beyond capacity triggers reallocation with new headroom (slower)


@dataclass
class PrefrontalConfig(NeuralComponentConfig):
    """Configuration specific to prefrontal cortex.

    PFC implements DOPAMINE-GATED STDP:
    - STDP creates eligibility traces from spike timing
    - Dopamine gates what enters working memory and what gets learned
    - High DA → update WM and learn new associations
    - Low DA → maintain WM and protect existing patterns

    **Size Specification** (Semantic-First):
    - Sizes passed via sizes dict: input_size, n_neurons
    - Computed at instantiation: output_size (= n_neurons), total_neurons (= n_neurons)
    - Config contains only behavioral parameters (learning rates, time constants, etc.)
    """

    # Working memory parameters
    wm_decay_tau_ms: float = 500.0  # How fast WM decays (slow!)
    wm_noise_std: float = 0.01  # Noise in WM maintenance

    # Gating parameters
    gate_threshold: float = 0.5  # DA level to open update gate
    gate_strength: float = 2.0  # How strongly gating affects updates

    # Dopamine parameters
    dopamine_tau_ms: float = 100.0  # DA decay time constant
    dopamine_baseline: float = 0.2  # Tonic DA level

    # Learning rates
    wm_lr: float = 0.1  # Learning rate for WM update weights
    rule_lr: float = LEARNING_RATE_STDP  # Learning rate for rule weights
    # Note: STDP parameters (stdp_lr, tau_plus_ms, tau_minus_ms, a_plus, a_minus)
    # and heterosynaptic_ratio (0.3) are inherited from NeuralComponentConfig

    # Recurrent connections for WM maintenance
    recurrent_strength: float = 0.8  # Self-excitation for persistence
    recurrent_inhibition: float = 0.2  # Lateral inhibition

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION (SFA)
    # =========================================================================
    # PFC pyramidal neurons show adaptation. This helps prevent runaway
    # activity during sustained working memory maintenance.
    # Inherited from base, with PFC-specific overrides:
    adapt_increment: float = 0.2  # Moderate (maintains WM while adapting)
    adapt_tau: float = 150.0  # Slower decay (longer timescale for WM)

    # =========================================================================
    # SHORT-TERM PLASTICITY (STP)
    # =========================================================================
    # Feedforward connections: Facilitation for temporal filtering
    stp_feedforward_enabled: bool = True
    """Enable short-term plasticity on feedforward (input) connections."""

    # Recurrent connections: Depression prevents frozen attractors
    stp_recurrent_enabled: bool = True
    """Enable short-term plasticity on recurrent connections."""

    # =========================================================================
    # PHASE 3: HIERARCHICAL GOALS & TEMPORAL ABSTRACTION
    # =========================================================================
    # Hierarchical goal management and hyperbolic discounting
    use_hierarchical_goals: bool = True
    """Enable hierarchical goal structures (Phase 3).

    When True:
        - Maintains goal hierarchy stack in working memory
        - Tracks active goals and subgoals
        - Supports options framework for reusable policies
        - Requires goal_hierarchy_config
    """

    goal_hierarchy_config: Optional["GoalHierarchyConfig"] = None
    """Configuration for goal hierarchy manager (Phase 3)."""

    use_hyperbolic_discounting: bool = True
    """Enable hyperbolic temporal discounting (Phase 3).

    When True:
        - Hyperbolic (not exponential) discounting of delayed rewards
        - Context-dependent k parameter (cognitive load, stress, fatigue)
        - Adaptive k learning from experience
        - Requires hyperbolic_config
    """

    hyperbolic_config: Optional["HyperbolicDiscountingConfig"] = None
    """Configuration for hyperbolic discounter (Phase 3)."""

    # =========================================================================
    # HETEROGENEOUS WORKING MEMORY (Phase 1B Enhancement)
    # =========================================================================
    # Biological reality: PFC neurons show heterogeneous maintenance properties
    # - Stable neurons: Strong recurrence, long time constants (~1-2s)
    # - Flexible neurons: Weak recurrence, short time constants (~100-200ms)
    #
    # This heterogeneity enables:
    # - Stable neurons: Maintain context/goals over long delays
    # - Flexible neurons: Rapid updating for new information
    # - Mixed selectivity: Distributed representations across neuron types
    #
    # References:
    # - Rigotti et al. (2013): Mixed selectivity in prefrontal cortex
    # - Murray et al. (2017): Stable population coding for working memory
    # - Wasmuht et al. (2018): Intrinsic neuronal dynamics in PFC
    use_heterogeneous_wm: bool = False  # Enable heterogeneous WM neurons
    stability_cv: float = 0.3  # Coefficient of variation for recurrent strength
    tau_mem_min: float = 100.0  # Minimum membrane time constant (ms) - flexible neurons
    tau_mem_max: float = 500.0  # Maximum membrane time constant (ms) - stable neurons

    # =========================================================================
    # D1/D2 DOPAMINE RECEPTOR SUBTYPES (Phase 1B Enhancement)
    # =========================================================================
    # Biological reality: PFC has both D1 (excitatory) and D2 (inhibitory) receptors
    # - D1-dominant neurons (~60%): "Go" pathway, enhance signals with DA
    # - D2-dominant neurons (~40%): "NoGo" pathway, suppress noise with DA
    #
    # This enables:
    # - D1: Update WM when DA high (new information is important)
    # - D2: Maintain WM when DA low (protect current state)
    # - Opponent modulation: D1 and D2 have opposite DA responses
    #
    # References:
    # - Seamans & Yang (2004): D1 and D2 dopamine systems in PFC
    # - Durstewitz & Seamans (2008): Neurocomputational perspective on PFC
    # - Cools & D'Esposito (2011): Inverted-U dopamine in working memory
    use_d1_d2_subtypes: bool = False  # Enable D1/D2 receptor subtypes
    d1_fraction: float = 0.6  # Fraction of neurons that are D1-dominant (60%)
    d1_da_gain: float = 0.5  # DA gain for D1 neurons (excitatory, 1.0 + gain*DA)
    d2_da_gain: float = 0.3  # DA suppression for D2 neurons (inhibitory, 1.0 - gain*DA)
    d2_output_weight: float = 0.5  # Weight of D2 output in competition (D1 - weight*D2)

    def __post_init__(self) -> None:
        """Auto-compute output_size and total_neurons from n_neurons."""
        # Properties handle this now - no manual assignment needed


@dataclass
class GoalHierarchyConfig:
    """Configuration for goal hierarchy manager."""

    max_depth: int = 4  # Maximum hierarchy depth (0=actions, 4=high-level)
    max_active_goals: int = 5  # Limit on parallel goals (working memory constraint)

    # Goal selection
    use_value_based_selection: bool = True  # Select high-value goals
    epsilon_exploration: float = DEFAULT_EPSILON_EXPLORATION  # Explore low-value goals sometimes

    # Goal dynamics
    goal_persistence: float = 0.9  # Resist goal switching (stability)
    deadline_pressure_scale: float = 0.1  # Boost value as deadline approaches

    # Options
    enable_option_learning: bool = True  # Learn reusable policies
    option_discovery_threshold: float = 0.8  # Success rate to cache option

    # Device
    device: str = "cpu"


@dataclass
class HyperbolicDiscountingConfig:
    """Configuration for hyperbolic discounting."""

    # Base discounting
    base_k: float = 0.01  # Base hyperbolic discount rate
    k_min: float = 0.001  # Minimum k (most patient)
    k_max: float = 0.20  # Maximum k (most impulsive)

    # Context modulation
    cognitive_load_scale: float = 0.5  # How much load affects k
    stress_scale: float = 0.3  # How much stress affects k
    fatigue_scale: float = 0.2  # How much fatigue affects k

    # Learning
    learn_k: bool = True  # Adapt k based on outcomes
    k_learning_rate: float = 0.01  # Step size for k adaptation

    # Device
    device: str = "cpu"


# ============================================================================
# SENSORY PROCESSING & INTEGRATION
# ============================================================================


@dataclass
class ThalamicRelayConfig(NeuralComponentConfig):
    """Configuration for thalamic relay nucleus.

    Thalamus sits between sensory input and cortex, providing:
    - Sensory gating (alpha-based suppression)
    - Mode switching (burst vs tonic)
    - Gain modulation (norepinephrine)
    - Spatial filtering

    **Pure Behavioral Configuration**:
    Contains ONLY behavioral parameters (learning rates, gains, thresholds).
    Sizes (relay_size, trn_size, input_size) are passed separately at instantiation.

    **Usage**:
    ```python
    config = ThalamicRelayConfig(relay_strength=1.5, alpha_suppression_strength=0.3)
    sizes = LayerSizeCalculator().thalamus_from_relay(relay_size=80)
    thalamus = ThalamicRelay(config=config, sizes=sizes, device="cpu")
    ```
    """

    # Relay parameters
    relay_strength: float = THALAMUS_RELAY_STRENGTH
    """Base relay gain (thalamus amplifies weak inputs)."""

    # Mode switching
    burst_threshold: float = THALAMUS_BURST_THRESHOLD
    """Membrane potential threshold for burst mode (hyperpolarized)."""

    tonic_threshold: float = THALAMUS_TONIC_THRESHOLD
    """Membrane potential threshold for tonic mode (depolarized)."""

    burst_spike_count: int = THALAMUS_BURST_SPIKE_COUNT
    """Number of spikes in a burst (typically 2-5)."""

    burst_gain: float = THALAMUS_BURST_GAIN
    """Amplification factor for burst mode (alerting signal)."""

    # Attention gating (alpha oscillation)
    alpha_suppression_strength: float = THALAMUS_ALPHA_SUPPRESSION
    """How strongly alpha suppresses unattended inputs (0-1)."""

    alpha_gate_threshold: float = THALAMUS_ALPHA_GATE_THRESHOLD
    """Alpha phase threshold for suppression (0 = trough, π = peak)."""

    trn_inhibition_strength: float = THALAMUS_TRN_INHIBITION
    """Strength of TRN → relay inhibition."""

    trn_recurrent_strength: float = THALAMUS_TRN_RECURRENT
    """TRN recurrent inhibition (for oscillations)."""

    # Sensory filtering
    spatial_filter_width: float = THALAMUS_SPATIAL_FILTER_WIDTH
    """Gaussian filter width for center-surround (as fraction of input)."""

    center_excitation: float = THALAMUS_CENTER_EXCITATION
    """Center enhancement in receptive field."""

    surround_inhibition: float = THALAMUS_SURROUND_INHIBITION
    """Surround suppression in receptive field."""

    # Corticothalamic feedback
    l6a_to_trn_strength: float = 0.8
    """Strength of L6a → TRN feedback (inhibitory modulation, type I)."""

    l6b_to_relay_strength: float = 0.6
    """Strength of L6b → relay feedback (excitatory modulation, type II)."""

    # Internal thalamic delays (critical for gamma oscillation emergence)
    trn_to_relay_delay_ms: float = 4.0
    """TRN → relay inhibitory delay (~3-5ms for GABAergic transmission)."""

    relay_to_cortex_delay_ms: float = 2.0
    """Relay → cortex thalamocortical delay (~2ms, handled by AxonalProjection)."""

    # Gap junctions (TRN interneuron synchronization)
    gap_junctions_enabled: bool = True
    """Enable gap junction coupling in TRN for fast synchronization."""

    gap_junction_strength: float = 0.15
    """Gap junction conductance (biological: 0.05-0.3, Landisman 2002)."""

    # Short-Term Plasticity (STP) - HIGH PRIORITY for sensory gating
    stp_enabled: bool = True
    """Enable STP for sensory relay and L6 feedback pathways.

    Biological justification (HIGH PRIORITY):
    - Sensory relay depression: Filters repetitive stimuli, responds to novelty
    - L6 feedback depression: Modulates gain control dynamically
    - CRITICAL for realistic sensory gating and attention
    - References: Castro-Alamancos (2002), Swadlow & Gusev (2001)
    """

    stp_sensory_relay_type: STPType = STPType.DEPRESSING_MODERATE
    """Sensory input → relay depression (U=0.4, moderate).

    Implements novelty detection: Sustained inputs depress, novel stimuli get
    through. Critical for attention capture and change detection.
    """

    stp_l6_feedback_type: STPType = STPType.DEPRESSING_STRONG
    """L6 cortical feedback → relay depression (U=0.7, strong).

    Implements dynamic gain control: Sustained cortical feedback reduces
    thalamic transmission, enabling efficient filtering.
    """


@dataclass
class CortexRobustnessConfig:
    """Cortex-specific robustness mechanisms.

    This config contains mechanisms NOT already handled by UnifiedHomeostasis:
    - E/I Balance: Critical for recurrent cortical stability
    - Criticality: Optional research/diagnostics tool
    - Metabolic: Optional sparse coding objective

    Note: The following are handled by UnifiedHomeostasis (base class):
    - Weight normalization (budget constraints)
    - Activity regulation (threshold adaptation)
    - Competitive dynamics (winner-take-all)

    Divisive normalization removed: ConductanceLIF neurons provide natural
    gain control via shunting inhibition, making explicit divisive norm redundant.

    **Recommended presets:**
    - minimal(): Just E/I balance (essential for recurrence)
      → Use for: Most cortical regions, minimal overhead

    - full(): All mechanisms enabled
      → Use for: Research, diagnostics, sparse coding goals

    **When to customize:**
    - Debugging: Disable all, enable E/I balance only
    - Sparse coding: Enable metabolic constraints
    - Research: Enable criticality monitoring for branching ratio

    Attributes:
        enable_ei_balance: Enable E/I balance regulation
            Maintains healthy ratio between excitation and inhibition.
            Critical for recurrent cortical circuits (prevents oscillations).

        enable_criticality: Enable criticality monitoring
            Tracks branching ratio, can correct toward critical state.
            More expensive, research/diagnostics use only.

        enable_metabolic: Enable metabolic constraints
            Penalizes excessive activity, encourages sparse coding.
            Useful when energy efficiency is an explicit goal.

        ei_balance: E/I balance configuration
        criticality: Criticality monitoring configuration
        metabolic: Metabolic constraint configuration
    """

    # Enable/disable flags
    enable_ei_balance: bool = True
    enable_criticality: bool = False  # Research/diagnostics only
    enable_metabolic: bool = False  # Sparse coding objective

    # Sub-configurations
    ei_balance: EIBalanceConfig = field(default_factory=EIBalanceConfig)
    criticality: CriticalityConfig = field(default_factory=CriticalityConfig)
    metabolic: MetabolicConfig = field(default_factory=MetabolicConfig)

    @classmethod
    def minimal(cls) -> CortexRobustnessConfig:
        """Create minimal config with only essential mechanisms.

        Enables E/I balance only (critical for recurrent stability).

        Use cases:
        - Most cortical regions (default choice)
        - Quick prototyping and debugging
        - Minimal computational overhead
        - Essential recurrence stability without extras

        Performance impact: ~10-15% overhead vs no robustness
        """
        return cls(
            enable_ei_balance=True,  # Essential for recurrence
            enable_criticality=False,
            enable_metabolic=False,
        )

    @classmethod
    def full(cls) -> CortexRobustnessConfig:
        """Create full config with ALL robustness mechanisms.

        Maximum robustness with all mechanisms enabled.

        Use cases:
        - Research exploring criticality dynamics
        - Sparse coding objectives (metabolic constraints)
        - Maximum diagnostics and monitoring

        Performance impact: ~20-30% overhead vs minimal
        """
        return cls(
            enable_ei_balance=True,
            enable_criticality=True,
            enable_metabolic=True,
        )

    def get_enabled_mechanisms(self) -> List[str]:
        """Get list of enabled mechanism names."""
        enabled: List[str] = []
        if self.enable_ei_balance:
            enabled.append("ei_balance")
        if self.enable_criticality:
            enabled.append("criticality")
        if self.enable_metabolic:
            enabled.append("metabolic")
        return enabled

    def summary(self) -> str:
        """Get a summary of the robustness configuration."""
        lines = [
            "Robustness Configuration:",
            f"  E/I Balance: {'ON' if self.enable_ei_balance else 'OFF'}",
            f"  Divisive Norm: {'ON' if hasattr(self, 'enable_divisive_norm') and self.enable_divisive_norm else 'OFF'}",
            f"  Intrinsic Plasticity: {'ON' if hasattr(self, 'enable_intrinsic_plasticity') and self.enable_intrinsic_plasticity else 'OFF'}",
            f"  Criticality: {'ON' if self.enable_criticality else 'OFF'}",
            f"  Metabolic: {'ON' if self.enable_metabolic else 'OFF'}",
        ]

        if self.enable_ei_balance:
            lines.append(f"    E/I target ratio: {self.ei_balance.target_ratio}")
        if (
            hasattr(self, "enable_divisive_norm")
            and self.enable_divisive_norm
            and hasattr(self, "divisive_norm")
        ):
            lines.append(f"    Divisive sigma: {self.divisive_norm.sigma}")  # type: ignore[attr-defined]
        if (
            hasattr(self, "enable_intrinsic_plasticity")
            and self.enable_intrinsic_plasticity
            and hasattr(self, "intrinsic_plasticity")
        ):
            lines.append(f"    IP target rate: {self.intrinsic_plasticity.target_rate}")  # type: ignore[attr-defined]
        if self.enable_criticality:
            lines.append(f"    Target branching: {self.criticality.target_branching}")
        if self.enable_metabolic:
            lines.append(f"    Energy budget: {self.metabolic.energy_budget}")

        return "\n".join(lines)


@dataclass
class LayeredCortexConfig(NeuralComponentConfig):
    """Configuration for layered cortical microcircuit.

    **BEHAVIORAL CONFIGURATION ONLY**

    This config contains ONLY behavioral parameters (learning rates, sparsity, etc.).
    Layer sizes are provided separately during instantiation via LayerSizeCalculator.

    Usage with BrainBuilder:
        >>> from thalia.config import LayerSizeCalculator
        >>> from thalia.core.brain_builder import BrainBuilder
        >>>
        >>> calc = LayerSizeCalculator()
        >>> cortex_sizes = calc.cortex_from_scale(scale_factor=128)
        >>>
        >>> builder = BrainBuilder(brain_config)
        >>> builder.add_component("cortex", "cortex", **cortex_sizes)
        >>> brain = builder.build()

    Direct region instantiation (internal use by builder):
        >>> config = LayeredCortexConfig(stdp_lr=0.001, sparsity=0.1)
        >>> sizes = calc.cortex_from_scale(128)
        >>> cortex = LayeredCortex(config=config, sizes=sizes, device="cpu")

    OUTPUT COMPUTATION:
        output_size is computed as l23_size + l5_size (dual pathways:
        cortico-cortical via L2/3 and subcortical via L5).
    """

    # NO SIZE FIELDS - sizes passed separately to __init__

    # Layer sparsity (fraction of neurons active)
    l4_sparsity: float = 0.15  # Moderate sparsity
    l23_sparsity: float = 0.10  # Sparser (more selective)
    l5_sparsity: float = 0.20  # Less sparse (motor commands)
    l6a_sparsity: float = 0.12  # L6a → TRN (slightly more sparse than L2/3)
    l6b_sparsity: float = 0.15  # L6b → relay (moderate sparsity)

    # Recurrence in L2/3
    l23_recurrent_strength: float = 0.3  # Lateral connection strength
    l23_recurrent_decay: float = 0.9  # Recurrent activity decay

    # Feedforward connection strengths
    # These need to be strong enough that sparse activity can drive next layer above threshold.
    # Weight initialization uses abs(randn) * scale, where scale = 1/expected_active.
    # abs(randn) has mean ~0.8, so we need ~1.5-2.0x strength to compensate.
    # With ~10-15% sparsity and random weights, we need ~2.0x strength for input layer
    # and ~1.5x for subsequent layers to reliably activate postsynaptic neurons.
    input_to_l4_strength: float = 2.0  # External input → L4 (was 1.0, too weak for sparse input)
    l4_to_l23_strength: float = 1.5  # L4 → L2/3 (was 0.4, too weak)
    l23_to_l5_strength: float = 1.5  # L2/3 → L5 (was 0.4, too weak)
    l23_to_l6a_strength: float = 0.8  # L2/3 → L6a (reduced for low gamma 25-35Hz)
    l23_to_l6b_strength: float = 2.0  # L2/3 → L6b (higher for high gamma 60-80Hz)

    # Top-down modulation (for attention pathway)
    l23_top_down_strength: float = 0.2  # Feedback to L2/3

    # L6 corticothalamic feedback strengths (different pathways)
    l6a_to_trn_strength: float = 0.8  # L6a → TRN (inhibitory modulation, low gamma)
    l6b_to_relay_strength: float = 0.6  # L6b → relay (excitatory modulation, high gamma)

    # Spillover transmission (volume transmission)
    # Enable in cortex L2/3 and L5 where experimentally documented
    # (Agnati et al. 2010, Fuxe & Agnati 1991, Zoli et al. 1999)
    # Cortical spillover contributes to lateral excitation and feature binding
    enable_spillover: bool = True  # Override base config (disabled by default)
    spillover_mode: str = "connectivity"  # Use shared inputs for neighborhood
    spillover_strength: float = 0.15  # 15% of direct synaptic strength (biological range)

    # Gap junctions (L2/3 interneuron synchronization)
    # Basket cells and chandelier cells in L2/3 have dense gap junction networks
    # Critical for cortical gamma oscillations (30-80 Hz) and precise spike timing
    # ~70-80% of cortical gap junctions are interneuron-interneuron (Bennett 2004)
    gap_junctions_enabled: bool = True
    """Enable gap junction coupling in L2/3 interneurons."""

    gap_junction_strength: float = 0.12
    """Gap junction conductance for L2/3 interneurons (biological: 0.05-0.2)."""

    gap_junction_threshold: float = 0.25
    """Connectivity threshold for gap junction coupling (shared inputs)."""

    gap_junction_max_neighbors: int = 8
    """Maximum gap junction neighbors per interneuron (biological: 4-12)."""

    # Note: STDP parameters (stdp_lr, tau_plus_ms, tau_minus_ms, a_plus, a_minus)
    # are inherited from NeuralComponentConfig
    # Override with cortical values from constants:
    a_plus: float = STDP_A_PLUS_CORTEX  # LTP amplitude
    a_minus: float = STDP_A_MINUS_CORTEX  # LTD amplitude

    # Weight bounds for L2/3 recurrent connections (signed, compact E/I approximation)
    # Unlike feedforward connections, recurrent lateral connections use signed weights
    # to approximate the mixed excitatory/inhibitory microcircuit within a cortical layer.
    # Positive weights = local excitation, negative weights = lateral inhibition.
    l23_recurrent_w_min: float = -1.5  # Allows inhibitory-like connections
    l23_recurrent_w_max: float = 1.0  # Symmetric by default

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION (SFA)
    # =========================================================================
    # Cortical pyramidal neurons show strong spike-frequency adaptation.
    # Inherited from base: adapt_increment=0.0, adapt_tau=100.0
    # Override for L2/3 strong adaptation:
    adapt_increment: float = ADAPT_INCREMENT_CORTEX_L23  # Very strong adaptation for decorrelation
    # adapt_tau: 100.0 (use base default)

    # =========================================================================
    # CORTEX-SPECIFIC DYNAMICS
    # =========================================================================
    # These parameters control cortical circuit mechanisms that are specific
    # to layered cortex architecture (not universal like homeostasis).
    #
    # Note: Intrinsic plasticity (threshold adaptation) is handled by
    # UnifiedHomeostasis via activity_target in the base NeuralComponentConfig.

    # Feedforward Inhibition (FFI) parameters
    # FFI detects stimulus changes and transiently suppresses recurrent activity
    # This is how the cortex naturally "clears" old representations when new input arrives
    # Always enabled (fundamental cortical mechanism)
    ffi_threshold: float = 0.3  # Input change threshold to trigger FFI
    ffi_strength: float = 0.8  # How much FFI suppresses L2/3 recurrent activity
    ffi_tau: float = 5.0  # FFI decay time constant (ms)

    # =========================================================================
    # INTER-LAYER AXONAL DELAYS
    # =========================================================================
    # Biological signal propagation times within cortical laminae:
    # - L4→L2/3: ~2ms (short vertical projection)
    # - L2/3→L5: ~2ms (longer vertical projection)
    # - L2/3→L6: ~2-3ms (within column, vertical projection)
    # - L6→TRN: ~10ms (corticothalamic feedback, long-range)
    # Total laminar processing: ~4-6ms (much faster than processing timescales)
    #
    # Internal delays enable realistic temporal dynamics and support oscillation emergence:
    # - L6a with 2ms internal + 10ms feedback = 12ms loop → ~83 Hz (high gamma range)
    # - L6b with 3ms internal + 5ms feedback = 8ms loop → ~125 Hz (very high gamma)
    # - With neural refractory periods and integration, actual frequencies settle to
    #   low gamma (25-35 Hz) for L6a and high gamma (60-80 Hz) for L6b
    l4_to_l23_delay_ms: float = 2.0  # L4→L2/3 axonal delay (short vertical)
    l23_to_l5_delay_ms: float = 2.0  # L2/3→L5 axonal delay (longer vertical)
    l23_to_l6a_delay_ms: float = 2.0  # L2/3→L6a axonal delay (type I pathway, slow)
    l23_to_l6b_delay_ms: float = 3.0  # L2/3→L6b axonal delay (type II pathway, fast)

    # L6 feedback delays (key for gamma frequency tuning)
    l6a_to_trn_delay_ms: float = 10.0  # L6a→TRN feedback delay (~10ms biological, slow pathway)
    l6b_to_relay_delay_ms: float = 5.0  # L6b→relay feedback delay (~5ms biological, fast pathway)

    # Gamma-based attention (spike-native phase gating for L2/3)
    # Always enabled for spike-native attention
    gamma_attention_width: float = 0.3  # Phase window width

    # =========================================================================
    # BCM SLIDING THRESHOLD (Metaplasticity)
    # =========================================================================
    # The BCM rule provides a sliding threshold for synaptic modification that
    # automatically adjusts based on postsynaptic activity history. This is
    # particularly important for cortical learning because:
    # 1. Prevents runaway potentiation in highly active neurons
    # 2. Maintains selectivity during feature learning
    # 3. Enables competitive dynamics between feature detectors
    #
    # In visual cortex, BCM explains orientation selectivity development:
    # neurons that respond strongly to one orientation have high thresholds,
    # making them less likely to respond to other orientations.
    bcm_enabled: bool = False
    bcm_config: Optional[BCMConfig] = None  # BCM configuration (if enabled)

    # =========================================================================
    # ROBUSTNESS MECHANISMS (Cortex-Specific)
    # =========================================================================
    # Optional configuration for mechanisms NOT already in UnifiedHomeostasis:
    # - E/I Balance: Maintains healthy excitation/inhibition ratio (critical for recurrence)
    # - Criticality: Maintains network near critical point (research/diagnostics)
    # - Metabolic: Energy-based regularization for sparsity (optional goal)
    #
    # Note: Activity regulation and threshold adaptation are handled by
    # UnifiedHomeostasis (in NeuralComponentConfig base class).
    robustness: Optional[CortexRobustnessConfig] = field(default=None)

    # =========================================================================
    # LAYER-SPECIFIC HETEROGENEITY (Phase 2A Enhancement)
    # =========================================================================
    # Biological reality: Each cortical layer has distinct cell types with
    # different electrophysiological properties:
    # - L4 spiny stellate: Fast, small tau_mem (~10ms), low threshold
    # - L2/3 pyramidal: Medium tau_mem (~20ms), moderate threshold
    # - L5 thick-tuft pyramidal: Slow tau_mem (~30ms), high threshold, burst-capable
    # - L6 corticothalamic: Variable tau_mem (~15-25ms), moderate threshold
    #
    # This heterogeneity enables:
    # - L4: Fast sensory processing and feature detection
    # - L2/3: Integration and association over longer timescales
    # - L5: Decision-making and sustained output generation
    # - L6: Feedback control with tuned dynamics
    #
    # References:
    # - Connors & Gutnick (1990): Intrinsic firing patterns of diverse neocortical neurons
    # - Markram et al. (2015): Reconstruction and simulation of neocortical microcircuitry
    # - Ramaswamy & Markram (2015): Anatomy and physiology of the thick-tufted layer 5 pyramidal neuron
    use_layer_heterogeneity: bool = False
    """Enable layer-specific neuron properties (Phase 2A).

    When True:
        - Each layer has distinct tau_mem, v_threshold, adaptation properties
        - Reflects biological diversity of cortical cell types
        - Improves layer-specific computational roles
        - Requires layer_properties config
    """

    # Layer-specific membrane time constants (ms)
    # These control integration timescales for each layer
    layer_tau_mem: Dict[str, float] = field(
        default_factory=lambda: {
            "l4": 10.0,  # Fast integration for sensory input
            "l23": 20.0,  # Moderate integration for association
            "l5": 30.0,  # Slow integration for output generation
            "l6a": 15.0,  # Fast for TRN feedback (low gamma)
            "l6b": 25.0,  # Moderate for relay feedback (high gamma)
        }
    )
    """Membrane time constants per layer (Phase 2A).

    Biological ranges:
    - L4 spiny stellate: 8-12ms (fast sensory processing)
    - L2/3 pyramidal: 18-25ms (integration)
    - L5 pyramidal: 25-35ms (sustained output)
    - L6 pyramidal: 12-20ms (feedback control)
    """

    # Layer-specific voltage thresholds (mV)
    # Higher threshold = more selective, requires more input
    layer_v_threshold: Dict[str, float] = field(
        default_factory=lambda: {
            "l4": -52.0,  # Low threshold for sensitive input detection
            "l23": -55.0,  # Moderate threshold for balanced processing
            "l5": -50.0,  # Lower threshold for reliable output (compensated by high tau)
            "l6a": -55.0,  # Moderate for attention gating
            "l6b": -52.0,  # Low for fast gain modulation
        }
    )
    """Voltage thresholds per layer (Phase 2A).

    Biological values:
    - L4: -50 to -55mV (sensitive to input)
    - L2/3: -53 to -58mV (selective integration)
    - L5: -48 to -52mV (reliable output despite high tau)
    - L6: -50 to -55mV (varied for feedback roles)
    """

    # Layer-specific adaptation strengths
    # Controls spike-frequency adaptation per layer
    layer_adaptation: Dict[str, float] = field(
        default_factory=lambda: {
            "l4": 0.05,  # Minimal adaptation for faithful sensory relay
            "l23": 0.15,  # Strong adaptation for decorrelation (inherited default)
            "l5": 0.10,  # Moderate adaptation for sustained output
            "l6a": 0.08,  # Light adaptation for feedback
            "l6b": 0.12,  # Moderate adaptation for gain control
        }
    )
    """Adaptation increments per layer (Phase 2A).

    Biological justification:
    - L4: Minimal (faithful relay of sensory input)
    - L2/3: Strong (prevents runaway recurrence, decorrelates features)
    - L5: Moderate (allows burst patterns while preventing runaway)
    - L6: Light-moderate (supports feedback dynamics)
    """


class PredictiveCodingErrorType(Enum):
    """Types of prediction errors."""

    POSITIVE = "positive"  # Actual > Predicted (under-prediction)
    NEGATIVE = "negative"  # Actual < Predicted (over-prediction)
    SIGNED = "signed"  # Single population with +/- values


@dataclass
class PredictiveCodingConfig:
    """Configuration for a predictive coding layer.

    Attributes:
        n_input: Size of input (from lower layer or sensory)
        n_representation: Size of internal representation (prediction neurons)

        # Prediction dynamics
        prediction_tau_ms: Time constant for prediction integration (slow, NMDA-like)
        error_tau_ms: Time constant for error neurons (fast, AMPA-like)

        # Learning parameters
        learning_rate: Base learning rate for prediction weight updates
        precision_learning_rate: Learning rate for precision updates

        # Precision (attention/confidence) parameters
        initial_precision: Starting precision (inverse variance)
        precision_min: Minimum precision (prevents division by zero)
        precision_max: Maximum precision (prevents over-confidence)

        # Architecture choices
        error_type: How errors are represented (separate +/- or signed)
        sparse_coding: Apply sparsity constraint on representations
        sparsity_target: Target activation fraction if sparse_coding=True

        dt_ms: Simulation timestep
        device: Computation device
    """

    n_input: int = 256
    n_representation: int = 128

    # Dynamics
    prediction_tau_ms: float = 50.0  # Slow (NMDA-like) for stable predictions
    error_tau_ms: float = 5.0  # Fast (AMPA-like) for quick error signaling

    # Learning
    learning_rate: float = 0.01
    precision_learning_rate: float = LEARNING_RATE_PRECISION

    # Precision (attention)
    initial_precision: float = 1.0
    precision_min: float = 0.1
    precision_max: float = 10.0

    # Temporal variance tracking for precision learning
    # Precision is updated based on variance of errors over recent history
    error_history_size: int = 50  # Number of timesteps to track for variance
    precision_update_interval: int = 10  # Update precision every N timesteps

    # Architecture
    error_type: PredictiveCodingErrorType = PredictiveCodingErrorType.SIGNED
    sparse_coding: bool = True
    sparsity_target: float = 0.1

    device: str = "cpu"


@dataclass
class PredictiveCortexConfig(LayeredCortexConfig):
    """Configuration for predictive cortex.

    Extends LayeredCortexConfig with predictive coding parameters.
    """

    # Predictive coding parameters
    prediction_enabled: bool = True
    prediction_tau_ms: float = 50.0  # Slow predictions (NMDA-like)
    error_tau_ms: float = 5.0  # Fast errors (AMPA-like)
    prediction_learning_rate: float = 0.01

    # Precision (attention) parameters
    use_precision_weighting: bool = True
    initial_precision: float = 1.0
    precision_learning_rate: float = LEARNING_RATE_PRECISION

    # Note: Gamma attention inherited from LayeredCortex base class (always enabled)
    # Configure width via gamma_attention_width


@dataclass
class MultimodalIntegrationConfig(NeuralComponentConfig, HebbianLearningConfig):
    """Configuration for multimodal integration region.

    Inherits Hebbian learning parameters from HebbianLearningConfig:
    - learning_rate: Base learning rate for cross-modal plasticity
    - learning_enabled: Global learning enable/disable
    - weight_min, weight_max: Weight bounds
    - decay_rate, sparsity_penalty, use_oja_rule: Hebbian variants

    Args:
        visual_input_size: Size of visual input
        auditory_input_size: Size of auditory input
        language_input_size: Size of language/semantic input
        visual_pool_ratio: Fraction of neurons for visual (0-1)
        auditory_pool_ratio: Fraction of neurons for auditory (0-1)
        language_pool_ratio: Fraction of neurons for language (0-1)
        integration_pool_ratio: Fraction of neurons for integration (0-1)
        cross_modal_strength: Strength of cross-modal connections (0-1)
        within_modal_strength: Strength of within-modal connections (0-1)
        integration_strength: Strength from pools → integration neurons
        salience_competition_strength: Winner-take-all competition strength
    """

    # Input sizes
    visual_input_size: int = 0
    auditory_input_size: int = 0
    language_input_size: int = 0

    # Pool sizes (explicit, computed from ratios via helper)
    visual_pool_size: int = field(default=0)
    auditory_pool_size: int = field(default=0)
    language_pool_size: int = field(default=0)
    integration_pool_size: int = field(default=0)

    # Connection strengths
    cross_modal_strength: float = 0.4
    within_modal_strength: float = 0.6
    integration_strength: float = 0.8
    salience_competition_strength: float = 0.5

    # Override default learning rate with region-specific value
    learning_rate: float = LEARNING_RATE_HEBBIAN_SLOW

    # Gamma synchronization parameters (for cross-modal binding)
    gamma_freq_hz: float = 40.0  # Gamma frequency for binding (typically 40 Hz)
    coherence_window: float = 0.785  # ~π/4 radians phase tolerance
    phase_coupling_strength: float = 0.1  # Mutual phase nudging strength
    gate_threshold: float = 0.3  # Minimum coherence for binding
    use_gamma_binding: bool = True  # Enable gamma synchronization


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Memory & Learning
    "HippocampusConfig",
    "CerebellumConfig",
    # Action Selection & Executive Control
    "StriatumConfig",
    "PrefrontalConfig",
    "GoalHierarchyConfig",
    "HyperbolicDiscountingConfig",
    # Sensory Processing & Integration
    "ThalamicRelayConfig",
    "CortexRobustnessConfig",
    "LayeredCortexConfig",
    "PredictiveCodingErrorType",
    "PredictiveCodingConfig",
    "PredictiveCortexConfig",
    "MultimodalIntegrationConfig",
]
