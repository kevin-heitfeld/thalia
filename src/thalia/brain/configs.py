"""Configurations for brain regions and overall brain parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

from thalia.components.synapses.stp import STPType
from thalia.constants import DEFAULT_DT_MS

if TYPE_CHECKING:
    from .oscillator import OscillatorCoupling


# ============================================================================
# Enums
# ============================================================================


class RegionType(Enum):
    """Types of brain regions."""

    CEREBELLUM = "cerebellum"
    CORTEX = "cortex"
    HIPPOCAMPUS = "hippocampus"
    PFC = "pfc"
    STRIATUM = "striatum"
    THALAMUS = "thalamus"


class CortexLayer(Enum):
    """Cortical layers enumeration."""

    L23 = "l23"
    L4 = "l4"
    L5 = "l5"
    L6A = "l6a"
    L6B = "l6b"


# ============================================================================
# Base Neural Region Config
# ============================================================================


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
    dt_ms: float = DEFAULT_DT_MS  # Simulation timestep in milliseconds.

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
    gain_learning_rate: float = 0.005  # Learning rate for gain adaptation
    target_firing_rate: float = 0.05  # Target firing rate for homeostatic plasticity
    gain_tau_ms: float = 2000.0  # Time constant for firing rate averaging
    baseline_noise_current: float = 0.15  # Baseline noise current for bootstrap

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
    synaptic_scaling_enabled: bool = True  # Enable global synaptic scaling
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
    weight_budget: float = 1.0
    """Target sum of weights per neuron (row normalization constraint)."""

    activity_target: float = 0.1
    """Target fraction of neurons active per timestep."""

    soft_normalization: bool = True
    """Use soft (multiplicative) normalization instead of hard constraint."""

    normalization_rate: float = 0.1
    """Rate of convergence toward target (soft normalization only)."""

    activity_tau_ms: float = 1000.0
    """Time constant for activity rate estimation."""

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION (SFA)
    # =========================================================================
    adapt_increment: float = 0.0
    """Adaptation current increase per spike (0 = disabled)."""

    adapt_tau: float = 100.0
    """Adaptation decay time constant in milliseconds."""

    # =========================================================================
    # SPIKE-TIMING DEPENDENT PLASTICITY (STDP)
    # =========================================================================
    learning_strategy: str = "STDP"
    """Which plasticity rule to use (STDP, PHASE_STDP, TRIPLET_STDP, etc.)."""

    tau_plus_ms: float = 20.0
    """LTP time constant in milliseconds."""

    tau_minus_ms: float = 20.0
    """LTD time constant in milliseconds."""

    a_plus: float = 1.0
    """LTP amplitude."""

    a_minus: float = 1.0
    """LTD amplitude."""


# ============================================================================
# Medial Septum Config
# ============================================================================


@dataclass
class MedialSeptumConfig(NeuralRegionConfig):
    """Configuration for medial septum theta pacemaker.

    The medial septum generates theta rhythm (4-10 Hz) through intrinsic
    bursting in cholinergic and GABAergic neurons. This rhythm phase-locks
    hippocampal circuits, creating encoding/retrieval separation.

    Key biological features:
    - Intrinsic pacemaker (no external oscillator needed)
    - Two phase-locked populations (ACh at 0°, GABA at 180°)
    - Pulsed output (not sinusoidal)
    - Frequency modulated by neuromodulators
    """

    # =========================================================================
    # NEURON POPULATIONS
    # =========================================================================
    n_ach: int = 100
    """Number of cholinergic neurons (excite hippocampal pyramidal)."""

    n_gaba: int = 100
    """Number of GABAergic neurons (inhibit hippocampal interneurons)."""

    # =========================================================================
    # PACEMAKER DYNAMICS
    # =========================================================================
    base_frequency_hz: float = 8.0
    """Base theta frequency in Hz. Modulated by ACh, NE, DA (range: 4-10 Hz)."""

    burst_duty_cycle: float = 0.3
    """Fraction of cycle spent in burst phase (vs inter-burst silence)."""

    burst_amplitude: float = 5.0
    """Peak drive current during burst (nA equivalent)."""

    inter_burst_amplitude: float = 0.5
    """Baseline drive current between bursts (maintains readiness)."""

    # =========================================================================
    # CHOLINERGIC NEURON PROPERTIES
    # =========================================================================
    ach_tau_mem: float = 30.0
    """ACh neuron membrane time constant (ms). Slow for bursting dynamics."""

    ach_threshold: float = 1.2
    """ACh neuron threshold (higher than typical - requires strong drive)."""

    ach_reset: float = 0.0
    """ACh neuron reset potential after spike."""

    ach_adaptation_tau: float = 100.0
    """ACh adaptation time constant (ms). Strong to create burst termination."""

    ach_adaptation_increment: float = 0.15
    """ACh adaptation increment per spike (terminates burst)."""

    # =========================================================================
    # GABAERGIC NEURON PROPERTIES
    # =========================================================================
    gaba_tau_mem: float = 20.0
    """GABA neuron membrane time constant (ms). Faster than ACh."""

    gaba_threshold: float = 1.0
    """GABA neuron threshold (lower than ACh - more excitable)."""

    gaba_reset: float = 0.0
    """GABA neuron reset potential after spike."""

    gaba_adaptation_tau: float = 80.0
    """GABA adaptation time constant (ms). Moderate."""

    gaba_adaptation_increment: float = 0.12
    """GABA adaptation increment per spike."""


# ============================================================================
# Cerebellum Config
# ============================================================================


@dataclass
class CerebellumConfig(NeuralRegionConfig):
    """Configuration specific to cerebellar regions.

    The cerebellum implements ERROR-CORRECTIVE learning through:
    1. Parallel fiber → Purkinje cell connections (learned)
    2. Climbing fiber error signals from inferior olive
    3. LTD when climbing fiber active with parallel fiber

    Key biological features:
    - Error signal triggers immediate learning (not delayed like RL)
    - Can learn arbitrary input-output mappings quickly
    - Uses eligibility traces for temporal credit assignment
    """

    # =========================================================================
    # ADAPTIVE GAIN CONTROL (HOMEOSTATIC INTRINSIC PLASTICITY)
    # =========================================================================
    # Purkinje cells have high spontaneous firing rates (~40-50 Hz in biology)
    target_firing_rate: float = 0.35  # 35% target (Purkinje cells are highly active)
    gain_learning_rate: float = 0.005  # Reduced 10x to prevent runaway (was 0.05, tried 0.0005 - too slow)
    gain_tau_ms: float = 2000.0  # 2s averaging window (slow for motor stability)
    baseline_noise_current: float = 0.30  # High spontaneous activity (Purkinje cells very active, increased from 0.15)

    # =========================================================================
    # ADAPTVE THRESHOLD PLASTICITY (complementary to gain adaptation)
    # =========================================================================
    threshold_learning_rate: float = 0.03  # Moderate threshold adaptation
    threshold_min: float = 0.05  # Lower floor to allow more aggressive adaptation for under-firing
    threshold_max: float = 1.2  # Moderate max (Purkinje cells naturally active)

    # =========================================================================
    # ARCHITECTURE: GRANULE→PURKINJE→DCN CIRCUIT
    # =========================================================================
    # Uses granule→Purkinje→DCN circuit instead of direct
    # parallel fiber→Purkinje mapping. Provides:
    # - 4× sparse expansion in granule layer (pattern separation)
    # - Dendritic computation in Purkinje cells (complex/simple spikes)
    # - DCN integration (Purkinje sculpts tonic output)

    granule_sparsity: float = 0.03  # Fraction of granule cells active (3%)
    purkinje_n_dendrites: int = 100  # Simplified dendritic compartments

    # =========================================================================
    # COMPLEX SPIKE DYNAMICS
    # =========================================================================
    # Climbing fibers trigger complex spikes in Purkinje cells - bursts of 2-7
    # spikes with very short inter-spike intervals (1-2ms). These bursts encode
    # ERROR MAGNITUDE: larger errors → longer bursts → more calcium influx.
    #
    # This provides GRADED ERROR SIGNALING instead of binary (error/no-error),
    # enabling more nuanced learning: small errors trigger small corrections,
    # large errors trigger large corrections.
    #
    # Mechanism:
    # 1. Climbing fiber error → complex spike burst (not single spike)
    # 2. Each spikelet triggers dendritic calcium influx (~0.2 units)
    # 3. Total calcium = n_spikes × ca2_per_spike
    # 4. LTD magnitude ∝ total calcium (graded learning)
    use_complex_spike_bursts: bool = True
    min_complex_spike_count: int = 2  # Minimum spikes in complex spike burst
    max_complex_spike_count: int = 7  # Maximum spikes in complex spike burst
    complex_spike_isi_ms: float = 1.5  # Inter-spike interval within complex spike (ms)
    ca2_per_spikelet: float = 0.2  # Calcium influx per complex spikelet

    # =========================================================================
    # ELIGIBILITY TRACES
    # =========================================================================
    eligibility_tau_ms: float = 20.0  # Short tau for immediate error correction

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
    gap_junction_strength: float = 0.18  # Strong coupling for IO synchronization
    gap_junction_threshold: float = 0.20  # Low threshold for dense connectivity
    gap_junction_max_neighbors: int = 12  # Many neighbors for global sync

    # =========================================================================
    # HETEROSYNAPTIC COMPETITION
    # =========================================================================
    # Cerebellum uses weaker heterosynaptic competition for faster convergence:
    heterosynaptic_ratio: float = 0.2  # Override base (0.3) - weaker competition

    # =========================================================================
    # LEARNING RATES
    # =========================================================================
    learning_rate_ltp: float = 0.01  # LTP rate for parallel fiber→Purkinje
    learning_rate_ltd: float = 0.01  # LTD rate for parallel fiber→Purkinje
    error_threshold: float = 0.01  # Minimum error to trigger learning

    # =========================================================================
    # SHORT-TERM PLASTICITY (STP)
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
    # BIOLOGICAL IMPORTANCE: This is perhaps the MOST important STP in the brain
    # for motor learning and timing. The cerebellar cortex is the brain's master
    # clock, and STP is essential for its temporal precision.
    stp_pf_purkinje_type: STPType = STPType.DEPRESSING  # Parallel fiber depression
    stp_mf_granule_type: STPType = STPType.FACILITATING  # Mossy fiber facilitation


# ============================================================================
# Cortex Config
# ============================================================================


@dataclass
class CortexConfig(NeuralRegionConfig):
    """Configuration for layered cortical microcircuit."""

    # =========================================================================
    # ADAPTIVE GAIN CONTROL (HOMEOSTATIC INTRINSIC PLASTICITY)
    # =========================================================================
    # EXPERIMENT: Raising from 0.015 to 0.03 to match burst-firing regime
    # Previous: 0.015 (1.5%) - observed to cause burst-only regime (3% burst > 1.5% target)
    # Hypothesis: Post-burst adaptation reduces excitability, preventing sustained activity
    # New: 0.03 (3%) to match observed burst average and enable sustained firing
    target_firing_rate: float = 0.03  # Adjusted to match ~3% burst activity (was 0.015)
    gain_learning_rate: float = 0.005  # Reduced 10x to prevent runaway (was 0.05, tried 0.0005 - too slow)
    gain_tau_ms: float = 1000.0
    baseline_noise_current: float = 0.05

    # =========================================================================
    # ADAPTVE THRESHOLD PLASTICITY (complementary to gain adaptation)
    # =========================================================================
    threshold_learning_rate: float = 0.03
    threshold_min: float = 0.2
    threshold_max: float = 1.5

    # =========================================================================
    # CORTEX-SPECIFIC DYNAMICS
    # =========================================================================
    # These parameters control cortical circuit mechanisms that are specific
    # to layered cortex architecture (not universal like homeostasis).

    # Recurrence in L2/3
    l23_recurrent_strength: float = 0.3  # Lateral connection strength
    l23_recurrent_decay: float = 0.9  # Recurrent activity decay

    # Weight bounds for L2/3 recurrent connections (signed, compact E/I approximation)
    # Unlike feedforward connections, recurrent lateral connections use signed weights
    # to approximate the mixed excitatory/inhibitory microcircuit within a cortical layer.
    # Positive weights = local excitation, negative weights = lateral inhibition.
    l23_recurrent_w_min: float = -1.5  # Allows inhibitory-like connections
    l23_recurrent_w_max: float = 1.0  # Symmetric by default

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

    # Gamma-based attention (spike-native phase gating for L2/3)
    # Always enabled for spike-native attention
    gamma_attention_width: float = 0.3  # Phase window width

    # =========================================================================
    # FEEDFORWARD INHIBITION (FFI)
    # =========================================================================
    # FFI detects stimulus changes and transiently suppresses recurrent activity
    # This is how the cortex naturally "clears" old representations when new input arrives
    # Always enabled (fundamental cortical mechanism)
    ffi_threshold: float = 0.3  # Input change threshold to trigger FFI
    ffi_strength: float = 0.8  # How much FFI suppresses L2/3 recurrent activity
    ffi_tau: float = 5.0  # FFI decay time constant (ms)

    # =========================================================================
    # GAP JUNCTIONS (L2/3 Interneuron Synchronization)
    # =========================================================================
    # Basket cells and chandelier cells in L2/3 have dense gap junction networks
    # Critical for cortical gamma oscillations (30-80 Hz) and precise spike timing
    # ~70-80% of cortical gap junctions are interneuron-interneuron
    gap_junction_strength: float = 0.12  # Moderate coupling for L2/3 interneurons
    gap_junction_threshold: float = 0.25
    gap_junction_max_neighbors: int = 8

    # =========================================================================
    # LAYER-SPECIFIC HETEROGENEITY
    # =========================================================================
    # Biological reality: Each cortical layer has distinct cell types with
    # different electrophysiological properties:
    # - L2/3 pyramidal: Medium tau_mem (~20ms), moderate threshold
    # - L4 spiny stellate: Fast, small tau_mem (~10ms), low threshold
    # - L5 thick-tuft pyramidal: Slow tau_mem (~30ms), high threshold, burst-capable
    # - L6 corticothalamic: Variable tau_mem (~15-25ms), moderate threshold
    #
    # This heterogeneity enables:
    # - L2/3: Integration and association over longer timescales
    # - L4: Fast sensory processing and feature detection
    # - L5: Decision-making and sustained output generation
    # - L6: Feedback control with tuned dynamics

    # Layer-specific membrane time constants (ms)
    # These control integration timescales for each layer
    layer_tau_mem: Dict[CortexLayer, float] = field(
        default_factory=lambda: {
            CortexLayer.L23: 20.0,  # Moderate integration for association
            CortexLayer.L4:  10.0,  # Fast integration for sensory input
            CortexLayer.L5:  30.0,  # Slow integration for output generation
            CortexLayer.L6A: 15.0,  # Fast for TRN feedback (low gamma)
            CortexLayer.L6B: 25.0,  # Moderate for relay feedback (high gamma)
        }
    )
    """Membrane time constants per layer.

    Biological ranges:
    - L4 spiny stellate: 8-12ms (fast sensory processing)
    - L2/3 pyramidal: 18-25ms (integration)
    - L5 pyramidal: 25-35ms (sustained output)
    - L6 pyramidal: 12-20ms (feedback control)
    """

    # Layer-specific voltage thresholds (mV)
    # Higher threshold = more selective, requires more input
    layer_v_threshold: Dict[CortexLayer, float] = field(
        default_factory=lambda: {
            CortexLayer.L23: -55.0,  # Moderate threshold for balanced processing
            CortexLayer.L4:  -52.0,  # Low threshold for sensitive input detection
            CortexLayer.L5:  -50.0,  # Lower threshold for reliable output (compensated by high tau)
            CortexLayer.L6A: -55.0,  # Moderate for attention gating
            CortexLayer.L6B: -52.0,  # Low for fast gain modulation
        }
    )
    """Voltage thresholds per layer.

    Biological values:
    - L4: -50 to -55mV (sensitive to input)
    - L2/3: -53 to -58mV (selective integration)
    - L5: -48 to -52mV (reliable output despite high tau)
    - L6: -50 to -55mV (varied for feedback roles)
    """

    # Layer-specific adaptation strengths
    # Controls spike-frequency adaptation per layer
    layer_adaptation: Dict[CortexLayer, float] = field(
        default_factory=lambda: {
            CortexLayer.L23: 0.15,  # Strong adaptation for decorrelation (inherited default)
            CortexLayer.L4:  0.05,  # Minimal adaptation for faithful sensory relay
            CortexLayer.L5:  0.10,  # Moderate adaptation for sustained output
            CortexLayer.L6A: 0.08,  # Light adaptation for feedback
            CortexLayer.L6B: 0.12,  # Moderate adaptation for gain control
        }
    )
    """Adaptation increments per layer.

    Biological justification:
    - L4: Minimal (faithful relay of sensory input)
    - L2/3: Strong (prevents runaway recurrence, decorrelates features)
    - L5: Moderate (allows burst patterns while preventing runaway)
    - L6: Light-moderate (supports feedback dynamics)
    """

    # =========================================================================
    # SPARSITY
    # =========================================================================
    l4_sparsity: float = 0.15  # Moderate sparsity
    l23_sparsity: float = 0.10  # Sparser (more selective)
    l5_sparsity: float = 0.20  # Less sparse (motor commands)
    l6a_sparsity: float = 0.12  # L6a → TRN (slightly more sparse than L2/3)
    l6b_sparsity: float = 0.15  # L6b → relay (moderate sparsity)

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION (SFA)
    # =========================================================================
    # Cortical pyramidal neurons show strong spike-frequency adaptation.
    # Inherited from base: adapt_increment=0.0, adapt_tau=100.0
    # Override for L2/3 strong adaptation:
    adapt_increment: float = 0.30  # Very strong adaptation for decorrelation
    # adapt_tau: 100.0 (use base default)

    # =========================================================================
    # SPIKE-TIMING DEPENDENT PLASTICITY (STDP)
    # =========================================================================
    a_plus: float = 0.01
    a_minus: float = 0.012

    # =========================================================================
    # TEMPORAL DYNAMICS & DELAYS
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

    # =========================================================================
    # PREDICTIVE CODING: L5/L6 → L4 FEEDBACK (NATURAL EMERGENCE)
    # =========================================================================
    # Deep layers (L5+L6) generate top-down predictions that inhibit L4
    # When prediction matches input, L4 is suppressed (no error)
    # When prediction mismatches input, L4 fires (error signal)
    # L2/3 naturally becomes error propagation pathway
    #
    # Biological basis:
    # - L5 deep pyramidal neurons project back to L4
    # - L6 corticothalamic neurons modulate L4 via local collaterals
    # - These connections are primarily inhibitory (via interneurons)
    l5_to_l4_pred_strength: float = 0.5  # L5→L4 prediction inhibition strength
    l6_to_l4_pred_strength: float = 0.3  # L6→L4 prediction inhibition strength
    prediction_learning_rate: float = 0.005  # Anti-Hebbian learning for predictions
    l5_to_l4_delay_ms: float = 1.0  # L5→L4 feedback delay (short, local)
    l6_to_l4_delay_ms: float = 1.5  # L6→L4 feedback delay (slightly longer)

    # Precision weighting: Scale predictions by confidence (population activity)
    # Higher activity in deep layers = stronger/more confident predictions
    # Biological basis: Attention modulates prediction precision via gain control
    precision_min: float = 0.5  # Minimum precision weight (even low activity has some effect)
    precision_max: float = 1.5  # Maximum precision weight (high activity strengthens prediction)


# ============================================================================
# Hippocampus Config
# ============================================================================


@dataclass
class HippocampusConfig(NeuralRegionConfig):
    """Configuration for hippocampus (trisynaptic circuit).

    The hippocampus has ~5x expansion from EC to DG, then compression back.

    **Size Specification**:
    Pure behavioral configuration. Sizes (input_size, dg_size, ca3_size, ca2_size, ca1_size)
    are passed separately via the `sizes` dict parameter to __init__().
    """

    # =========================================================================
    # CA3 RECURRENT STRENGTH AND BISTABILITY
    # =========================================================================
    ca3_recurrent_strength: float = 0.4  # Strength of recurrent connections

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

    # =========================================================================
    # CONSOLIDATION
    # =========================================================================
    # Biological reality: Memory consolidation operates over multiple timescales
    # - Fast trace (synaptic tagging): Minutes, ~60s tau
    # - Slow trace (systems consolidation): Hours, ~3600s tau
    # - Consolidation: Gradual transfer from fast (episodic) to slow (semantic)
    #
    # This implements "systems consolidation theory":
    # - Fast learning in hippocampus captures episodic details
    # - Slow consolidation transfers statistical regularities to neocortex
    # - Gradual interleaving prevents catastrophic forgetting
    fast_trace_tau_ms: float = 60_000.0  # Fast trace decay (1 minute = 60,000ms)
    slow_trace_tau_ms: float = 3_600_000.0  # Slow trace decay (1 hour = 3,600,000ms)
    consolidation_rate: float = 0.001  # Transfer rate from fast to slow (0.1% per timestep)
    slow_trace_contribution: float = 0.1  # Weight of slow trace in learning (10%)

    # =========================================================================
    # FEEDBACK INHIBITION
    # =========================================================================
    # Feedback inhibition from interneurons scales with total CA3 activity.
    # When many CA3 neurons fire, inhibition increases, making it harder
    # for the same neurons to fire again.
    ca3_feedback_inhibition: float = 0.3  # Feedback inhibition strength (normalized per neuron)

    # =========================================================================
    # FEEDFORWARD INHIBITION (FFI)
    # =========================================================================
    ffi_threshold: float = 0.3  # Input change threshold to trigger FFI
    ffi_strength: float = 0.3  # Reduced from 0.8 - DG needs to be more active for learning
    ffi_tau: float = 5.0  # FFI decay time constant (ms)

    # =========================================================================
    # GAP JUNCTIONS (Electrical Synapses)
    # =========================================================================
    # Gap junctions between CA1 interneurons (basket cells, bistratified cells)
    # provide fast electrical coupling for theta-gamma synchronization.
    # Critical for precise spike timing in episodic memory encoding/retrieval.
    #
    # CA1 interneurons (~10-15% of CA1 population) have dense gap junction
    # networks that synchronize inhibition during theta-gamma nested oscillations.
    gap_junction_strength: float = 0.01  # Coupling strength (biological: 0.05-0.2) - strongly reduced to prevent spontaneous oscillations from overwhelming task-driven input during early learning
    gap_junction_threshold: float = 0.25  # Neighborhood connectivity threshold
    gap_junction_max_neighbors: int = 8  # Max neighbors per interneuron (biological: 4-12)

    # =========================================================================
    # HETEROSYNAPTIC PLASTICITY
    # =========================================================================
    # Synapses to inactive postsynaptic neurons weaken when nearby neurons
    # fire strongly. This prevents winner-take-all dynamics from freezing.
    heterosynaptic_ratio: float = 0.1  # LTD for inactive synapses

    # =========================================================================
    # HOMEOSTATIC INTRINSIC PLASTICITY
    # =========================================================================
    # Adaptive gain control to maintain target firing rates (Turrigiano 2008)
    # Prevents bootstrap problem and maintains stable activity during learning
    target_firing_rate: float = 0.15  # 15% target for hippocampal neurons (increased for learning)
    gain_learning_rate: float = 0.003  # Reduced 10x to prevent runaway (was 0.03, tried 0.0003 - too slow)
    gain_tau_ms: float = 1500.0  # 1.5s averaging window (slower than cortex)
    baseline_noise_current: float = 0.30  # Strong spontaneous activity to bootstrap silent network (increased from 0.15)

    # Adaptive threshold plasticity (complementary to gain adaptation)
    threshold_learning_rate: float = 0.02  # Moderate threshold adaptation
    threshold_min: float = 0.05  # Lower floor to allow more aggressive adaptation for under-firing
    threshold_max: float = 1.5  # Allow some increase above default

    # =========================================================================
    # LEARNING RATES
    # =========================================================================
    learning_rate: float = 0.1  # Fast one-shot learning for CA3 recurrent
    ca3_ca2_learning_rate: float = 0.001  # Very weak CA3→CA2 (stability mechanism)
    ec_ca2_learning_rate: float = 0.01  # Strong EC→CA2 direct (temporal encoding)
    ca2_ca1_learning_rate: float = 0.005  # Moderate CA2→CA1 (social context to output)
    ec_ca1_learning_rate: float = 0.5  # Strong learning for EC→CA1 alignment

    # =========================================================================
    # NMDA RECEPTOR COINCIDENCE DETECTION
    # =========================================================================
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

    # =========================================================================
    # OSCILLATIONS
    # =========================================================================
    # PHASE PRECESSION
    # Phase diversity initialization: adds timing jitter to initial weights
    # This seeds the emergence of phase preferences (otherwise all neurons identical)
    phase_diversity_init: bool = True  # Initialize weights with timing diversity
    phase_jitter_std_ms: float = 5.0  # Std dev of timing jitter (0-10ms)

    # THETA-PHASE RESETS
    # Reset persistent activity at the start of each theta cycle to prevent
    # stale attractors from dominating. In real brains, theta troughs
    # (encoding phase) partially reset the network.
    theta_reset_persistent: bool = True  # Reset ca3_persistent at theta trough
    theta_reset_fraction: float = 0.5  # How much to decay (0=none, 1=full)

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
    # SIZES
    # =========================================================================
    # EC Layer III input size (for direct EC→CA1 pathway)
    # If 0, uses the same input as EC layer II
    # If >0, expects separate raw sensory input for the temporoammonic path
    ec_l3_input_size: int = 0

    # =========================================================================
    # SPARSITY
    # =========================================================================
    dg_sparsity: float = 0.10  # 10% active neurons (increased to prevent bottleneck while maintaining separation)
    ca3_sparsity: float = 0.10  # 10% active
    ca2_sparsity: float = 0.12  # 12% active (slightly higher than CA3)
    ca1_sparsity: float = 0.15  # 15% active

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
    # SPIKE-TIMING DEPENDENT PLASTICITY (STDP)
    # =========================================================================
    a_plus: float = 0.01
    a_minus: float = 0.01

    # =========================================================================
    # TEMPORAL DYNAMICS & DELAYS
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


# ============================================================================
# Prefrontal Cortex Config
# ============================================================================


@dataclass
class PrefrontalConfig(NeuralRegionConfig):
    """Configuration specific to prefrontal cortex.

    PFC implements DOPAMINE-GATED STDP:
    - STDP creates eligibility traces from spike timing
    - Dopamine gates what enters working memory and what gets learned
    - High DA → update WM and learn new associations
    - Low DA → maintain WM and protect existing patterns

    **Size Specification** (Semantic-First):
    - Sizes passed via sizes dict: input_size, n_neurons
    - Config contains only behavioral parameters (learning rates, time constants, etc.)
    """

    # =========================================================================
    # D1/D2 DOPAMINE RECEPTOR SUBTYPES
    # =========================================================================
    # Biological reality: PFC has both D1 (excitatory) and D2 (inhibitory) receptors
    # - D1-dominant neurons (~60%): "Go" pathway, enhance signals with DA
    # - D2-dominant neurons (~40%): "NoGo" pathway, suppress noise with DA
    #
    # This enables:
    # - D1: Update WM when DA high (new information is important)
    # - D2: Maintain WM when DA low (protect current state)
    # - Opponent modulation: D1 and D2 have opposite DA responses
    use_d1_d2_subtypes: bool = False  # Enable D1/D2 receptor subtypes
    d1_fraction: float = 0.6  # Fraction of neurons that are D1-dominant (60%)
    d1_da_gain: float = 0.5  # DA gain for D1 neurons (excitatory, 1.0 + gain*DA)
    d2_da_gain: float = 0.3  # DA suppression for D2 neurons (inhibitory, 1.0 - gain*DA)
    d2_output_weight: float = 0.5  # Weight of D2 output in competition (D1 - weight*D2)

    # =========================================================================
    # GAP JUNCTIONS
    # =========================================================================
    gap_junction_strength: float = 0.1  # Moderate coupling for PFC interneurons
    gap_junction_threshold: float = 0.3  # Neighborhood connectivity threshold
    gap_junction_max_neighbors: int = 8  # Max neighbors per interneuron (biological: 4-12)

    # =========================================================================
    # HOMEOSTATIC INTRINSIC PLASTICITY
    # =========================================================================
    # Adaptive gain control to maintain target firing rates (Turrigiano 2008)
    # Essential for PFC to overcome bootstrap problem and maintain working memory
    target_firing_rate: float = 0.05  # 5% target (sparse working memory representations)
    gain_learning_rate: float = 0.004  # Reduced 10x to prevent runaway (was 0.04, tried 0.0004 - too slow)
    gain_tau_ms: float = 2000.0  # 2s averaging window (slow for working memory stability)
    baseline_noise_current: float = 0.20  # Moderate spontaneous activity (increased from 0.05 for cold-start)

    # Adaptive threshold plasticity (complementary to gain adaptation)
    threshold_learning_rate: float = 0.02  # Slow for working memory stability
    threshold_min: float = 0.05  # Lower floor for under-firing regions
    threshold_max: float = 1.5  # Allow some increase above default

    # =========================================================================
    # LEARNING RATES
    # =========================================================================
    rule_lr: float = 0.001  # Learning rate for rule weights

    # =========================================================================
    # NEUROMODULATION: DOPAMINE
    # =========================================================================
    dopamine_tau_ms: float = 100.0  # DA decay time constant
    dopamine_baseline: float = 0.2  # Tonic DA level

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION
    # =========================================================================
    # PFC pyramidal neurons show adaptation. This helps prevent runaway
    # activity during sustained working memory maintenance.
    # Inherited from base, with PFC-specific overrides:
    adapt_increment: float = 0.2  # Moderate (maintains WM while adapting)
    adapt_tau: float = 150.0  # Slower decay (longer timescale for WM)

    # =========================================================================
    # WORKING MEMORY
    # =========================================================================
    # Biological reality: PFC neurons show heterogeneous maintenance properties
    # - Stable neurons: Strong recurrence, long time constants (~1-2s)
    # - Flexible neurons: Weak recurrence, short time constants (~100-200ms)
    #
    # This heterogeneity enables:
    # - Stable neurons: Maintain context/goals over long delays
    # - Flexible neurons: Rapid updating for new information
    # - Mixed selectivity: Distributed representations across neuron types
    stability_cv: float = 0.3  # Coefficient of variation for recurrent strength
    tau_mem_min: float = 100.0  # Minimum membrane time constant (ms) - flexible neurons
    tau_mem_max: float = 500.0  # Maximum membrane time constant (ms) - stable neurons

    # Working memory parameters
    wm_decay_tau_ms: float = 500.0  # How fast WM decays (slow!)
    wm_noise_std: float = 0.01  # Noise in WM maintenance

    # Gating parameters
    gate_threshold: float = 0.5  # DA level to open update gate

    # Recurrent connections for WM maintenance
    recurrent_strength: float = 0.8  # Self-excitation for persistence
    recurrent_inhibition: float = 0.2  # Lateral inhibition


# ============================================================================
# Striatum Config
# ============================================================================


@dataclass
class StriatumConfig(NeuralRegionConfig):
    """Configuration specific to striatal regions (behavior only, no sizes).

    Behavioral parameters:
    - Learning rates
    - Action selection (lateral_inhibition, softmax_temperature)
    - Exploration (ucb_coefficient, adaptive_exploration)
    - Neuromodulation (tonic_dopamine, d1/d2 sensitivity)
    - Homeostatic plasticity (normalization_rate)
    - Baseline pressure (baseline_pressure_rate, baseline_target_net)
    - Others as needed

    Key Features:
    =============
    1. THREE-FACTOR LEARNING: Δw = eligibility × dopamine
    2. D1/D2 OPPONENT PATHWAYS: Go/No-Go balance
    3. POPULATION CODING: Multiple neurons per action
    4. ADAPTIVE EXPLORATION: UCB + uncertainty-driven exploration
    5. HOMEOSTATIC PLASTICITY: Maintain stable activity and weight norms
    """

    # =========================================================================
    # ADAPTIVE GAIN CONTROL (HOMEOSTATIC INTRINSIC PLASTICITY)
    # =========================================================================
    # For striatum: Maintain COMBINED D1+D2 rate, not independent rates
    # Biology: D1 and D2 naturally balance via competition and FSI inhibition
    target_firing_rate: float = 0.08  # 8% target per pathway (16% combined for D1+D2)
    gain_learning_rate: float = 0.004  # Reduced 10x to prevent runaway (was 0.04, tried 0.0004 - too slow)
    gain_tau_ms: float = 1500.0  # 1.5s averaging window
    baseline_noise_current: float = 0.15  # Moderate spontaneous activity (increased from 0.06 for cold-start)

    # =========================================================================
    # ADAPTVE THRESHOLD PLASTICITY (for MSN neurons)
    # =========================================================================
    threshold_learning_rate: float = 0.02  # Moderate threshold adaptation
    threshold_min: float = 0.05  # Lower floor to allow more aggressive adaptation for under-firing
    threshold_max: float = 1.5  # Allow some increase above default

    # =========================================================================
    # ARCHITECTURE: D1/D2 PATHWAY RATIO
    # =========================================================================
    d1_d2_ratio: float = 0.5  # 50/50 split between D1 and D2 MSNs

    # =========================================================================
    # BASELINE PRESSURE (drift towards balanced D1/D2)
    # =========================================================================
    baseline_pressure_enabled: bool = True
    baseline_pressure_rate: float = 0.015
    baseline_target_net: float = 0.0

    # =========================================================================
    # ELIGIBILITY TRACES: MULTISCALE
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
    # EXPLORATION STRATEGIES
    # =========================================================================
    # Softmax action selection with temperature scaling
    softmax_action_selection: bool = True
    softmax_temperature: float = 2.0

    # Upper Confidence Bound (UCB) exploration
    # Encourages exploration of less-visited actions
    ucb_exploration: bool = True
    ucb_coefficient: float = 2.0

    # Adaptive exploration based on recent performance
    # Increases exploration when performance drops
    adaptive_exploration: bool = True
    performance_window: int = 10

    # Uncertainty-driven exploration
    uncertainty_temperature: float = 0.05
    min_exploration_boost: float = 0.05

    # =========================================================================
    # FEEDFORWARD INHIBITION (FFI): FAST-SPIKING INTERNEURONS (FSI)
    # =========================================================================
    # FSI are ~2% of striatal neurons (vs 95% MSNs) but critical for timing:
    # - Feedforward inhibition sharpens action selection
    # - Gap junction networks enable ultra-fast synchronization (<0.1ms)
    # - Synchronize MSN activity during beta oscillations (13-30 Hz)
    # - Critical for action initiation timing and motor control
    fsi_ratio: float = 0.02  # FSI as fraction of total striatal neurons (2%)

    # =========================================================================
    # GAP JUNCTIONS
    # =========================================================================
    # Gap junction configuration for FSI networks
    gap_junction_strength: float = 0.15  # Biological: 0.05-0.3
    gap_junction_threshold: float = 0.25  # Neighborhood inference threshold
    gap_junction_max_neighbors: int = 10  # Biological: 4-12 neighbors

    # =========================================================================
    # GOAL-CONDITIONED VALUES
    # =========================================================================
    # Enable PFC goal context to modulate striatal action values
    # Biology: PFC → Striatum projections gate action selection by goal context
    use_goal_conditioning: bool = True  # Enable goal-conditioned value learning

    # =========================================================================
    # HOMEOSTATIC PLASTICITY
    # =========================================================================
    normalization_rate: float = 0.01  # Reduced 10x to prevent runaway (was 0.1, tried 0.001 - too slow)

    # =========================================================================
    # LATERAL INHIBITION (ACTION SELECTION)
    # =========================================================================
    lateral_inhibition: bool = True
    inhibition_strength: float = 2.0

    # =========================================================================
    # LEARNING RATES
    # =========================================================================
    # Striatum three-factor learning rate (dopamine-gated plasticity)
    # NOTE: With eligibility trace fix (no double-scaling), this is the actual learning rate
    # Biological range: 0.0001-0.001 for stable opponent pathway dynamics
    learning_rate: float = 0.01  # 10× increase from 0.001 for faster weight changes

    # =========================================================================
    # NEUROMODULATION: TONIC DOPAMINE
    # =========================================================================
    tonic_dopamine: float = 0.3
    tonic_modulates_d1_gain: bool = True
    tonic_d1_gain_scale: float = 0.5
    tonic_modulates_exploration: bool = True
    tonic_exploration_scale: float = 0.1

    min_tonic_dopamine: float = 0.1
    max_tonic_dopamine: float = 0.5

    # =========================================================================
    # OSCILLATIONS: BETA MODULATION OF D1/D2 BALANCE
    # =========================================================================
    # Beta amplitude modulates D1/D2 balance for action maintenance vs switching
    # High beta → action persistence (D1 dominant, D2 suppressed)
    # Low beta → action flexibility (D2 effective, D1 reduced)
    beta_modulation_strength: float = 0.3  # [0, 1] - strength of beta influence

    # =========================================================================
    # SHORT-TERM PLASTICITY (STP)
    # =========================================================================
    # Biologically, different striatal input pathways have distinct STP properties:
    # - Cortex→MSNs: DEPRESSING (U=0.4) - prevents sustained cortical input from
    #   saturating striatum, enables novelty detection (fresh inputs get stronger)
    # - Thalamus→MSNs: WEAK FACILITATION (U=0.25) - phasic input amplification,
    #   balances phasic (thalamus) and tonic (cortex) command signals

    # Heterogeneous STP
    # Biological: Within same pathway, U varies 10-fold across synapses (Dobrunz & Stevens 1997)
    # Enables more realistic synaptic diversity and temporal dynamics
    heterogeneous_stp: bool = False  # Enable per-synapse STP parameter sampling
    stp_variability: float = 0.3  # Coefficient of variation (0.2-0.5 typical)
    stp_seed: Optional[int] = None  # Random seed for reproducibility

    # =========================================================================
    # TEMPORAL DYNAMICS & DELAYS
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


# ============================================================================
# Thalamus Config
# ============================================================================


@dataclass
class ThalamusConfig(NeuralRegionConfig):
    """Configuration for thalamic relay nucleus.

    Thalamus sits between sensory input and cortex, providing:
    - Sensory gating (alpha-based suppression)
    - Mode switching (burst vs tonic)
    - Gain modulation (norepinephrine)
    - Spatial filtering

    **Pure Behavioral Configuration**:
    Contains ONLY behavioral parameters (learning rates, gains, thresholds).
    Sizes (relay_size, trn_size, input_size) are passed separately at instantiation.
    """

    # =========================================================================
    # ADAPTIVE GAIN CONTROL (HOMEOSTATIC INTRINSIC PLASTICITY)
    # =========================================================================
    target_firing_rate: float = 0.08  # Target firing rate for homeostatic plasticity (8% sparsity, increased for better signal).
    gain_learning_rate: float = 0.003  # Learning rate for gain adaptation (faster for cold-start, then stabilizes).
    gain_tau_ms: float = 1000.0  # Time constant for firing rate averaging (fast enough to prevent hysteresis, ~1s).
    baseline_noise_current: float = 0.3

    # =========================================================================
    # ADAPTVE THRESHOLD PLASTICITY (complementary to gain adaptation)
    # =========================================================================
    threshold_learning_rate: float = 0.03
    threshold_min: float = 0.05
    threshold_max: float = 1.5

    # =========================================================================
    # THALAMIC RELAY NUCLEUS PARAMETERS
    # =========================================================================

    # Relay parameters
    relay_strength: float = 2.0  # Increased to drive more cortex activity (target 4% vs current 0.58%)
    """Base relay gain (thalamus amplifies weak inputs).

    Increased from 1.5 to 3.5 to overcome cold-start silence:
    - Center-surround filter causes 94% signal loss (0.2 sparsity × 0.3 scale)
    - Alpha gating reduces by additional 0-50%
    - Need ~3-5x gain to compensate for these attenuations
    """

    # =========================================================================
    # BURST vs TONIC MODE SWITCHING
    # =========================================================================
    burst_threshold: float = -0.2
    """Membrane potential threshold for burst mode (hyperpolarized)."""

    tonic_threshold: float = 0.3
    """Membrane potential threshold for tonic mode (depolarized)."""

    burst_gain: float = 2.0
    """Amplification factor for burst mode (alerting signal)."""

    # =========================================================================
    # CORTICAL FEEDBACK MODULATION
    # =========================================================================
    l6a_to_trn_strength: float = 0.8
    """Strength of L6a → TRN feedback (inhibitory modulation, type I)."""

    l6b_to_relay_strength: float = 0.6
    """Strength of L6b → relay feedback (excitatory modulation, type II)."""

    # =========================================================================
    # GAP JUNCTIONS: TRN INTERNEURONS
    # =========================================================================
    gap_junction_strength: float = 0.15  # Moderate coupling for TRN interneurons
    gap_junction_threshold: float = 0.3  # Neighborhood connectivity threshold
    gap_junction_max_neighbors: int = 8  # Max neighbors per interneuron (biological: 4-12)

    # =========================================================================
    # OSCILLATIONS: ALPHA RHYTHM MODULATION
    # =========================================================================
    alpha_suppression_strength: float = 0.1
    """How strongly alpha suppresses unattended inputs (0-1).

    Reduced from 0.5 to 0.3 for better training:
    - Still provides attention gating (30% max suppression)
    - Reduces compound attenuation with spatial filtering
    - Allows faster learning during early training
    """

    trn_inhibition_strength: float = 0.15
    """Strength of TRN → relay inhibition."""

    trn_recurrent_strength: float = 0.4
    """TRN recurrent inhibition (for oscillations)."""

    # =========================================================================
    # SHORT-TERM PLASTICITY (STP)
    # =========================================================================
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

    # =========================================================================
    # SPATIAL FILTERING: CENTER-SURROUND RECEPTIVE FIELDS
    # =========================================================================
    spatial_filter_width: float = 0.15
    """Gaussian filter width for center-surround (as fraction of input)."""

    center_excitation: float = 3.0
    """Center enhancement in receptive field."""

    surround_inhibition: float = 0.5
    """Surround suppression in receptive field."""


# ============================================================================
# Brain Config (global parameters for entire brain instance)
# ============================================================================


@dataclass
class BrainConfig:
    """Complete brain configuration for a single brain instance.

    Each brain instance is fully self-contained with
    its own device, dt, oscillator frequencies, etc.

    This enables:
    - Multiple independent brains with different devices (GPU vs CPU)
    - Different temporal resolutions per brain (dt_ms)
    - Different oscillator frequencies per brain
    """

    device: str = "cpu"  # Device to run on: 'cpu', 'cuda', 'cuda:0', etc.
    seed: Optional[int] = None  # Random seed for reproducibility. None = no seeding.

    # =========================================================================
    # TIMING
    # =========================================================================
    dt_ms: float = DEFAULT_DT_MS
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

    See thalia.brain.oscillator.OscillatorCoupling for full parameters.
    """

    # =========================================================================
    # VALIDATION AND SUMMARY
    # =========================================================================

    def summary(self) -> str:
        """Return formatted summary of brain configuration."""
        lines = [
            "=== Brain Configuration ===",
            f"  Device: {self.device}",
            f"  Data type: {self.dtype}",
            f"  Timestep: {self.dt_ms} ms",
            "",
            "  Oscillator Frequencies:",
            f"    Delta:  {self.delta_frequency_hz:>5.1f} Hz",
            f"    Theta:  {self.theta_frequency_hz:>5.1f} Hz",
            f"    Alpha:  {self.alpha_frequency_hz:>5.1f} Hz",
            f"    Beta:   {self.beta_frequency_hz:>5.1f} Hz",
            f"    Gamma:  {self.gamma_frequency_hz:>5.1f} Hz",
            f"    Ripple: {self.ripple_frequency_hz:>5.1f} Hz",
            "",
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
