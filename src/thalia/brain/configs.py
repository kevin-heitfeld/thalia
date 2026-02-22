"""Configurations for brain regions and overall brain parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from thalia.constants import DEFAULT_DT_MS


# ============================================================================
# Enums
# ============================================================================


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
    gain_learning_rate: float = 0.02  # INCREASED from 0.005 to respond faster to activity collapse
    target_firing_rate: float = 0.05  # Target firing rate for homeostatic plasticity
    gain_tau_ms: float = 2000.0  # Time constant for firing rate averaging
    baseline_noise_conductance_enabled: bool = False  # DISABLED - stochastic synaptic background (miniature EPSPs)

    # =========================================================================
    # NEUROMODULATION
    # =========================================================================
    enable_neuromodulation: bool = False  # Enable/disable neuromodulator receptor updates (DA/NE/ACh)

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

    # =========================================================================
    # SPIKE-TIMING DEPENDENT PLASTICITY (STDP)
    # =========================================================================
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
    # PACEMAKER DYNAMICS
    # =========================================================================
    base_frequency_hz: float = 8.0
    """Base theta frequency in Hz. Modulated by ACh, NE, DA (range: 4-10 Hz)."""

    burst_duty_cycle: float = 0.08
    """Fraction of cycle spent in burst phase (vs inter-burst silence).

    CRITICAL: Set to 0.08 for NARROW burst windows.
    - At 8 Hz (125ms period), burst phase = 125 * 0.08 = 10ms
    - Inter-burst silence = 125 * 0.92 = 115ms
    - 10ms burst window allows 2 spikes with 5ms refractory
    - Narrow window prevents early/late firings outside pacemaker rhythm
    """

    burst_amplitude: float = 0.025
    """Peak drive conductance during burst (normalized by g_L).

    CRITICAL: Minimal supra-threshold value to trigger brief bursts.
    - Must overcome threshold (~1.2-1.3) but not create sustained firing
    - Value 0.025 creates just enough drive for 2-3 spikes
    - Adaptation accumulates quickly to terminate burst
    - Results in clean 8 Hz rhythm with short intra-burst intervals
    """

    inter_burst_amplitude: float = 0.001
    """Baseline drive conductance between bursts (subthreshold).

    CRITICAL: Reduced to 0.001 (near zero) to ensure complete silence.
    During inter-burst phase:
    - Membrane potential decays back to rest
    - Adaptation variables decay to baseline
    - Neurons are prepared for next burst
    Should be well below threshold (< 0.1) to prevent spontaneous firing.
    """

    # =========================================================================
    # CHOLINERGIC NEURON PROPERTIES
    # =========================================================================
    ach_tau_mem: float = 35.0
    """ACh neuron membrane time constant (ms). Moderate for responsive bursting.

    REDUCED to 35ms for faster integration during short (10ms) burst windows:
    - Quick enough to respond within burst window
    - Still slower than typical neurons (20ms) for pacemaker-like behavior
    - Matches burst window duration for clean firing
    """

    ach_threshold: float = 1.3
    """ACh neuron threshold (slightly elevated to require stronger drive)."""

    ach_reset: float = 0.0
    """ACh neuron reset potential after spike."""

    ach_adaptation_tau: float = 80.0
    """ACh adaptation time constant (ms). INCREASED for burst termination.

    Critical: Must balance two requirements:
    1. Persist through 25ms burst duration to accumulate and terminate firing
    2. Decay during 100ms inter-burst interval to allow next burst

    With tau=80ms:
    - During burst (25ms): ~27% decay → adaptation accumulates strongly
    - Between bursts (100ms): ~71% decay → mostly resets for next burst
    - Ensures burst self-terminates after 2-4 spikes
    """

    ach_adaptation_increment: float = 0.40
    """ACh adaptation increment per spike.

    INCREASED to 0.40 for rapid burst termination.
    - Each spike adds 0.40 to adaptation current
    - After 2-3 spikes: adaptation ~ 0.80-1.20, strongly suppressing firing
    - Burst self-terminates quickly, then adaptation decays during inter-burst
    - Creates clean 8 Hz rhythm with brief (~10-15ms) bursts
    """

    # =========================================================================
    # GABAERGIC NEURON PROPERTIES
    # =========================================================================
    gaba_tau_mem: float = 30.0
    """GABA neuron membrane time constant (ms). Moderate for responsive theta bursting."""

    gaba_threshold: float = 1.2
    """GABA neuron threshold."""

    gaba_reset: float = 0.0
    """GABA neuron reset potential after spike."""

    gaba_adaptation_tau: float = 75.0
    """GABA adaptation time constant (ms). Similar to ACh for matched burst dynamics."""

    gaba_adaptation_increment: float = 0.40
    """GABA adaptation increment per spike. INCREASED for rapid burst termination."""


# ============================================================================
# Reward Encoder Config
# ============================================================================


@dataclass
class RewardEncoderConfig(NeuralRegionConfig):
    """Configuration for reward encoder region.

    The reward encoder provides a spike-based interface for delivering external
    reward signals to VTA. It uses population coding to convert scalar reward
    values into naturalistic spike patterns, abstracting away the complexity
    of hypothalamic and limbic reward pathways.

    Biological motivation:
    - Real brains process rewards through lateral hypothalamus, amygdala, etc.
    - This region serves as a simplified placeholder for those pathways
    - Enables clean external interface while maintaining spike-based signaling
    - Can be expanded to full limbic system in future versions
    """

    reward_noise: float = 0.1
    """Variability in spike encoding for biological realism (10% noise)."""


# ============================================================================
# Substantia Nigra pars Reticulata (SNr) Config
# ============================================================================


@dataclass
class SubstantiaNigraConfig(NeuralRegionConfig):
    """Configuration for SNr (substantia nigra pars reticulata) region.

    The SNr is the primary output nucleus of the basal ganglia, consisting of
    tonically-active GABAergic neurons that gate thalamic output and provide
    value feedback to VTA for dopamine-based reinforcement learning.

    Key features:
    - Tonic firing at 50-70 Hz baseline
    - Disinhibition mechanism: Striatum D1 reduces SNr → releases thalamus
    - Value encoding: Firing rate inversely proportional to state value
    - Closed-loop TD learning: Striatum → SNr → VTA → Striatum
    """

    baseline_drive: float = 0.0
    """Tonic drive conductance

    Biological SNr neurons fire tonically at 50-70 Hz baseline due to intrinsic
    currents (persistent Na+, T-type Ca2+) and recurrent excitation. In
    conductance-based model, this represents a tonic excitatory conductance
    (normalized by g_L).

    This value (0.008) is calibrated with:
    - Moderate noise (std=0.05)
    - Realistic membrane tau (15ms)
    - Standard leak (g_L=0.10)
    - High threshold (1.25) to reduce excitability
    - Strong striatal inhibition (D1: 0.08 weight, ~0.69 g_inh at 8.65% firing)
    - Weak D2 excitation (D2: 0.01 weight, ~0.09 g_exc at 8.65% firing)
    To produce biologically plausible 50-70 Hz (5-7% in 1ms bins) tonic baseline.

    BIOLOGY: With conductance-based neurons, shunting inhibition is
    EXTREMELY powerful (diverts current proportional to g_inh × V_drive). The
    large D1 inhibition (~0.69) creates massive shunting that requires very
    low baseline drive to achieve target firing rate.

    Calculation: With D1 inhibition ~0.69 g_inh and D2 excitation ~0.09 g_exc,
    baseline drive 0.008 provides total excitation 0.008 + 0.09 = 0.098.
    With massive shunting from 0.69 g_inh and high threshold (1.25), this
    produces ~5-7% baseline.

    Calibration history:
    - 0.110 → 46.26% (HYPERACTIVE)
    - 0.025 → 23.46% (still too high - 4x target)
    - 0.008 → targeting 5-7% with threshold 1.25
    """

    tau_mem: float = 15.0
    """Membrane time constant for realistic integration (10-20ms typical for SNr)."""

    v_threshold: float = 1.25
    """Firing threshold. Set to 1.25 to maintain 5-7% target with very low baseline drive (0.008) and strong D1 inhibition (~0.69)."""

    tau_ref: float = 2.0
    """Refractory period for realistic max frequency (~500 Hz ceiling, actual 50-70 Hz)."""


# ============================================================================
# Ventral Tegmental Area (VTA) Config
# ============================================================================


@dataclass
class VTAConfig(NeuralRegionConfig):
    """Configuration for VTA (ventral tegmental area) region.

    The VTA is the brain's primary dopamine source, computing reward prediction
    errors (RPE) through burst/pause dynamics. It forms the core of the
    reinforcement learning system by broadcasting teaching signals to all regions.

    Key features:
    - Dopamine neurons: Tonic (4-5 Hz) + phasic (burst/pause)
    - RPE computation: δ = r - V(s)
    - Closed-loop TD learning with SNr feedback
    - Adaptive normalization to prevent saturation
    - Strong baseline inhibition to maintain biological firing rates
    """

    rpe_gain: float = 15.0
    """Gain for converting RPE to membrane current (mV per RPE unit).

    Positive RPE → depolarization → burst (15-20 Hz)
    Negative RPE → hyperpolarization → pause (<1 Hz)
    """

    gamma: float = 0.99
    """TD learning discount factor for future reward weighting."""

    rpe_normalization: bool = True
    """Enable adaptive RPE normalization to prevent saturation."""


# ============================================================================
# Locus Coeruleus (LC) Config
# ============================================================================


@dataclass
class LCConfig(NeuralRegionConfig):
    """Configuration for LC (locus coeruleus) region.

    The LC is the brain's primary norepinephrine source, broadcasting arousal
    and uncertainty signals that modulate attention, gain, and exploration.
    LC neurons exhibit synchronized bursting due to dense gap junction coupling.

    Key features:
    - Norepinephrine neurons: Tonic (1-3 Hz) + phasic bursts (10-15 Hz for 500ms)
    - Gap junction coupling → synchronized population bursts
    - Uncertainty signal → exploratory behavior, network reset
    - Global projections → all brain regions
    """

    uncertainty_gain: float = 20.0
    """Gain for converting uncertainty to membrane current (mV per uncertainty unit).

    High uncertainty → depolarization → synchronized burst (10-15 Hz)
    Low uncertainty → hyperpolarization → pause or low tonic
    """

    gap_junction_strength: float = 0.05
    """Gap junction coupling strength (voltage-gated to prevent pacemaker quenching)."""

    gap_junction_radius: int = 50
    """Radius for gap junction connectivity (neurons within this range are coupled)."""

    uncertainty_normalization: bool = True
    """Enable adaptive uncertainty normalization to prevent saturation."""


# ============================================================================
# Nucleus Basalis (NB) Config
# ============================================================================


@dataclass
class NBConfig(NeuralRegionConfig):
    """Configuration for NB (nucleus basalis) region.

    The NB is the brain's primary source of cortical acetylcholine, broadcasting
    attention and encoding/retrieval mode signals. NB neurons exhibit fast,
    brief bursts in response to prediction errors and attention shifts.

    Key features:
    - Acetylcholine neurons: Tonic (2-5 Hz) + fast bursts (10-20 Hz for 50-100ms)
    - Brief bursts (shorter than DA/NE) due to fast SK adaptation
    - Prediction error magnitude → encoding mode, attention enhancement
    - Selective projections → cortex and hippocampus (not striatum)
    """

    pe_gain: float = 25.0
    """Gain for converting prediction error to membrane current (mV per PE unit).

    High |PE| → depolarization → fast brief burst (10-20 Hz for 50-100ms)
    Low |PE| → baseline tonic firing
    """

    pe_normalization: bool = True
    """Enable adaptive prediction error normalization to prevent saturation."""


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
    # At 1ms timestep: 40-50 Hz = 4-5% firing rate per timestep
    target_firing_rate: float = 0.045  # 4.5% target = 45 Hz (biological Purkinje baseline)
    gain_learning_rate: float = 0.005  # Reduced 10x to prevent runaway (was 0.05, tried 0.0005 - too slow)
    gain_tau_ms: float = 2000.0  # 2s averaging window (slow for motor stability)

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

    granule_connectivity: float = 0.03  # Fraction of granule cells active (3%)
    purkinje_n_dendrites: int = 100  # Simplified dendritic compartments

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


# ============================================================================
# Cortex Config
# ============================================================================


@dataclass
class CortexConfig(NeuralRegionConfig):
    """Configuration for layered cortical microcircuit."""

    # =========================================================================
    # ADAPTIVE GAIN CONTROL (HOMEOSTATIC INTRINSIC PLASTICITY)
    # =========================================================================
    target_firing_rate: float = 0.03
    gain_learning_rate: float = 0.004
    gain_tau_ms: float = 2000.0

    # =========================================================================
    # ADAPTVE THRESHOLD PLASTICITY (complementary to gain adaptation)
    # =========================================================================
    threshold_learning_rate: float = 0.03
    threshold_min: float = 0.2
    threshold_max: float = 1.5

    # =========================================================================
    # FEEDFORWARD INHIBITION (FFI)
    # =========================================================================
    # FFI detects stimulus changes and transiently suppresses recurrent activity
    # This is how the cortex naturally "clears" old representations when new input arrives
    # Always enabled (fundamental cortical mechanism)
    ffi_threshold: float = 0.3  # Input change threshold to trigger FFI
    ffi_strength: float = 0.18  # INCREASED from 0.12 to break 1 Hz cortical synchrony
    # Stronger FFI makes cortex more stimulus-driven and less self-sustaining
    ffi_tau: float = 5.0  # FFI decay time constant (ms)

    # =========================================================================
    # GAP JUNCTIONS (L2/3 Interneuron Synchronization)
    # =========================================================================
    # Basket cells and chandelier cells in L2/3 have dense gap junction networks
    # Critical for cortical gamma oscillations (30-80 Hz) and precise spike timing
    # ~70-80% of cortical gap junctions are interneuron-interneuron
    # INCREASED: Gap junctions are the PRIMARY mechanism for emergent gamma (PING)
    gap_junction_strength: float = 0.25  # INCREASED from 0.12 for stronger gamma synchronization
    gap_junction_threshold: float = 0.20  # REDUCED from 0.25 for easier activation
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

    # Layer-specific neuron parameters
    # Each layer gets a complete parameter set for neuron creation
    layer_overrides: Dict[CortexLayer, Dict[str, float]] = field(
        default_factory=lambda: {
            CortexLayer.L23: {
                "tau_mem": 20.0,          # Moderate integration for association
                "v_threshold": 1.8,       # INCREASED from 1.5 to reduce hyperactivity (25.3% → 5-10%)
                "adapt_increment": 0.45,  # VERY STRONG adaptation (prevents runaway recurrence)
                "tau_adapt": 150.0,       # Medium-slow decay (100-200ms biological)
            },
            CortexLayer.L4: {
                "tau_mem": 10.0,          # Fast integration for sensory input
                "v_threshold": 2.5,       # INCREASED to 2.5 to achieve target 1-3% FR (was 5.04%)
                "adapt_increment": 0.35,  # INCREASED to 0.35 for stronger spike frequency adaptation
                "tau_adapt": 80.0,        # Fast decay (50-100ms biological)
            },
            CortexLayer.L5: {
                "tau_mem": 30.0,          # Slow integration for output generation
                "v_threshold": 1.2,       # INCREASED from 0.9 to reduce hyperactivity (19% → 5-10%)
                "adapt_increment": 0.20,  # INCREASED from 0.10 for stronger adaptation
                "tau_adapt": 120.0,       # Medium decay (80-150ms biological)
            },
            CortexLayer.L6A: {
                "tau_mem": 15.0,          # Fast for TRN feedback (low gamma)
                "v_threshold": 1.0,       # Standard for attention gating
                "adapt_increment": 0.08,  # Light adaptation for feedback
                "tau_adapt": 100.0,       # Medium-fast decay (80-120ms biological)
            },
            CortexLayer.L6B: {
                "tau_mem": 25.0,          # Moderate for relay feedback (high gamma)
                "v_threshold": 0.9,       # Lower for fast gain modulation
                "adapt_increment": 0.12,  # Moderate adaptation for gain control
                "tau_adapt": 100.0,       # Medium-fast decay (80-120ms biological)
            },
        }
    )
    """Layer-specific neuron parameters.

    Each layer has distinct electrophysiological properties:

    **L2/3 (Integration & Association)**:
    - tau_mem: 20ms (moderate, ~10 Hz resonance)
    - v_threshold: 1.8 (high, selective integration)
    - adapt_increment: 0.45 (very strong, prevents runaway)
    - tau_adapt: 150ms (slow, sustained decorrelation)

    **L4 (Fast Sensory Processing)**:
    - tau_mem: 10ms (fast, ~20 Hz resonance)
    - v_threshold: 0.9 (low, sensitive detection)
    - adapt_increment: 0.05 (minimal, faithful relay)
    - tau_adapt: 80ms (fast, rapid reset)

    **L5 (Output Generation)**:
    - tau_mem: 30ms (slow, ~6 Hz resonance)
    - v_threshold: 1.2 (moderate-high, reliable output)
    - adapt_increment: 0.20 (moderate, burst patterns)
    - tau_adapt: 120ms (medium, stable output)

    **L6A (TRN Feedback, Low Gamma)**:
    - tau_mem: 15ms (fast feedback control)
    - v_threshold: 1.0 (standard)
    - adapt_increment: 0.08 (light, feedback dynamics)
    - tau_adapt: 100ms (medium-fast)

    **L6B (Relay Feedback, High Gamma)**:
    - tau_mem: 25ms (moderate feedback)
    - v_threshold: 0.9 (lower, fast modulation)
    - adapt_increment: 0.12 (moderate, gain control)
    - tau_adapt: 100ms (medium-fast)

    Biological ranges from intracellular recordings:
    - tau_mem: L4 spiny stellate 8-12ms, L2/3 pyramidal 18-25ms, L5 pyramidal 25-35ms
    - v_threshold: -50 to -58mV biological (normalized 0.9-1.8 in our model)
    - adapt_increment: Layer-specific Ca2+-dependent K+ conductances
    - tau_adapt: 50-200ms depending on cell type and recording conditions
    """

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
    l5_to_l4_delay_ms: float = 1.0  # L5→L4 feedback delay (short, local)
    l6_to_l4_delay_ms: float = 1.5  # L6→L4 feedback delay (slightly longer)
    l4_to_l23_delay_ms: float = 2.0  # L4→L2/3 axonal delay (short vertical)
    l23_to_l23_delay_ms: float = 9.0  # L2/3→L2/3 recurrent delay (longer horizontal)
    l23_to_l5_delay_ms: float = 2.0  # L2/3→L5 axonal delay (longer vertical)
    l23_to_l6a_delay_ms: float = 2.0  # L2/3→L6a axonal delay (type I pathway, slow)
    l23_to_l6b_delay_ms: float = 3.0  # L2/3→L6b axonal delay (type II pathway, fast)

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
    prediction_learning_rate: float = 0.005  # Anti-Hebbian learning for predictions

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
    """

    # =========================================================================
    # CA3 RECURRENT BISTABILITY
    # =========================================================================
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
    ca3_persistent_gain: float = 0.0

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
    # FEEDFORWARD INHIBITION (FFI)
    # =========================================================================
    ffi_threshold: float = 0.3  # Input change threshold to trigger FFI
    ffi_strength: float = 0.1  # Moderate strength to control CA3 hyperactivity
    ffi_tau: float = 5.0  # FFI decay time constant (ms)

    # =========================================================================
    # TONIC INHIBITION (Ambient GABA)
    # =========================================================================
    # Real hippocampus has extrasynaptic GABA_A receptors (α5 subunit) that
    # provide continuous background inhibition from GABA spillover.
    # This prevents runaway excitation even when phasic inhibition fails.
    tonic_inhibition: float = 0.01  # Constant background GABA conductance (normalized)

    # =========================================================================
    # GAP JUNCTIONS (Electrical Synapses)
    # =========================================================================
    # Gap junctions between CA1 interneurons (basket cells, bistratified cells)
    # provide fast electrical coupling for theta-gamma synchronization.
    # Critical for precise spike timing in episodic memory encoding/retrieval.
    #
    # CA1 interneurons (~10-15% of CA1 population) have dense gap junction
    # networks that synchronize inhibition during theta-gamma nested oscillations.
    gap_junction_strength: float = 0.01  # Coupling strength (biological: 0.05-0.2)
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
    # Adaptive gain control to maintain target firing rates
    target_firing_rate: float = 0.15  # 15% target for hippocampal neurons (increased for learning)
    gain_learning_rate: float = 0.0001
    gain_tau_ms: float = 1500.0  # 1.5s averaging window (slower than cortex)

    # Adaptive threshold plasticity (complementary to gain adaptation)
    threshold_learning_rate: float = 0.02  # Moderate threshold adaptation
    threshold_min: float = 0.05  # Lower floor to allow more aggressive adaptation for under-firing
    threshold_max: float = 1.5  # Allow some increase above default

    # =========================================================================
    # LEARNING RATES
    # =========================================================================
    learning_rate: float = 0.1  # Fast one-shot learning for CA3 recurrent
    ca2_ca1_learning_rate: float = 0.005  # Moderate CA2→CA1 (social context to output)

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
    # SPIKE-FREQUENCY ADAPTATION (SFA)
    # =========================================================================
    # Real CA3 pyramidal neurons show strong adaptation: Ca²⁺ influx during
    # spikes activates K⁺ channels (I_AHP) that hyperpolarize the neuron.
    # This prevents the same neurons from dominating activity.
    # Inherited from base with hippocampus-specific override:
    adapt_increment: float = 0.75  # DRASTICALLY increased from 0.5 to 0.75 (50% stronger) to suppress hyperactivity
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
    dg_to_ca3_delay_ms: float = 3.0    # DG→CA3 axonal delay (mossy fibers)
    ca3_to_ca3_delay_ms: float = 10.0  # CA3→CA3 recurrent delay (CRITICAL for preventing synchronization)
    ca3_to_ca2_delay_ms: float = 2.0   # CA3→CA2 axonal delay
    ca2_to_ca1_delay_ms: float = 2.0   # CA2→CA1 axonal delay
    ca3_to_ca1_delay_ms: float = 3.0   # CA3→CA1 axonal delay (Schaffer collaterals)


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
    d1_fraction: float = 0.6  # Fraction of neurons that are D1-dominant (60%)
    d1_da_gain: float = 0.3  # REDUCED from 0.5: DA gain for D1 neurons (excitatory)
    d2_da_gain: float = 0.3  # DA suppression for D2 neurons (inhibitory, 1.0 - gain*DA)

    # =========================================================================
    # GAP JUNCTIONS
    # =========================================================================
    gap_junction_strength: float = 0.1  # Moderate coupling for TRN synchronization (enables 8-13 Hz alpha from Relay↔TRN loops)
    gap_junction_threshold: float = 0.3  # Neighborhood connectivity threshold
    gap_junction_max_neighbors: int = 8  # Max neighbors per interneuron (biological: 4-12)

    # =========================================================================
    # HOMEOSTATIC INTRINSIC PLASTICITY
    # =========================================================================
    # Adaptive gain control to maintain target firing rates
    # CRITICAL FIX: PFC needs HIGHER baseline firing (7-10%) for working memory maintenance.
    # Setting target too low (5%) caused homeostatic collapse: activity 7% → gain reduced
    # → activity drops → gain reduced further → -84% collapse in 100ms.
    target_firing_rate: float = 0.10  # 10% target (working memory needs sustained activity)
    gain_learning_rate: float = 0.001  # Very slow adaptation (was 0.004, caused collapse)
    gain_tau_ms: float = 5000.0  # 5s averaging window (very slow for WM stability)

    # Adaptive threshold plasticity (complementary to gain adaptation)
    threshold_learning_rate: float = 0.02  # Slow for working memory stability
    threshold_min: float = 0.05  # Lower floor for under-firing regions
    threshold_max: float = 1.5  # Allow some increase above default

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION
    # =========================================================================
    # PFC pyramidal neurons show adaptation. This helps prevent runaway
    # activity during sustained working memory maintenance.
    adapt_increment: float = 0.05  # Very weak (allows sustained WM activity)
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
    recurrent_delay_ms: float = 10.0  # PFC recurrent delay (prevents instant feedback oscillations)

    # Gating parameters
    gate_threshold: float = 0.5  # DA level to open update gate


# ============================================================================
# Striatum Config
# ============================================================================


@dataclass
class StriatumConfig(NeuralRegionConfig):
    """Configuration specific to striatal regions.

    Behavioral parameters:
    - Learning rates
    - Exploration
    - Neuromodulation (tonic_dopamine, d1/d2 sensitivity)
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
    # BIOLOGY: Homeostatic plasticity operates on minutes-to-hours timescale
    # Increased tau from 1.5s to 30s (should be hours, but 30s for testing)
    # Reduced learning rate 10x to prevent rapid changes
    target_firing_rate: float = 0.08  # 8% target per pathway (16% combined for D1+D2)
    gain_learning_rate: float = 0.0004  # REDUCED 10x: slow homeostatic adaptation
    gain_tau_ms: float = 30000.0  # INCREASED 20x: 30s averaging window (biological: minutes to hours)

    # =========================================================================
    # ADAPTVE THRESHOLD PLASTICITY (for MSN neurons)
    # =========================================================================
    threshold_learning_rate: float = 0.02  # Moderate threshold adaptation
    threshold_min: float = 0.05  # Lower floor to allow more aggressive adaptation for under-firing
    threshold_max: float = 1.5  # Allow some increase above default

    # =========================================================================
    # ELIGIBILITY TRACES: MULTISCALE
    # =========================================================================
    # Biological: Synaptic tags (eligibility traces) have multiple timescales:
    # - Fast traces (~500ms): Immediate pre-post spike coincidence tagging
    # - Slow traces (~60s): Consolidated tags from fast traces, enables credit
    #   assignment over multiple seconds (e.g., delayed rewards in RL tasks)
    # Combined eligibility: fast_trace + α × slow_trace enables both rapid and
    # delayed credit assignment.
    fast_eligibility_tau_ms: float = 500.0  # Fast trace decay (~500ms)
    slow_eligibility_tau_ms: float = 60000.0  # Slow trace decay (~60s)
    eligibility_consolidation_rate: float = 0.01  # Transfer rate from fast to slow (1% per timestep)
    slow_trace_weight: float = 0.3  # Weight of slow trace in combined eligibility

    # =========================================================================
    # EXPLORATION STRATEGIES
    # =========================================================================
    # Adaptive exploration based on recent performance
    # Increases exploration when performance drops
    performance_window: int = 10

    # =========================================================================
    # GAP JUNCTIONS
    # =========================================================================
    # Gap junction configuration for FSI networks
    gap_junction_strength: float = 0.15  # Biological: 0.05-0.3
    gap_junction_threshold: float = 0.25  # Neighborhood inference threshold
    gap_junction_max_neighbors: int = 10  # Biological: 4-12 neighbors

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
    min_tonic_dopamine: float = 0.1
    max_tonic_dopamine: float = 0.5

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
    """

    # =========================================================================
    # ADAPTIVE GAIN CONTROL (HOMEOSTATIC INTRINSIC PLASTICITY)
    # =========================================================================
    target_firing_rate: float = 0.08  # Target firing rate for homeostatic plasticity
    gain_learning_rate: float = 0.0001  # Let conductance scales work first
    gain_tau_ms: float = 10000.0  # Very slow adaptation

    # =========================================================================
    # ADAPTVE THRESHOLD PLASTICITY (complementary to gain adaptation)
    # =========================================================================
    threshold_learning_rate: float = 0.03
    threshold_min: float = 0.05
    threshold_max: float = 1.5

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
    # GAP JUNCTIONS: TRN INTERNEURONS
    # =========================================================================
    gap_junction_strength: float = 0.10  # Moderate coupling for alpha synchronization (8-13 Hz)
    gap_junction_threshold: float = 0.3  # Neighborhood connectivity threshold
    gap_junction_max_neighbors: int = 8  # Max neighbors per interneuron (biological: 4-12)

    # =========================================================================
    # SPATIAL FILTERING: CENTER-SURROUND RECEPTIVE FIELDS
    # =========================================================================
    spatial_filter_width: float = 0.15
    """Gaussian filter width for center-surround (as fraction of input)."""

    center_excitation: float = 3.0
    """Center enhancement in receptive field."""

    surround_inhibition: float = 0.5
    """Surround suppression in receptive field."""

    # =========================================================================
    # TEMPORAL DYNAMICS & DELAYS
    # =========================================================================
    trn_recurrent_delay_ms: float = 8.0
    """TRN→TRN recurrent inhibition delay (prevents instant feedback).

    Biological: 2-3ms local inhibition
    Set to 8ms to stabilize dynamics (must be > refractory period of 5ms)

    Note: For alpha oscillations (8-12 Hz), the relay→TRN→relay loop needs ~80-120ms total.
    This delay contributes to the overall loop timing. Additional delays come from:
    - Relay → TRN: ~2-5ms (synaptic)
    - TRN → Relay: ~3-5ms (inhibitory synaptic)
    - Gap junction synchronization: <1ms (electrical)
    - Total loop: Gap junctions + recurrent inhibition create alpha rhythm
    """


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
    All decay factors, and delays derive from this value.

    Can be changed adaptively during simulation via brain.set_timestep(new_dt).
    Typical values:
    - 1.0ms: Standard biological timescale (Brian2, most research)
    - 0.1ms: High-resolution for detailed temporal dynamics
    - 10ms: Fast replay for memory consolidation
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
