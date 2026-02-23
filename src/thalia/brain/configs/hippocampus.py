"""Hippocampus Configurations"""

from __future__ import annotations

from dataclasses import dataclass

from .neural_region import NeuralRegionConfig


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
    tau_plus_ms: float = 20.0  # LTP time constant (ms)
    tau_minus_ms: float = 20.0  # LTD time constant (ms)
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
    # IONIC PACEMAKER CURRENTS
    # =========================================================================
    # Parameters for the biologically realistic oscillatory mechanism that
    # replaces the sinusoidal drive.  The theta rhythm (4-10 Hz) emerges from
    # the interaction of three slow ionic currents plus reciprocal connectivity.

    i_nap_conductance: float = 0.012
    """Tonic I_NaP (persistent Na⁺) drive conductance for ACh neurons.

    Provides the sub-threshold depolarising bias that initiates each ACh burst.
    Must be below the ACh threshold (~1.3) but sufficient to reach threshold
    when combined with recurrent excitation.  Modulated by NE (+40%) and ACh
    self-modulation (+30%) from neuromodulator broadcast.
    """

    i_h_conductance: float = 0.006
    """Peak I_h (HCN) rebound conductance for GABA neurons.

    Opens during GABA quiescence (time constant ~150ms) to provide a slow
    depolarising ramp that, combined with ACh→GABA excitation, triggers the
    GABA burst at the theta trough (≈180° offset from ACh burst).
    """

    i_ahp_tau_ms: float = 100.0
    """Time constant of the I_AHP (slow K⁺) current (ms).

    AHP decays exponentially between bursts.  With 8 Hz theta (125ms period)
    and this 100ms tau, roughly 71% of AHP dissipates between bursts, allowing
    the next cycle to fire.  Shortening to ~50ms gives faster theta; lengthening
    or increasing the increment gives slower / more adaptation.
    """

    i_ahp_increment: float = 0.008
    """Increment added to I_AHP conductance on each spike.

    Each spike adds this value to the per-neuron AHP conductance (g_inh channel).
    With tau_ahp=100ms, 2-3 ACh spikes accumulate ≈0.016-0.024 g_inh — enough
    to terminate the burst when combined with the GABA feedback inhibition.
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
