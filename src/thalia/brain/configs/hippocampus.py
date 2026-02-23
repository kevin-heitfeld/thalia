"""Hippocampus Configurations"""

from __future__ import annotations

from dataclasses import dataclass

from .neural_region import NeuralRegionConfig


@dataclass
class EntorhinalCortexConfig(NeuralRegionConfig):
    """Configuration for entorhinal cortex — hippocampal gateway.

    The entorhinal cortex (EC) is the primary interface between neocortex and
    the hippocampal formation.  It provides spatial and temporal context via two
    anatomically distinct pathways:

    **Perforant path (EC_II → DG, CA3)**
        Layer II stellate cells project strongly to dentate gyrus granule cells
        and less densely to CA3.  This pathway carries allocentric spatial
        information (grid cells) and drives pattern separation in DG.
        Synapses are initially facilitating, reflecting the high-frequency grid
        cell bursts during active exploration.

    **Temporoammonic path (EC_III → CA1)**
        Layer III pyramidal cells bypass the trisynaptic loop and project
        directly to CA1 distal dendrites.  This pathway is thought to carry
        a "current sensory context" signal that disambiguates retrieved memories
        from ongoing perception.  Synapses are initially depressing (novelty
        emphasis; strong initial drive, then adaptation).

    **Back-projection (CA1 → EC_V → cortex)**
        Layer V neurons integrate CA1 outputs and relay the compressed memory
        index back to association cortex, closing the hippocortico-neocortical
        loop essential for memory consolidation.

    Key features:
    - Grid-cell–like tiling (approximated by heterogeneous noise-driven LIF)
    - Layer-specific STP (facilitating perforant; depressing temporoammonic)
    - Direct CA1 readout pathway for memory-indexing output to cortex

    References:
    - Hafting et al. (2005): Grid cells in entorhinal cortex
    - Amaral & Witter (1989): Three-dimensional organisation of the hippocampal
      formation — a review of anatomical data
    - Brun et al. (2008): Impaired spatial representation in the dentate gyrus
      following entorhinal cortex lesions
    - Deshmukh & Knierim (2011): Representation of non-spatial and spatial
      information in the lateral entorhinal cortex
    """

    # =========================================================================
    # EC_II — PERFORANT PATH (Grid/Place cells → DG, CA3)
    # =========================================================================
    ec_ii_tau_mem_ms: float = 20.0
    """EC layer II stellate cell membrane time constant (ms).

    Stellate cells have relatively fast integration (~20 ms) consistent
    with their role as feedforward relay neurons.
    """

    ec_ii_threshold: float = 1.1
    """EC_II firing threshold (normalised).

    Slightly below CA3 threshold — EC drives hippocampus, not the other
    way around.
    """

    ec_ii_adapt_increment: float = 0.15
    """Spike-frequency adaptation increment for EC_II neurons.

    Moderate adaptation to prevent sustained high-rate responses; grid cells
    show burst-then-adapt firing at field crossing.
    """

    ec_ii_adapt_tau_ms: float = 100.0
    """EC_II adaptation decay time constant (ms)."""

    ec_ii_tonic_drive: float = 0.02
    """Background excitatory conductance to EC_II (maintains low tonic activity).

    Models the persistent depolarisation from layer I inputs and recurrent
    EC activity.  Value is normalised input conductance per neuron.
    """

    # =========================================================================
    # EC_III — TEMPOROAMMONIC PATH (Time cells → CA1)
    # =========================================================================
    ec_iii_tau_mem_ms: float = 25.0
    """EC layer III pyramidal cell membrane time constant (ms).

    Layer III cells are larger, slower integrators than layer II stellate
    cells.  Their longer time constant (~25 ms) supports the temporal coding
    required for 'time cell' activity along the theta cycle.
    """

    ec_iii_threshold: float = 1.2
    """EC_III firing threshold (normalised)."""

    ec_iii_adapt_increment: float = 0.20
    """EC_III adaptation increment.  Slightly stronger than EC_II to implement
    the depressing (novelty-emphasis) character of the temporoammonic synapse.
    """

    ec_iii_adapt_tau_ms: float = 80.0
    """EC_III adaptation decay time constant (ms)."""

    ec_iii_tonic_drive: float = 0.015
    """Background excitatory conductance to EC_III (lower than EC_II;
    time-cell activity is sparser).
    """

    # =========================================================================
    # EC_V — BACK-PROJECTION (Receives CA1 → outputs to cortex)
    # =========================================================================
    ec_v_tau_mem_ms: float = 30.0
    """EC layer V pyramidal cell membrane time constant (ms).

    Deeper-layer cells tend to be larger with slower integration, consistent
    with their role aggregating hippocampal outputs over several theta cycles.
    """

    ec_v_threshold: float = 1.15
    """EC_V firing threshold (normalised)."""

    ec_v_adapt_increment: float = 0.12
    """EC_V adaptation increment.  Mild — layer V cells maintain moderate
    sustained activity to drive cortical consolidation.
    """

    ec_v_adapt_tau_ms: float = 150.0
    """EC_V adaptation decay time constant (ms).

    Long adaptation time constant supports persistent activity over
    multiple theta cycles, acting as a short-term memory buffer.
    """

    ec_v_tonic_drive: float = 0.01
    """Background excitatory conductance to EC_V (very low — driven mainly
    by CA1 back-projection input rather than tonic drive).
    """


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
