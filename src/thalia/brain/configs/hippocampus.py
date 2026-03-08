"""Hippocampus Configurations"""

from __future__ import annotations

from dataclasses import dataclass, field

from .neural_region import NeuralRegionConfig


@dataclass
class HippocampalLayerConfig:
    v_threshold: float
    adapt_increment: float
    tau_adapt: float
    total_inhib_fraction: float
    v_threshold_olm: float
    v_threshold_bistratified: float


def get_default_dg_layer_config() -> HippocampalLayerConfig:
    """Per-layer configuration for the Dentate Gyrus (pattern separation).

    All threshold and adaptation parameters that were previously scattered as
    magic literals in ``Hippocampus.__init__`` are collected here so they can
    be swept and inspected without editing region code.
    """
    return HippocampalLayerConfig(
        # Firing threshold for DG granule cells (normalised).
        # High value enforces sparse activity (<1 Hz).  History: raised 0.9→1.6→1.8
        # to drive population fraction toward the 2–5 % biological target.
        v_threshold=1.8,
        # Spike-frequency adaptation increment.  Strong to enforce sparsity (Ca²⁺-K⁺).
        adapt_increment=0.30,
        # Adaptation decay time constant (ms).  Slow to persist across pattern presentations.
        tau_adapt=120.0,
        # Fraction of DG pyramidal count allocated to inhibitory interneurons.
        total_inhib_fraction=0.20,
        # OLM cell firing threshold.  Tuned to ~0.3–0.8 Hz at sparse DG activity.
        v_threshold_olm=1.00,
        # Bistratified cell firing threshold.
        v_threshold_bistratified=1.00,
    )


def get_default_ca3_layer_config() -> HippocampalLayerConfig:
    """Per-layer configuration for CA3 (pattern completion / autoassociative memory)."""
    return HippocampalLayerConfig(
        # CA3 firing threshold (normalised).
        # Lowered 1.0→0.50: EC_II drive reaches V_inf ≈ 0.53 at biological input rates
        # with STP, so the threshold must be reachable from combined EC + DG input.
        v_threshold=0.50,
        # Spike-frequency adaptation increment.  Biological AHP range 0.20–0.30.
        adapt_increment=0.25,
        # Adaptation decay time constant (ms).
        tau_adapt=100.0,
        # Fraction of CA3 pyramidal count allocated to inhibitory interneurons.
        total_inhib_fraction=0.25,
        # OLM cell threshold.  Lower than DG because sparse CA3 firing (0.75–2 Hz)
        # produces V_inf ≈ 0.18–0.45; DG-level thresholds are unreachable.
        v_threshold_olm=0.35,
        # Bistratified cell threshold.
        v_threshold_bistratified=0.30,
    )


def get_default_ca2_layer_config() -> HippocampalLayerConfig:
    """Per-layer configuration for CA2 (social / temporal context memory)."""
    return HippocampalLayerConfig(
        # CA2 firing threshold.  Slightly above CA3 for selectivity; reduced from 1.6
        # which caused near-silence.
        v_threshold=1.1,
        # Moderate adaptation for temporal selectivity.
        adapt_increment=0.25,
        # Adaptation decay time constant (ms).
        tau_adapt=100.0,
        # Lighter inhibition than CA3; CA2 is a smaller, more tightly gated hub.
        total_inhib_fraction=0.15,
        # OLM cell threshold.
        v_threshold_olm=0.35,
        # Bistratified cell threshold.
        v_threshold_bistratified=0.30,
    )


def get_default_ca1_layer_config() -> HippocampalLayerConfig:
    """Per-layer configuration for CA1 (output / coincidence detection layer)."""
    return HippocampalLayerConfig(
        # CA1 firing threshold.  Lowered 0.50→0.30: EC_III V_inf ≈ 0.18 at STP-depleted
        # 11 Hz; threshold must be reachable from combined EC_III + CA3 Schaffer + PFC drive.
        v_threshold=0.30,
        # Moderate adaptation to prevent runaway output activity.
        adapt_increment=0.20,
        # Faster adapt decay than CA3/CA2 for responsive output-layer dynamics.
        tau_adapt=80.0,
        # Stronger inhibition supports theta modulation and coincidence gating.
        total_inhib_fraction=0.30,
        # OLM cell threshold.
        v_threshold_olm=0.35,
        # Bistratified cell threshold.
        v_threshold_bistratified=0.30,
    )


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

    ec_ii_tonic_drive: float = 0.005
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

    ec_iii_tonic_drive: float = 0.005
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

    Per-layer neuron and inhibitory-network parameters are grouped into four
    sub-configs (``dg``, ``ca3``, ``ca2``, ``ca1``) so individual thresholds
    and adaptation constants can be swept or inspected without touching region
    code.
    """

    # =========================================================================
    # PER-LAYER SUB-CONFIGS
    # =========================================================================
    dg:  HippocampalLayerConfig = field(default_factory=get_default_dg_layer_config)
    """Dentate Gyrus neuron and inhibitory-network parameters."""

    ca3: HippocampalLayerConfig = field(default_factory=get_default_ca3_layer_config)
    """CA3 neuron and inhibitory-network parameters."""

    ca2: HippocampalLayerConfig = field(default_factory=get_default_ca2_layer_config)
    """CA2 neuron and inhibitory-network parameters."""

    ca1: HippocampalLayerConfig = field(default_factory=get_default_ca1_layer_config)
    """CA1 neuron and inhibitory-network parameters."""

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
    ffi_strength: float = 0.05  # Halved (0.1→0.05): FFI was over-gating already-silent CA3/CA1
    ffi_tau: float = 5.0  # FFI decay time constant (ms)

    # =========================================================================
    # TONIC INHIBITION (Ambient GABA)
    # =========================================================================
    # Real hippocampus has extrasynaptic GABA_A receptors (α5 subunit) that
    # provide continuous background inhibition from GABA spillover.
    # This prevents runaway excitation even when phasic inhibition fails.
    tonic_inhibition: float = 0.0008  # Reduced (0.002→0.0008): was creating an inhibitory floor that blocked sub-threshold CA3/CA1

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
    nmda_threshold: float = 0.4
    nmda_steepness: float = 12.0  # Sharp discrimination above threshold
    ampa_ratio: float = 0.20  # Raised (0.05→0.20): with CA3 silent the NMDA gate is closed; AMPA enables CA1 to fire first

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

    # =========================================================================
    # SHARP-WAVE RIPPLE (SWR) REPLAY
    # =========================================================================
    # SWRs occur spontaneously during slow-wave sleep and quiet waking. They are
    # characterised by synchronous CA3 population bursts (>5% cells in 1-5ms) that
    # strongly drive CA1 via already-potentiated Schaffer collaterals, enabling
    # rapid offline sequence replay essential for systems memory consolidation.
    # References: Buzsaki 1989 (Neuroscience); Wilson & McNaughton 1994 (Science).
    ripple_threshold: float = 0.05
    """CA3 population firing rate (fraction) that triggers SWR onset.

    A burst involving >5% of CA3 cells within a single timestep (1 ms) is
    characteristic of a sharp-wave event.  Increase to 0.08 in denser networks.
    """

    ripple_duration_max_ms: float = 100.0
    """Maximum duration (ms) of the SWR replay window.

    Biologically SWRs last 50-150 ms.  The window is refreshed on each
    high-rate CA3 timestep, so longer sustained bursts extend beyond this.
    """

    ripple_boost_factor: float = 1.5
    """Multiplicative boost applied to the CA3→CA1 Schaffer collateral drive during
    a detected sharp-wave ripple.

    Range: 1.0 (disabled) – 2.5.  Default 1.5 produces a ~50% increase in CA1
    excitability, sufficient to reliably cross threshold on previously-encoded
    sequences without causing runaway activity.
    """

    ripple_replay_injection: float = 0.3
    """Fraction of the captured CA3 onset pattern re-injected as excitatory
    conductance during each timestep of the replay window.

    Provides a depolarising bias that keeps the CA3 attractor active across
    the SWR duration.  Range: 0.1 (subtle) – 0.5 (strong sustained replay).
    """


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
    """ACh adaptation time constant (ms).

    Critical: Must balance two requirements:
    1. Persist through 25ms burst duration to accumulate and terminate firing
    2. Decay during 100ms inter-burst interval to allow next burst

    With tau=80ms:
    - During burst (25ms): ~27% decay → adaptation accumulates strongly
    - Between bursts (100ms): ~71% decay → mostly resets for next burst
    - Ensures burst self-terminates after 2-4 spikes
    """

    ach_adaptation_increment: float = 0.40
    """ACh adaptation increment per spike."""

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
    """GABA adaptation increment per spike."""


@dataclass
class SubiculumConfig(NeuralRegionConfig):
    """Configuration for the subiculum — hippocampal output gateway.

    The subiculum is the principal output structure of the hippocampal formation,
    positioned between CA1 and the entorhinal cortex.  It receives the bulk of
    CA1 output (~75% of CA1 projections) and converts CA1 complex-spike bursts into
    regular-spiking output distributed to entorhinal cortex, PFC, BLA, and
    hypothalamus.

    Distinct from CA1:
    - Contains three physiological cell types (regular-spiking, burst-firing,
      weak-burst), collapsed here into one population with heterogeneous parameters.
    - Does NOT send back-projections to CA3 (unlike CA1).
    - Has strong place-cell properties with different spatial scale than CA1.
    - Generates theta-frequency bursts driven by CA1 Schaffer input.

    Key features:
    - Single PRINCIPAL population (excitatory pyramidal)
    - Implicit PV basket-cell inhibition via lateral inhibition coefficient
    - Receives CA1 input; outputs to EC_V, PFC L5, BLA PRINCIPAL

    References:
    - O'Mara et al. (2001): The subiculum: the heart of the hippocampal outflow
    - Witter et al. (2000): Anatomical organisation of the parahippocampal-
      hippocampal network
    - Aggleton (2012): Multiple anatomical systems embedded within the primate
      medial temporal lobe: implications for hippocampal function
    """

    tau_mem: float = 25.0
    """Subicular pyramidal membrane time constant (ms).

    Intermediate between CA1 (~20 ms) and EC_V (~30 ms), consistent with
    their role as buffered output neurons.
    """

    v_threshold: float = 1.1
    """Firing threshold (normalised).  Same as EC_II — subiculum is a relay
    and should be easily driven by strong CA1 input.
    """

    adapt_increment: float = 0.10
    """Spike-frequency adaptation increment (I_KCA).

    Mild adaptation — subicular cells show burst-then-regular firing;
    a small increment allows burst onset without full silencing.
    """

    tau_adapt: float = 150.0
    """Adaptation decay time constant (ms).

    Long time constant to sustain the regular-spiking mode across a
    theta cycle (~125 ms).
    """

    tonic_drive: float = 0.003
    """Background excitatory conductance (normalised).

    Keeps cells slightly depolarised below threshold; subiculum is tonically
    active during awake exploration at ~5-10 Hz.
    """

    lateral_inhibition_ratio: float = 0.70
    """Implicit lateral inhibition: fraction of excitatory drive fed back as GABA_A.

    Approximates PV basket-cell feedback (within ~1 ms at dt=1ms) to prevent
    runaway synchronous bursting across the entire population.  Biological
    range: 0.5–0.8 for hippocampal output structures.
    """
