"""Hippocampus Configurations"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from thalia.brain.gap_junctions import GapJunctionConfig
from thalia.brain.regions.population_names import EntorhinalCortexPopulation, HippocampusPopulation

from .neural_region import (
    HomeostaticGainConfig,
    HomeostaticThresholdConfig,
    NeuralPopulationConfig,
    NeuralRegionConfig,
)


@dataclass
class EntorhinalCortexPopulationConfig(NeuralPopulationConfig):
    tonic_drive: float


@dataclass
class HippocampalPopulationConfig(NeuralPopulationConfig):
    total_inhib_fraction: float
    v_threshold_olm: float
    v_threshold_bistratified: float


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
    """

    population_overrides: Dict[EntorhinalCortexPopulation, EntorhinalCortexPopulationConfig] = field(
        default_factory=lambda: {
            EntorhinalCortexPopulation.EC_II: EntorhinalCortexPopulationConfig(
                tau_mem_ms=20.0,
                v_threshold=1.1,
                v_reset=-0.08,
                adapt_increment=0.15,
                tau_adapt_ms=100.0,
                tonic_drive=0.005, # History: 0.005→45 Hz gamma; 0.002→2.81 Hz; 0.0025→3.5 Hz;
                                   # 0.0035→still sub-target. Raised back to 0.005: DG PV overexcitation
                                   # was the real bottleneck (fixed via halved E→I w_max), so EC_II
                                   # can now safely drive at theta (5-8 Hz) without downstream blowup.
                noise_std=0.08,
            ),
            EntorhinalCortexPopulation.EC_III: EntorhinalCortexPopulationConfig(
                tau_mem_ms=25.0,
                v_threshold=1.2,
                v_reset=-0.08,
                adapt_increment=0.20,
                tau_adapt_ms=80.0,
                tonic_drive=0.004,  # Raised 0.003→0.004: EC_III drives CA1 directly (temporoammonic path);
                                    # at 0.003 it fired ~3 Hz (target 3-15 Hz). 0.004 targets 5-10 Hz.
                noise_std=0.08,
            ),
            EntorhinalCortexPopulation.EC_V: EntorhinalCortexPopulationConfig(
                tau_mem_ms=30.0,
                v_threshold=1.15,
                v_reset=-0.08,
                adapt_increment=0.18,
                tau_adapt_ms=150.0,
                tonic_drive=0.004,  # Reduced 0.010→0.004: EC_V is a back-projection relay, should not drive autonomously at gamma.
                noise_std=0.08,
            ),
            EntorhinalCortexPopulation.EC_INHIBITORY: EntorhinalCortexPopulationConfig(
                tau_mem_ms=8.0,
                v_threshold=0.90,
                v_reset=0.0,
                adapt_increment=0.52,
                tau_adapt_ms=30.0,
                tonic_drive=0.0,  # Inhibitory neurons are driven entirely by network input, no tonic drive
                noise_std=0.08,
            ),
        }
    )


@dataclass
class HippocampusConfig(NeuralRegionConfig):
    """Configuration for hippocampus (trisynaptic circuit)."""

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
    tonic_inhibition: float = 0.0008

    # =========================================================================
    # GAP JUNCTIONS (Electrical Synapses)
    # =========================================================================
    # Gap junctions between CA1 interneurons (basket cells, bistratified cells)
    # provide fast electrical coupling for theta-gamma synchronization.
    # Critical for precise spike timing in episodic memory encoding/retrieval.

    # CA1 interneurons (~10-15% of CA1 population) have dense gap junction
    # networks that synchronize inhibition during theta-gamma nested oscillations.
    gap_junctions: GapJunctionConfig = field(default_factory=lambda: GapJunctionConfig(
        coupling_strength=0.01,  # Coupling strength (biological: 0.05-0.2)
        connectivity_threshold=0.25,  # Neighborhood connectivity threshold
        max_neighbors=8,  # Max neighbors per interneuron (biological: 4-12)
    ))

    # =========================================================================
    # HETEROSYNAPTIC PLASTICITY
    # =========================================================================
    # Synapses to inactive postsynaptic neurons weaken when nearby neurons
    # fire strongly. This prevents winner-take-all dynamics from freezing.
    heterosynaptic_ratio: float = 0.1  # LTD for inactive synapses

    # =========================================================================
    # HOMEOSTATIC INTRINSIC PLASTICITY
    # =========================================================================
    homeostatic_gain: HomeostaticGainConfig = field(default_factory=lambda: HomeostaticGainConfig(
        lr_per_ms=0.001,  # 10× increase: was 0.0001 (g_L_scale barely moved 0.1% in 5000 steps)
        tau_ms=1500.0,  # 1.5s averaging window (slower than cortex)
    ))

    # Adaptive threshold plasticity (complementary to gain adaptation)
    homeostatic_threshold: HomeostaticThresholdConfig = field(default_factory=lambda: HomeostaticThresholdConfig(
        lr_per_ms=0.001,  # Reduced 0.02→0.001: effective tau must be ≥ 1000 ms (biological minimum for homeostasis)
        threshold_min=0.05,  # Lower floor to allow more aggressive adaptation for under-firing
        threshold_max=3.0,
    ))
    homeostatic_target_rates: dict[str, float] = field(default_factory=lambda: {
        HippocampusPopulation.DG: 0.001,
        HippocampusPopulation.CA3: 0.002,
        HippocampusPopulation.CA2: 0.003,
        HippocampusPopulation.CA1: 0.003,
    })

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
    # INHIBITORY STDP (Vogels et al. 2011)
    # =========================================================================
    # Symmetric learning rule for I→E synapses (PV→Pyr, OLM→Pyr).
    istdp_learning_rate: float = 0.001
    istdp_alpha: float = 0.12  # Target-rate offset
    istdp_tau_ms: float = 20.0  # Trace time constant

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
    ca2_to_ca3_delay_ms: float = 2.0   # CA2→CA3 back-projection delay (modulatory)
    ca3_to_ca1_delay_ms: float = 3.0   # CA3→CA1 axonal delay (Schaffer collaterals)

    # =========================================================================
    # SHARP-WAVE RIPPLE (SWR) REPLAY
    # =========================================================================
    # SWRs occur spontaneously during slow-wave sleep and quiet waking. They are
    # characterised by synchronous CA3 population bursts (>5% cells in 1-5ms) that
    # strongly drive CA1 via already-potentiated Schaffer collaterals, enabling
    # rapid offline sequence replay essential for systems memory consolidation.
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

    # =========================================================================
    # PER-LAYER SUB-CONFIGS
    # =========================================================================
    population_overrides: Dict[HippocampusPopulation, HippocampalPopulationConfig] = field(
        default_factory=lambda: {
            HippocampusPopulation.DG: HippocampalPopulationConfig(
                tau_mem_ms=20.0,
                v_threshold=1.8,
                v_reset=-0.10,
                adapt_increment=0.50,
                tau_adapt_ms=120.0,
                noise_std=0.04,
                total_inhib_fraction=0.20,
                v_threshold_olm=1.00,
                v_threshold_bistratified=1.50,
            ),
            HippocampusPopulation.CA3: HippocampalPopulationConfig(
                tau_mem_ms=20.0,
                v_threshold=0.74,
                v_reset=-0.12,
                adapt_increment=0.35,
                tau_adapt_ms=100.0,
                noise_std=0.12,
                total_inhib_fraction=0.30,
                v_threshold_olm=0.06,           # Raised 0.01→0.06: CA3 now fires at 1.63 Hz (was 0.12 Hz); ultra-low threshold caused all OLM cells to co-activate synchronously
                v_threshold_bistratified=0.15,  # Raised 0.04→0.15: epileptiform bursting (6%) — threshold range [0.03,0.05] too narrow, all neurons fire together; 0.15 spreads thresholds to [0.11, 0.20]
            ),
            HippocampusPopulation.CA2: HippocampalPopulationConfig(
                tau_mem_ms=20.0,
                v_threshold=0.60,
                v_reset=-0.10,
                adapt_increment=0.20,
                tau_adapt_ms=100.0,
                noise_std=0.08,
                total_inhib_fraction=0.15,
                v_threshold_olm=0.35,
                v_threshold_bistratified=0.30,
            ),
            HippocampusPopulation.CA1: HippocampalPopulationConfig(
                tau_mem_ms=20.0,
                v_threshold=0.30,
                v_reset=-0.12,
                adapt_increment=0.35,
                tau_adapt_ms=80.0,
                noise_std=0.08,
                total_inhib_fraction=0.30,
                v_threshold_olm=0.08,           # Raised 0.05→0.08: with increased CA1 firing (2.06 Hz), OLM cells were over-driven
                v_threshold_bistratified=0.20,  # Raised 0.12→0.20: epileptiform bursting (7.3%) — threshold too low for current CA1 activity level; wider spread reduces synchrony
            ),
        }
    )


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

    i_nap_conductance: float = 0.010
    """Tonic I_NaP (persistent Na⁰) drive conductance for ACh neurons."""

    i_h_conductance: float = 0.006
    """Peak I_h (HCN) rebound conductance for GABA neurons."""

    i_ahp_tau_ms: float = 100.0
    """Time constant of the I_AHP (slow K⁺) current (ms)."""

    i_ahp_increment: float = 0.015
    """Increment added to I_AHP conductance on each spike."""

    # =========================================================================
    # CHOLINERGIC NEURON PROPERTIES
    # =========================================================================
    ach_tau_mem: float = 35.0
    """ACh neuron membrane time constant (ms). Moderate for responsive bursting."""

    ach_threshold: float = 1.3
    """ACh neuron threshold (slightly elevated to require stronger drive)."""

    ach_reset: float = 0.0
    """ACh neuron reset potential after spike."""

    ach_adaptation_tau: float = 80.0
    """ACh adaptation time constant (ms)."""

    ach_adaptation_increment: float = 0.40
    """ACh adaptation increment per spike."""

    # =========================================================================
    # GABAERGIC NEURON PROPERTIES
    # =========================================================================
    gaba_tau_mem: float = 30.0
    """GABA neuron membrane time constant (ms). Moderate for responsive theta bursting."""

    gaba_threshold: float = 0.85  # Lowered 1.2→1.0→0.85: MS:GABA at 4.38 Hz (target 5-15 Hz) after first reduction; further lowering needed
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
    """

    tau_mem_ms: float = 25.0
    """Subicular pyramidal membrane time constant (ms).

    Intermediate between CA1 (~20 ms) and EC_V (~30 ms), consistent with
    their role as buffered output neurons.
    """

    v_threshold: float = 1.1
    """Firing threshold (normalised).  Reverted 1.5→1.1: v_threshold=1.5
    killed PV firing (8.19→1.97 Hz) more than principal, worsening E/I to 52.2.
    """

    adapt_increment: float = 0.20
    """Spike-frequency adaptation increment (I_KCA)."""

    tau_adapt_ms: float = 150.0
    """Adaptation decay time constant (ms).

    Long time constant to sustain the regular-spiking mode across a
    theta cycle (~125 ms).
    """

    tonic_drive: float = 0.003
    """Background excitatory conductance (normalised).

    Keeps cells slightly depolarised below threshold; subiculum is tonically
    active during awake exploration at ~5-10 Hz.
    """
