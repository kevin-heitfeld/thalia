"""Cortex configuration module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from thalia.brain.gap_junctions import GapJunctionConfig
from thalia.brain.regions.population_names import CortexPopulation

from .neural_region import (
    HomeostaticGainConfig,
    HomeostaticThresholdConfig,
    NeuralPopulationConfig,
    NeuralRegionConfig,
)


@dataclass(frozen=True)
class LayerFractions:
    """Per-layer neuromodulator concentration scaling factors (L2/3, L4, L5, L6a, L6b).

    Each value is a multiplicative fraction applied to the global neuromodulator
    concentration to obtain the per-layer effective concentration.
    """

    l23: float
    l4: float
    l5: float
    l6a: float
    l6b: float

    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        """Return (l23, l4, l5, l6a, l6b) for passing to _process_neuromodulator."""
        return (self.l23, self.l4, self.l5, self.l6a, self.l6b)


def _default_da_fractions() -> LayerFractions:
    """Default mesocortical DA layer fractions (standard cortex).

    VTA mesocortical DA innervation density varies across layers.
    L5 receives the heaviest dopaminergic input (primary cortical target),
    L2/3 receives moderate input, L4 receives sparse input.
    Fractions relative to L5 primary:
      L2/3 = 25%, L4 = 10%, L5 = 100%, L6a = 45%, L6b = 45%.
    """
    return LayerFractions(
        l23=0.075,   # 7.5%  (25% of L5 primary; standard cortex)
        l4=0.03,     # 3%    (10% of L5 primary)
        l5=0.30,     # 30%   (primary mesocortical target)
        l6a=0.135,   # 13.5% (45% of L5)
        l6b=0.135,   # 13.5% (45% of L5)
    )


def _default_ne_fractions() -> LayerFractions:
    """Default LC NE layer fractions.

    LC NE innervation is densest in superficial layers, particularly L1
    (apical dendrite zone) and L2/3 (Morrison et al. 1982; Berridge &
    Waterhouse 2003).  NE α1 receptors are highest in superficial layers;
    deep layers receive moderate NE.
    """
    return LayerFractions(
        l23=1.30,    # Dense NE in superficial layers (apical tuft zone)
        l4=0.80,     # Moderate NE in granular layer
        l5=1.00,     # Baseline — moderate α1 density
        l6a=0.70,    # Lowest NE innervation in deep layers
        l6b=0.70,    # Lowest NE innervation in deep layers
    )


def _default_ach_fractions() -> LayerFractions:
    """Default NB ACh layer fractions.

    NB ACh fibers are densest in L1 (targeting NGC interneurons via nicotinic
    receptors) and L5 (muscarinic M1 on deep pyramidal neurons; Mesulam 2004;
    Zaborszky et al. 2018).  L4 receives moderate ACh (nicotinic modulation
    of thalamocortical terminals).  L6 has moderate ACh density.
    """
    return LayerFractions(
        l23=1.00,    # Moderate ACh — nicotinic on VIP interneurons
        l4=1.20,     # Dense nicotinic on thalamocortical terminals
        l5=1.40,     # Densest: muscarinic M1 on deep pyramidals
        l6a=0.80,    # Moderate ACh density
        l6b=0.80,    # Moderate ACh density
    )


@dataclass
class CorticalPopulationConfig(NeuralPopulationConfig):
    pass


@dataclass
class CorticalColumnConfig(NeuralRegionConfig):
    """Configuration for layered cortical microcircuit."""

    # =========================================================================
    # HOMEOSTATIC INTRINSIC PLASTICITY
    # =========================================================================
    homeostatic_gain: HomeostaticGainConfig = field(default_factory=lambda: HomeostaticGainConfig(
        lr_per_ms=0.001,
        tau_ms=2000.0,
    ))
    homeostatic_threshold: HomeostaticThresholdConfig = field(default_factory=lambda: HomeostaticThresholdConfig(
        lr_per_ms=0.001,
        threshold_min=0.2,
        threshold_max=1.5,
    ))
    homeostatic_target_rates: dict[str, float] = field(default_factory=lambda: {
        CortexPopulation.L23_PYR: 0.003,
        CortexPopulation.L4_PYR: 0.005,
        CortexPopulation.L5_PYR: 0.004,
        CortexPopulation.L6A_PYR: 0.004,
        CortexPopulation.L6B_PYR: 0.004,
        CortexPopulation.L1_NGC: 0.005,
    })

    # =========================================================================
    # FEEDFORWARD INHIBITION (FFI)
    # =========================================================================
    # FFI detects stimulus changes and transiently suppresses recurrent activity
    # This is how the cortex naturally "clears" old representations when new input arrives
    # Always enabled (fundamental cortical mechanism)
    ffi_threshold: float = 0.3  # Input change threshold to trigger FFI
    ffi_strength: float = 0.18
    # Stronger FFI makes cortex more stimulus-driven and less self-sustaining
    ffi_tau: float = 5.0  # FFI decay time constant (ms)

    # =========================================================================
    # GAP JUNCTIONS (L2/3 Interneuron Synchronization)
    # =========================================================================
    # Basket cells and chandelier cells in L2/3 have dense gap junction networks
    # Critical for cortical gamma oscillations (30-80 Hz) and precise spike timing
    # ~70-80% of cortical gap junctions are interneuron-interneuron
    gap_junctions: GapJunctionConfig = field(default_factory=lambda: GapJunctionConfig(
        coupling_strength=0.25,
        connectivity_threshold=0.20,
        max_neighbors=8,
    ))

    # =========================================================================
    # INHIBITORY INTERNEURON OVERRIDES
    # =========================================================================
    total_inhib_fraction: float = 0.25

    # =========================================================================
    # SPIKE-TIMING DEPENDENT PLASTICITY (STDP)
    # =========================================================================
    tau_plus_ms: float = 20.0  # LTP time constant (ms) — coincidence detection window
    tau_minus_ms: float = 20.0  # LTD time constant (ms)
    a_plus: float = 0.01
    a_minus: float = 0.012

    # =========================================================================
    # INHIBITORY STDP (Vogels et al. 2011)
    # =========================================================================
    # Symmetric learning rule for I→E synapses that homeostatically tunes
    # inhibition to balance excitation.  Applied to PV→Pyr and SST→Pyr.
    istdp_learning_rate: float = 0.001  # iSTDP learning rate (η)
    istdp_alpha: float = 0.12  # Target-rate offset (sets E/I balance set-point)
    istdp_tau_ms: float = 20.0  # Trace time constant (ms)

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

    # =========================================================================
    # NEUROMODULATOR LAYER FRACTIONS
    # =========================================================================
    da_fractions: LayerFractions = field(default_factory=_default_da_fractions)
    """Mesocortical DA innervation density per layer."""

    ne_fractions: LayerFractions = field(default_factory=_default_ne_fractions)
    """LC NE innervation density per layer."""

    ach_fractions: LayerFractions = field(default_factory=_default_ach_fractions)
    """NB ACh innervation density per layer."""

    # =========================================================================
    # L2/3 RECURRENT CONNECTION PARAMETERS
    # =========================================================================
    # Connectivity and weight scale for the L2/3→L2/3 recurrent synapse.
    l23_recurrent_connectivity: float = 0.15   # Reduced 0.25→0.15: lower connectivity reduces total recurrent
                                               # drive to L2/3; with 25% connectivity and noise_std=0.30,
                                               # L2/3 self-ignited before L4 received thalamic input.
    l23_recurrent_weight_scale: float = 0.002  # Reduced 0.003→0.002: further weakens L2/3 self-sustaining recurrence.
                                               # L2/3 must wait for L4 feedforward drive to activate.

    # =========================================================================
    # NMDA PLATEAU POTENTIALS (L2/3)
    # =========================================================================
    # Enable NMDA-dependent plateau potentials for L2/3 pyramidal dendrites.
    # When enabled, sustained apical NMDA input that relieves Mg²⁺ block triggers a
    # 100-300 ms plateau depolarization, supporting persistent activity for WM.
    l23_enable_nmda_plateau: bool = False

    # =========================================================================
    # POPULATION-SPECIFIC HETEROGENEITY
    # =========================================================================
    population_overrides: Dict[CortexPopulation, CorticalPopulationConfig] = field(
        default_factory=lambda: {
            CortexPopulation.L23_PYR: CorticalPopulationConfig(
                tau_mem_ms=20.0,       # Moderate integration for association
                v_threshold=2.0,       # Lowered 2.5→2.0: high threshold with weak recurrence produced clock-like CV=0.19; 2.0 enables fluctuation-driven firing
                v_reset=-0.15,         # AHP: 15% below E_L; enables visible SFA, prevents bursting
                adapt_increment=1.5,   # Strong per-spike adaptation to prevent runaway with stronger recurrence
                tau_adapt_ms=150.0,    # Medium-slow decay (100-200ms biological)
                noise_std=0.10,        # 0.30→0.18→0.12→0.06→0.10: 0.06 removed desynchronizing noise,
                                       # contributing to pathological cortical synchrony (ρ=0.75 L23).
                                       # 0.10 (5% of v_threshold=2.0) balances desynchronization with
                                       # letting L4 feedforward dominate.
            ),
            CortexPopulation.L4_PYR: CorticalPopulationConfig(
                tau_mem_ms=10.0,       # Fast integration for sensory input
                v_threshold=0.50,      # 1.1→0.65→0.55→0.45→0.50: 0.45 made L4 too excitable under
                                       # synchronized thalamic input, causing L4 ρ=0.74-0.90.
                                       # 0.50 still lower than 0.55 to support adequate L4 firing.
                v_reset=-0.10,         # AHP: mild for faithful fast relay
                adapt_increment=0.12,  # Reduced 0.50→0.12: compensate tau_adapt 80→500ms; g_adapt_ss ≈ 0.12×5.5×0.5 = 0.33 (~6.6× g_L)
                tau_adapt_ms=500.0,    # Slow AHP range: tau must exceed SFA quarter (1250ms) for visible SFA index
                noise_std=0.08,        # 12% of v_threshold=0.65; default is appropriate here
            ),
            CortexPopulation.L5_PYR: CorticalPopulationConfig(
                tau_mem_ms=30.0,       # Slow integration for output generation
                v_threshold=1.2,       # Moderate-high threshold for reliable output (not too easily driven)
                v_reset=-0.12,         # AHP: moderate for output neurons
                adapt_increment=0.15,  # Raised 0.12→0.15: stronger per-spike AHP for tonic rate suppression.
                tau_adapt_ms=1500.0,   # Raised 300→1500: adaptation must still be building at the SFA measurement
                                       # window start (t=2000ms). At tau=1500ms, 73% of SS at t=2s, 95% at t=4.5s
                                       # → visible SFA>1.3 (first-quarter mean ~79% SS, last-quarter ~95% SS).
                noise_std=0.12,        # 10% of v_threshold=1.2; promotes AI-state irregularity
            ),
            CortexPopulation.L6A_PYR: CorticalPopulationConfig(
                tau_mem_ms=15.0,       # Fast for TRN feedback (low gamma)
                v_threshold=1.4,       # Higher threshold to prevent excessive feedback excitation (calibrated for stable low gamma)
                v_reset=-0.10,         # AHP: mild for feedback neurons
                adapt_increment=0.25,  # Raised 0.18→0.25: at 2.9 Hz, g_adapt_ss=0.073 (1.45× g_L); L6A SFA=0.77-0.83 needs stronger adaptation
                tau_adapt_ms=100.0,    # Medium-fast decay (80-120ms biological)
                noise_std=0.12,        # ~8.5% of v_threshold=1.4
            ),
            CortexPopulation.L6B_PYR: CorticalPopulationConfig(
                tau_mem_ms=25.0,       # Moderate for relay feedback (high gamma)
                v_threshold=1.1,       # Lower threshold to allow fast modulation of L4 (calibrated for stable high gamma)
                v_reset=-0.10,         # AHP: mild for relay modulation
                adapt_increment=0.22,  # Moderate adaptation to support relay dynamics without runaway activity
                tau_adapt_ms=100.0,    # Medium-fast decay (80-120ms biological)
                noise_std=0.10,        # ~9% of v_threshold=1.1
            ),

            # =========================================================================
            # INHIBITORY INTERNEURON OVERRIDES
            # =========================================================================
            # L2/3 interneurons are more excitable and less adaptive than pyramidal neurons,
            # enabling rapid feedforward and feedback inhibition to control recurrent activity
            # and support oscillations. PV+ basket cells are the most excitable (fast-spiking),
            # followed by VIP+ and SST+ interneurons, with NGCs being the least excitable but
            # most adapting.
            CortexPopulation.L23_INHIBITORY_PV: CorticalPopulationConfig(
                tau_mem_ms=5.0,
                v_threshold=0.55,  # Lowered 0.75→0.55: PV/FSI have lowest rheobase in cortex.
                                   # With g_L=0.10, g_E_thresh drops from 0.033→0.022 (1.47× PYR).
                v_reset=0.0,
                adapt_increment=0.10,
                tau_adapt_ms=100.0,
                noise_std=0.08,
            ),
            CortexPopulation.L23_INHIBITORY_SST: CorticalPopulationConfig(
                tau_mem_ms=15.0,
                v_threshold=0.50,      # Lowered 0.65→0.50: SST still at 1.88 Hz (target 5-25 Hz); Pyr→SST
                                       # facilitating STP was providing ~10% efficacy at low Pyr rates.
                                       # Lower threshold lets even weak drive sustain SST tonic activity.
                v_reset=0.0,
                adapt_increment=0.10,
                tau_adapt_ms=90.0,
                noise_std=0.08,
            ),
            CortexPopulation.L23_INHIBITORY_VIP: CorticalPopulationConfig(
                tau_mem_ms=10.0,
                v_threshold=0.80,      # Lowered 1.15→0.80: SST→SST mutual inhibition now prevents the
                                       # SST synchrony cascade that drove VIP epileptiform at 0.95.
                                       # g_E_thresh drops from 0.031→0.018 (1.2× PYR), biologically reasonable.
                v_reset=0.0,
                adapt_increment=0.10,  # Lowered 0.20→0.10: moderate adaptation, VIP are irregular-spiking.
                tau_adapt_ms=70.0,
                noise_std=0.15,        # Keep raised noise for VIP-SST decorrelation.
            ),
            CortexPopulation.L23_INHIBITORY_NGC: CorticalPopulationConfig(
                tau_mem_ms=15.0,
                v_threshold=0.80,
                v_reset=0.0,
                adapt_increment=0.30,
                tau_adapt_ms=100.0,
                noise_std=0.08,
            ),

            # L4 interneurons are less excitable and more adaptive than L2/3 interneurons,
            # reflecting their role in modulating thalamic input rather than controlling recurrent
            # activity. PV+ interneurons are still the most excitable, but all L4 interneurons have
            # higher thresholds and stronger adaptation than their L2/3 counterparts.
            CortexPopulation.L4_INHIBITORY_PV: CorticalPopulationConfig(
                tau_mem_ms=5.0,
                v_threshold=0.55,  # Lowered 0.75→0.55: PV/FSI lowest rheobase (g_E_thresh 0.022, 1.47× PYR).
                v_reset=0.0,
                adapt_increment=0.10,
                tau_adapt_ms=100.0,
                noise_std=0.08,
            ),
            CortexPopulation.L4_INHIBITORY_SST: CorticalPopulationConfig(
                tau_mem_ms=15.0,
                v_threshold=0.75,  # Partial rollback 0.60→0.75 (original 0.80): L4 SST was in-range (3-6 Hz);
                                   # lowering to 0.60 caused epileptiform bursting across all cortices/layers.
                                   # 0.75 gives slight excitability boost from Pyr→SST U=0.25 without bursting.
                v_reset=0.0,
                adapt_increment=0.10,
                tau_adapt_ms=90.0,
                noise_std=0.08,
            ),
            CortexPopulation.L4_INHIBITORY_VIP: CorticalPopulationConfig(
                tau_mem_ms=10.0,
                v_threshold=0.80,  # Lowered 1.15→0.80: SST→SST fix prevents synchrony cascade.
                v_reset=0.0,
                adapt_increment=0.10,  # Lowered 0.20→0.10: moderate VIP adaptation.
                tau_adapt_ms=70.0,
                noise_std=0.15,
            ),
            CortexPopulation.L4_INHIBITORY_NGC: CorticalPopulationConfig(
                tau_mem_ms=15.0,
                v_threshold=0.80,
                v_reset=0.0,
                adapt_increment=0.30,
                tau_adapt_ms=100.0,
                noise_std=0.08,
            ),

            # L5 interneurons have similar properties to L2/3 interneurons, but with slightly stronger
            # adaptation to help control the more excitable L5 pyramidal neurons. PV+ interneurons are
            # still the most excitable, followed by VIP+ and SST+ interneurons, with NGCs being the least
            # excitable but most adapting.
            CortexPopulation.L5_INHIBITORY_PV: CorticalPopulationConfig(
                tau_mem_ms=5.0,
                v_threshold=0.55,  # Lowered 0.75→0.55: PV/FSI lowest rheobase (g_E_thresh 0.022, 1.47× PYR).
                v_reset=0.0,
                adapt_increment=0.10,
                tau_adapt_ms=100.0,
                noise_std=0.08,
            ),
            CortexPopulation.L5_INHIBITORY_SST: CorticalPopulationConfig(
                tau_mem_ms=15.0,
                v_threshold=0.75,  # Partial rollback 0.60→0.75 (original 0.80): was in-range, don't over-drive.
                v_reset=0.0,
                adapt_increment=0.10,
                tau_adapt_ms=90.0,
                noise_std=0.08,
            ),
            CortexPopulation.L5_INHIBITORY_VIP: CorticalPopulationConfig(
                tau_mem_ms=10.0,
                v_threshold=0.80,  # Lowered 1.15→0.80: SST→SST fix prevents synchrony cascade.
                v_reset=0.0,
                adapt_increment=0.10,  # Lowered 0.20→0.10.
                tau_adapt_ms=70.0,
                noise_std=0.15,
            ),
            CortexPopulation.L5_INHIBITORY_NGC: CorticalPopulationConfig(
                tau_mem_ms=15.0,
                v_threshold=0.80,
                v_reset=0.0,
                adapt_increment=0.30,
                tau_adapt_ms=100.0,
                noise_std=0.08,
            ),

            # L6 interneurons are less well-characterized, but likely have similar properties to L5 interneurons,
            # with slightly stronger adaptation to control the feedback projections from L6 pyramidal neurons.
            # PV+ interneurons are still the most excitable, followed by VIP+ and SST+ interneurons, with NGCs
            # being the least excitable but most adapting.
            CortexPopulation.L6A_INHIBITORY_PV: CorticalPopulationConfig(
                tau_mem_ms=5.0,
                v_threshold=0.55,  # Lowered 0.75→0.55: PV/FSI lowest rheobase.
                v_reset=0.0,
                adapt_increment=0.05,
                tau_adapt_ms=100.0,
                noise_std=0.08,
            ),
            CortexPopulation.L6A_INHIBITORY_SST: CorticalPopulationConfig(
                tau_mem_ms=15.0,
                v_threshold=0.75,  # Partial rollback 0.60→0.75 (original 0.80): was in-range, don't over-drive.
                v_reset=0.0,
                adapt_increment=0.10,
                tau_adapt_ms=90.0,
                noise_std=0.08,
            ),
            CortexPopulation.L6A_INHIBITORY_VIP: CorticalPopulationConfig(
                tau_mem_ms=10.0,
                v_threshold=0.80,  # Lowered 1.15→0.80: SST→SST fix prevents synchrony cascade.
                v_reset=0.0,
                adapt_increment=0.10,  # Lowered 0.20→0.10.
                tau_adapt_ms=70.0,
                noise_std=0.15,
            ),
            CortexPopulation.L6A_INHIBITORY_NGC: CorticalPopulationConfig(
                tau_mem_ms=15.0,
                v_threshold=0.80,
                v_reset=0.0,
                adapt_increment=0.30,
                tau_adapt_ms=100.0,
                noise_std=0.08,
            ),

            # L6 interneurons are less well-characterized, but likely have similar properties to L5 interneurons,
            # with slightly stronger adaptation to control the feedback projections from L6 pyramidal neurons.
            CortexPopulation.L6B_INHIBITORY_PV: CorticalPopulationConfig(
                tau_mem_ms=5.0,
                v_threshold=0.55,  # Lowered 0.75→0.55: PV/FSI lowest rheobase.
                v_reset=0.0,
                adapt_increment=0.10,
                tau_adapt_ms=100.0,
                noise_std=0.08,
            ),
            CortexPopulation.L6B_INHIBITORY_SST: CorticalPopulationConfig(
                tau_mem_ms=15.0,
                v_threshold=0.75,  # Partial rollback 0.60→0.75 (original 0.80): was in-range, don't over-drive.
                v_reset=0.0,
                adapt_increment=0.10,
                tau_adapt_ms=90.0,
                noise_std=0.08,
            ),
            CortexPopulation.L6B_INHIBITORY_VIP: CorticalPopulationConfig(
                tau_mem_ms=10.0,
                v_threshold=0.80,  # Lowered 1.15→0.80: SST→SST fix prevents synchrony cascade.
                v_reset=0.0,
                adapt_increment=0.10,  # Lowered 0.20→0.10.
                tau_adapt_ms=70.0,
                noise_std=0.15,
            ),
            CortexPopulation.L6B_INHIBITORY_NGC: CorticalPopulationConfig(
                tau_mem_ms=15.0,
                v_threshold=0.80,
                v_reset=0.0,
                adapt_increment=0.30,
                tau_adapt_ms=100.0,
                noise_std=0.08,
            ),
        }
    )
