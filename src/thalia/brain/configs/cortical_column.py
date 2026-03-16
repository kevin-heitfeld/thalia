"""Cortex configuration module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from thalia.brain.regions.population_names import CortexPopulation

from .neural_region import NeuralRegionConfig


@dataclass
class CorticalPopulationConfig:
    tau_mem_ms: float
    v_threshold: float
    v_reset: float
    adapt_increment: float
    tau_adapt: float


@dataclass
class CorticalColumnConfig(NeuralRegionConfig):
    """Configuration for layered cortical microcircuit."""

    # =========================================================================
    # HOMEOSTATIC INTRINSIC PLASTICITY
    # =========================================================================
    # Adaptive gain control to maintain target firing rates
    gain_learning_rate: float = 0.004
    gain_tau_ms: float = 2000.0

    # Adaptive threshold plasticity (complementary to gain adaptation)
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
    ffi_strength: float = 0.18
    # Stronger FFI makes cortex more stimulus-driven and less self-sustaining
    ffi_tau: float = 5.0  # FFI decay time constant (ms)

    # =========================================================================
    # GAP JUNCTIONS (L2/3 Interneuron Synchronization)
    # =========================================================================
    # Basket cells and chandelier cells in L2/3 have dense gap junction networks
    # Critical for cortical gamma oscillations (30-80 Hz) and precise spike timing
    # ~70-80% of cortical gap junctions are interneuron-interneuron
    gap_junction_strength: float = 0.25
    gap_junction_threshold: float = 0.20
    gap_junction_max_neighbors: int = 8

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
    # MESOCORTICAL DOPAMINE LAYER FRACTIONS
    # =========================================================================
    # VTA mesocortical DA innervation density varies across layers.
    # L5 receives the heaviest dopaminergic input (primary cortical target),
    # L2/3 receives moderate input, L4 receives sparse input.
    # Fractions are multiplied against the full DA concentration vector to obtain
    # per-layer effective DA concentrations in forward().
    # Biology: Goldman-Rakic et al. 1992; Williams & Goldman-Rakic 1993.
    da_l23_fraction: float = 0.075   # 7.5%  (25% of L5 primary; standard cortex)
    da_l4_fraction: float = 0.03     # 3%    (10% of L5 primary)
    da_l5_fraction: float = 0.30     # 30%   (primary mesocortical target)
    da_l6a_fraction: float = 0.135   # 13.5% (45% of L5)
    da_l6b_fraction: float = 0.135   # 13.5% (45% of L5)

    # =========================================================================
    # L2/3 RECURRENT CONNECTION PARAMETERS
    # =========================================================================
    # Connectivity and weight scale for the L2/3→L2/3 recurrent synapse.
    l23_recurrent_connectivity: float = 0.25     # Sparse recurrence (standard cortex)
    l23_recurrent_weight_scale: float = 0.0008   # Weak weights (prevents runaway in sensory cortex)

    # =========================================================================
    # POPULATION-SPECIFIC HETEROGENEITY
    # =========================================================================
    population_overrides: Dict[CortexPopulation, CorticalPopulationConfig] = field(
        default_factory=lambda: {
            CortexPopulation.L23_PYR: CorticalPopulationConfig(
                tau_mem_ms=20.0,       # Moderate integration for association
                v_threshold=2.5,       # Raised 2.0→2.5: PFC L23 from internal recurrence (target ≤3)
                v_reset=-0.15,         # AHP: 15% below E_L; enables visible SFA, prevents bursting
                adapt_increment=1.5,   # Raised 1.0→1.5: PFC L23 at 3.89 Hz, need more adaptation to suppress recurrence
                tau_adapt=150.0,       # Medium-slow decay (100-200ms biological)
            ),
            CortexPopulation.L4_PYR: CorticalPopulationConfig(
                tau_mem_ms=10.0,       # Fast integration for sensory input
                v_threshold=0.65,      # Lowered 1.1→0.65: thalamus at 30 Hz with STP eff=0.029
                                       # gives V_inf≈0.70; threshold must be below achievable V_inf.
                                       # adapt_increment=0.20, tau_adapt=80ms → ~3 Hz steady-state
                v_reset=-0.10,         # AHP: mild for faithful fast relay
                adapt_increment=0.20,  # Moderate adaptation to prevent overfiring from strong input
                tau_adapt=80.0,        # Fast decay (50-100ms biological)
            ),
            CortexPopulation.L5_PYR: CorticalPopulationConfig(
                tau_mem_ms=30.0,       # Slow integration for output generation
                v_threshold=1.2,       # Moderate-high threshold for reliable output (not too easily driven)
                v_reset=-0.12,         # AHP: moderate for output neurons
                adapt_increment=0.20,  # Moderate adaptation to enable burst firing without runaway activity
                tau_adapt=120.0,       # Medium decay (80-150ms biological)
            ),
            CortexPopulation.L6A_PYR: CorticalPopulationConfig(
                tau_mem_ms=15.0,       # Fast for TRN feedback (low gamma)
                v_threshold=1.4,       # Higher threshold to prevent excessive feedback excitation (calibrated for stable low gamma)
                v_reset=-0.10,         # AHP: mild for feedback neurons
                adapt_increment=0.25,  # Raised 0.18→0.25: at 2.9 Hz, g_adapt_ss=0.073 (1.45× g_L); L6A SFA=0.77-0.83 needs stronger adaptation
                tau_adapt=100.0,       # Medium-fast decay (80-120ms biological)
            ),
            CortexPopulation.L6B_PYR: CorticalPopulationConfig(
                tau_mem_ms=25.0,       # Moderate for relay feedback (high gamma)
                v_threshold=1.1,       # Lower threshold to allow fast modulation of L4 (calibrated for stable high gamma)
                v_reset=-0.10,         # AHP: mild for relay modulation
                adapt_increment=0.22,  # Moderate adaptation to support relay dynamics without runaway activity
                tau_adapt=100.0,       # Medium-fast decay (80-120ms biological)
            ),

            # =========================================================================
            # INHIBITORY INTERNEURON OVERRIDES
            # =========================================================================
            CortexPopulation.L23_INHIBITORY_PV:  CorticalPopulationConfig(tau_mem_ms= 5.0, v_threshold=0.75, v_reset=0.0, adapt_increment=0.10, tau_adapt=100.0),
            CortexPopulation.L23_INHIBITORY_SST: CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.80, v_reset=0.0, adapt_increment=0.10, tau_adapt= 90.0),
            CortexPopulation.L23_INHIBITORY_VIP: CorticalPopulationConfig(tau_mem_ms=10.0, v_threshold=0.95, v_reset=0.0, adapt_increment=0.12, tau_adapt= 70.0),
            CortexPopulation.L23_INHIBITORY_NGC: CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.80, v_reset=0.0, adapt_increment=0.30, tau_adapt=100.0),

            CortexPopulation.L4_INHIBITORY_PV:   CorticalPopulationConfig(tau_mem_ms= 5.0, v_threshold=0.75, v_reset=0.0, adapt_increment=0.10, tau_adapt=100.0),
            CortexPopulation.L4_INHIBITORY_SST:  CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.80, v_reset=0.0, adapt_increment=0.10, tau_adapt= 90.0),
            CortexPopulation.L4_INHIBITORY_VIP:  CorticalPopulationConfig(tau_mem_ms=10.0, v_threshold=0.95, v_reset=0.0, adapt_increment=0.12, tau_adapt= 70.0),
            CortexPopulation.L4_INHIBITORY_NGC:  CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.80, v_reset=0.0, adapt_increment=0.30, tau_adapt=100.0),

            CortexPopulation.L5_INHIBITORY_PV:   CorticalPopulationConfig(tau_mem_ms= 5.0, v_threshold=0.75, v_reset=0.0, adapt_increment=0.10, tau_adapt=100.0),
            CortexPopulation.L5_INHIBITORY_SST:  CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.80, v_reset=0.0, adapt_increment=0.10, tau_adapt= 90.0),
            CortexPopulation.L5_INHIBITORY_VIP:  CorticalPopulationConfig(tau_mem_ms=10.0, v_threshold=0.95, v_reset=0.0, adapt_increment=0.12, tau_adapt= 70.0),
            CortexPopulation.L5_INHIBITORY_NGC:  CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.80, v_reset=0.0, adapt_increment=0.30, tau_adapt=100.0),

            CortexPopulation.L6A_INHIBITORY_PV:  CorticalPopulationConfig(tau_mem_ms= 5.0, v_threshold=0.75, v_reset=0.0, adapt_increment=0.05, tau_adapt=100.0),
            CortexPopulation.L6A_INHIBITORY_SST: CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.80, v_reset=0.0, adapt_increment=0.10, tau_adapt= 90.0),
            CortexPopulation.L6A_INHIBITORY_VIP: CorticalPopulationConfig(tau_mem_ms=10.0, v_threshold=0.95, v_reset=0.0, adapt_increment=0.12, tau_adapt= 70.0),
            CortexPopulation.L6A_INHIBITORY_NGC: CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.80, v_reset=0.0, adapt_increment=0.30, tau_adapt=100.0),

            CortexPopulation.L6B_INHIBITORY_PV:  CorticalPopulationConfig(tau_mem_ms= 5.0, v_threshold=0.75, v_reset=0.0, adapt_increment=0.10, tau_adapt=100.0),
            CortexPopulation.L6B_INHIBITORY_SST: CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.80, v_reset=0.0, adapt_increment=0.10, tau_adapt= 90.0),
            CortexPopulation.L6B_INHIBITORY_VIP: CorticalPopulationConfig(tau_mem_ms=10.0, v_threshold=0.95, v_reset=0.0, adapt_increment=0.12, tau_adapt= 70.0),
            CortexPopulation.L6B_INHIBITORY_NGC: CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.80, v_reset=0.0, adapt_increment=0.30, tau_adapt=100.0),
        }
    )
