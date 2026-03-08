"""Cortex configuration module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from thalia.brain.regions.population_names import CortexPopulation

from .neural_region import NeuralRegionConfig


@dataclass
class CorticalColumnConfig(NeuralRegionConfig):
    """Configuration for layered cortical microcircuit."""

    # =========================================================================
    # ADAPTIVE GAIN CONTROL (HOMEOSTATIC INTRINSIC PLASTICITY)
    # =========================================================================
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
    population_overrides: Dict[CortexPopulation, Dict[str, float]] = field(
        default_factory=lambda: {
            CortexPopulation.L23_PYR: {
                "tau_mem": 20.0,          # Moderate integration for association
                "v_threshold": 1.8,       # High threshold for selective integration
                "adapt_increment": 0.45,  # VERY STRONG adaptation (prevents runaway recurrence)
                "tau_adapt": 150.0,       # Medium-slow decay (100-200ms biological)
            },
            CortexPopulation.L4_PYR: {
                "tau_mem": 10.0,          # Fast integration for sensory input
                "v_threshold": 0.65,      # Lowered 1.1→0.65: thalamus at 30 Hz with STP eff=0.029
                                          # gives V_inf≈0.70; threshold must be below achievable V_inf.
                                          # adapt_increment=0.20, tau_adapt=80ms → ~3 Hz steady-state
                "adapt_increment": 0.20,  # Moderate adaptation to prevent overfiring from strong input
                "tau_adapt": 80.0,        # Fast decay (50-100ms biological)
            },
            CortexPopulation.L5_PYR: {
                "tau_mem": 30.0,          # Slow integration for output generation
                "v_threshold": 1.2,       # Moderate-high threshold for reliable output (not too easily driven)
                "adapt_increment": 0.20,  # Moderate adaptation to enable burst firing without runaway activity
                "tau_adapt": 120.0,       # Medium decay (80-150ms biological)
            },
            CortexPopulation.L6A_PYR: {
                "tau_mem": 15.0,          # Fast for TRN feedback (low gamma)
                "v_threshold": 1.4,       # Higher threshold to prevent excessive feedback excitation (calibrated for stable low gamma)
                "adapt_increment": 0.18,  # Moderate adaptation to support feedback dynamics without runaway activity
                "tau_adapt": 100.0,       # Medium-fast decay (80-120ms biological)
            },
            CortexPopulation.L6B_PYR: {
                "tau_mem": 25.0,          # Moderate for relay feedback (high gamma)
                "v_threshold": 1.1,       # Lower threshold to allow fast modulation of L4 (calibrated for stable high gamma)
                "adapt_increment": 0.22,  # Moderate adaptation to support relay dynamics without runaway activity
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
    # Subclasses (e.g. PrefrontalCortexConfig) can increase these to create stronger
    # WM attractors.
    l23_recurrent_connectivity: float = 0.25     # Sparse recurrence (standard cortex)
    l23_recurrent_weight_scale: float = 0.0008   # Weak weights (prevents runaway in sensory cortex)


@dataclass
class PrefrontalCortexConfig(CorticalColumnConfig):
    """Configuration specific to prefrontal cortex.

    PFC is the agranular frontal cortex specialised for executive control and
    working memory.  It is structurally a CorticalColumn with PFC-specific
    parameter overrides:

    * Dense mesocortical DA to L2/3 — D1-gated WM update pathway.
    * Dense, strong L2/3 recurrence — attractor dynamics for WM persistence.
    * Long tau_mem in L2/3 (200 ms) and L5 (150 ms) — temporal integration.
    * D1/D2 dopamine receptor subpopulations on L2/3 neurons.
    """

    # =========================================================================
    # PFC-SPECIFIC CORTICAL COLUMN OVERRIDES
    # =========================================================================
    # PFC is agranular: thin L4, thick L2/3 and L5.
    # Long tau_mem in L2/3 supports WM attractor persistence.
    # Weak adapt_increment in L2/3 allows sustained spiking (no rapid adaptation).

    population_overrides: Dict[CortexPopulation, Dict[str, float]] = field(
        default_factory=lambda: {
            CortexPopulation.L23_PYR: {
                "tau_mem": 200.0,          # Very long integration for WM persistence
                "v_threshold": 1.8,        # Same as default — selective firing
                "adapt_increment": 0.05,   # Very weak — allows sustained WM activity
                "tau_adapt": 200.0,        # Slow — WM timescale
            },
            CortexPopulation.L4_PYR: {
                "tau_mem": 10.0,           # Fast — same as default
                "v_threshold": 0.65,       # Low threshold — same as default
                "adapt_increment": 0.20,
                "tau_adapt": 80.0,
            },
            CortexPopulation.L5_PYR: {
                "tau_mem": 150.0,          # Long — planning / output integration
                "v_threshold": 1.2,
                "adapt_increment": 0.15,   # Weaker — supports sustained output
                "tau_adapt": 180.0,        # Slow — planning timescale
            },
            CortexPopulation.L6A_PYR: {
                "tau_mem": 15.0,
                "v_threshold": 1.4,
                "adapt_increment": 0.18,
                "tau_adapt": 100.0,
            },
            CortexPopulation.L6B_PYR: {
                "tau_mem": 25.0,
                "v_threshold": 1.1,
                "adapt_increment": 0.22,
                "tau_adapt": 100.0,
            },
        }
    )

    # Dense mesocortical DA to L2/3: WM gating via D1 receptors.
    # Standard cortex: 7.5%.  PFC L2/3: 30% (matches L5 primary innervation).
    # Biology: Goldman-Rakic et al. 1992; mesocortical DA densely innervates
    # deep L3 / L5 of dlPFC which routes through our L2/3 WM population.
    da_l23_fraction: float = 0.30

    # Dense L2/3 recurrence for WM attractor dynamics.
    # Standard cortex: connectivity=0.25, weight_scale=0.0008 (sparse, weak).
    # PFC: all-to-all (1.0) at 5× weight scale → robust WM attractors.
    l23_recurrent_connectivity: float = 1.0
    l23_recurrent_weight_scale: float = 0.004

    # =========================================================================
    # D1/D2 DOPAMINE RECEPTOR SUBTYPES
    # =========================================================================
    # PFC L2/3 neurons split into D1-dominant (excitatory DA response) and
    # D2-dominant (inhibitory DA response) populations.
    # D1 (~60%): Enhance WM maintenance and gating on high-DA bursts.
    # D2 (~40%): Suppress noise; protect WM on low/baseline DA.
    # Biology: Goldman-Rakic et al. 2000; Seamans & Yang 2004.
    d1_fraction: float = 0.6     # Fraction of L2/3 neurons that are D1-dominant
    d1_da_gain: float = 0.3      # DA gain for D1 neurons (excitatory)
    d2_da_gain: float = 0.3      # DA suppression for D2 neurons (inhibitory)

    # =========================================================================
    # HOMEOSTATIC INTRINSIC PLASTICITY (PFC overrides)
    # =========================================================================
    # PFC requires slower homeostasis to preserve WM patterns across delays.
    gain_learning_rate: float = 0.001    # Very slow (prevents WM collapse)
    gain_tau_ms: float = 5000.0          # 5 s averaging window
    threshold_learning_rate: float = 0.02
    threshold_min: float = 0.05          # Lower floor for under-firing regions
