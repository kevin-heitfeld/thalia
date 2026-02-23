"""Cortex configuration module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict

from .neural_region import NeuralRegionConfig


# TODO: Use CortexPopulation enum instead of string keys for layer overrides
# (requires refactor of region code)
class CortexLayer(StrEnum):
    """Cortical layers enumeration."""

    L23 = "l23"
    L4 = "l4"
    L5 = "l5"
    L6A = "l6a"
    L6B = "l6b"


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
    # SPIKE-TIMING DEPENDENT PLASTICITY (STDP)
    # =========================================================================
    # PFC uses dopamine-gated STDP; these values are passed to the eligibility
    # trace manager and modulated by DA concentration at the time of readout.
    tau_plus_ms: float = 20.0  # LTP time constant (ms)
    tau_minus_ms: float = 20.0  # LTD time constant (ms)
    a_plus: float = 0.01
    a_minus: float = 0.01

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
