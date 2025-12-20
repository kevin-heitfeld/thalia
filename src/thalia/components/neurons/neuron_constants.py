"""
Standard neuron parameter values used across Thalia.

This module defines biologically-motivated constants for neuron parameters,
eliminating magic numbers scattered throughout the codebase.

Biological Basis:
=================

Membrane Time Constants (tau_mem):
----------------------------------
- STANDARD (20ms): Typical pyramidal neurons in cortex
- FAST (10ms): Fast-spiking interneurons (parvalbumin+)
- SLOW (30ms): Slowly-adapting neurons, some hippocampal cells

Synaptic Time Constants:
------------------------
- AMPA (5ms): Fast excitatory transmission (glutamate)
- GABA_A (10ms): Fast inhibition
- NMDA (100ms): Slow excitatory, important for learning
- GABA_B (150ms): Slow inhibition, modulatory

Reversal Potentials (Conductance-Based Models):
-----------------------------------------------
Based on ion concentrations and Nernst equation:
- E_LEAK (0mV normalized): Reference potential
- E_EXCITATORY (+3mV normalized): Depolarizing (Na+/Ca2+ influx)
- E_INHIBITORY (-0.5mV normalized): Hyperpolarizing (Cl- influx)

References:
-----------
- Dayan & Abbott (2001): Theoretical Neuroscience, Chapter 5-6
- Gerstner et al. (2014): Neuronal Dynamics, Chapter 1
- Johnston & Wu (1994): Foundations of Cellular Neurophysiology

Usage:
======
    from thalia.core.neuron_constants import (
        TAU_MEM_STANDARD, V_THRESHOLD_STANDARD,
        E_EXCITATORY, E_INHIBITORY
    )

    config = LIFConfig(
        tau_mem=TAU_MEM_STANDARD,
        v_threshold=V_THRESHOLD_STANDARD,
    )

Author: Thalia Project
Date: December 11, 2025
"""

# =============================================================================
# MEMBRANE DYNAMICS TIME CONSTANTS (milliseconds)
# =============================================================================

TAU_MEM_STANDARD = 20.0
"""Standard membrane time constant (ms).

Typical value for pyramidal neurons in cortex and hippocampus.
Determines how quickly membrane potential decays toward rest.
Larger values = longer temporal integration window.
"""

TAU_MEM_FAST = 10.0
"""Fast membrane time constant (ms).

Used for fast-spiking interneurons (parvalbumin-positive cells).
Enables rapid, precise spike timing for inhibitory control.
"""

TAU_MEM_SLOW = 30.0
"""Slow membrane time constant (ms).

Used for slowly-adapting neurons, some CA1 pyramidal cells.
Provides longer temporal integration for pattern completion.
"""

# =============================================================================
# SYNAPTIC TIME CONSTANTS (milliseconds)
# =============================================================================

TAU_SYN_EXCITATORY = 5.0
"""Excitatory synaptic time constant (ms).

AMPA receptor kinetics. Fast excitatory transmission.
Typical rise time ~1ms, decay ~5ms.
"""

TAU_SYN_INHIBITORY = 10.0
"""Inhibitory synaptic time constant (ms).

GABA_A receptor kinetics. Fast inhibition.
Slightly slower than AMPA for temporal precision.
"""

TAU_SYN_NMDA = 100.0
"""NMDA receptor time constant (ms).

Slow excitatory transmission, critical for learning.
Voltage-dependent Mg2+ block enables coincidence detection.
"""

TAU_SYN_GABA_B = 150.0
"""GABA_B receptor time constant (ms).

Slow inhibition, modulatory effects.
G-protein coupled, affects excitability over hundreds of ms.
"""

# =============================================================================
# VOLTAGE THRESHOLDS AND POTENTIALS (normalized units)
# =============================================================================

V_THRESHOLD_STANDARD = 1.0
"""Standard spike threshold (normalized).

When membrane potential reaches this value, neuron fires a spike.
Normalized to 1.0 for numerical stability.
Biological equivalent: ~-50mV (vs resting ~-70mV).
"""

V_RESET_STANDARD = 0.0
"""Standard reset potential (normalized).

Membrane potential immediately after spike.
Normalized to 0.0 (matches resting potential in our convention).
Biological equivalent: ~-70mV.
"""

V_REST_STANDARD = 0.0
"""Standard resting membrane potential (normalized).

Equilibrium potential with no input.
Normalized to 0.0 as reference point.
Biological equivalent: ~-70mV for pyramidal neurons.
"""

# =============================================================================
# REVERSAL POTENTIALS (Conductance-Based Models, normalized)
# =============================================================================

E_LEAK = 0.0
"""Leak reversal potential (normalized).

Reversal potential for leak conductance.
Sets resting potential when no synaptic input present.
"""

E_EXCITATORY = 3.0
"""Excitatory reversal potential (normalized).

Reversal potential for glutamate receptors (AMPA/NMDA).
Depolarizing: drives membrane toward spike threshold.
Biological equivalent: ~0mV (Na+/K+ equilibrium).
"""

E_INHIBITORY = -0.5
"""Inhibitory reversal potential (normalized).

Reversal potential for GABA receptors.
Hyperpolarizing: drives membrane away from threshold.
Biological equivalent: ~-75mV (Cl- equilibrium).
"""

# =============================================================================
# CONDUCTANCES (normalized units)
# =============================================================================

G_LEAK_STANDARD = 0.05
"""Standard leak conductance (normalized).

Determines membrane time constant: tau_mem = C_mem / g_leak.
With standard value, tau_mem ≈ 20ms for typical membrane capacitance.
"""

G_LEAK_FAST = 0.10
"""Fast-spiking leak conductance (normalized).

Higher leak = shorter time constant (tau_mem ≈ 10ms).
Used for parvalbumin-positive interneurons.
"""

G_LEAK_SLOW = 0.033
"""Slow-adapting leak conductance (normalized).

Lower leak = longer time constant (tau_mem ≈ 30ms).
Used for some hippocampal and prefrontal neurons.
"""

# =============================================================================
# REFRACTORY PERIODS (milliseconds)
# =============================================================================

TAU_REF_STANDARD = 5.0
"""Standard absolute refractory period (ms).

Typical refractory period for cortical pyramidal neurons (3-7ms biological range).
Increased from 2ms to 5ms to prevent instant re-firing and allow proper
temporal integration. This ensures oscillations are driven by feedback loop
delays rather than just refractory period cycling.

Biological basis: Na+ channel inactivation + K+ afterhyperpolarization
"""

TAU_REF_FAST = 1.0
"""Fast refractory period (ms).

Used for fast-spiking interneurons.
Shorter refractory period enables higher firing rates.
"""

TAU_REF_SLOW = 7.0
"""Slow refractory period (ms).

Used for slowly-adapting neurons with strong adaptation.
Longer refractory period limits maximum firing rate and enforces
longer inter-spike intervals for proper temporal coding.
"""

# =============================================================================
# ADAPTATION PARAMETERS
# =============================================================================

TAU_ADAPT_STANDARD = 100.0
"""Standard adaptation time constant (ms).

Controls decay of spike-frequency adaptation current.
Enables neurons to reduce firing with sustained input.
"""

ADAPT_INCREMENT_NONE = 0.0
"""No adaptation (default).

Set adapt_increment to 0 to disable spike-frequency adaptation.
"""

ADAPT_INCREMENT_MODERATE = 0.05
"""Moderate adaptation increment.

After each spike, adaptation current increases by this amount.
Moderate adaptation for typical pyramidal neurons.
"""

ADAPT_INCREMENT_STRONG = 0.10
"""Strong adaptation increment.

Strong spike-frequency adaptation for CA3 pyramidal neurons.
Prevents runaway recurrent excitation in attractor networks.
"""

ADAPT_INCREMENT_CORTEX_L23 = 0.30
"""Very strong adaptation for cortical L2/3 pyramidal neurons.

Cortical L2/3 pyramidal neurons show particularly strong spike-frequency
adaptation, critical for:
1. Preventing frozen attractors in recurrent networks
2. Temporal decorrelation of neural responses
3. Maintaining response selectivity during sustained stimulation

This strong adaptation works synergistically with short-term depression (STD)
to enable pattern transitions and prevent network from getting "stuck" on
the same representation.

References:
- McCormick et al. (1985): Comparative electrophysiology of pyramidal cells
- Sanchez-Vives et al. (2000): Adaptation in neocortical neurons
"""

# =============================================================================
# NOISE PARAMETERS
# =============================================================================

NOISE_STD_NONE = 0.0
"""No membrane noise (deterministic).

Default: no stochastic noise in membrane dynamics.
"""

NOISE_STD_LOW = 0.01
"""Low membrane noise.

Small stochastic fluctuations, enables occasional spontaneous spikes.
"""

NOISE_STD_MODERATE = 0.05
"""Moderate membrane noise.

Realistic level of membrane noise from channel fluctuations.
"""

# =============================================================================
# WEIGHT INITIALIZATION SCALES
# =============================================================================

WEIGHT_INIT_SCALE_SMALL = 0.1
"""Small weight initialization scale.

Used for PFC modulation weights and other fine-tuning connections.
Prevents overwhelming initial connectivity.
"""

WEIGHT_INIT_SCALE_MODERATE = 0.2
"""Moderate weight initialization scale.

Standard scale for most weight initializations in regions.
Balances initial activity without saturation.
"""

WEIGHT_INIT_SCALE_SPARSITY_DEFAULT = 0.3
"""Default sparsity for sparse connectivity.

Typical sparse connectivity ratio (30% connected).
Balances computational efficiency with biological realism.
"""

# =============================================================================
# TASK & TRAINING PARAMETERS
# =============================================================================

SPIKE_PROBABILITY_LOW = 0.1
"""Low spike probability threshold.

Used for conservative motor output or exploration.
Typical for early learning phases.
"""

SPIKE_PROBABILITY_MEDIUM = 0.15
"""Medium spike probability threshold.

Balanced threshold for standard task performance.
"""

SPIKE_PROBABILITY_HIGH = 0.2
"""High spike probability threshold.

Used for confident motor output or high-activity phases.
"""

PROPRIOCEPTION_NOISE_SCALE = 0.05
"""Proprioceptive feedback noise scale.

Realistic sensory noise in position/velocity feedback.
Based on psychophysical measurements of proprioceptive accuracy.
"""

DECISION_THRESHOLD_DEFAULT = 0.5
"""Default decision threshold (normalized).

Midpoint threshold for binary decisions.
Standard value for balanced sensitivity.
"""

STIMULUS_STRENGTH_HIGH = 0.8
"""High stimulus strength.

Strong sensory input, near-maximal activation.
Used for salient or superthreshold stimuli.
"""

# =============================================================================
# NEUROMODULATOR GAIN PARAMETERS
# =============================================================================

NE_MAX_GAIN = 1.5

# ============================================================================
# DEPRECATED CONSTANTS - Use thalia.neuromodulation.constants instead
# ============================================================================

# NE_GAIN_RANGE has been removed. Use compute_ne_gain() from thalia.neuromodulation.constants
# Migration: Replace `1.0 + NE_GAIN_RANGE * ne_level` with `compute_ne_gain(ne_level)`

TONIC_D1_GAIN_SCALE = 0.5
"""Tonic dopamine modulation of D1 pathway gain.

Scales how much tonic DA increases D1 responsiveness.
D1 gain = 1.0 + tonic_da * TONIC_D1_GAIN_SCALE
Example: tonic_da=0.3, scale=0.5 → gain=1.15
"""

# =============================================================================
# THETA OSCILLATION MODULATION PARAMETERS
# =============================================================================

THETA_BASELINE_MIN = 0.7
"""Minimum theta baseline modulation factor.

Theta baseline modulation range: 0.7-1.0
Computed as: THETA_BASELINE_MIN + THETA_BASELINE_RANGE * encoding_phase
Where encoding_phase ∈ [0, 1] from theta oscillation.
"""

THETA_BASELINE_RANGE = 0.3
"""Theta baseline modulation range.

Full range of theta baseline modulation (0.3).
Combined with THETA_BASELINE_MIN (0.7) gives 0.7-1.0 range.
"""

THETA_CONTRAST_MIN = 0.8
"""Minimum theta contrast modulation factor.

Theta contrast modulation range: 0.8-1.0
Computed as: THETA_CONTRAST_MIN + THETA_CONTRAST_RANGE * retrieval_phase
Where retrieval_phase ∈ [0, 1] from theta oscillation.
"""

THETA_CONTRAST_RANGE = 0.2
"""Theta contrast modulation range.

Full range of theta contrast modulation (0.2).
Combined with THETA_CONTRAST_MIN (0.8) gives 0.8-1.0 range.
"""

BASELINE_EXCITATION_SCALE = 1.2
"""Baseline excitation scale factor.

Scales theta-modulated baseline to set excitation level.
baseline_exc = BASELINE_EXCITATION_SCALE * theta_baseline_mod
Results in 0.84-1.2 range when theta_baseline_mod ∈ [0.7, 1.0]
"""

# =============================================================================
# LEARNING THRESHOLD PARAMETERS
# =============================================================================

INTRINSIC_LEARNING_THRESHOLD = 0.3
"""Intrinsic reward threshold for learning.

Minimum absolute intrinsic reward to trigger learning updates.
Used to filter out noise and focus learning on significant events.
Biological basis: Requires sufficient novelty/surprise signal.
"""

MATCH_THRESHOLD = 0.5
"""Pattern match threshold.

Used for pattern completion and recognition tasks.
Values above this threshold indicate successful pattern match.
"""

# =============================================================================
# CONVENIENCE PRESETS
# =============================================================================

# Standard pyramidal neuron (cortex, hippocampus)
STANDARD_PYRAMIDAL = {
    "tau_mem": TAU_MEM_STANDARD,
    "v_rest": V_REST_STANDARD,
    "v_reset": V_RESET_STANDARD,
    "v_threshold": V_THRESHOLD_STANDARD,
    "tau_ref": TAU_REF_STANDARD,
    "g_leak": G_LEAK_STANDARD,
}

# Fast-spiking interneuron (parvalbumin+)
FAST_SPIKING_INTERNEURON = {
    "tau_mem": TAU_MEM_FAST,
    "v_rest": V_REST_STANDARD,
    "v_reset": V_RESET_STANDARD,
    "v_threshold": V_THRESHOLD_STANDARD,
    "tau_ref": TAU_REF_FAST,
    "g_leak": G_LEAK_FAST,
}

# Slowly-adapting neuron (some hippocampal cells)
SLOW_ADAPTING = {
    "tau_mem": TAU_MEM_SLOW,
    "v_rest": V_REST_STANDARD,
    "v_reset": V_RESET_STANDARD,
    "v_threshold": V_THRESHOLD_STANDARD,
    "tau_ref": TAU_REF_SLOW,
    "g_leak": G_LEAK_SLOW,
}
