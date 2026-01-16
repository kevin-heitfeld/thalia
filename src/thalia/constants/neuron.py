"""
Neuron Constants - Membrane time constants, thresholds, refractory periods.

Consolidated from components/neurons/neuron_constants.py and utils/time_constants.py.

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

Author: Thalia Project
Date: January 16, 2026 (Architecture Review Tier 1.2 - Full Consolidation)
"""

import math

# =============================================================================
# TIME UNIT CONVERSIONS
# =============================================================================

MS_PER_SECOND = 1000.0
"""Milliseconds per second (1000.0 ms/s)."""

SECONDS_PER_MS = 1.0 / 1000.0
"""Seconds per millisecond (0.001 s/ms)."""

# =============================================================================
# MEMBRANE TIME CONSTANTS (milliseconds)
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

# Aliases for backward compatibility
TAU_MEMBRANE_MS = TAU_MEM_STANDARD
"""Alias for TAU_MEM_STANDARD."""

TAU_MEMBRANE_FAST_MS = TAU_MEM_FAST
"""Alias for TAU_MEM_FAST."""

TAU_MEMBRANE_SLOW_MS = TAU_MEM_SLOW
"""Alias for TAU_MEM_SLOW."""

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

# Aliases for backward compatibility and clarity
TAU_EXCITATORY_CONDUCTANCE = TAU_SYN_EXCITATORY
"""Excitatory conductance decay time constant (ms).

Alias for TAU_SYN_EXCITATORY. Used in conductance-based neuron models
to determine how quickly excitatory conductance decays after synaptic input.
"""

TAU_INHIBITORY_CONDUCTANCE = TAU_SYN_INHIBITORY
"""Inhibitory conductance decay time constant (ms).

Alias for TAU_SYN_INHIBITORY. Used in conductance-based neuron models
to determine how quickly inhibitory conductance decays after synaptic input.
"""

TAU_SYNAPTIC_FAST_MS = 2.0
"""Fast synaptic time constant for AMPA receptors (2ms).

Alternative value used in some configurations. Faster than TAU_SYN_EXCITATORY.
"""

TAU_SYNAPTIC_SLOW_MS = 10.0
"""Slow synaptic time constant for NMDA receptors (10ms).

Alternative value used in some configurations. Faster than TAU_SYN_NMDA.
"""

TAU_SYNAPTIC_GABA_MS = 5.0
"""GABAergic inhibitory synaptic time constant (5ms).

Alternative value used in some configurations. Faster than TAU_SYN_INHIBITORY.
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

# Aliases for backward compatibility
REFRACTORY_PERIOD_MS = 2.0
"""Absolute refractory period after spike (2ms).

Legacy value. Use TAU_REF_STANDARD for new code.
"""

REFRACTORY_PERIOD_RELATIVE_MS = 5.0
"""Relative refractory period (5ms total).

Legacy value indicating total refractory period including relative phase.
"""

# =============================================================================
# MEMBRANE CAPACITANCE (normalized units)
# =============================================================================

MEMBRANE_CAPACITANCE_STANDARD = 1.0
"""Standard membrane capacitance (normalized).

Determines relationship between conductance and time constant:
tau_mem = C_mem / g_leak

Normalized to 1.0 for simplicity. Biological value ~1 μF/cm².
"""

# Alias for clarity in code
C_MEM_STANDARD = MEMBRANE_CAPACITANCE_STANDARD

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

# Legacy aliases (millivolt versions for backward compatibility)
SPIKE_THRESHOLD_MV = -50.0
"""Spike threshold voltage (-50mV).

Legacy value in absolute millivolts. Use V_THRESHOLD_STANDARD for normalized models.
"""

RESTING_POTENTIAL_MV = -70.0
"""Resting membrane potential (-70mV).

Legacy value in absolute millivolts. Use V_REST_STANDARD for normalized models.
"""

RESET_POTENTIAL_MV = -65.0
"""Reset potential after spike (-65mV).

Legacy value in absolute millivolts. Use V_RESET_STANDARD for normalized models.
"""

# =============================================================================
# SPIKE DETECTION THRESHOLDS
# =============================================================================

SPIKE_DETECTION_THRESHOLD = 0.5
"""Binary spike detection threshold.

Used to convert continuous values to binary spikes.
Values > 0.5 are considered spikes (1), values <= 0.5 are not (0).
Common in temporal coding and rate-to-spike conversion.
"""

SPIKE_ACTIVITY_THRESHOLD = 0.5
"""Threshold for neuron activity detection.

Used in diagnostics to count "active" neurons (those that spiked).
A spike value > 0.5 indicates the neuron was active in that timestep.
"""

SPIKE_RATE_NORMALIZATION_FACTOR = 1000.0
"""Convert spike rate to Hz: rate_hz = spike_rate * 1000.0 / dt_ms."""

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

# Legacy aliases (nanosiemens versions)
CONDUCTANCE_LEAK_NS = 10.0
"""Leak conductance (10 nS).

Legacy value in absolute units. Use G_LEAK_STANDARD for normalized models.
"""

CONDUCTANCE_EXCITATORY_NS = 0.5
"""Excitatory synaptic conductance (0.5 nS).

Legacy value in absolute units. Use model-specific parameters for conductance-based models.
"""

CONDUCTANCE_INHIBITORY_NS = 1.0
"""Inhibitory synaptic conductance (1.0 nS).

Legacy value in absolute units. Use model-specific parameters for conductance-based models.
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
"""Maximum norepinephrine gain multiplier.

Used to scale responsiveness during high arousal states.
"""

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
# PHASE AND OSCILLATION CONSTANTS
# =============================================================================

TAU = 2.0 * math.pi
"""Full circle in radians (τ ≈ 6.283185307179586).

Tau (τ) represents one complete turn, making circle mathematics more intuitive.
"""

TWO_PI = TAU  # Alias for compatibility
"""Alias for TAU. Use TAU for new code."""

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

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Time unit conversions
    "MS_PER_SECOND",
    "SECONDS_PER_MS",
    # Membrane time constants
    "TAU_MEM_STANDARD",
    "TAU_MEM_FAST",
    "TAU_MEM_SLOW",
    "TAU_MEMBRANE_MS",
    "TAU_MEMBRANE_FAST_MS",
    "TAU_MEMBRANE_SLOW_MS",
    # Refractory periods
    "TAU_REF_STANDARD",
    "TAU_REF_FAST",
    "TAU_REF_SLOW",
    "REFRACTORY_PERIOD_MS",
    "REFRACTORY_PERIOD_RELATIVE_MS",
    # Synaptic time constants
    "TAU_SYN_EXCITATORY",
    "TAU_SYN_INHIBITORY",
    "TAU_SYN_NMDA",
    "TAU_SYN_GABA_B",
    "TAU_EXCITATORY_CONDUCTANCE",
    "TAU_INHIBITORY_CONDUCTANCE",
    "TAU_SYNAPTIC_FAST_MS",
    "TAU_SYNAPTIC_SLOW_MS",
    "TAU_SYNAPTIC_GABA_MS",
    # Voltage thresholds
    "V_THRESHOLD_STANDARD",
    "V_RESET_STANDARD",
    "V_REST_STANDARD",
    "SPIKE_THRESHOLD_MV",
    "RESTING_POTENTIAL_MV",
    "RESET_POTENTIAL_MV",
    # Spike detection
    "SPIKE_DETECTION_THRESHOLD",
    "SPIKE_ACTIVITY_THRESHOLD",
    "SPIKE_RATE_NORMALIZATION_FACTOR",
    # Reversal potentials
    "E_LEAK",
    "E_EXCITATORY",
    "E_INHIBITORY",
    # Conductances
    "G_LEAK_STANDARD",
    "G_LEAK_FAST",
    "G_LEAK_SLOW",
    "MEMBRANE_CAPACITANCE_STANDARD",
    "C_MEM_STANDARD",
    "CONDUCTANCE_LEAK_NS",
    "CONDUCTANCE_EXCITATORY_NS",
    "CONDUCTANCE_INHIBITORY_NS",
    # Adaptation parameters
    "TAU_ADAPT_STANDARD",
    "ADAPT_INCREMENT_NONE",
    "ADAPT_INCREMENT_MODERATE",
    "ADAPT_INCREMENT_STRONG",
    "ADAPT_INCREMENT_CORTEX_L23",
    # Noise parameters
    "NOISE_STD_NONE",
    "NOISE_STD_LOW",
    "NOISE_STD_MODERATE",
    # Weight initialization
    "WEIGHT_INIT_SCALE_SMALL",
    "WEIGHT_INIT_SCALE_MODERATE",
    "WEIGHT_INIT_SCALE_SPARSITY_DEFAULT",
    # Task parameters
    "SPIKE_PROBABILITY_LOW",
    "SPIKE_PROBABILITY_MEDIUM",
    "SPIKE_PROBABILITY_HIGH",
    "PROPRIOCEPTION_NOISE_SCALE",
    "DECISION_THRESHOLD_DEFAULT",
    "STIMULUS_STRENGTH_HIGH",
    # Neuromodulator gains
    "NE_MAX_GAIN",
    "TONIC_D1_GAIN_SCALE",
    # Theta modulation
    "THETA_BASELINE_MIN",
    "THETA_BASELINE_RANGE",
    "THETA_CONTRAST_MIN",
    "THETA_CONTRAST_RANGE",
    "BASELINE_EXCITATION_SCALE",
    # Learning thresholds
    "INTRINSIC_LEARNING_THRESHOLD",
    "MATCH_THRESHOLD",
    # Phase constants
    "TAU",
    "TWO_PI",
    # Convenience presets
    "STANDARD_PYRAMIDAL",
    "FAST_SPIKING_INTERNEURON",
    "SLOW_ADAPTING",
]
