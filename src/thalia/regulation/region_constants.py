"""
Region-specific parameter constants for specialized brain regions.

This module defines biologically-motivated constants for region-specific
parameters in thalamus, striatum, and other specialized brain regions,
eliminating magic numbers scattered throughout the codebase.

Biological Basis:
=================

Thalamus - Relay and Filtering:
-------------------------------
The thalamus acts as a gatekeeper and amplifier for sensory information:

**Mode Switching**:
- Burst mode: Occurs when thalamic neurons are hyperpolarized (-0.2 normalized)
  - Produces 2-5 spikes in rapid succession (~3 typical)
  - Acts as "wake-up call" to cortex, 2x amplification
  - Detected via T-type calcium channels (Crunelli & Hughes, 2010)
- Tonic mode: Occurs when neurons are depolarized (+0.3 normalized)
  - Single spikes, faithful relay of input
  - Allows precise temporal information transfer

**Attention Gating (Alpha Oscillations)**:
- Alpha rhythm (8-12 Hz) suppresses unattended sensory streams
- Pulvinar nucleus coordinates attention via alpha phase
- 50% suppression strength at trough (Sherman & Guillery, 2013)

**TRN (Thalamic Reticular Nucleus)**:
- Inhibitory shell around thalamus (20% of relay neurons)
- Provides feedback inhibition (30% strength)
- Recurrent inhibition (40% strength) generates spindle oscillations
- Gates attention and sensory selection (Halassa & Kastner, 2017)

**Spatial Filtering**:
- Center-surround receptive fields enhance contrast
- Center excitation: 1.5x amplification
- Surround inhibition: 0.5x suppression
- Gaussian filter width: 15% of input dimension

Striatum - Reinforcement Learning:
----------------------------------
The striatum implements reward-based learning through dopamine modulation:

**TD(λ) Multi-Step Credit Assignment**:
- λ (lambda): Temporal bridge parameter
  - 0.0: Only immediate rewards (TD(0))
  - 0.9: Bridge ~10 timesteps (standard value, Sutton & Barto 2018)
  - 0.95: Bridge ~20 timesteps (longer horizon)
  - 1.0: Full episode returns (Monte Carlo)

**Discount Factor (γ/gamma)**:
- Standard RL value: 0.99
- Values 100 steps ahead worth 37% of immediate
- Balances short-term and long-term rewards

**Eligibility Traces**:
- Minimum trace: 1e-6 (computational efficiency)
- Bridge between synaptic events (ms) and rewards (seconds)
- Yagishita et al. (2014): 1-2 second biological persistence

Weight Initialization Scales:
-----------------------------
Different network architectures require different initialization scales:

**Sparse Random Initialization**:
- Relay pathways: 30% sparsity, 0.3 scale (moderate connectivity)
- TRN inhibitory: 20% sparsity, 0.4 scale (strong, sparse inhibition)
- Recurrent networks: 30% sparsity, 0.2 scale (stability)
- Center-surround: 20% sparsity, 0.15 scale (spatial locality)

References:
-----------
- Crunelli & Hughes (2010): The slow (<1 Hz) rhythm of non-REM sleep
- Sherman & Guillery (2013): Exploring the Thalamus and Its Role in Cortical Function
- Halassa & Kastner (2017): Thalamic functions in distributed cognitive control
- Sutton & Barto (2018): Reinforcement Learning: An Introduction
- Yagishita et al. (2014): A critical time window for dopamine actions
- Schultz et al. (1997): A neural substrate of prediction and reward

Usage:
======
    from thalia.regulation.region_constants import (
        THALAMUS_BURST_THRESHOLD,
        THALAMUS_ALPHA_SUPPRESSION,
        STRIATUM_TD_LAMBDA,
        STRIATUM_GAMMA
    )

    relay_config = ThalamicRelayConfig(
        burst_threshold=THALAMUS_BURST_THRESHOLD,
        alpha_suppression_strength=THALAMUS_ALPHA_SUPPRESSION
    )

    td_config = TDLambdaConfig(
        lambda_=STRIATUM_TD_LAMBDA,
        gamma=STRIATUM_GAMMA
    )

Author: Thalia Project
Date: December 13, 2025
"""

# =============================================================================
# THALAMUS - MODE SWITCHING
# =============================================================================

THALAMUS_BURST_THRESHOLD = -0.2
"""Membrane potential threshold for burst mode (normalized).

When thalamic relay neurons are hyperpolarized below this value,
T-type calcium channels de-inactivate, enabling burst firing.
Biological equivalent: ~-65mV (vs resting -70mV).
"""

THALAMUS_TONIC_THRESHOLD = 0.3
"""Membrane potential threshold for tonic mode (normalized).

When neurons are depolarized above this value, they operate in
tonic relay mode with faithful spike-by-spike transmission.
Biological equivalent: ~-55mV.
"""

THALAMUS_BURST_SPIKE_COUNT = 3
"""Number of spikes in a thalamic burst (typical value).

Bursts typically contain 2-5 spikes. 3 is the most common.
Acts as "alerting signal" to cortex.
"""

THALAMUS_BURST_GAIN = 2.0
"""Amplification factor for burst mode.

Bursts are roughly 2x more effective at driving cortical neurons
than single spikes due to temporal summation.
"""

# =============================================================================
# THALAMUS - ATTENTION GATING (ALPHA OSCILLATIONS)
# =============================================================================

THALAMUS_ALPHA_SUPPRESSION = 0.5
"""Alpha oscillation suppression strength (0-1).

During alpha trough, unattended inputs are suppressed by 50%.
Implements attention-based gating of sensory information.
Based on pulvinar alpha modulation studies.
"""

THALAMUS_ALPHA_GATE_THRESHOLD = 0.0
"""Alpha phase threshold for suppression (normalized).

0 = trough (maximum suppression)
π = peak (minimum suppression)
"""

# =============================================================================
# THALAMUS - TRN (THALAMIC RETICULAR NUCLEUS)
# =============================================================================

THALAMUS_TRN_RATIO = 0.2
"""TRN neurons as fraction of relay neurons.

TRN forms an inhibitory shell around thalamus.
Approximately 20% the size of relay population.
"""

THALAMUS_TRN_INHIBITION = 0.3
"""Strength of TRN → relay inhibition.

TRN provides feedback inhibition to relay neurons,
gating sensory information flow.
"""

THALAMUS_TRN_RECURRENT = 0.4
"""TRN recurrent inhibition strength.

Recurrent connections within TRN generate spindle
oscillations (7-14 Hz) during sleep and drowsiness.
"""

# =============================================================================
# THALAMUS - SPATIAL FILTERING
# =============================================================================

THALAMUS_SPATIAL_FILTER_WIDTH = 0.15
"""Gaussian filter width for center-surround receptive fields.

Expressed as fraction of input dimension (15% = ~7 pixels for 50x50 input).
Implements contrast enhancement via lateral interactions.
"""

THALAMUS_CENTER_EXCITATION = 1.5
"""Center excitation strength in receptive field.

Center of receptive field amplifies signals by 1.5x.
"""

THALAMUS_SURROUND_INHIBITION = 0.5
"""Surround inhibition strength in receptive field.

Surround suppresses signals by 0.5x (50% reduction).
Creates contrast enhancement at edges and boundaries.
"""

# =============================================================================
# THALAMUS - RELAY PARAMETERS
# =============================================================================

THALAMUS_RELAY_STRENGTH = 1.2
"""Base relay gain for thalamic amplification.

Thalamus amplifies weak sensory inputs by 20% to ensure
cortical neurons can detect faint signals.
"""

# =============================================================================
# THALAMUS - NEUROMODULATION
# =============================================================================

THALAMUS_NE_GAIN_SCALE = 0.5
"""Norepinephrine gain modulation scale.

NE modulates thalamic gain: gain = 1.0 + 0.5 × NE
With NE ∈ [0,1], this gives gain ∈ [1.0, 1.5]
Implements arousal-dependent sensory amplification.
"""

# =============================================================================
# THALAMUS - MODE DETECTION
# =============================================================================

THALAMUS_MODE_THRESHOLD = 0.5
"""Threshold for burst/tonic mode detection.

Mode state: 0 = burst, 1 = tonic
Threshold at 0.5 discriminates between modes.
"""

# =============================================================================
# THALAMUS - WEIGHT INITIALIZATION SPARSITY
# =============================================================================

THALAMUS_RELAY_SPARSITY = 0.3
"""Sparsity for thalamus → cortex relay connections.

30% of potential connections are active.
"""

THALAMUS_RELAY_SCALE = 0.3
"""Weight scale for relay connections.

Moderate strength for stable relay.
"""

THALAMUS_TRN_FEEDBACK_SPARSITY = 0.2
"""Sparsity for cortex → TRN feedback connections.

20% sparsity for strong, sparse feedback control.
"""

THALAMUS_TRN_FEEDBACK_SCALE = 0.4
"""Weight scale for TRN feedback connections.

Strong feedback for effective gating.
"""

THALAMUS_TRN_FEEDFORWARD_SPARSITY = 0.3
"""Sparsity for TRN → relay feedforward connections.

30% sparsity for broad inhibitory influence.
"""

THALAMUS_SPATIAL_CENTER_SPARSITY = 0.2
"""Sparsity for center-surround spatial filters.

20% sparsity for spatially localized receptive fields.
"""

THALAMUS_SPATIAL_CENTER_SCALE = 0.15
"""Weight scale for center-surround connections.

Small scale for subtle spatial filtering effects.
"""

# =============================================================================
# STRIATUM - TD(λ) LEARNING PARAMETERS
# =============================================================================

STRIATUM_TD_LAMBDA = 0.9
"""TD(λ) trace decay parameter (0-1).

Controls temporal credit assignment window:
- 0.0: Only immediate rewards (TD(0))
- 0.9: Bridge ~10 timesteps (standard, Sutton & Barto 2018)
- 0.95: Bridge ~20 timesteps (longer horizon)
- 1.0: Full episode returns (Monte Carlo)

At 1ms timesteps, λ=0.9 bridges ~10ms of computation.
For longer-scale tasks, see TAU_ELIGIBILITY_LONG in learning_constants.py.
"""

STRIATUM_GAMMA = 0.99
"""Discount factor for future rewards (0-1).

Standard reinforcement learning value. Exponentially discounts
future rewards: reward 100 steps ahead is worth 0.99^100 ≈ 0.37
of immediate reward.

Balances immediate and delayed gratification.
"""

STRIATUM_TD_MIN_TRACE = 1e-6
"""Minimum eligibility trace value.

Below this threshold, traces are zeroed for computational efficiency.
Prevents accumulation of infinitesimal traces.
"""

# =============================================================================
# STRIATUM - LEARNING MODES
# =============================================================================

STRIATUM_TD_ACCUMULATING = True
"""Whether to use accumulating (True) or replacing (False) traces.

Accumulating traces: e(t) = γλe(t-1) + ∇V(t)
  - Traces accumulate over repeated state visits
  - Standard in RL theory

Replacing traces: e(t) = max(γλe(t-1), ∇V(t))
  - Most recent visit resets trace
  - Can be more efficient for certain tasks
"""

# =============================================================================
# STRIATUM - THRESHOLD FOR SPIKING DETECTION
# =============================================================================

STRIATUM_RELAY_THRESHOLD = 0.5
"""Threshold for converting continuous activations to spikes.

Used in relay pathways when thalamus outputs need to be
converted to binary spike trains. Values > 0.5 become spikes.
"""
