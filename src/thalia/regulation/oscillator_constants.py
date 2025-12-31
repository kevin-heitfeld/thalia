"""
Oscillator-related constants for theta-gamma-alpha coupling.

This module defines standard constants for oscillator phase-based modulation,
ensuring biological consistency and eliminating magic numbers scattered
throughout the codebase.

Biological Basis:
=================

Theta Rhythm (4-12 Hz):
-----------------------
- Peak (phase=0): Encoding mode, DG→CA3 pattern separation dominant
- Trough (phase=π): Retrieval mode, EC→CA1 pattern completion dominant
- Reference: Hasselmo et al. (2002), Buzsáki & Draguhn (2004)

Gamma Rhythm (30-80 Hz):
-------------------------
- Phase-amplitude coupled with theta
- Attention and binding mechanism
- Reference: Fries (2009), Buzsáki & Wang (2012)

Alpha Rhythm (8-13 Hz):
------------------------
- Feedback and top-down modulation
- Inhibitory control mechanism
- Reference: Jensen & Mazaheri (2010)

Author: Thalia Project
Date: December 21, 2025 (Architecture Review Tier 1.1)
"""

from __future__ import annotations

import math

# =============================================================================
# Theta-Phase Modulation (Encoding/Retrieval)
# =============================================================================

THETA_ENCODING_PHASE_SCALE: float = 0.5
"""Scale factor for theta-phase encoding modulation.

encoding_mod = THETA_ENCODING_PHASE_SCALE * (1.0 + cos(theta_phase))
Result range: [0.0, 1.0], peaks at theta peak (phase=0).
"""

THETA_RETRIEVAL_PHASE_SCALE: float = 0.5
"""Scale factor for theta-phase retrieval modulation.

retrieval_mod = THETA_RETRIEVAL_PHASE_SCALE * (1.0 - cos(theta_phase))
Result range: [0.0, 1.0], peaks at theta trough (phase=π).
"""

# =============================================================================
# Hippocampal Pathway Gating (DG→CA3, EC→CA3, CA3 recurrence)
# =============================================================================

DG_CA3_GATE_MIN: float = 0.1
"""Minimum DG→CA3 pathway strength during retrieval.

Biological basis: DG→CA3 pattern separation weakened during retrieval
to allow EC→CA3 pattern completion to dominate.
"""

DG_CA3_GATE_RANGE: float = 0.9
"""Range of DG→CA3 pathway modulation (min to max).

Full strength: DG_CA3_GATE_MIN + DG_CA3_GATE_RANGE * encoding_mod
Result range: [0.1 (retrieval), 1.0 (encoding)]
"""

EC_CA3_GATE_MIN: float = 0.3
"""Minimum EC→CA3 pathway strength during encoding.

Biological basis: EC→CA3 pattern completion weakened during encoding
to allow DG→CA3 pattern separation to dominate.
"""

EC_CA3_GATE_RANGE: float = 0.7
"""Range of EC→CA3 pathway modulation (min to max).

Full strength: EC_CA3_GATE_MIN + EC_CA3_GATE_RANGE * retrieval_mod
Result range: [0.3 (encoding), 1.0 (retrieval)]
"""

CA3_RECURRENT_GATE_MIN: float = 0.2
"""Minimum CA3 recurrent pathway strength during encoding.

Biological basis: CA3 recurrence weakened during encoding to prioritize
new pattern storage over pattern completion.
"""

CA3_RECURRENT_GATE_RANGE: float = 0.8
"""Range of CA3 recurrent pathway modulation (min to max).

Full strength: CA3_RECURRENT_GATE_MIN + CA3_RECURRENT_GATE_RANGE * retrieval_mod
Result range: [0.2 (encoding), 1.0 (retrieval)]
"""

# =============================================================================
# CA3→CA1 and EC→CA1 Gating
# =============================================================================

CA3_CA1_ENCODING_SCALE: float = 0.5
"""Scale factor for CA3→CA1 encoding contribution.

CA3 contribution scaled by (0.5 + 0.5 * encoding_mod) during encoding.
"""

CA1_SPARSITY_RETRIEVAL_BOOST: float = 0.5
"""Sparsity threshold increase during retrieval.

Biological basis: Higher firing threshold during retrieval to maintain
sparse coding despite increased EC input.
"""

# =============================================================================
# Cortical Gating (L4, L2/3, L5)
# =============================================================================

L4_INPUT_ENCODING_SCALE: float = 0.5
"""Scale factor for L4 input gain during encoding.

L4 input modulated by (0.5 + 0.5 * encoding_mod).
Range: [0.5 (retrieval), 1.0 (encoding)]
"""

L23_RECURRENT_RETRIEVAL_SCALE: float = 0.5
"""Scale factor for L2/3 recurrent connections during retrieval.

Recurrent strength: (0.5 + 0.5 * retrieval_mod)
Range: [0.5 (encoding), 1.0 (retrieval)]
"""

# =============================================================================
# Prefrontal Gating (Feedforward/Recurrent)
# =============================================================================

PFC_FEEDFORWARD_BASE_GAIN: float = 0.5
"""Base feedforward gain in prefrontal cortex."""

PFC_FEEDFORWARD_MODULATION_RANGE: float = 0.5
"""Modulation range for feedforward connections.

Full gain: PFC_FEEDFORWARD_BASE_GAIN + PFC_FEEDFORWARD_MODULATION_RANGE * encoding_mod
Range: [0.5, 1.0]
"""

PFC_RECURRENT_BASE_GAIN: float = 0.5
"""Base recurrent gain in prefrontal cortex."""

PFC_RECURRENT_MODULATION_RANGE: float = 0.5
"""Modulation range for recurrent connections.

Full gain: PFC_RECURRENT_BASE_GAIN + PFC_RECURRENT_MODULATION_RANGE * retrieval_mod
Range: [0.5, 1.0]
"""

# =============================================================================
# Cerebellum Input Gating
# =============================================================================

CEREBELLUM_INPUT_BASE_GAIN: float = 0.7
"""Base input gain for cerebellar granule cells."""

CEREBELLUM_INPUT_MODULATION_RANGE: float = 0.3
"""Modulation range for cerebellar input.

Full gain: CEREBELLUM_INPUT_BASE_GAIN + CEREBELLUM_INPUT_MODULATION_RANGE * encoding_mod
Range: [0.7, 1.0]
"""

# =============================================================================
# Neuromodulator-Oscillator Interactions
# =============================================================================

ACH_RECURRENT_SUPPRESSION: float = 0.7
"""Maximum suppression of recurrent connections by high acetylcholine.

High ACh (>0.5) suppresses recurrence by up to 70%, favoring
afferent over recurrent input during encoding.

Reference: Hasselmo & McGaughy (2004)
"""

ACH_THRESHOLD_FOR_SUPPRESSION: float = 0.5
"""ACh level threshold above which recurrent suppression begins.

Below this threshold: No suppression (full recurrence)
Above this threshold: Linear suppression up to ACH_RECURRENT_SUPPRESSION
"""

ACH_ENCODING_BOOST_BASE: float = 0.5
"""Base encoding gain when ACh is low."""

ACH_ENCODING_BOOST_RANGE: float = 0.5
"""Additional encoding gain at high ACh levels."""

# =============================================================================
# Gamma-Phase Gating (Attention)
# =============================================================================

GAMMA_ATTENTION_THRESHOLD: float = 0.5
"""Threshold for gamma-phase attention gating.

Neurons only process input during specific gamma phases (attention windows).
"""

GAMMA_LEARNING_MODULATION_SCALE: float = 0.5
"""Scale factor for gamma-phase learning rate modulation.

effective_lr = base_lr * (SCALE + SCALE * gamma_mod)
Range: [50%, 100%] of base learning rate

Also used for gamma-phase input gain modulation:
effective_input = input * (SCALE + SCALE * gamma_amplitude)
Range: [50%, 100%] of base input
"""

# =============================================================================
# Theta-Gamma Coupling
# =============================================================================

THETA_GAMMA_PHASE_DIFF_SIGMA: float = 0.5
"""Sigma (standard deviation) for Gaussian phase-difference gating.

gating = exp(-(phase_diff)^2 / (2 * sigma^2))

Used for theta-gamma cross-frequency coupling in hippocampus.
"""

# =============================================================================
# Striatal Learning Modulation
# =============================================================================

STRIATUM_PFC_MODULATION_LR: float = 0.001
"""Learning rate for prefrontal modulation of striatal pathways.

Used for learning PFC→D1 and PFC→D2 modulatory connections.
"""

# =============================================================================
# Prefrontal Cortex Oscillator Modulation
# =============================================================================

PFC_FEEDFORWARD_GAIN_MIN: float = 0.5
"""Minimum gain for feedforward input to prefrontal cortex.

During retrieval phase (low encoding_mod), feedforward input is weakened
to protect working memory contents.
"""

PFC_FEEDFORWARD_GAIN_RANGE: float = 0.5
"""Range of feedforward gain modulation (0.5 to 1.0 when combined with min).

During encoding phase (high encoding_mod), feedforward gain reaches 1.0
to allow new information into working memory.
"""

PFC_RECURRENT_GAIN_MIN: float = 0.5
"""Minimum gain for recurrent connections in prefrontal cortex.

During encoding phase (low retrieval_mod), recurrence is weakened to
allow WM update.
"""

PFC_RECURRENT_GAIN_RANGE: float = 0.5
"""Range of recurrent gain modulation (0.5 to 1.0 when combined with min).

During retrieval phase (high retrieval_mod), recurrent gain reaches 1.0
to maintain working memory contents.
"""

# =============================================================================
# Utility Functions (for reference, actual implementation in oscillator_utils.py)
# =============================================================================

def compute_encoding_retrieval_modulation(theta_phase: float) -> tuple[float, float]:
    """Reference implementation for theta-phase encoding/retrieval modulation.

    This is provided for documentation. Actual implementation is in
    thalia.utils.oscillator_utils to avoid circular imports.

    Args:
        theta_phase: Theta phase in radians [0, 2π]

    Returns:
        (encoding_mod, retrieval_mod): Both in range [0.0, 1.0]
    """
    encoding_mod = THETA_ENCODING_PHASE_SCALE * (1.0 + math.cos(theta_phase))
    retrieval_mod = THETA_RETRIEVAL_PHASE_SCALE * (1.0 - math.cos(theta_phase))
    return encoding_mod, retrieval_mod


__all__ = [
    # Theta modulation
    "THETA_ENCODING_PHASE_SCALE",
    "THETA_RETRIEVAL_PHASE_SCALE",
    # Hippocampal gating
    "DG_CA3_GATE_MIN",
    "DG_CA3_GATE_RANGE",
    "EC_CA3_GATE_MIN",
    "EC_CA3_GATE_RANGE",
    "CA3_RECURRENT_GATE_MIN",
    "CA3_RECURRENT_GATE_RANGE",
    "CA3_CA1_ENCODING_SCALE",
    "CA1_SPARSITY_RETRIEVAL_BOOST",
    # Cortical gating
    "L4_INPUT_ENCODING_SCALE",
    "L23_RECURRENT_RETRIEVAL_SCALE",
    # Prefrontal gating
    "PFC_FEEDFORWARD_GAIN_MIN",
    "PFC_FEEDFORWARD_GAIN_RANGE",
    "PFC_RECURRENT_GAIN_MIN",
    "PFC_RECURRENT_GAIN_RANGE",
    # Cerebellum gating
    "CEREBELLUM_INPUT_BASE_GAIN",
    "CEREBELLUM_INPUT_MODULATION_RANGE",
    # Neuromodulator interactions
    "ACH_RECURRENT_SUPPRESSION",
    "ACH_THRESHOLD_FOR_SUPPRESSION",
    "ACH_ENCODING_BOOST_BASE",
    "ACH_ENCODING_BOOST_RANGE",
    # Gamma attention
    "GAMMA_ATTENTION_THRESHOLD",
    "GAMMA_LEARNING_MODULATION_SCALE",
    # Theta-gamma coupling
    "THETA_GAMMA_PHASE_DIFF_SIGMA",
    # Striatal learning
    "STRIATUM_PFC_MODULATION_LR",
    # Utility
    "compute_encoding_retrieval_modulation",
]
