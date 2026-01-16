"""
Oscillator Constants - Theta, gamma, alpha frequencies and coupling.

Consolidated from regulation/oscillator_constants.py.

Author: Thalia Project
Date: January 16, 2026 (Architecture Review Tier 1.2)
"""

from __future__ import annotations

# =============================================================================
# Theta-Phase Modulation (Encoding/Retrieval)
# =============================================================================

THETA_ENCODING_PHASE_SCALE = 0.5
"""Scale factor for theta-phase encoding modulation."""

THETA_RETRIEVAL_PHASE_SCALE = 0.5
"""Scale factor for theta-phase retrieval modulation."""

# =============================================================================
# Hippocampal Pathway Gating (DG→CA3, EC→CA3, CA3 recurrence)
# =============================================================================

DG_CA3_GATE_MIN = 0.1
"""Minimum DG→CA3 pathway strength during retrieval."""

DG_CA3_GATE_RANGE = 0.9
"""Range of DG→CA3 pathway modulation (min to max)."""

EC_CA3_GATE_MIN = 0.3
"""Minimum EC→CA3 pathway strength during encoding."""

EC_CA3_GATE_RANGE = 0.7
"""Range of EC→CA3 pathway modulation (min to max)."""

CA3_RECURRENT_GATE_MIN = 0.2
"""Minimum CA3 recurrent pathway strength during encoding."""

CA3_RECURRENT_GATE_RANGE = 0.8
"""Range of CA3 recurrent pathway modulation (min to max)."""

# =============================================================================
# CA3→CA1 and EC→CA1 Gating
# =============================================================================

CA3_CA1_ENCODING_SCALE = 0.5
"""Scale factor for CA3→CA1 encoding contribution."""

CA1_SPARSITY_RETRIEVAL_BOOST = 0.5
"""Sparsity threshold increase during retrieval."""

# =============================================================================
# Cortical Gating (L4, L2/3, L5)
# =============================================================================

L4_INPUT_ENCODING_SCALE = 0.5
"""Scale factor for L4 input gain during encoding."""

L23_RECURRENT_RETRIEVAL_SCALE = 0.5
"""Scale factor for L2/3 recurrent connections during retrieval."""

# =============================================================================
# Prefrontal Gating (Feedforward/Recurrent)
# =============================================================================

PFC_FEEDFORWARD_BASE_GAIN = 0.5
"""Base feedforward gain in prefrontal cortex."""

PFC_FEEDFORWARD_MODULATION_RANGE = 0.5
"""Modulation range for feedforward connections."""

PFC_RECURRENT_BASE_GAIN = 0.5
"""Base recurrent gain in prefrontal cortex."""

PFC_RECURRENT_MODULATION_RANGE = 0.5
"""Modulation range for recurrent connections."""

PFC_FEEDFORWARD_GAIN_MIN = 0.5
"""Minimum gain for feedforward input to prefrontal cortex."""

PFC_FEEDFORWARD_GAIN_RANGE = 0.5
"""Range of feedforward gain modulation."""

PFC_RECURRENT_GAIN_MIN = 0.5
"""Minimum gain for recurrent connections in prefrontal cortex."""

PFC_RECURRENT_GAIN_RANGE = 0.5
"""Range of recurrent gain modulation."""

# =============================================================================
# Cerebellum Input Gating
# =============================================================================

CEREBELLUM_INPUT_BASE_GAIN = 0.7
"""Base input gain for cerebellar granule cells."""

CEREBELLUM_INPUT_MODULATION_RANGE = 0.3
"""Modulation range for cerebellar input."""

# =============================================================================
# Neuromodulator-Oscillator Interactions
# =============================================================================

ACH_RECURRENT_SUPPRESSION = 0.7
"""Maximum suppression of recurrent connections by high acetylcholine."""

ACH_THRESHOLD_FOR_SUPPRESSION = 0.5
"""ACh level threshold above which recurrent suppression begins."""

ACH_ENCODING_BOOST_BASE = 0.5
"""Base encoding gain when ACh is low."""

ACH_ENCODING_BOOST_RANGE = 0.5
"""Additional encoding gain at high ACh levels."""

# =============================================================================
# Gamma Modulation (Attention and Learning)
# =============================================================================

GAMMA_ATTENTION_THRESHOLD = 0.5
"""Threshold for gamma-phase attention gating."""

GAMMA_LEARNING_MODULATION_SCALE = 0.5
"""Scale for gamma-modulated learning rate (default: 0.5)."""

GAMMA_ATTENTION_SCALE = 1.0
"""Scale for gamma attention gating."""

# =============================================================================
# Theta-Gamma Coupling
# =============================================================================

THETA_GAMMA_PHASE_DIFF_SIGMA = 0.5
"""Sigma for Gaussian phase-difference gating in theta-gamma coupling."""

# =============================================================================
# Striatal Learning Modulation
# =============================================================================

STRIATUM_PFC_MODULATION_LR = 0.001
"""Learning rate for prefrontal modulation of striatal pathways."""

# =============================================================================
# Alpha Modulation (Feedback Control)
# =============================================================================

ALPHA_FEEDBACK_GAIN_MIN = 0.3
"""Minimum alpha-modulated feedback gain."""

ALPHA_FEEDBACK_GAIN_MAX = 1.0
"""Maximum alpha-modulated feedback gain."""

# =============================================================================
# Acetylcholine Recurrent Suppression (Duplicates for backwards compat)
# =============================================================================

ACH_RECURRENT_SUPPRESSION_MIN = 0.3
"""Minimum recurrent weight gain during encoding (high ACh)."""

ACH_RECURRENT_SUPPRESSION_MAX = 1.0
"""Maximum recurrent weight gain during retrieval (low ACh)."""

# =============================================================================
# Phase Locking Thresholds
# =============================================================================

PHASE_LOCKING_THRESHOLD_LOW = 0.3
"""Low phase locking quality threshold."""

PHASE_LOCKING_THRESHOLD_MEDIUM = 0.5
"""Medium phase locking quality threshold."""

PHASE_LOCKING_THRESHOLD_HIGH = 0.7
"""High phase locking quality threshold."""


__all__ = [
    "THETA_ENCODING_PHASE_SCALE",
    "THETA_RETRIEVAL_PHASE_SCALE",
    "DG_CA3_GATE_MIN",
    "DG_CA3_GATE_RANGE",
    "EC_CA3_GATE_MIN",
    "EC_CA3_GATE_RANGE",
    "CA3_RECURRENT_GATE_MIN",
    "CA3_RECURRENT_GATE_RANGE",
    "CA3_CA1_ENCODING_SCALE",
    "CA1_SPARSITY_RETRIEVAL_BOOST",
    "L4_INPUT_ENCODING_SCALE",
    "L23_RECURRENT_RETRIEVAL_SCALE",
    "PFC_FEEDFORWARD_BASE_GAIN",
    "PFC_FEEDFORWARD_MODULATION_RANGE",
    "PFC_RECURRENT_BASE_GAIN",
    "PFC_RECURRENT_MODULATION_RANGE",
    "PFC_FEEDFORWARD_GAIN_MIN",
    "PFC_FEEDFORWARD_GAIN_RANGE",
    "PFC_RECURRENT_GAIN_MIN",
    "PFC_RECURRENT_GAIN_RANGE",
    "CEREBELLUM_INPUT_BASE_GAIN",
    "CEREBELLUM_INPUT_MODULATION_RANGE",
    "ACH_RECURRENT_SUPPRESSION",
    "ACH_THRESHOLD_FOR_SUPPRESSION",
    "ACH_ENCODING_BOOST_BASE",
    "ACH_ENCODING_BOOST_RANGE",
    "GAMMA_ATTENTION_THRESHOLD",
    "GAMMA_LEARNING_MODULATION_SCALE",
    "GAMMA_ATTENTION_SCALE",
    "THETA_GAMMA_PHASE_DIFF_SIGMA",
    "STRIATUM_PFC_MODULATION_LR",
    "ALPHA_FEEDBACK_GAIN_MIN",
    "ALPHA_FEEDBACK_GAIN_MAX",
    "ACH_RECURRENT_SUPPRESSION_MIN",
    "ACH_RECURRENT_SUPPRESSION_MAX",
    "PHASE_LOCKING_THRESHOLD_LOW",
    "PHASE_LOCKING_THRESHOLD_MEDIUM",
    "PHASE_LOCKING_THRESHOLD_HIGH",
]
