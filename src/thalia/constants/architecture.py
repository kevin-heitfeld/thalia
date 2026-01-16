"""
Architecture Constants - Structural ratios and expansion factors.

Consolidated from regulation/region_architecture_constants.py.
These define biological structure ratios based on neuroanatomy.

Author: Thalia Project
Date: January 16, 2026 (Architecture Review Tier 1.2 - Complete Migration)
"""

from __future__ import annotations

# =============================================================================
# HIPPOCAMPUS ARCHITECTURE
# =============================================================================

HIPPOCAMPUS_DG_EXPANSION_FACTOR = 3.0
"""Dentate Gyrus expansion factor relative to entorhinal cortex input (3x more neurons)."""

HIPPOCAMPUS_SPARSITY_TARGET = 0.03
"""Target sparsity for dentate gyrus (3% active neurons for pattern separation)."""

# =============================================================================
# CORTICAL LAYER ARCHITECTURE
# =============================================================================

CORTEX_L4_DA_FRACTION = 0.2
"""Layer 4 dopamine sensitivity fraction (sensory input layer, low DA)."""

CORTEX_L23_DA_FRACTION = 0.3
"""Layer 2/3 dopamine sensitivity fraction (association layer, moderate DA)."""

CORTEX_L5_DA_FRACTION = 0.4
"""Layer 5 dopamine sensitivity fraction (motor output layer, high DA)."""

CORTEX_L6_DA_FRACTION = 0.1
"""Layer 6 dopamine sensitivity fraction (feedback/attention layer, low DA)."""

# =============================================================================
# CEREBELLUM ARCHITECTURE
# =============================================================================

CEREBELLUM_GRANULE_EXPANSION = 100.0
"""Granule cell expansion factor (extremely large in biology, scaled for modeling)."""

CEREBELLUM_PURKINJE_PER_DCN = 10.0
"""Purkinje cells per deep cerebellar nucleus neuron (10:1 convergence ratio)."""

# =============================================================================
# PREFRONTAL CORTEX ARCHITECTURE
# =============================================================================

PFC_WM_CAPACITY_RATIO = 0.3
"""Working memory capacity as ratio of PFC size (30% of neurons participate in WM)."""

# =============================================================================
# MULTISENSORY ARCHITECTURE
# =============================================================================

MULTISENSORY_VISUAL_RATIO = 0.3
"""Visual pool fraction in multisensory integration areas."""

MULTISENSORY_AUDITORY_RATIO = 0.3
"""Auditory pool fraction in multisensory integration areas."""

MULTISENSORY_LANGUAGE_RATIO = 0.2
"""Language pool fraction in multisensory integration areas."""

MULTISENSORY_INTEGRATION_RATIO = 0.2
"""Integration pool fraction in multisensory areas."""

# =============================================================================
# NEURAL GROWTH CONSTANTS
# =============================================================================

GROWTH_NEW_WEIGHT_SCALE = 0.2
"""Scaling factor for new weights during neurogenesis (20% of w_max)."""

ACTIVITY_HISTORY_DECAY = 0.99
"""Exponential decay factor for activity history tracking."""

ACTIVITY_HISTORY_INCREMENT = 0.01
"""Increment weight for new activity in exponential moving average."""

# =============================================================================
# METACOGNITION THRESHOLDS
# =============================================================================

METACOG_ABSTENTION_STAGE1 = 0.5
"""Binary abstention threshold for Stage 1 (toddler, simple binary threshold)."""

METACOG_ABSTENTION_STAGE2 = 0.3
"""Low confidence threshold for Stage 2 (preschool, confidence estimation)."""

METACOG_ABSTENTION_STAGE3 = 0.4
"""Uncertainty threshold for Stage 3 (elementary, uncertainty estimation)."""

METACOG_ABSTENTION_STAGE4 = 0.3
"""Calibrated threshold for Stage 4 (adolescent, calibrated confidence)."""

METACOG_CALIBRATION_LR = 0.01
"""Learning rate for metacognitive calibration network."""


__all__ = [
    "HIPPOCAMPUS_DG_EXPANSION_FACTOR",
    "HIPPOCAMPUS_SPARSITY_TARGET",
    "CORTEX_L4_DA_FRACTION",
    "CORTEX_L23_DA_FRACTION",
    "CORTEX_L5_DA_FRACTION",
    "CORTEX_L6_DA_FRACTION",
    "CEREBELLUM_GRANULE_EXPANSION",
    "CEREBELLUM_PURKINJE_PER_DCN",
    "PFC_WM_CAPACITY_RATIO",
    "MULTISENSORY_VISUAL_RATIO",
    "MULTISENSORY_AUDITORY_RATIO",
    "MULTISENSORY_LANGUAGE_RATIO",
    "MULTISENSORY_INTEGRATION_RATIO",
    "GROWTH_NEW_WEIGHT_SCALE",
    "ACTIVITY_HISTORY_DECAY",
    "ACTIVITY_HISTORY_INCREMENT",
    "METACOG_ABSTENTION_STAGE1",
    "METACOG_ABSTENTION_STAGE2",
    "METACOG_ABSTENTION_STAGE3",
    "METACOG_ABSTENTION_STAGE4",
    "METACOG_CALIBRATION_LR",
]
