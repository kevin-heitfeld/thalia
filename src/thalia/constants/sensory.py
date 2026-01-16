"""
Sensory pathway constants for biological sensory processing.

This module defines constants for retinal, cochlear, and somatosensory processing
that reflect biological sensor dynamics and adaptation.

Consolidated from pathways/sensory_constants.py.

Author: Thalia Project
Date: January 16, 2026 (Architecture Review Tier 1.2)
"""

from __future__ import annotations

# ============================================================================
# RETINAL PROCESSING CONSTANTS
# ============================================================================

# Temporal adaptation (photoreceptor dynamics)
RETINA_ADAPTATION_DECAY = 0.9
"""Decay rate for photoreceptor adaptation state (90% retention per timestep)."""

RETINA_ADAPTATION_RATE = 0.1
"""Rate at which new input influences adaptation (10% per timestep)."""

# Center-surround (DoG filter)
DOG_FILTER_SIZE = 7
"""Spatial extent of Difference-of-Gaussians filter (pixels)."""

DOG_SIGMA_CENTER = 1.0
"""Standard deviation of center Gaussian (in pixels)."""

DOG_SIGMA_SURROUND = 2.0
"""Standard deviation of surround Gaussian (in pixels)."""

# ============================================================================
# COCHLEAR PROCESSING CONSTANTS
# ============================================================================

# Cochlear filterbank
COCHLEA_MIN_FREQ_HZ = 80.0
"""Minimum frequency for cochlear filterbank (human hearing: ~20-20,000 Hz)."""

COCHLEA_MAX_FREQ_HZ = 7600.0
"""Maximum frequency for cochlear filterbank (speech-optimized range)."""

COCHLEA_INTEGRATION_WINDOW_MS = 25.0
"""Temporal integration window for hair cell responses (ms)."""

# Hair cell dynamics
HAIR_CELL_COMPRESSION_EXPONENT = 0.3
"""Compressive nonlinearity exponent (models outer hair cell dynamics)."""

HAIR_CELL_ADAPTATION_SUPPRESSION = 0.5
"""Gain reduction from adaptation (50% suppression of sustained input)."""

# Auditory nerve adaptation
AUDITORY_NERVE_ADAPTATION_DECAY = 0.95
"""Decay rate for auditory nerve adaptation (95% retention per timestep)."""

AUDITORY_NERVE_ADAPTATION_RATE = 0.05
"""Rate at which new input influences adaptation (5% per timestep)."""

# ============================================================================
# TEMPORAL CODING CONSTANTS
# ============================================================================

LATENCY_EPSILON = 1e-6
"""Small epsilon to prevent division by zero in latency calculations."""

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Retinal processing
    "RETINA_ADAPTATION_DECAY",
    "RETINA_ADAPTATION_RATE",
    "DOG_FILTER_SIZE",
    "DOG_SIGMA_CENTER",
    "DOG_SIGMA_SURROUND",
    # Cochlear processing
    "COCHLEA_MIN_FREQ_HZ",
    "COCHLEA_MAX_FREQ_HZ",
    "COCHLEA_INTEGRATION_WINDOW_MS",
    "HAIR_CELL_COMPRESSION_EXPONENT",
    "HAIR_CELL_ADAPTATION_SUPPRESSION",
    "AUDITORY_NERVE_ADAPTATION_DECAY",
    "AUDITORY_NERVE_ADAPTATION_RATE",
    # Temporal coding
    "LATENCY_EPSILON",
]
