"""
Training and Evaluation Threshold Constants

This module centralizes threshold values used across training, curriculum,
and evaluation code. Replaces scattered magic numbers with named constants
that document their biological/empirical basis.
"""

from __future__ import annotations

# =============================================================================
# Curriculum Progression
# =============================================================================

CURRICULUM_LOAD_THRESHOLD = 0.9
"""Cognitive overload threshold (90%).

When cognitive load exceeds this, learning becomes inefficient. Used by
CognitiveLoadMonitor to prevent overwhelming the system with too much info.

Basis: Working memory capacity limits (Cowan 2001, ~4 chunks)
"""

CURRICULUM_MARGIN = 0.1
"""Safety margin below threshold (10%).

Target cognitive load = CURRICULUM_LOAD_THRESHOLD - CURRICULUM_MARGIN.
Provides buffer to prevent frequent threshold crossings.
"""

# =============================================================================
# CURRICULUM TRAINING CONSTANTS
# =============================================================================


# Safety System Thresholds (Graceful Degradation)
CRITICAL_SYSTEMS = {"working_memory", "oscillators", "replay"}  # Cannot degrade
DEGRADABLE_SYSTEMS = {"language", "grammar", "reading"}  # Can degrade gracefully
LIMITED_DEGRADATION = {"vision", "phonology"}  # Limited degradation allowed
