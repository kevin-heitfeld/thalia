"""
Memory Module for THALIA.

This module provides higher-level memory abstractions built on the
hippocampal circuit for sequence and context memory.

Components:
- Consolidation: Memory pressure detection and sleep-based replay
"""

from __future__ import annotations

from .consolidation import (
    ConsolidationMetrics,
    ConsolidationSnapshot,
    ConsolidationTrigger,
    ConsolidationTriggerConfig,
    MemoryPressureConfig,
    MemoryPressureDetector,
    SleepStage,
    SleepStageConfig,
    SleepStageController,
)

__all__ = [
    # Basic Consolidation
    "MemoryPressureDetector",
    "MemoryPressureConfig",
    "SleepStageController",
    "SleepStageConfig",
    "SleepStage",
    "ConsolidationMetrics",
    "ConsolidationSnapshot",
    "ConsolidationTrigger",
    "ConsolidationTriggerConfig",
]
