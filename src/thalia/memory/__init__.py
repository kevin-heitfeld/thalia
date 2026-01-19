"""
Memory Module for THALIA.

This module provides higher-level memory abstractions built on the
hippocampal circuit for sequence and context memory.

Components:
- SequenceMemory: Token sequence storage and recall
- ContextBuffer: Working memory for recent context
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
from .context import ContextBuffer, ContextBufferConfig
from .sequence import SequenceContext, SequenceMemory

__all__ = [
    "SequenceMemory",
    "SequenceContext",
    "ContextBuffer",
    "ContextBufferConfig",
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
