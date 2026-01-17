"""
Memory Module for THALIA.

This module provides higher-level memory abstractions built on the
hippocampal circuit for sequence and context memory.

Components:
- SequenceMemory: Token sequence storage and recall
- ContextBuffer: Working memory for recent context
- EpisodicStore: Long-term episodic memory storage
- Consolidation: Memory pressure detection and sleep-based replay
- Advanced Consolidation: Schema extraction, semantic reorganization, interference resolution
"""

from __future__ import annotations

from thalia.memory.consolidation.advanced_consolidation import (
    InterferenceResolution,
    InterferenceResolutionConfig,
    Schema,
    SchemaExtractionConfig,
    SchemaExtractionConsolidation,
    SemanticReorganization,
    SemanticReorganizationConfig,
    run_advanced_consolidation,
)
from thalia.memory.consolidation.consolidation import (
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
from thalia.memory.context import (
    ContextBuffer,
    ContextBufferConfig,
)
from thalia.memory.sequence import (
    SequenceContext,
    SequenceMemory,
)

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
    # Advanced Consolidation
    "SchemaExtractionConsolidation",
    "SchemaExtractionConfig",
    "Schema",
    "SemanticReorganization",
    "SemanticReorganizationConfig",
    "InterferenceResolution",
    "InterferenceResolutionConfig",
    "run_advanced_consolidation",
]
