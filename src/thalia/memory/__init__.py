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

Configuration:
- Use ``thalia.config.SequenceMemoryConfig`` for new code
- Legacy ``SequenceMemoryConfig`` here is deprecated
"""

from thalia.memory.sequence import (
    SequenceMemory,
    SequenceContext,
)
from thalia.memory.context import (
    ContextBuffer,
    ContextBufferConfig,
)
from thalia.memory.consolidation import (
    MemoryPressureDetector,
    MemoryPressureConfig,
    SleepStageController,
    SleepStageConfig,
    SleepStage,
    ConsolidationMetrics,
    ConsolidationSnapshot,
    ConsolidationTrigger,
    ConsolidationTriggerConfig,
)
from thalia.memory.advanced_consolidation import (
    SchemaExtractionConsolidation,
    SchemaExtractionConfig,
    Schema,
    SemanticReorganization,
    SemanticReorganizationConfig,
    InterferenceResolution,
    InterferenceResolutionConfig,
    run_advanced_consolidation,
)

# Re-export from canonical location for backwards compatibility
from thalia.config import SequenceMemoryConfig

__all__ = [
    "SequenceMemory",
    "SequenceMemoryConfig",
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
