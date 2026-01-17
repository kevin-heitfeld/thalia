"""Memory Consolidation.

Mechanisms for detecting memory pressure, triggering consolidation,
and advanced consolidation processes including schema extraction,
semantic reorganization, and interference resolution.
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
from thalia.memory.consolidation.manager import (
    ConsolidationManager,
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
    # Advanced Consolidation
    "SchemaExtractionConsolidation",
    "SchemaExtractionConfig",
    "Schema",
    "SemanticReorganization",
    "SemanticReorganizationConfig",
    "InterferenceResolution",
    "InterferenceResolutionConfig",
    "run_advanced_consolidation",
    # Manager
    "ConsolidationManager",
]
