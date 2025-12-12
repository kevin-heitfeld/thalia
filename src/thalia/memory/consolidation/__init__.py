"""Memory Consolidation.

Mechanisms for detecting memory pressure, triggering consolidation,
and advanced consolidation processes including schema extraction,
semantic reorganization, and interference resolution.
"""

from thalia.memory.consolidation.consolidation import (
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
from thalia.memory.consolidation.advanced_consolidation import (
    SchemaExtractionConsolidation,
    SchemaExtractionConfig,
    Schema,
    SemanticReorganization,
    SemanticReorganizationConfig,
    InterferenceResolution,
    InterferenceResolutionConfig,
    run_advanced_consolidation,
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
