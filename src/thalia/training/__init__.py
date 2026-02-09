"""
Curriculum Management for THALIA Training.

This module provides curriculum strategies and stage management:
- Interleaved curriculum sampling
- Spaced repetition
- Difficulty calibration
- Stage transition protocols
- Stage evaluation and milestone checking
- Safety system (stage gates, monitoring, graceful degradation)
"""

from __future__ import annotations

from .critical_periods import (
    CriticalPeriodWindow,
    CriticalPeriodConfig,
    CriticalPeriodGating,
)
from .curriculum import (
    CurriculumStage,
    CurriculumDifficultyCalibrator,
    DifficultyCalibratorConfig,
    InterleavedCurriculumSampler,
    InterleavedCurriculumSamplerConfig,
    ProductiveFailureConfig,
    ProductiveFailurePhase,
    SpacedRepetitionScheduler,
    SpacedRepetitionSchedulerConfig,
    StageTransitionConfig,
    StageTransitionProtocol,
    TestingPhaseConfig,
    TestingPhaseProtocol,
    TransitionWeekConfig,
)
from .curriculum_growth import (
    get_curriculum_growth_config,
)
from .curriculum_trainer import (
    ActiveMechanism,
    CognitiveLoadMonitor,
    CurriculumTrainer,
    MechanismPriority,
    TaskConfig,
    TrainingResult,
)
from .logger import (
    CurriculumLogger,
    LogLevel,
    StageLog,
)
from .noise_scheduler import (
    NoiseProfile,
    NoiseScheduler,
    NoiseSchedulerConfig,
    NoiseType,
)
from .safety_system import (
    CurriculumSafetySystem,
    SafetyStatus,
)
from .stage_configs import (
    StageConfig,
    get_bootstrap_config,
    get_sensorimotor_config,
)
from .stage_gates import (
    GateDecision,
    GateResult,
    GracefulDegradationManager,
)
from .stage_monitoring import (
    ContinuousMonitor,
    InterventionType,
    MonitoringMetrics,
    Stage1Monitor,
)

__all__ = [
    # Critical periods
    "CriticalPeriodWindow",
    "CriticalPeriodConfig",
    "CriticalPeriodGating",
    # Curriculum mechanics
    "CurriculumStage",
    "InterleavedCurriculumSampler",
    "InterleavedCurriculumSamplerConfig",
    "SpacedRepetitionScheduler",
    "SpacedRepetitionSchedulerConfig",
    "TestingPhaseProtocol",
    "TestingPhaseConfig",
    "ProductiveFailurePhase",
    "ProductiveFailureConfig",
    "CurriculumDifficultyCalibrator",
    "DifficultyCalibratorConfig",
    "StageTransitionProtocol",
    "StageTransitionConfig",
    "TransitionWeekConfig",
    "get_curriculum_growth_config",
    # Stage manager
    "CurriculumTrainer",
    "StageConfig",
    "TaskConfig",
    "TrainingResult",
    "MechanismPriority",
    "ActiveMechanism",
    "CognitiveLoadMonitor",
    # Stage configs
    "get_bootstrap_config",
    "get_sensorimotor_config",
    # Noise scheduling
    "NoiseScheduler",
    "NoiseSchedulerConfig",
    "NoiseProfile",
    "NoiseType",
    # Safety system
    "CurriculumSafetySystem",
    "SafetyStatus",
    "GracefulDegradationManager",
    "GateResult",
    "GateDecision",
    "ContinuousMonitor",
    "Stage1Monitor",
    "InterventionType",
    "MonitoringMetrics",
    # Logger
    "CurriculumLogger",
    "LogLevel",
    "StageLog",
]
