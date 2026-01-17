"""
Curriculum Management for THALIA Training.

This module provides curriculum strategies and stage management:
- Interleaved curriculum sampling
- Spaced repetition
- Difficulty calibration
- Stage transition protocols
- Stage evaluation and milestone checking
- Safety system (stage gates, monitoring, graceful degradation)

Author: Thalia Project
Date: December 12, 2025
"""

from __future__ import annotations

from thalia.training.curriculum.curriculum import (
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
from thalia.training.curriculum.logger import (
    CurriculumLogger,
    LogLevel,
    StageLog,
)
from thalia.training.curriculum.noise_scheduler import (
    NoiseProfile,
    NoiseScheduler,
    NoiseSchedulerConfig,
    NoiseType,
)
from thalia.training.curriculum.safety_system import (
    CurriculumSafetySystem,
    SafetyStatus,
)
from thalia.training.curriculum.stage_evaluation import (
    check_system_health,
    evaluate_stage_phonology,
    evaluate_stage_sensorimotor,
    evaluate_stage_toddler,
    generate_evaluation_report,
)
from thalia.training.curriculum.stage_gates import (
    GateDecision,
    GateResult,
    GracefulDegradationManager,
    Stage1SurvivalGate,
)
from thalia.training.curriculum.stage_manager import (
    ActiveMechanism,
    CognitiveLoadMonitor,
    CurriculumTrainer,
    MechanismPriority,
    StageConfig,
    TaskConfig,
    TrainingResult,
)
from thalia.training.curriculum.stage_monitoring import (
    ContinuousMonitor,
    InterventionType,
    MonitoringMetrics,
    Stage1Monitor,
)

__all__ = [
    # Curriculum mechanics
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
    # Stage manager (formerly curriculum_trainer)
    "CurriculumTrainer",
    "StageConfig",
    "TaskConfig",
    "TrainingResult",
    "MechanismPriority",
    "ActiveMechanism",
    "CognitiveLoadMonitor",
    # Noise scheduling (NEW)
    "NoiseScheduler",
    "NoiseSchedulerConfig",
    "NoiseProfile",
    "NoiseType",
    # Safety system (NEW)
    "CurriculumSafetySystem",
    "SafetyStatus",
    "Stage1SurvivalGate",
    "GracefulDegradationManager",
    "GateResult",
    "GateDecision",
    "ContinuousMonitor",
    "Stage1Monitor",
    "InterventionType",
    "MonitoringMetrics",
    # Stage evaluation
    "evaluate_stage_sensorimotor",
    "evaluate_stage_phonology",
    "evaluate_stage_toddler",
    "check_system_health",
    "generate_evaluation_report",
    # Logger
    "CurriculumLogger",
    "LogLevel",
    "StageLog",
]
