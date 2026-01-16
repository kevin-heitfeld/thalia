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


from thalia.constants.training import (
    AttentionStage,
    get_attention_weights,
)
from thalia.training.curriculum.curriculum import (
    InterleavedCurriculumSampler,
    InterleavedCurriculumSamplerConfig,
    SpacedRepetitionScheduler,
    SpacedRepetitionSchedulerConfig,
    TestingPhaseProtocol,
    TestingPhaseConfig,
    ProductiveFailurePhase,
    ProductiveFailureConfig,
    CurriculumDifficultyCalibrator,
    DifficultyCalibratorConfig,
    StageTransitionProtocol,
    StageTransitionConfig,
    TransitionWeekConfig,
)
from thalia.training.curriculum.logger import (
    CurriculumLogger,
    LogLevel,
    StageLog,
)
from thalia.training.curriculum.noise_scheduler import (
    NoiseScheduler,
    NoiseSchedulerConfig,
    NoiseProfile,
    NoiseType,
)
from thalia.training.curriculum.safety_system import (
    CurriculumSafetySystem,
    SafetyStatus,
)
from thalia.training.curriculum.stage_evaluation import (
    evaluate_stage_sensorimotor,
    evaluate_stage_phonology,
    evaluate_stage_toddler,
    check_system_health,
    generate_evaluation_report,
)
from thalia.training.curriculum.stage_gates import (
    Stage1SurvivalGate,
    GracefulDegradationManager,
    GateResult,
    GateDecision,
)
from thalia.training.curriculum.stage_manager import (
    CurriculumTrainer,
    StageConfig,
    TaskConfig,
    TrainingResult,
    MechanismPriority,
    ActiveMechanism,
    CognitiveLoadMonitor,
)
from thalia.training.curriculum.stage_monitoring import (
    ContinuousMonitor,
    Stage1Monitor,
    InterventionType,
    MonitoringMetrics,
)

__all__ = [
    # Constants (NEW)
    "AttentionStage",
    "get_attention_weights",
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
