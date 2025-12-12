"""
Curriculum Management for THALIA Training.

This module provides curriculum strategies and stage management:
- Interleaved curriculum sampling
- Spaced repetition
- Difficulty calibration
- Stage transition protocols
- Stage evaluation and milestone checking

Author: Thalia Project
Date: December 12, 2025
"""

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
from thalia.training.curriculum.stage_manager import (
    CurriculumTrainer,
    StageConfig,
    TaskConfig,
    TrainingResult,
    MechanismPriority,
    ActiveMechanism,
    CognitiveLoadMonitor,
)
from thalia.training.curriculum.stage_evaluation import (
    evaluate_stage_sensorimotor,
    evaluate_stage_phonology,
    evaluate_stage_toddler,
    check_system_health,
    generate_evaluation_report,
)
from thalia.training.curriculum.logger import (
    CurriculumLogger,
    LogLevel,
    StageLog,
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
