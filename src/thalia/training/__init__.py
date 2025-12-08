"""
Training Infrastructure for THALIA.

This module provides training utilities that use LOCAL learning rules
rather than backpropagation. All learning is biologically inspired:

- STDP: Spike-Timing Dependent Plasticity
- BCM: Bienenstock-Cooper-Munro sliding threshold
- Three-Factor: Eligibility traces + neuromodulatory signals
- Hebbian: Activity correlation-based learning

Components:
- LocalTrainer: Main training loop with local rules
- DataPipeline: Text data loading and batching
- Metrics: Learning progress tracking
- Curriculum: Advanced curriculum strategies (interleaving, spaced repetition, etc.)
- CurriculumTrainer: Multi-stage developmental training orchestration
- StageEvaluation: Milestone checking for stage transitions

Configuration:
- Use ``thalia.config.TrainingConfig`` for new code
- Legacy ``TrainingConfig`` here is deprecated
"""

from thalia.training.local_trainer import (
    LocalTrainer,
    TrainingMetrics,
)
from thalia.training.data_pipeline import (
    TextDataPipeline,
    DataConfig,
)
from thalia.training.curriculum import (
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
from thalia.training.curriculum_trainer import (
    CurriculumTrainer,
    StageConfig,
    TaskConfig,
    TrainingResult,
    MechanismPriority,
    ActiveMechanism,
    CognitiveLoadMonitor,
)
from thalia.training.stage_evaluation import (
    evaluate_stage_sensorimotor,
    evaluate_stage_phonology,
    evaluate_stage_toddler,
    check_system_health,
    generate_evaluation_report,
)
from thalia.training.curriculum_logger import (
    CurriculumLogger,
    LogLevel,
    StageLog,
)
from thalia.training.metacognition import (
    MetacognitiveCalibrator,
    CalibrationSample,
    CalibrationPrediction,
    CalibrationMetrics,
    create_simple_task_generator,
)

# Re-export from canonical location for backwards compatibility
from thalia.config import TrainingConfig
from thalia.config.curriculum_growth import CurriculumStage

__all__ = [
    # Core training
    "LocalTrainer",
    "TrainingConfig",
    "TrainingMetrics",
    "TextDataPipeline",
    "DataConfig",
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
    # Curriculum trainer
    "CurriculumTrainer",
    "CurriculumStage",
    "StageConfig",
    "TaskConfig",
    "TrainingResult",
    # Cognitive load monitoring
    "MechanismPriority",
    "ActiveMechanism",
    "CognitiveLoadMonitor",
    # Stage evaluation
    "evaluate_stage_sensorimotor",
    "evaluate_stage_phonology",
    "evaluate_stage_toddler",
    "check_system_health",
    "generate_evaluation_report",
    # Logging
    "CurriculumLogger",
    "LogLevel",
    "StageLog",
    # Metacognition
    "MetacognitiveCalibrator",
    "CalibrationSample",
    "CalibrationPrediction",
    "CalibrationMetrics",
    "create_simple_task_generator",
]

