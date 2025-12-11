"""
Training Infrastructure for THALIA.

This module provides training utilities that use LOCAL learning rules
rather than backpropagation. All learning is biologically inspired:

- STDP: Spike-Timing Dependent Plasticity
- BCM: Bienenstock-Cooper-Munro sliding threshold
- Three-Factor: Eligibility traces + neuromodulatory signals
- Hebbian: Activity correlation-based learning

Components:
- DataPipeline: Text data loading and batching
- Metrics: Learning progress tracking
- Curriculum: Advanced curriculum strategies (interleaving, spaced repetition, etc.)
- CurriculumTrainer: Multi-stage developmental training orchestration
- StageEvaluation: Milestone checking for stage transitions
- TrainingMonitor: Interactive matplotlib-based monitoring (works in notebooks & locally)
"""

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
from thalia.training.task_loaders import (
    BaseTaskLoader,
    SensorimotorTaskLoader,
    SensorimotorConfig,
    PhonologyTaskLoader,
    PhonologyConfig,
    TaskLoaderRegistry,
    create_sensorimotor_loader,
    create_phonology_loader,
)
from thalia.training.monitor import (
    TrainingMonitor,
    quick_monitor,
)
from thalia.training.live_diagnostics import (
    LiveDiagnostics,
    quick_diagnostics,
)

__all__ = [
    # Core training
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
    # Task loaders
    "BaseTaskLoader",
    "SensorimotorTaskLoader",
    "SensorimotorConfig",
    "PhonologyTaskLoader",
    "PhonologyConfig",
    "TaskLoaderRegistry",
    "create_sensorimotor_loader",
    "create_phonology_loader",
    # Monitoring
    "TrainingMonitor",
    "quick_monitor",
    "LiveDiagnostics",
    "quick_diagnostics",
]
