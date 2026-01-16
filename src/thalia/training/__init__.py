"""
Training Infrastructure for THALIA.

This module provides training utilities that use LOCAL learning rules
rather than backpropagation. All learning is biologically inspired:

- STDP: Spike-Timing Dependent Plasticity
- BCM: Bienenstock-Cooper-Munro sliding threshold
- Three-Factor: Eligibility traces + neuromodulatory signals
- Hebbian: Activity correlation-based learning

Organization:
- curriculum/: Stage management, curriculum strategies, evaluation
- datasets/: Task loaders, data pipeline, constants
- evaluation/: Metacognition, metrics
- visualization/: Monitoring, live diagnostics

Author: Thalia Project
Date: December 12, 2025
"""

from __future__ import annotations


# Import from reorganized subdirectories
from thalia.training.curriculum import (
    # Curriculum mechanics
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
    # Stage manager
    CurriculumTrainer,
    StageConfig,
    TaskConfig,
    TrainingResult,
    MechanismPriority,
    ActiveMechanism,
    CognitiveLoadMonitor,
    # Noise scheduling
    NoiseScheduler,
    NoiseSchedulerConfig,
    NoiseProfile,
    NoiseType,
    # Stage evaluation
    evaluate_stage_sensorimotor,
    evaluate_stage_phonology,
    evaluate_stage_toddler,
    check_system_health,
    generate_evaluation_report,
    # Logger
    CurriculumLogger,
    LogLevel,
    StageLog,
)
from thalia.training.datasets import (
    # Task loaders
    BaseTaskLoader,
    SensorimotorTaskLoader,
    SensorimotorConfig,
    PhonologyTaskLoader,
    PhonologyConfig,
    TaskLoaderRegistry,
    create_sensorimotor_loader,
    create_phonology_loader,
    # Data pipeline
    TextDataPipeline,
    DataConfig,
)
from thalia.training.evaluation import (
    MetacognitiveCalibrator,
    CalibrationSample,
    CalibrationPrediction,
    CalibrationMetrics,
    create_simple_task_generator,
)
from thalia.training.visualization import (
    TrainingMonitor,
    quick_monitor,
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
    # Noise scheduling
    "NoiseScheduler",
    "NoiseSchedulerConfig",
    "NoiseProfile",
    "NoiseType",
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
