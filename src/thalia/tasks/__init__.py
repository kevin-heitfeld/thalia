"""
Cognitive tasks for Thalia's developmental curriculum.
"""

from __future__ import annotations


from thalia.tasks.executive_function import (
    ExecutiveFunctionTasks,
    TaskType,
    StimulusType,
    GoNoGoConfig,
    DelayedGratificationConfig,
    DCCSConfig,
    TaskResult,
)

from thalia.tasks.working_memory import (
    NBackTask,
    ThetaGammaEncoder,
    WorkingMemoryTaskConfig,
    theta_gamma_n_back,
    create_n_back_sequence,
)

from thalia.tasks.sensorimotor import (
    SensorimotorTaskType,
    MovementDirection,
    MotorControlConfig,
    ReachingConfig,
    ManipulationConfig,
    MotorControlTask,
    ReachingTask,
    ManipulationTask,
    SensorimotorTaskLoader,
)

__all__ = [
    "ExecutiveFunctionTasks",
    "TaskType",
    "StimulusType",
    "GoNoGoConfig",
    "DelayedGratificationConfig",
    "DCCSConfig",
    "TaskResult",
    "NBackTask",
    "ThetaGammaEncoder",
    "WorkingMemoryTaskConfig",
    "theta_gamma_n_back",
    "create_n_back_sequence",
    # Sensorimotor (Stage -0.5)
    "SensorimotorTaskType",
    "MovementDirection",
    "MotorControlConfig",
    "ReachingConfig",
    "ManipulationConfig",
    "MotorControlTask",
    "ReachingTask",
    "ManipulationTask",
    "SensorimotorTaskLoader",
]
