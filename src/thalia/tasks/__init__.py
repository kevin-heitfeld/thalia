"""
Cognitive tasks for Thalia's developmental curriculum.
"""

from __future__ import annotations

from thalia.tasks.executive_function import (
    DCCSConfig,
    DelayedGratificationConfig,
    ExecutiveFunctionTasks,
    GoNoGoConfig,
    StimulusType,
    TaskResult,
    TaskType,
)
from thalia.tasks.sensorimotor import (
    ManipulationConfig,
    ManipulationTask,
    MotorControlConfig,
    MotorControlTask,
    MovementDirection,
    ReachingConfig,
    ReachingTask,
    SensorimotorTaskLoader,
    SensorimotorTaskType,
)
from thalia.tasks.working_memory import (
    NBackTask,
    ThetaGammaEncoder,
    WorkingMemoryTaskConfig,
    create_n_back_sequence,
    theta_gamma_n_back,
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
