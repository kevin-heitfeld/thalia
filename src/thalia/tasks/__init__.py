"""
Cognitive tasks for Thalia's developmental curriculum.
"""

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
]
