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
"""

from thalia.training.local_trainer import (
    LocalTrainer,
    TrainingConfig,
    TrainingMetrics,
)
from thalia.training.data_pipeline import (
    TextDataPipeline,
    DataConfig,
)

__all__ = [
    "LocalTrainer",
    "TrainingConfig",
    "TrainingMetrics",
    "TextDataPipeline",
    "DataConfig",
]
