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

# Re-export from canonical location for backwards compatibility
from thalia.config import TrainingConfig

__all__ = [
    "LocalTrainer",
    "TrainingConfig",
    "TrainingMetrics",
    "TextDataPipeline",
    "DataConfig",
]
