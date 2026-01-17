"""
Dataset Management for THALIA Training.

This module provides dataset loading and task generation:
- Base task loader protocol
- Sensorimotor task loader
- Phonology task loader
- Task constants (probabilities, weights)
- Data pipeline for text processing

Author: Thalia Project
Date: December 12, 2025
"""

from __future__ import annotations

from thalia.training.datasets.loaders import (
    BaseTaskLoader,
    PhonologyConfig,
    PhonologyTaskLoader,
    SensorimotorConfig,
    SensorimotorTaskLoader,
    TaskLoaderRegistry,
    create_phonology_loader,
    create_sensorimotor_loader,
)
from thalia.training.datasets.pipeline import (
    DataConfig,
    TextDataPipeline,
)

__all__ = [
    # Task loaders
    "BaseTaskLoader",
    "SensorimotorTaskLoader",
    "SensorimotorConfig",
    "PhonologyTaskLoader",
    "PhonologyConfig",
    "TaskLoaderRegistry",
    "create_sensorimotor_loader",
    "create_phonology_loader",
    # Data pipeline
    "TextDataPipeline",
    "DataConfig",
]
