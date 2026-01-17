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

from thalia.constants.task import (
    DATASET_WEIGHT_GAZE,
    DATASET_WEIGHT_MNIST,
    DATASET_WEIGHT_PHONOLOGY,
    DATASET_WEIGHT_TEMPORAL,
    MATCH_PROBABILITY_DEFAULT,
    REWARD_SCALE_PREDICTION,
    SENSORIMOTOR_WEIGHT_MANIPULATION,
    SENSORIMOTOR_WEIGHT_MOTOR_CONTROL,
    SENSORIMOTOR_WEIGHT_PREDICTION,
    SENSORIMOTOR_WEIGHT_REACHING,
    SPIKE_PROBABILITY_HIGH,
    SPIKE_PROBABILITY_LOW,
    SPIKE_PROBABILITY_MEDIUM,
)
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
    # Constants
    "SPIKE_PROBABILITY_LOW",
    "SPIKE_PROBABILITY_MEDIUM",
    "SPIKE_PROBABILITY_HIGH",
    "SENSORIMOTOR_WEIGHT_MOTOR_CONTROL",
    "SENSORIMOTOR_WEIGHT_REACHING",
    "SENSORIMOTOR_WEIGHT_MANIPULATION",
    "SENSORIMOTOR_WEIGHT_PREDICTION",
    "DATASET_WEIGHT_MNIST",
    "DATASET_WEIGHT_TEMPORAL",
    "DATASET_WEIGHT_PHONOLOGY",
    "DATASET_WEIGHT_GAZE",
    "REWARD_SCALE_PREDICTION",
    "MATCH_PROBABILITY_DEFAULT",
    # Data pipeline
    "TextDataPipeline",
    "DataConfig",
]
