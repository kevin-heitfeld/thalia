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
    SensorimotorTaskLoader,
    SensorimotorConfig,
    PhonologyTaskLoader,
    PhonologyConfig,
    TaskLoaderRegistry,
    create_sensorimotor_loader,
    create_phonology_loader,
)
from thalia.constants.task import (
    SPIKE_PROBABILITY_LOW,
    SPIKE_PROBABILITY_MEDIUM,
    SPIKE_PROBABILITY_HIGH,
    SENSORIMOTOR_WEIGHT_MOTOR_CONTROL,
    SENSORIMOTOR_WEIGHT_REACHING,
    SENSORIMOTOR_WEIGHT_MANIPULATION,
    SENSORIMOTOR_WEIGHT_PREDICTION,
    DATASET_WEIGHT_MNIST,
    DATASET_WEIGHT_TEMPORAL,
    DATASET_WEIGHT_PHONOLOGY,
    DATASET_WEIGHT_GAZE,
    REWARD_SCALE_PREDICTION,
    MATCH_PROBABILITY_DEFAULT,
)
from thalia.training.datasets.pipeline import (
    TextDataPipeline,
    DataConfig,
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
