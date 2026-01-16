"""
Evaluation Tools for THALIA Training.

This module provides evaluation and metacognitive assessment:
- Metacognitive calibration
- Task generation for evaluation
- Prediction accuracy metrics

Author: Thalia Project
Date: December 12, 2025
"""

from __future__ import annotations


from thalia.training.evaluation.metacognition import (
    MetacognitiveCalibrator,
    CalibrationSample,
    CalibrationPrediction,
    CalibrationMetrics,
    create_simple_task_generator,
)

__all__ = [
    "MetacognitiveCalibrator",
    "CalibrationSample",
    "CalibrationPrediction",
    "CalibrationMetrics",
    "create_simple_task_generator",
]
