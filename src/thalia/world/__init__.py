"""
World Model - Predictive processing and internal simulation.

This module provides the network's internal model of the world:
- PredictiveLayer: Generates predictions and computes errors
- WorldModel: Multi-layer predictive hierarchy
- ActionSimulator: Evaluates potential actions through simulation
- PredictiveCodingNetwork: Classic predictive coding architecture

Key concepts:
- Prediction: The model constantly predicts incoming sensory data
- Prediction Error: Difference between prediction and actual input
- Surprise: Unexpected events that update the model
- Simulation: Running the model forward without actual sensory input
"""

from .world_model import (
    WorldModel,
    WorldModelConfig,
    PredictiveLayer,
    PredictiveLayerConfig,
    PredictionMode,
    ActionSimulator,
    SimulationResult,
    PredictiveCodingNetwork,
)

__all__ = [
    "WorldModel",
    "WorldModelConfig",
    "PredictiveLayer",
    "PredictiveLayerConfig",
    "PredictionMode",
    "ActionSimulator",
    "SimulationResult",
    "PredictiveCodingNetwork",
]
