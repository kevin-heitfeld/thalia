"""
Environments Module for THALIA.

Provides wrappers for standard RL environments (Gymnasium, MuJoCo)
adapted for spiking neural networks.

Components:
- SensorimotorWrapper: Gymnasium + MuJoCo environments
  * Proprioception encoding (joint angles/velocities → spikes)
  * Motor decoding (spikes → torques)
  * Motor babbling, reaching tasks
  * Cerebellum training support
"""

from __future__ import annotations

from thalia.environments.sensorimotor_wrapper import (
    SensorimotorConfig,
    SensorimotorWrapper,
    SpikeEncoding,
)

__all__ = [
    "SensorimotorWrapper",
    "SensorimotorConfig",
    "SpikeEncoding",
]
