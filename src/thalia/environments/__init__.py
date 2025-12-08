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

Usage:
    from thalia.environments import SensorimotorWrapper
    
    wrapper = SensorimotorWrapper('Reacher-v4')
    obs_spikes = wrapper.reset()
    motor_spikes = brain.motor_cortex(obs_spikes)
    obs_spikes, reward, done, truncated = wrapper.step(motor_spikes)
"""

from thalia.environments.sensorimotor_wrapper import (
    SensorimotorWrapper,
    SensorimotorConfig,
    SpikeEncoding,
)

__all__ = [
    "SensorimotorWrapper",
    "SensorimotorConfig",
    "SpikeEncoding",
]
