"""
Sensorimotor Environment Wrapper - Gymnasium + MuJoCo integration.

This module wraps Gymnasium MuJoCo environments for Thalia's spiking neural networks,
providing the foundation for Stage -0.5 (Sensorimotor Grounding) training.

Key Features:
=============

1. PROPRIOCEPTION ENCODING
   Joint angles, velocities → spike patterns:
   - Rate coding: higher velocity = higher firing rate
   - Population coding: distributed representation
   - Realistic sensory noise

2. MOTOR DECODING
   Spike patterns → motor commands (torques):
   - Population vector decoding
   - Smoothing for realistic control
   - Action bounds enforcement

3. TASK SUPPORT
   - Motor babbling (exploration)
   - Reaching tasks (goal-directed)
   - Object manipulation
   - Forward/inverse model training (cerebellum)

4. CURRICULUM INTEGRATION
   - Difficulty scaling
   - Progressive task complexity
   - Success tracking

Supported Environments:
=======================

**Reacher-v4** (PRIMARY):
- 2-joint arm reaching to random targets
- Observation: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), target_x, target_y, vel1, vel2]
- Action: [torque1, torque2]
- Perfect for sensorimotor grounding

**Pusher-v4**:
- 3-DOF arm pushing objects
- More complex manipulation
- Stage -0.5 advanced tasks

**HalfCheetah-v4**:
- Full-body locomotion
- Complex coordination
- Optional stretch goal

Usage:
======

    from thalia.environments.sensorimotor_wrapper import (
        SensorimotorWrapper,
        SensorimotorConfig,
    )

    # Create wrapper
    wrapper = SensorimotorWrapper(
        env_name='Reacher-v4',
        config=SensorimotorConfig(
            spike_encoding='rate',
            n_neurons_per_dof=50,
        )
    )

    # Reset environment
    obs_spikes = wrapper.reset()
    # → torch.Tensor[n_neurons_total] (binary spikes)

    # Interact
    for step in range(1000):
        # Brain produces motor spikes
        motor_spikes = brain.motor_cortex(obs_spikes)

        # Step environment
        obs_spikes, reward, terminated, truncated = wrapper.step(motor_spikes)

        if terminated or truncated:
            obs_spikes = wrapper.reset()

    # Motor babbling task
    wrapper.motor_babbling(brain, n_steps=10000)

    # Reaching task
    success_rate = wrapper.reaching_task(brain, n_trials=100)

Biological Mapping:
===================

**Proprioception → Dorsal Stream**:
- Joint angles/velocities → primary somatosensory cortex (S1)
- Population coding in cortical columns
- ~50 neurons per degree of freedom

**Motor Commands → Motor Cortex**:
- Population vector coding (Georgopoulos)
- M1 → spinal cord → muscles
- Smoothing via basal ganglia

**Cerebellum**:
- Forward model: predict sensory consequences of actions
- Inverse model: compute actions to achieve desired state
- Error-corrective learning (supervised)

References:
===========
- Georgopoulos et al. (1986): Population vector coding
- Wolpert & Kawato (1998): Internal models in cerebellum
- Todorov & Jordan (2002): Optimal feedback control

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

# ============================================================================
# Configuration
# ============================================================================


class SpikeEncoding(Enum):
    """Spike encoding strategies for proprioception."""

    RATE = "rate"  # Rate coding (firing rate ∝ value)
    POPULATION = "population"  # Population coding (Gaussian tuning curves)
    TEMPORAL = "temporal"  # Temporal coding (spike timing)


@dataclass
class SensorimotorConfig:
    """Configuration for sensorimotor wrapper."""

    # Environment
    env_name: str = "Reacher-v4"
    render_mode: Optional[str] = None  # 'human' for visualization

    # Spike encoding
    spike_encoding: str = "rate"  # 'rate', 'population', 'temporal'
    n_neurons_per_dof: int = 50  # Neurons per degree of freedom
    max_firing_rate: float = 100.0  # Hz (for rate coding)
    dt_ms: float = 0.001  # Simulation timestep (1ms)

    # Motor decoding
    motor_decoding: str = "population_vector"  # 'population_vector', 'rate'
    motor_smoothing: float = 0.3  # Exponential smoothing factor

    # Noise (biological realism)
    sensory_noise_std: float = 0.02  # Proprioceptive noise
    motor_noise_std: float = 0.01  # Motor command noise

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Sensorimotor Wrapper
# ============================================================================


class SensorimotorWrapper:
    """Wrap Gymnasium MuJoCo environments for Thalia's spiking networks.

    Converts continuous observations (joint angles, velocities) to spike patterns,
    and spike patterns to motor commands (torques).

    Example:
        >>> wrapper = SensorimotorWrapper('Reacher-v4')
        >>> obs_spikes = wrapper.reset()
        >>> obs_spikes.shape
        torch.Size([400])  # 8 DOF × 50 neurons/DOF
        >>>
        >>> motor_spikes = torch.rand(100) > 0.9  # 10% firing
        >>> obs_spikes, reward, done, truncated = wrapper.step(motor_spikes)
    """

    def __init__(self, env_name: Optional[str] = None, config: Optional[SensorimotorConfig] = None):
        """Initialize wrapper.

        Args:
            env_name: Gymnasium environment name (overrides config)
            config: Configuration object
        """
        self.config = config or SensorimotorConfig()

        if env_name is not None:
            self.config.env_name = env_name

        # Create Gymnasium environment
        self.env = gym.make(
            self.config.env_name,
            render_mode=self.config.render_mode,
        )

        # Get observation/action dimensions
        assert self.env.observation_space.shape is not None, "Observation space must have shape"
        assert self.env.action_space.shape is not None, "Action space must have shape"
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Calculate spike dimensions
        self.n_sensory_neurons = self.obs_dim * self.config.n_neurons_per_dof
        self.n_motor_neurons = self.action_dim * self.config.n_neurons_per_dof

        # Motor smoothing state
        self._last_action = np.zeros(self.action_dim)

        # Episode statistics
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_history: List[Dict[str, Any]] = []

        # Device
        self.device = torch.device(self.config.device)

        print(f"Initialized {self.config.env_name}:")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Sensory neurons: {self.n_sensory_neurons}")
        print(f"  Motor neurons: {self.n_motor_neurons}")

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """Reset environment and return initial observation as spikes.

        Args:
            seed: Random seed for reproducibility

        Returns:
            obs_spikes: Binary spike tensor [n_sensory_neurons]
        """
        obs, info = self.env.reset(seed=seed)

        # Reset episode tracking
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._last_action = np.zeros(self.action_dim)

        # Encode observation as spikes
        obs_spikes = self._encode_observation(obs)

        return obs_spikes

    def step(self, motor_spikes: torch.Tensor) -> Tuple[torch.Tensor, float, bool, bool]:
        """Execute one step in the environment.

        Args:
            motor_spikes: Binary spike tensor [n_motor_neurons]

        Returns:
            Tuple of (obs_spikes, reward, terminated, truncated)
        """
        # Decode motor spikes to action
        action = self._decode_motor_command(motor_spikes)

        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update episode tracking
        self._episode_reward += float(reward)
        self._episode_steps += 1

        # Encode observation as spikes
        obs_spikes = self._encode_observation(obs)

        return obs_spikes, float(reward), terminated, truncated

    def _encode_observation(self, obs: np.ndarray) -> torch.Tensor:
        """Encode continuous observation as spike pattern.

        Args:
            obs: Observation from environment [obs_dim]

        Returns:
            spikes: Binary spike tensor [n_sensory_neurons]
        """
        if self.config.spike_encoding == "rate":
            return self._rate_encoding(obs)
        elif self.config.spike_encoding == "population":
            return self._population_encoding(obs)
        elif self.config.spike_encoding == "temporal":
            return self._temporal_encoding(obs)
        else:
            raise ValueError(f"Unknown encoding: {self.config.spike_encoding}")

    def _rate_encoding(self, obs: np.ndarray) -> torch.Tensor:
        """Rate coding: firing rate proportional to value.

        Each DOF has n_neurons_per_dof neurons. Firing probability
        increases linearly with normalized observation value.

        Args:
            obs: Observation [obs_dim]

        Returns:
            spikes: Binary spikes [n_sensory_neurons]
        """
        # Add sensory noise
        obs_noisy = obs + np.random.randn(self.obs_dim) * self.config.sensory_noise_std

        # Normalize to [0, 1] (assuming obs is roughly [-1, 1])
        obs_norm = (np.clip(obs_noisy, -1, 1) + 1) / 2.0

        # Compute firing rates (Hz)
        firing_rates = obs_norm * self.config.max_firing_rate

        # Convert to spike probabilities (per timestep)
        spike_probs = firing_rates * self.config.dt_ms

        # Tile to full population
        spike_probs_full = np.repeat(spike_probs, self.config.n_neurons_per_dof)

        # Sample spikes
        spikes = np.random.rand(self.n_sensory_neurons) < spike_probs_full

        return torch.tensor(spikes, dtype=torch.float32, device=self.device)

    def _population_encoding(self, obs: np.ndarray) -> torch.Tensor:
        """Population coding: Gaussian tuning curves.

        Each neuron has a preferred value and fires maximally when
        observation matches that value.

        Args:
            obs: Observation [obs_dim]

        Returns:
            spikes: Binary spikes [n_sensory_neurons]
        """
        # Create tuning curve centers (evenly spaced in [-1, 1])
        centers = np.linspace(-1, 1, self.config.n_neurons_per_dof)
        tuning_width = 0.3  # Gaussian width

        spikes = []
        for value in obs:
            # Compute tuning curve responses (Gaussian)
            responses = np.exp(-((value - centers) ** 2) / (2 * tuning_width**2))

            # Convert to spike probabilities
            spike_probs = responses * self.config.max_firing_rate * self.config.dt_ms

            # Sample spikes
            neuron_spikes = np.random.rand(self.config.n_neurons_per_dof) < spike_probs
            spikes.append(neuron_spikes)

        spikes_flat = np.concatenate(spikes)
        return torch.tensor(spikes_flat, dtype=torch.float32, device=self.device)

    def _temporal_encoding(self, obs: np.ndarray) -> torch.Tensor:
        """Temporal coding: spike timing encodes value.

        Higher values → earlier spikes within time window.

        Args:
            obs: Observation [obs_dim]

        Returns:
            spikes: Binary spikes [n_sensory_neurons]
        """
        # Simplified: just use rate coding for now
        # Full temporal coding requires time window management
        return self._rate_encoding(obs)

    def _decode_motor_command(self, motor_spikes: torch.Tensor) -> np.ndarray:
        """Decode motor spikes to continuous action.

        Args:
            motor_spikes: Binary spikes [n_motor_neurons]

        Returns:
            action: Continuous action [action_dim]
        """
        if self.config.motor_decoding == "population_vector":
            action = self._population_vector_decode(motor_spikes)
        elif self.config.motor_decoding == "rate":
            action = self._rate_decode(motor_spikes)
        else:
            raise ValueError(f"Unknown decoding: {self.config.motor_decoding}")

        # Apply exponential smoothing
        alpha = self.config.motor_smoothing
        action_smoothed = alpha * action + (1 - alpha) * self._last_action
        self._last_action = action_smoothed

        # Add motor noise
        action_noisy = (
            action_smoothed + np.random.randn(self.action_dim) * self.config.motor_noise_std
        )

        # Clip to action bounds
        action_clipped = np.clip(
            action_noisy,
            self.env.action_space.low,  # type: ignore[attr-defined]
            self.env.action_space.high,  # type: ignore[attr-defined]
        )

        return action_clipped

    def _population_vector_decode(self, motor_spikes: torch.Tensor) -> np.ndarray:
        """Population vector decoding (Georgopoulos).

        Each neuron votes for a preferred direction/value.
        Average weighted by spike counts.

        Args:
            motor_spikes: Binary spikes [n_motor_neurons]

        Returns:
            action: Decoded action [action_dim]
        """
        spikes_np = motor_spikes.cpu().numpy()

        # Reshape to [action_dim, n_neurons_per_dof]
        spikes_reshaped = spikes_np.reshape(self.action_dim, self.config.n_neurons_per_dof)

        # Preferred values (evenly spaced in action space)
        action_low = self.env.action_space.low  # type: ignore[attr-defined]
        action_high = self.env.action_space.high  # type: ignore[attr-defined]

        action = np.zeros(self.action_dim)
        for i in range(self.action_dim):
            # Preferred values for this DOF
            preferred = np.linspace(action_low[i], action_high[i], self.config.n_neurons_per_dof)

            # Weighted average by spike counts
            spike_counts = spikes_reshaped[i]
            if spike_counts.sum() > 0:
                action[i] = np.average(preferred, weights=spike_counts)
            else:
                action[i] = 0.0  # No spikes → neutral

        return action

    def _rate_decode(self, motor_spikes: torch.Tensor) -> np.ndarray:
        """Rate decoding: firing rate → action value.

        Args:
            motor_spikes: Binary spikes [n_motor_neurons]

        Returns:
            action: Decoded action [action_dim]
        """
        spikes_np = motor_spikes.cpu().numpy()

        # Reshape and average
        spikes_reshaped = spikes_np.reshape(self.action_dim, self.config.n_neurons_per_dof)
        firing_rates = spikes_reshaped.mean(axis=1)  # Average over neurons

        # Scale to action range
        action_low = self.env.action_space.low  # type: ignore[attr-defined]
        action_high = self.env.action_space.high  # type: ignore[attr-defined]
        action: np.ndarray = action_low + firing_rates * (action_high - action_low)

        return action

    def motor_babbling(self, n_steps: int = 1000) -> Dict[str, Any]:
        """Motor babbling: random exploration to learn sensorimotor mappings.

        Generates random motor commands and observes sensory consequences.
        Foundation for cerebellum forward/inverse models.

        Args:
            n_steps: Number of babbling steps

        Returns:
            Statistics: rewards, distances, etc.
        """
        self.reset()

        rewards = []
        for _ in range(n_steps):
            # Random motor spikes (uniform ~10% firing)
            motor_spikes = torch.rand(self.n_motor_neurons, device=self.device) < 0.1

            obs_spikes, reward, terminated, truncated = self.step(motor_spikes)
            rewards.append(reward)

            if terminated or truncated:
                self.reset()

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "n_steps": n_steps,
        }

    def reaching_task(self, brain, n_trials: int = 100) -> Dict[str, Any]:
        """Reaching task: evaluate brain's reaching performance.

        Args:
            brain: Thalia brain with motor output
            n_trials: Number of trials

        Returns:
            Statistics: success rate, rewards, etc.
        """
        rewards = []
        successes = []

        for trial in range(n_trials):
            obs_spikes = self.reset()
            trial_reward = 0.0

            for step in range(50):  # Max 50 steps per trial
                # Brain produces motor command
                motor_spikes = brain(obs_spikes)  # Placeholder

                obs_spikes, reward, terminated, truncated = self.step(motor_spikes)
                trial_reward += reward

                if terminated or truncated:
                    break

            rewards.append(trial_reward)
            successes.append(trial_reward > -5.0)  # Success threshold

        return {
            "success_rate": np.mean(successes),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "n_trials": n_trials,
        }

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for current episode.

        Returns:
            Statistics dict
        """
        return {
            "reward": self._episode_reward,
            "steps": self._episode_steps,
        }

    def close(self):
        """Close environment."""
        self.env.close()

    def render(self):
        """Render environment (if render_mode is set)."""
        if self.config.render_mode is not None:
            return self.env.render()
        return None
