"""Reward Encoder - Simplified Reward Signal Interface for VTA.

This region provides a spike-based interface for delivering external reward signals
to the VTA. It abstracts away the complexity of hypothalamic and limbic reward
pathways (lateral hypothalamus, amygdala, lateral habenula) while maintaining
biological plausibility through population coding.

Biological Justification:
=========================
In real brains, reward signals reach VTA through:
- Lateral Hypothalamus (LH): Primary rewards (food, water)
- Basolateral Amygdala (BLA): Learned reward associations
- Pedunculopontine Nucleus (PPN): Sensory-motor reward prediction
- Lateral Habenula (LHb) → RMTg → VTA: Negative prediction errors

This region serves as a placeholder for these complex pathways, providing
a clean external interface while enabling future expansion to full limbic
system modeling.

Encoding Scheme:
================
Uses **population coding** where different neurons respond to different
reward magnitudes and valences:

- Neurons 0 to N/2-1: Positive reward neurons (respond to reward)
- Neurons N/2 to N-1: Negative reward neurons (respond to punishment)

Firing rate is proportional to reward magnitude, providing a naturalistic
spike-based signal to VTA rather than abstract scalar injection.

Usage:
======
```python
# In training loop or environment interface
reward_encoder = brain.regions["reward_encoder"]
reward_encoder.set_reward(external_reward=1.0)  # Positive reward

# Next timestep, reward is encoded as spikes
brain.step()  # VTA receives spike pattern via connection
```

Author: Thalia Project
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from thalia.brain.configs import RewardEncoderConfig
from thalia.typing import (
    PopulationName,
    PopulationSizes,
    RegionSpikesDict,
)

from ..neural_region import NeuralRegion
from ..region_registry import register_region


@register_region(
    "reward_encoder",
    aliases=["reward", "reward_input"],
    description="External reward signal encoder with population coding",
    version="1.0",
    author="Thalia Project",
    config_class=RewardEncoderConfig,
)
class RewardEncoder(NeuralRegion[RewardEncoderConfig]):
    """Reward signal encoder using population coding.

    Converts external scalar reward signals [-1, +1] into spike patterns
    suitable for delivery to VTA. Provides clean abstraction boundary
    between external environment and internal neuromodulation system.

    Input Populations:
    ------------------
    None - This is a source region driven by external signals (set_reward API).

    Output Populations:
    -------------------
    - "reward_signal": Spike pattern encoding current reward [n_neurons]

    Computational Properties:
    -------------------------
    - Population coding: Different neurons respond to different reward magnitudes
    - Positive rewards activate first half of neurons, negative rewards activate second half
    - Spike probability proportional to reward magnitude (with noise)
    """

    OUTPUT_POPULATIONS: Dict[PopulationName, str] = {
        "reward_signal": "n_neurons",
    }

    def __init__(self, config: RewardEncoderConfig, population_sizes: PopulationSizes):
        super().__init__(config, population_sizes)

        # Number of neurons
        self.n_neurons = config.n_neurons
        self.n_positive = config.n_neurons // 2  # Positive reward neurons
        self.n_negative = config.n_neurons - self.n_positive  # Negative reward neurons

        # Current reward value (set externally, consumed by forward())
        self._current_reward = 0.0

        # Track reward history for monitoring
        self._reward_history: list[float] = []

        super().__post_init__()

    def set_reward(self, reward: float):
        """Set external reward signal to be encoded on next forward() call.

        This is the public API for delivering reward from the environment
        or training loop into the brain.

        Args:
            reward: Reward value in range [-1, +1]
                   +1.0 = maximum positive reward
                    0.0 = neutral (no reward/punishment)
                   -1.0 = maximum punishment

        Example:
            ```python
            # After agent receives environment reward
            brain.regions["reward_encoder"].set_reward(env_reward)
            brain.step()  # Encode and deliver to VTA
            ```
        """
        # Clip to valid range
        reward = max(-1.0, min(1.0, reward))

        # Store for next forward() call
        self._current_reward = reward

        # Track history (useful for analysis)
        self._reward_history.append(reward)
        if len(self._reward_history) > 1000:
            self._reward_history.pop(0)  # Keep last 1000 only

    def forward(self, region_inputs: RegionSpikesDict) -> RegionSpikesDict:
        """Encode current reward as population spike pattern.

        Population coding:
        - First N/2 neurons: Fire proportional to positive reward magnitude
        - Last N/2 neurons: Fire proportional to negative reward magnitude
        """
        self._pre_forward(region_inputs)

        # Initialize spike tensor
        spikes = torch.zeros(self.n_neurons, device=self.device, dtype=torch.bool)

        reward = self._current_reward

        # Add slight noise for biological realism
        noise = (
            torch.randn(self.n_neurons, device=self.device) * self.config.reward_noise
        )

        if reward > 0:
            # Positive reward: Activate first N/2 neurons
            # Spike probability proportional to reward magnitude
            spike_prob = reward + noise[:self.n_positive]
            spike_prob = torch.clamp(spike_prob, 0.0, 1.0)

            positive_spikes = torch.rand(self.n_positive, device=self.device) < spike_prob
            spikes[: self.n_positive] = positive_spikes

        elif reward < 0:
            # Negative reward (punishment): Activate last N/2 neurons
            # Spike probability proportional to punishment magnitude
            spike_prob = abs(reward) + noise[self.n_positive :]
            spike_prob = torch.clamp(spike_prob, 0.0, 1.0)

            negative_spikes = torch.rand(self.n_negative, device=self.device) < spike_prob
            spikes[self.n_positive :] = negative_spikes

        # Reset reward after encoding (single-timestep pulse)
        # This ensures reward is only delivered once per set_reward() call
        self._current_reward = 0.0

        region_outputs: RegionSpikesDict = {
            "reward_signal": spikes,
        }

        return self._post_forward(region_outputs)

    def get_reward_history(self) -> list[float]:
        """Get recent reward history for analysis.

        Returns:
            List of recent reward values (up to last 1000)
        """
        return self._reward_history.copy()

    def get_mean_reward(self) -> float:
        """Get mean reward over recent history.

        Returns:
            Mean reward value, or 0.0 if no history
        """
        if not self._reward_history:
            return 0.0
        return sum(self._reward_history) / len(self._reward_history)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for this region."""
        return {
            "current_reward": self._current_reward,
            "mean_reward": self.get_mean_reward(),
            "reward_history_length": len(self._reward_history),
        }
