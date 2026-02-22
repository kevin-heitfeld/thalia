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

import torch

from thalia.brain.configs import RewardEncoderConfig
from thalia.brain.regions.population_names import RewardEncoderPopulation
from thalia.typing import (
    NeuromodulatorInput,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
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
    - "reward_signal": Spike pattern encoding current reward [reward_signal_size]

    Computational Properties:
    -------------------------
    - Population coding: Different neurons respond to different reward magnitudes
    - Positive rewards activate first half of neurons, negative rewards activate second half
    - Spike probability proportional to reward magnitude (with noise)
    """

    def __init__(self, config: RewardEncoderConfig, population_sizes: PopulationSizes, region_name: RegionName):
        super().__init__(config, population_sizes, region_name)

        # Number of neurons
        self.reward_signal_size: int = population_sizes[RewardEncoderPopulation.REWARD_SIGNAL.value]
        self.n_positive = self.reward_signal_size // 2  # Positive reward neurons
        self.n_negative = self.reward_signal_size - self.n_positive  # Negative reward neurons

        # Current reward value (set externally, consumed by forward())
        self._current_reward = 0.0

        # Store population sizes for get_population_size() queries
        self._population_sizes = population_sizes

        super().__post_init__()

    def get_population_size(self, population: str) -> int:
        """Override to provide population sizes without neuron objects.

        RewardEncoder generates spikes directly without simulating neurons,
        so it doesn't register neuron populations. We override this method
        to return sizes from the stored population_sizes dict.
        """
        if population not in self._population_sizes:
            raise ValueError(
                f"Population '{population}' not found in {self.__class__.__name__}. "
                f"Available populations: {list(self._population_sizes.keys())}"
            )
        return self._population_sizes[population]

    def set_reward(self, reward: float):
        """Set external reward signal to be encoded on next forward() call.

        Args:
            reward: Reward value in range [-1, +1]
                   +1.0 = maximum positive reward
                    0.0 = neutral (no reward/punishment)
                   -1.0 = maximum punishment
            ```
        """
        # Clip to valid range
        reward = max(-1.0, min(1.0, reward))

        # Store for next forward() call
        self._current_reward = reward

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Encode current reward as population spike pattern.

        Population coding:
        - First N/2 neurons: Fire proportional to positive reward magnitude
        - Last N/2 neurons: Fire proportional to negative reward magnitude

        Args:
            synaptic_inputs: Not used (reward is externally set via set_reward())
            neuromodulator_inputs: Not used (source region, doesn't consume neuromodulators)
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        # Initialize spike tensor
        spikes = torch.zeros(self.reward_signal_size, device=self.device, dtype=torch.bool)

        reward = self._current_reward

        # Add slight noise for biological realism
        noise = torch.randn(self.reward_signal_size, device=self.device) * self.config.reward_noise

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

        region_outputs: RegionOutput = {
            RewardEncoderPopulation.REWARD_SIGNAL.value: spikes,
        }

        return self._post_forward(region_outputs)
