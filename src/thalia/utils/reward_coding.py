"""Reward population coding utilities.

Converts scalar reward values into population-coded spike tensors suitable
for injecting directly into VTA.  Extracted from the former RewardEncoder
region so it can be used as a plain function without a region boundary.

Encoding scheme (same as old RewardEncoder):
- Neurons 0 to N/2-1: Positive reward neurons (fire for reward > 0)
- Neurons N/2 to N-1: Negative reward neurons (fire for reward < 0)
- Spike probability ∝ reward magnitude with optional Gaussian noise.
"""

from __future__ import annotations

import torch


def generate_reward_spikes(
    reward: float,
    n_neurons: int,
    noise_scale: float = 0.1,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Encode a scalar reward value as a population-coded spike tensor.

    Args:
        reward: Reward magnitude in ``[-1, +1]``.
                +1.0 = maximum positive reward
                 0.0 = neutral
                -1.0 = maximum punishment
        n_neurons: Total number of encoding neurons (must be ≥ 2).
                   The first half respond to positive rewards; the second
                   half respond to negative rewards / punishments.
        noise_scale: Standard deviation of Gaussian noise added to spike
                     probability before thresholding (default 0.1 = 10%).
        device: Torch device for the output tensor.

    Returns:
        Boolean spike tensor of shape ``[n_neurons]``.
    """
    # Clip to valid range
    reward = max(-1.0, min(1.0, reward))

    n_positive = n_neurons // 2
    n_negative = n_neurons - n_positive

    spikes = torch.zeros(n_neurons, device=device, dtype=torch.bool)
    noise = torch.randn(n_neurons, device=device) * noise_scale

    if reward > 0:
        spike_prob = torch.clamp(torch.tensor(reward, device=device) + noise[:n_positive], 0.0, 1.0)
        spikes[:n_positive] = torch.rand(n_positive, device=device) < spike_prob
    elif reward < 0:
        spike_prob = torch.clamp(torch.tensor(abs(reward), device=device) + noise[n_positive:], 0.0, 1.0)
        spikes[n_positive:] = torch.rand(n_negative, device=device) < spike_prob

    return spikes
