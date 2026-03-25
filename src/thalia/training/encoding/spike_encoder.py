"""Encode stimuli as spike trains targeting thalamic relay neurons."""

from __future__ import annotations

import torch

from thalia.typing import SynapseId, SynapticInput


def encode_population_rate(
    relay_size: int,
    neuron_indices: torch.Tensor,
    firing_prob: float,
    thalamus_region: str,
    device: torch.device | str = "cpu",
) -> SynapticInput:
    """Generate a Poisson spike vector for a subset of thalamic relay neurons.

    Args:
        relay_size: Total number of relay neurons in the target thalamus.
        neuron_indices: 1-D integer tensor of relay neuron indices to activate.
        firing_prob: Per-neuron per-timestep spike probability (0–1).
        thalamus_region: Name of the thalamus region (e.g. ``"thalamus_sensory"``).
        device: Torch device for the output tensor.

    Returns:
        A single-entry ``SynapticInput`` dict ready for ``brain.forward()``.
    """
    spikes = torch.zeros(relay_size, dtype=torch.bool, device=device)
    if len(neuron_indices) > 0 and firing_prob > 0.0:
        mask = torch.rand(len(neuron_indices), device=device) < firing_prob
        spikes[neuron_indices] = mask
    syn_id = SynapseId.external_sensory_to_thalamus_relay(thalamus_region)
    return {syn_id: spikes}
