"""
Neural Component State Dataclass.

This module defines the dataclass for representing the dynamic state of neural
components (regions, pathways, etc.) during simulation.

Architecture Note:
==================
This is a CORE type because it's used by the state management system across all
regions and pathways. It represents the interface for checkpointing and state
persistence, not region-specific implementation.

Author: Thalia Project
Date: January 16, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class NeuralComponentState:
    """Dynamic state of a neural component during simulation.

    This holds all the time-varying quantities that change during
    forward passes and learning for any neural component (region, pathway, etc.).

    Attributes:
        membrane: Membrane potentials of neurons
        spikes: Current spike output
        spike_history: Historical spike trains
        eligibility: Eligibility traces for reinforcement learning
        dopamine: Dopamine neuromodulator level (reward signal)
        acetylcholine: Acetylcholine level (attention/novelty)
        norepinephrine: Norepinephrine level (arousal/flexibility)
        firing_rate_estimate: Estimated firing rates for homeostasis
        bcm_threshold: BCM sliding threshold for homeostatic plasticity
        t: Current timestep counter
    """

    # Membrane potentials
    membrane: Optional[torch.Tensor] = None

    # Firing history
    spikes: Optional[torch.Tensor] = None
    spike_history: Optional[List[torch.Tensor]] = None

    # Eligibility traces (for RL)
    eligibility: Optional[torch.Tensor] = None

    # Neuromodulator levels (modulate plasticity)
    dopamine: float = 0.0  # Reward signal: high = consolidate, low = exploratory
    acetylcholine: float = 0.0  # Attention/novelty
    norepinephrine: float = 0.0  # Arousal/flexibility

    # Homeostatic variables
    firing_rate_estimate: Optional[torch.Tensor] = None
    bcm_threshold: Optional[torch.Tensor] = None

    # Timestep counter
    t: int = 0


__all__ = ["NeuralComponentState"]
