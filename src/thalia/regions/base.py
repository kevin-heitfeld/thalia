"""
Base classes for brain region modules.

This module defines state and utility classes for neural components.

Architecture (v3.0+):
- Brain regions inherit from NeuralRegion (thalia.core.neural_region)
- Custom pathways inherit from LearnableComponent (thalia.core.protocols.component)

Usage:
- Regions: `class MyRegion(NeuralRegion)`
- Custom pathways: `class MyPathway(LearnableComponent)`
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List

import torch


class LearningRule(Enum):
    """Types of learning rules used in different brain regions."""

    # Unsupervised learning (Cortex)
    HEBBIAN = auto()           # Basic Hebbian: Δw ∝ pre × post
    STDP = auto()              # Spike-Timing Dependent Plasticity
    BCM = auto()               # Bienenstock-Cooper-Munro with sliding threshold

    # Supervised learning (Cerebellum)
    ERROR_CORRECTIVE = auto()  # Delta rule: Δw ∝ pre × (target - actual)
    PERCEPTRON = auto()        # Binary error correction

    # Reinforcement learning (Striatum)
    THREE_FACTOR = auto()      # Δw ∝ eligibility × dopamine
    ACTOR_CRITIC = auto()      # Policy gradient with value function
    REWARD_MODULATED_STDP = auto()  # Δw ∝ STDP_eligibility × dopamine (striatum uses D1/D2 variant)

    # Episodic learning (Hippocampus)
    ONE_SHOT = auto()          # Single-exposure learning
    THETA_PHASE = auto()       # Phase-dependent encoding/retrieval

    # Predictive STDP: combines spiking with prediction error modulation (Cortex)
    PREDICTIVE_STDP = auto()   # Δw ∝ STDP × prediction_error (three-factor)


@dataclass
class NeuralComponentState:
    """Dynamic state of a neural component during simulation.

    This holds all the time-varying quantities that change during
    forward passes and learning for any neural component (region, pathway, etc.).
    """
    # Membrane potentials
    membrane: Optional[torch.Tensor] = None

    # Firing history
    spikes: Optional[torch.Tensor] = None
    spike_history: Optional[List[torch.Tensor]] = None

    # Eligibility traces (for RL)
    eligibility: Optional[torch.Tensor] = None

    # Neuromodulator levels (modulate plasticity)
    dopamine: float = 0.0           # Reward signal: high = consolidate, low = exploratory
    acetylcholine: float = 0.0      # Attention/novelty
    norepinephrine: float = 0.0     # Arousal/flexibility

    # Homeostatic variables
    firing_rate_estimate: Optional[torch.Tensor] = None
    bcm_threshold: Optional[torch.Tensor] = None

    # Timestep counter
    t: int = 0
