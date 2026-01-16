"""
Learning Rule Enumeration.

This module defines the types of learning rules used across different brain regions
in the Thalia architecture. Each rule represents a different biological learning
mechanism with distinct computational properties.

Architecture Note:
==================
This enum is a CORE type because it's used across multiple regions and by the
configuration system. It defines the behavioral signature of a region, not the
implementation details.

Author: Thalia Project
Date: January 16, 2026
"""

from __future__ import annotations

from enum import Enum, auto


class LearningRule(Enum):
    """Types of learning rules used in different brain regions.

    These represent biologically-plausible learning mechanisms, each with
    distinct computational properties and use cases.
    """

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


__all__ = ["LearningRule"]
