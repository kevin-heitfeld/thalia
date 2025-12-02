"""
Base classes for brain region modules.

This module defines the abstract interface that all brain regions implement,
ensuring consistent API while allowing specialized learning rules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn


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

    # Episodic learning (Hippocampus)
    ONE_SHOT = auto()          # Single-exposure learning
    THETA_PHASE = auto()       # Phase-dependent encoding/retrieval

    # Predictive coding (Cortex alternative)
    PREDICTIVE = auto()        # Δw minimizes prediction error


@dataclass
class RegionConfig:
    """Configuration for a brain region.

    This contains all the parameters needed to instantiate a region,
    with biologically-plausible defaults.
    """
    # Dimensions
    n_input: int
    n_output: int

    # Neuron model
    neuron_type: str = "lif"  # "lif", "conductance", "dendritic"

    # Learning parameters
    learning_rate: float = 0.01
    w_max: float = 1.0
    w_min: float = 0.0

    # Homeostasis
    target_firing_rate_hz: float = 5.0
    homeostatic_tau_ms: float = 1000.0

    # Timing
    dt_ms: float = 0.1

    # Device
    device: str = "cpu"

    # Additional region-specific parameters
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegionState:
    """Dynamic state of a brain region during simulation.

    This holds all the time-varying quantities that change during
    forward passes and learning.
    """
    # Membrane potentials
    membrane: Optional[torch.Tensor] = None

    # Firing history
    spikes: Optional[torch.Tensor] = None
    spike_history: Optional[List[torch.Tensor]] = None

    # Eligibility traces (for RL)
    eligibility: Optional[torch.Tensor] = None

    # Neuromodulator levels
    dopamine: float = 0.0
    acetylcholine: float = 0.0
    norepinephrine: float = 0.0

    # Homeostatic variables
    firing_rate_estimate: Optional[torch.Tensor] = None
    bcm_threshold: Optional[torch.Tensor] = None

    # Timestep counter
    t: int = 0


class BrainRegion(ABC):
    """Abstract base class for brain regions.

    Each brain region implements:
    1. A specific neural circuit architecture
    2. Appropriate learning rules for that region
    3. Neuromodulation effects specific to that region

    The key insight is that different brain regions use fundamentally
    different learning algorithms, optimized for different tasks.
    """

    def __init__(self, config: RegionConfig):
        """Initialize the brain region.

        Args:
            config: Configuration parameters for the region
        """
        self.config = config
        self.device = torch.device(config.device)

        # Initialize weights
        self.weights = self._initialize_weights()

        # Initialize neurons
        self.neurons = self._create_neurons()

        # Initialize state
        self.state = RegionState()

        # Learning rule for this region
        self.learning_rule = self._get_learning_rule()

    @abstractmethod
    def _get_learning_rule(self) -> LearningRule:
        """Return the primary learning rule for this region."""
        pass

    @abstractmethod
    def _initialize_weights(self) -> torch.Tensor:
        """Initialize the weight matrix for this region.

        Different regions may have different initialization strategies
        (e.g., sparse vs dense, structured vs random).
        """
        pass

    @abstractmethod
    def _create_neurons(self) -> Any:
        """Create the neuron model for this region.

        Returns the neuron object(s) used in this region.
        """
        pass

    @abstractmethod
    def forward(
        self,
        input_spikes: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Process input through the region.

        Args:
            input_spikes: Input spike tensor, shape (batch, n_input)
            **kwargs: Additional region-specific inputs

        Returns:
            Output spikes, shape (batch, n_output)
        """
        pass

    @abstractmethod
    def learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """Apply the region's learning rule.

        Args:
            input_spikes: Input that caused the output
            output_spikes: Output produced by the region
            **kwargs: Learning signal (target, reward, error, etc.)

        Returns:
            Dictionary with learning statistics (weight changes, etc.)
        """
        pass

    def reset(self) -> None:
        """Reset the region's dynamic state.

        Called between trials or episodes to clear transient state
        while preserving learned weights.
        """
        self.state = RegionState()
        if hasattr(self.neurons, 'reset'):
            self.neurons.reset()

    def get_weights(self) -> torch.Tensor:
        """Return the current weight matrix."""
        return self.weights.detach().clone()

    def set_weights(self, weights: torch.Tensor) -> None:
        """Set the weight matrix."""
        self.weights = weights.to(self.device)

    @property
    def n_input(self) -> int:
        return self.config.n_input

    @property
    def n_output(self) -> int:
        return self.config.n_output


class NeuromodulatorSystem(ABC):
    """Base class for neuromodulatory systems.

    Different brain regions are modulated by different neuromodulators:
    - Dopamine: Reward prediction error (striatum, PFC)
    - Acetylcholine: Attention, novelty (cortex, hippocampus)
    - Norepinephrine: Arousal, flexibility (widespread)
    - Serotonin: Mood, patience (widespread)

    Each neuromodulator affects learning and processing differently.
    """

    def __init__(self, tau_ms: float = 50.0, device: str = "cpu"):
        self.tau_ms = tau_ms
        self.device = torch.device(device)
        self.level = 0.0
        self.baseline = 0.0

    @abstractmethod
    def compute(self, **kwargs) -> float:
        """Compute the neuromodulator level based on current state.

        Returns:
            Current neuromodulator level (can be positive or negative
            relative to baseline, depending on the system).
        """
        pass

    def decay(self, dt_ms: float) -> None:
        """Decay neuromodulator level toward baseline."""
        decay_factor = 1.0 - dt_ms / self.tau_ms
        self.level = self.baseline + (self.level - self.baseline) * decay_factor

    def reset(self) -> None:
        """Reset to baseline level."""
        self.level = self.baseline
