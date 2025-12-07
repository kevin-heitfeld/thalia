"""
Base classes for brain region modules.

This module defines the abstract interface that all brain regions implement,
ensuring consistent API while allowing specialized learning rules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List

import torch

from thalia.config.base import RegionConfigBase


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
    REWARD_MODULATED_STDP = auto()  # Δw ∝ STDP_eligibility × dopamine (spike-based)

    # Episodic learning (Hippocampus)
    ONE_SHOT = auto()          # Single-exposure learning
    THETA_PHASE = auto()       # Phase-dependent encoding/retrieval

    # Predictive STDP: combines spiking with prediction error modulation (Cortex)
    PREDICTIVE_STDP = auto()   # Δw ∝ STDP × prediction_error (three-factor)


@dataclass
class RegionConfig(RegionConfigBase):
    """Configuration for a brain region.

    Inherits n_input, n_output, n_neurons, dt, device, dtype, seed from RegionConfigBase.

    This contains all the parameters needed to instantiate a region,
    with biologically-plausible defaults.
    """
    # Neuron model
    neuron_type: str = "lif"  # "lif", "conductance", "dendritic"

    # Learning parameters
    learning_rate: float = 0.01
    w_max: float = 1.0
    w_min: float = 0.0

    # Homeostasis
    target_firing_rate_hz: float = 5.0
    homeostatic_tau_ms: float = 1000.0

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

    # Neuromodulator levels (modulate plasticity)
    dopamine: float = 0.0           # Reward signal: high = consolidate, low = exploratory
    acetylcholine: float = 0.0      # Attention/novelty
    norepinephrine: float = 0.0     # Arousal/flexibility

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

    CONTINUOUS PLASTICITY
    =====================
    Unlike traditional ML models, brain regions learn CONTINUOUSLY during
    forward passes. The learning rate is modulated by neuromodulators:

    - Dopamine: Modulates learning rate. High dopamine = consolidate good patterns.
    - Acetylcholine: Modulates attention and novelty detection.
    - Norepinephrine: Modulates arousal and flexibility.

    The `forward()` method applies plasticity at each timestep, modulated by
    neuromodulators. There is no separate `learn()` method - this is intentional!
    In biological brains, learning IS dynamics, not a separate phase.
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

        # =================================================================
        # CONTINUOUS PLASTICITY SETTINGS
        # =================================================================
        self.plasticity_enabled: bool = True       # Can be disabled for eval
        self.base_learning_rate: float = config.learning_rate

    def set_dopamine(self, level: float) -> None:
        """Set dopamine level (modulates plasticity rate).

        Args:
            level: Dopamine level, typically in [-1, 1].
                   Positive = reward, consolidate current patterns
                   Negative = punishment, reduce current patterns
                   Zero = baseline learning rate
        """
        self.state.dopamine = level

    def decay_neuromodulators(
        self,
        dt_ms: float = 1.0,
        dopamine_tau_ms: float = 200.0,
        acetylcholine_tau_ms: float = 50.0,
        norepinephrine_tau_ms: float = 100.0,
    ) -> None:
        """Decay neuromodulator levels toward baseline.

        Call this at each timestep for realistic dynamics.
        Uses exponential decay toward zero (baseline).

        Args:
            dt_ms: Time step in milliseconds
            dopamine_tau_ms: Dopamine decay time constant (default 200ms)
            acetylcholine_tau_ms: ACh decay time constant (default 50ms)
            norepinephrine_tau_ms: NE decay time constant (default 100ms)

        Note:
            Subclasses can override defaults or use region-specific configs.
            E.g., Striatum uses striatum_config.dopamine_tau_ms.
        """
        import math
        self.state.dopamine *= math.exp(-dt_ms / dopamine_tau_ms)
        self.state.acetylcholine *= math.exp(-dt_ms / acetylcholine_tau_ms)
        self.state.norepinephrine *= math.exp(-dt_ms / norepinephrine_tau_ms)

    def get_effective_learning_rate(self, base_lr: Optional[float] = None) -> float:
        """Compute learning rate modulated by dopamine.

        The effective learning rate is:
            base_lr * (1 + dopamine)

        This means:
            - dopamine = 0: baseline learning
            - dopamine = 1: 2x learning rate (strong consolidation)
            - dopamine = -0.5: 0.5x learning rate (reduced learning)
            - dopamine = -1: no learning (fully suppressed)

        Args:
            base_lr: Base learning rate to modulate. If None, uses self.base_learning_rate

        Returns:
            Effective learning rate for this timestep
        """
        # Clamp to prevent negative learning rates
        modulation = max(0.0, 1.0 + self.state.dopamine)
        lr = base_lr if base_lr is not None else self.base_learning_rate
        return lr * modulation

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
        dt: float = 1.0,
        encoding_mod: float = 1.0,
        retrieval_mod: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Process input through the region AND apply continuous plasticity.

        This is the main computation method. Unlike traditional ML models,
        brain regions learn CONTINUOUSLY during forward passes. Plasticity
        is applied at each timestep, modulated by neuromodulators (dopamine).

        Args:
            input_spikes: Input spike tensor, shape (batch, n_input)
            dt: Time step in milliseconds
            encoding_mod: Theta modulation for encoding (0-1, high at theta trough)
            retrieval_mod: Theta modulation for retrieval (0-1, high at theta peak)
            **kwargs: Additional region-specific inputs

        Returns:
            Output spikes, shape (batch, n_output)
        """
        pass

    def reset_state(self) -> None:
        """Reset the region's dynamic state to initial conditions.

        This is primarily for:
        - Testing (deterministic initial state)
        - True episode boundaries (new game, new environment)
        - Initialization after construction

        WARNING: Do NOT use this between trials in a continuous task!
        Real brains don't "reset" between trials. Instead, neural activity
        decays naturally through membrane time constants. With continuous
        learning, state transitions happen via natural dynamics (decay, FFI).
        Use brain.new_sequence() only when starting completely unrelated sequences.

        This method clears ALL transient state (membrane potentials, traces,
        spike history) while preserving learned weights.
        """
        self.state = RegionState()
        if hasattr(self.neurons, 'reset_state'):
            self.neurons.reset_state()

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

    def reset_state(self) -> None:
        """Reset to baseline level."""
        self.level = self.baseline
