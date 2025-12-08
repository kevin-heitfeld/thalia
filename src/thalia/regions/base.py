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
from thalia.core.neuromodulator_mixin import NeuromodulatorMixin


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


class BrainRegion(NeuromodulatorMixin, ABC):
    """Abstract base class for brain regions.

    Each brain region implements:
    1. A specific neural circuit architecture
    2. Appropriate learning rules for that region
    3. Neuromodulation effects specific to that region

    The key insight is that different brain regions use fundamentally
    different learning algorithms, optimized for different tasks.

    COMPONENT PROTOCOL
    ==================
    BrainRegion implements the BrainComponent protocol, which defines
    the unified interface shared with BaseNeuralPathway. This ensures
    feature parity between regions and pathways.
    
    **CRITICAL**: When adding features to regions, also add to pathways!
    Both are equally important active learning components.
    
    All components use forward() for processing (standard PyTorch, ADR-007).
    
    See: src/thalia/core/component_protocol.py
         docs/patterns/component-parity.md
         docs/decisions/adr-007-pytorch-consistency.md

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
    
    NEUROMODULATION
    ===============
    Inherits from NeuromodulatorMixin which provides:
    - set_dopamine(), set_acetylcholine(), set_norepinephrine()
    - decay_neuromodulators() with configurable tau constants
    - get_effective_learning_rate() for dopamine-modulated plasticity
    
    See NeuromodulatorMixin for full interface and usage examples.
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

    def add_neurons(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Add neurons to region for growth and capacity expansion.
        
        This is a default implementation that should be overridden by
        specific region implementations for proper weight matrix expansion.
        
        Growth Strategy:
        1. Expand weight matrix: [n_output, n_input] → [n_output+n_new, n_input]
        2. Initialize new rows with sparse random connections
        3. Preserve existing weights exactly (no reinitialization)
        4. Update config.n_output and capacity metrics
        5. Expand neuron state arrays if needed
        6. Maintain functional continuity during growth
        
        Args:
            n_new: Number of neurons to add
            initialization: Weight initialization strategy
            sparsity: Sparsity for new connections
            
        Raises:
            NotImplementedError: If region doesn't support growth
        """
        raise NotImplementedError(
            f"Region {self.__class__.__name__} does not implement add_neurons(). "
            "Growth support requires region-specific implementation."
        )
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get current activity and health metrics.
        
        Returns dictionary with region-specific diagnostics:
        - firing_rate: Average firing rate
        - weight_stats: Weight statistics (mean, std, min, max)
        - spike_count: Total spikes this timestep
        - neuromodulator_levels: Current dopamine, ACh, NE
        
        Returns:
            Dict with diagnostic information
        """
        diagnostics = {
            'region_name': self.__class__.__name__,
            'n_neurons': self.n_output,
            'timestep': self.state.t,
        }
        
        # Spike statistics
        if self.state.spikes is not None:
            spikes = self.state.spikes
            diagnostics['spike_count'] = int(spikes.sum().item())
            diagnostics['firing_rate'] = float(spikes.float().mean().item())
        else:
            diagnostics['spike_count'] = 0
            diagnostics['firing_rate'] = 0.0
        
        # Weight statistics
        if hasattr(self, 'weights'):
            w = self.weights.detach()
            diagnostics['weight_stats'] = {
                'mean': float(w.mean().item()),
                'std': float(w.std().item()),
                'min': float(w.min().item()),
                'max': float(w.max().item()),
            }
        
        # Neuromodulator levels
        diagnostics['neuromodulators'] = self.get_neuromodulator_state()
        
        return diagnostics
    
    def check_health(self) -> 'HealthReport':
        """Check for pathological states.
        
        Detects:
        - Silence: Firing rate too low (<1%)
        - Runaway activity: Firing rate too high (>90%)
        - Weight saturation: Too many weights at limits
        - Dead neurons: No activity
        
        Returns:
            HealthReport with detected issues
        """
        from thalia.diagnostics.health_monitor import HealthReport, IssueReport, IssueSeverity
        
        issues = []
        
        # Check firing rate
        if self.state.spikes is not None:
            firing_rate = float(self.state.spikes.float().mean().item())
            
            if firing_rate < 0.01:  # Less than 1%
                issues.append(IssueReport(
                    severity=IssueSeverity.HIGH,
                    issue_type='silence',
                    message=f'Firing rate too low: {firing_rate:.1%}',
                    suggested_fix='Check input strength, reduce thresholds, or increase excitation'
                ))
            elif firing_rate > 0.90:  # More than 90%
                issues.append(IssueReport(
                    severity=IssueSeverity.HIGH,
                    issue_type='runaway',
                    message=f'Firing rate too high: {firing_rate:.1%}',
                    suggested_fix='Increase inhibition, increase thresholds, or reduce input strength'
                ))
        
        # Check weight saturation
        if hasattr(self, 'weights'):
            w = self.weights.detach()
            near_max = (w > self.config.w_max * 0.95).float().mean().item()
            near_min = (w < self.config.w_min + 0.05).float().mean().item()
            
            if near_max > 0.5 or near_min > 0.5:
                issues.append(IssueReport(
                    severity=IssueSeverity.MEDIUM,
                    issue_type='weight_saturation',
                    message=f'Weight saturation: {near_max:.1%} near max, {near_min:.1%} near min',
                    suggested_fix='Consider synaptic scaling or weight normalization'
                ))
        
        # Create report
        is_healthy = len(issues) == 0
        overall_severity = max([issue.severity.value for issue in issues]) if issues else 0.0
        
        if is_healthy:
            summary = f"{self.__class__.__name__}: Healthy"
        else:
            summary = f"{self.__class__.__name__}: {len(issues)} issue(s) detected"
        
        return HealthReport(
            is_healthy=is_healthy,
            overall_severity=overall_severity,
            issues=issues,
            summary=summary,
            metrics=self.get_diagnostics()
        )
    
    def get_capacity_metrics(self) -> "CapacityMetrics":
        """Get capacity utilization metrics for growth decisions.
        
        Default implementation provides basic metrics. Regions can override
        for more sophisticated analysis.
        
        Returns:
            CapacityMetrics with:
            - firing_rate: Average firing rate (0-1)
            - weight_saturation: Fraction of weights near max
            - synapse_usage: Fraction of active synapses
            - neuron_count: Total neurons
            - synapse_count: Total synapses
            - growth_recommended: Whether growth is advised
        """
        from ..core.growth import GrowthManager
        
        # Use GrowthManager for standard metrics computation
        manager = GrowthManager(region_name=self.__class__.__name__)
        metrics = manager.get_capacity_metrics(self)
        return metrics  # Return CapacityMetrics object, not dict

    @abstractmethod
    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for checkpointing.

        This method returns ALL state needed to resume training or inference
        from this exact point, including:

        1. **Learnable Parameters** (weights):
           - Weight matrices
           - Any additional learnable parameters specific to the region

        2. **Region State** (dynamic state from RegionState):
           - Current spikes
           - Membrane potentials
           - Conductances (if using conductance-based neurons)
           - Refractory state
           - Any region-specific dynamic state

        3. **Learning Rule State** (internal state of learning mechanisms):
           - BCM thresholds (for cortex)
           - Eligibility traces (for striatum)
           - STP efficacy values (u, x)
           - STDP traces
           - Homeostatic scaling factors

        4. **Oscillator State** (if applicable):
           - Theta phase and frequency
           - Gamma phase and frequency
           - Current slot/sequence position
           - Time tracking

        5. **Neuromodulator State**:
           - Current dopamine level
           - Current acetylcholine level
           - Current norepinephrine level
           - Baseline levels

        Returns:
            Dictionary with keys:
            - 'weights': Dict[str, torch.Tensor] - All learnable parameters
            - 'region_state': Dict[str, Any] - Current RegionState data
            - 'learning_state': Dict[str, Any] - Learning rule internal state
            - 'oscillator_state': Dict[str, Any] - Oscillator phases/state (if applicable)
            - 'neuromodulator_state': Dict[str, float] - Neuromodulator levels
            - 'config': RegionConfig - Configuration (for validation on load)

        Note:
            All tensor values should be detached and cloned to prevent
            unintended modifications. This method should never modify
            the region's actual state.

        Example:
            >>> state = region.get_full_state()
            >>> # Save state to checkpoint
            >>> BrainCheckpoint.save(brain, "checkpoint.thalia")
            >>> # Later...
            >>> new_region = RegionClass(config)
            >>> new_region.load_full_state(state)
            >>> # new_region now has identical state to original
        """
        pass

    @abstractmethod
    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Restore complete state from checkpoint.

        This method is the inverse of get_full_state(). It restores ALL
        state components to resume training or inference from the exact
        point where the state was captured.

        Args:
            state: Dictionary returned by get_full_state() containing:
                - 'weights': Learnable parameters
                - 'region_state': Dynamic state (spikes, membrane, etc.)
                - 'learning_state': Learning rule internal state
                - 'oscillator_state': Oscillator phases (if applicable)
                - 'neuromodulator_state': Neuromodulator levels
                - 'config': Configuration (for validation)

        Raises:
            ValueError: If state is incompatible with current configuration
            KeyError: If required state components are missing

        Note:
            This method should validate that the loaded state is compatible
            with the current region configuration (e.g., matching dimensions).
            If config mismatch is detected, raise ValueError with details.

        Example:
            >>> state = torch.load("region_state.pt")
            >>> region.load_full_state(state)
            >>> # Region continues from exact state where checkpoint was saved
        """
        pass


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
