"""
Base classes for brain region modules.

This module defines the abstract interface that all brain regions implement,
ensuring consistent API while allowing specialized learning rules.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn

from thalia.core.component_config import NeuralComponentConfig
from thalia.core.component_protocol import BrainComponentBase, BrainComponentMixin
from thalia.core.neuromodulator_mixin import NeuromodulatorMixin
from thalia.core.diagnostics_mixin import DiagnosticsMixin
from thalia.learning.strategy_mixin import LearningStrategyMixin
from thalia.mixins.growth_mixin import GrowthMixin
from thalia.components.coding.spike_utils import compute_firing_rate
from thalia.diagnostics.health_monitor import HealthReport
from thalia.core.growth import CapacityMetrics


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


class NeuralComponent(BrainComponentBase, nn.Module, NeuromodulatorMixin, LearningStrategyMixin, DiagnosticsMixin, GrowthMixin, BrainComponentMixin):
    """Abstract base class for ALL neural components (regions, pathways, populations).

    This unified base class reflects a key biological insight:
    **There is no fundamental difference between "regions" and "pathways".**

    Both are populations of neurons with:
    - Synaptic connections (weights)
    - Membrane dynamics (potentials, spikes, traces)
    - Learning rules (STDP, BCM, three-factor, etc.)
    - Neuromodulation (dopamine, acetylcholine, norepinephrine)

    The distinction is organizational, not architectural:
    - **Regions**: Named functional populations (Cortex, Hippocampus, Striatum)
    - **Pathways**: Connection populations between regions (Cortex→Hippocampus)
    - **Components**: Generic term for any neural population

    DESIGN PHILOSOPHY
    =================
    Previously, we had separate hierarchies for regions and pathways, creating
    artificial distinctions and code duplication. Now unified:

    - LayeredCortex(NeuralComponent) - named functional unit
    - Striatum(NeuralComponent) - named functional unit
    - SpikingPathway(NeuralComponent) - inter-region connections
    - All inherit from same base, implement same protocol

    COMPONENT PROTOCOL ENFORCEMENT
    ==============================
    NeuralComponent now inherits from BrainComponentBase which enforces the
    complete BrainComponent protocol interface. This ensures:
    - All required methods are implemented (compile-time checking)
    - Component parity between regions and pathways
    - Consistent API across all neural components

    Inheritance order matters:
    1. BrainComponentBase - Enforces abstract interface
    2. nn.Module - Provides PyTorch functionality
    3. BrainComponentMixin - Provides default implementations
    4. NeuromodulatorMixin - Provides neuromodulation
    5. LearningStrategyMixin - Provides learning strategies
    6. DiagnosticsMixin - Provides diagnostics helpers

    All components use forward() for processing (standard PyTorch, ADR-007).

    See: src/thalia/core/component_protocol.py
         docs/patterns/component-parity.md
         docs/patterns/component-interface-enforcement.md
         docs/decisions/adr-007-pytorch-consistency.md

    CONTINUOUS PLASTICITY
    =====================
    Unlike traditional ML models with separate training/inference phases,
    neural components learn CONTINUOUSLY during forward passes.

    **Biological Basis**: In real brains, synaptic plasticity occurs during
    every spike pairing. There is no "training mode" vs "inference mode" -
    learning IS dynamics.

    **Neuromodulation**: Learning rate is dynamically modulated by:
    - **Dopamine**: High DA = consolidate patterns (reward), low DA = explore
    - **Acetylcholine**: High ACh = encoding mode, low ACh = retrieval mode
    - **Norepinephrine**: High NE = increase gain and flexibility

    The forward() method processes input AND applies plasticity at each timestep.
    There is no separate learn() method - this is intentional!

    NEUROMODULATION
    ===============
    Inherits from NeuromodulatorMixin which provides:
    - set_dopamine(), set_acetylcholine(), set_norepinephrine()
    - set_neuromodulators() - consolidated setter
    - decay_neuromodulators() with configurable tau constants
    - get_effective_learning_rate() for dopamine-modulated plasticity

    See NeuromodulatorMixin for full interface and usage examples.
    """

    def __init__(self, config: NeuralComponentConfig):
        """Initialize the brain region.

        Args:
            config: Configuration parameters for the region
        """
        # MUST call nn.Module.__init__() first before setting any attributes
        nn.Module.__init__(self)

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.get_torch_dtype()  # Get dtype from config

        # Initialize weights (may be deferred for composition-based regions)
        # Subclasses that use composition (like PredictiveCortex) can override
        # _initialize_weights() to delegate, and should set self.weights themselves
        # after calling super().__init__()
        weights = self._initialize_weights()
        if weights is not None:
            self.weights = weights

        # Initialize neurons (may also be deferred for composition-based regions)
        neurons = self._create_neurons()
        if neurons is not None:
            self.neurons = neurons

        # Initialize state
        self.state = NeuralComponentState()

        # =================================================================
        # CONTINUOUS PLASTICITY SETTINGS
        # =================================================================
        self.plasticity_enabled: bool = True       # Can be disabled for eval
        self.base_learning_rate: float = config.learning_rate

        # =================================================================
        # AXONAL DELAY BUFFER (ALL neural components have conduction delays)
        # =================================================================
        # This makes regions and pathways architecturally identical - the only
        # difference is configuration (e.g., regions use 1-2ms, long-range
        # pathways use 5-10ms).
        #
        # Biological justification:
        # - Within-region delays: 0.5-2ms (local axons)
        # - Inter-region delays: 1-10ms (long-range axons)
        # - Thalamo-cortical: 8-15ms
        # - Striato-cortical: 10-20ms
        #
        # Delay buffer is a circular buffer that stores spike history.
        # Forward pass writes current spikes, reads delayed spikes.
        #
        # NOTE: We don't initialize delay_buffer here to avoid conflicts with
        # subclasses that use register_buffer() (like SpikingPathway).
        # It will be initialized on first use in _apply_axonal_delay().
        self.axonal_delay_ms = config.axonal_delay_ms
        self.avg_delay_steps = int(self.axonal_delay_ms / config.dt_ms)
        self.max_delay_steps = max(1, int(self.axonal_delay_ms * 2 / config.dt_ms) + 1)
        # delay_buffer and delay_buffer_idx will be initialized on first forward() call

    @abstractmethod
    def _initialize_weights(self) -> Optional[torch.Tensor]:
        """Initialize the weight matrix for this region.

        Different regions may have different initialization strategies
        (e.g., sparse vs dense, structured vs random).

        Returns:
            Weights tensor, or None for composition-based regions that
            defer weight initialization until after super().__init__()
        """

    @abstractmethod
    def _create_neurons(self) -> Optional[Any]:
        """Create the neuron model for this region.

        Returns the neuron object(s) used in this region, or None for
        composition-based regions that defer neuron creation until after
        super().__init__().
        """

    @abstractmethod
    def forward(
        self,
        input_spikes: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Process input through the region AND apply continuous plasticity.

        This is the main computation method. Unlike traditional ML models,
        brain regions learn CONTINUOUSLY during forward passes. Plasticity
        is applied at each timestep, modulated by neuromodulators (dopamine).

        Args:
            input_spikes: Input spike tensor [n_input] (1D per ADR-005)
            **kwargs: Additional region-specific inputs

        Returns:
            Output spikes [n_output] (1D per ADR-005)

        Note:
            Theta modulation and timestep (dt_ms) are computed internally from
            self._theta_phase and self.config.dt_ms (both set by Brain)
        """

    def _reset_tensors(self, *tensor_names: str) -> None:
        """Helper to zero multiple tensors by name.

        Args:
            *tensor_names: Names of tensor attributes to zero

        Example:
            self._reset_tensors('eligibility', 'input_trace', 'output_trace')
        """
        for name in tensor_names:
            if hasattr(self, name):
                tensor = getattr(self, name)
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    tensor.zero_()

    def _reset_subsystems(self, *subsystem_names: str) -> None:
        """Helper to reset multiple subsystems that have reset_state() methods.

        Args:
            *subsystem_names: Names of subsystem attributes to reset

        Example:
            self._reset_subsystems('neurons', 'd1_neurons', 'd2_neurons', 'eligibility')
        """
        for name in subsystem_names:
            if hasattr(self, name):
                subsystem = getattr(self, name)
                if subsystem is not None and hasattr(subsystem, 'reset_state'):
                    subsystem.reset_state()

    def _reset_scalars(self, **scalar_values: Any) -> None:
        """Helper to reset scalar attributes to specified values.

        Args:
            **scalar_values: Mapping of attribute name to reset value

        Example:
            self._reset_scalars(last_action=None, exploring=False, timestep=0)
        """
        for name, value in scalar_values.items():
            if hasattr(self, name):
                setattr(self, name, value)

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

        Only use for initialization or after device changes, NOT during training.
        This method clears ALL transient state (membrane potentials, traces,
        spike history) while preserving learned weights.
        """
        self.state = NeuralComponentState()

        # Reset common subsystems using helper
        self._reset_subsystems('neurons')

        # Reset delay buffer tensors and scalars using helpers
        self._reset_tensors('delay_buffer')
        self._reset_scalars(delay_buffer_idx=0)

    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Optional[Dict[str, float]] = None,
        theta_slot: int = 0,
        coupled_amplitudes: Optional[Dict[str, float]] = None,
    ) -> None:
        """Set current oscillator phases and effective amplitudes from brain.

        Effective amplitudes are pre-computed by OscillatorManager with automatic
        multiplicative coupling. Each oscillator's amplitude reflects ALL
        phase-amplitude coupling effects.

        Regions use oscillator information for:
        - Phase-dependent gating (e.g., theta encoding vs retrieval)
        - Attention modulation (e.g., alpha suppression)
        - Motor preparation (e.g., beta synchrony)
        - Feature binding (e.g., gamma synchrony)
        - Sequence encoding (e.g., theta_slot for working memory)
        - Amplitude-gated learning (e.g., effective gamma amplitude)

        Called every timestep, similar to set_dopamine.
        Default implementation stores phases but doesn't use them.
        Subclasses override to implement oscillator-dependent behavior.

        Args:
            phases: Dict mapping oscillator name to phase [0, 2π)
            signals: Dict mapping oscillator name to signal [-1, 1]
            theta_slot: Current theta slot [0, n_slots-1] for sequence encoding
            coupled_amplitudes: Effective amplitudes per oscillator (pre-computed)
                               {'delta': 1.0, 'theta': 0.73, 'gamma': 0.48, ...}
                               Values reflect automatic multiplicative coupling.

        Example:
            # In hippocampus, use theta slot for sequence encoding:
            slot = theta_slot  # 0-6, maps to gamma cycle

            # Use effective amplitude for gating:
            gamma_amp = coupled_amplitudes.get('gamma', 1.0)  # Pre-computed!
            excitability = gamma_amp * base_excitability
        """
        # Default: store but don't require all regions to use
        if not hasattr(self.state, '_oscillator_phases'):
            self.state._oscillator_phases = {}
            self.state._oscillator_signals = {}
        self.state._oscillator_phases = phases
        self.state._oscillator_signals = signals

    def get_weights(self) -> torch.Tensor:
        """Return the current weight matrix."""
        return self.weights.detach().clone()

    def set_weights(self, weights: torch.Tensor) -> None:
        """Set the weight matrix."""
        self.weights = weights.to(self.device)

    def _initialize_delay_buffer(self, n_neurons: int) -> None:
        """Initialize the axonal delay buffer.

        This should be called after n_output is known (typically in subclass __init__
        after weights are created).

        Args:
            n_neurons: Number of output neurons (size of delay buffer)
        """
        if not hasattr(self, 'delay_buffer') or self.delay_buffer is None:
            self.delay_buffer = torch.zeros(
                self.max_delay_steps,
                n_neurons,
                device=self.device,
                dtype=torch.bool,  # Use bool for spike storage (ADR-004)
            )
        if not hasattr(self, 'delay_buffer_idx'):
            self.delay_buffer_idx = 0

    def _apply_axonal_delay(
        self,
        output_spikes: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply axonal delay to output spikes using circular buffer.

        This implements the biological reality that ALL neural connections
        have conduction delays. Regions and pathways use the same mechanism,
        differing only in delay_ms configuration.

        Args:
            output_spikes: Immediate output from neurons [n_output]
            dt: Timestep in milliseconds

        Returns:
            delayed_spikes: Spikes from delay_ms ago [n_output]

        Usage in forward():
            immediate_spikes = self.neurons(synaptic_input)
            delayed_spikes = self._apply_axonal_delay(immediate_spikes, dt)
            return delayed_spikes
        """
        # Initialize buffer if needed (only if not already initialized by subclass)
        if not hasattr(self, 'delay_buffer') or self.delay_buffer is None:
            self._initialize_delay_buffer(output_spikes.shape[0])

        # Store current output in delay buffer
        self.delay_buffer[self.delay_buffer_idx] = output_spikes

        # Retrieve delayed spikes
        delayed_idx = (self.delay_buffer_idx - self.avg_delay_steps) % self.delay_buffer.shape[0]
        delayed_spikes = self.delay_buffer[delayed_idx]

        # Advance buffer index
        self.delay_buffer_idx = (self.delay_buffer_idx + 1) % self.delay_buffer.shape[0]

        return delayed_spikes

    @property
    def n_input(self) -> int:
        return self.config.n_input

    @property
    def n_output(self) -> int:
        return self.config.n_output

    @property
    def device(self) -> torch.device:
        """Device where tensors are stored (CPU or CUDA)."""
        return self._device if hasattr(self, '_device') else torch.device('cpu')

    @device.setter
    def device(self, value: torch.device) -> None:
        """Set device (allows assignment from __init__)."""
        self._device = value

    @property
    def dtype(self) -> torch.dtype:
        """Data type for floating point tensors."""
        return self._dtype if hasattr(self, '_dtype') else torch.float32

    @dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        """Set dtype (allows assignment from __init__)."""
        self._dtype = value

    # Growth methods (add_neurons) are provided by GrowthMixin
    # See src/thalia/mixins/growth_mixin.py for template method and helpers

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
            diagnostics['firing_rate'] = float(compute_firing_rate(spikes))
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
        from thalia.diagnostics.health_monitor import (
            HealthReport, IssueReport, HealthIssue, IssueSeverity
        )

        issues = []

        # Check firing rate
        if self.state.spikes is not None:
            firing_rate = float(compute_firing_rate(self.state.spikes))

            if firing_rate < 0.01:  # Less than 1%
                issues.append(IssueReport(
                    issue_type=HealthIssue.ACTIVITY_COLLAPSE,
                    severity=IssueSeverity.HIGH.value,
                    description=f'Firing rate too low: {firing_rate:.1%}',
                    recommendation='Check input strength, reduce thresholds, or increase excitation'
                ))
            elif firing_rate > 0.90:  # More than 90%
                issues.append(IssueReport(
                    issue_type=HealthIssue.SEIZURE_RISK,
                    severity=IssueSeverity.HIGH.value,
                    description=f'Firing rate too high: {firing_rate:.1%}',
                    recommendation='Increase inhibition, increase thresholds, or reduce input strength'
                ))

        # Check weight saturation
        if hasattr(self, 'weights'):
            w = self.weights.detach()
            near_max = (w > self.config.w_max * 0.95).float().mean().item()
            near_min = (w < self.config.w_min + 0.05).float().mean().item()

            if near_max > 0.5:
                issues.append(IssueReport(
                    issue_type=HealthIssue.WEIGHT_EXPLOSION,
                    severity=IssueSeverity.MEDIUM.value,
                    description=f'Weight saturation at maximum: {near_max:.1%} near max',
                    recommendation='Consider synaptic scaling or weight normalization'
                ))
            elif near_min > 0.5:
                issues.append(IssueReport(
                    issue_type=HealthIssue.WEIGHT_COLLAPSE,
                    severity=IssueSeverity.MEDIUM.value,
                    description=f'Weight saturation at minimum: {near_min:.1%} near min',
                    recommendation='Consider increasing learning rate or input strength'
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

        2. **Component State** (dynamic state from NeuralComponentState):
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
            - 'component_state': Dict[str, Any] - Current NeuralComponentState data
            - 'learning_state': Dict[str, Any] - Learning rule internal state
            - 'oscillator_state': Dict[str, Any] - Oscillator phases/state (if applicable)
            - 'neuromodulator_state': Dict[str, float] - Neuromodulator levels
            - 'config': NeuralComponentConfig - Configuration (for validation on load)

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

    # =========================================================================
    # GROWTH HELPERS - Provided by GrowthMixin
    # =========================================================================
    # Growth utilities (_expand_weights, _expand_state_tensors,
    # _recreate_neurons_with_state, and template method add_neurons) are now
    # provided by GrowthMixin. See src/thalia/mixins/growth_mixin.py.
    #
    # Multi-layer regions (Hippocampus, LayeredCortex) override add_neurons()
    # entirely but use the helper methods for weight/state expansion.

    @abstractmethod
    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Restore complete state from checkpoint.

        This method is the inverse of get_full_state(). It restores ALL
        state components to resume training or inference from the exact
        point where the state was captured.

        Args:
            state: Dictionary returned by get_full_state() containing:
                - 'weights': Learnable parameters
                - 'component_state': Dynamic state (spikes, membrane, etc.)
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
