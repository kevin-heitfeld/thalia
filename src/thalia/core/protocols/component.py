"""
Unified protocol for brain components (regions and pathways).

This module defines the common interface that both brain regions and neural
pathways implement, ensuring feature parity and preventing oversight.

Design Philosophy
=================
Brain regions and pathways are EQUALLY IMPORTANT:
- Both process information (forward() - standard PyTorch convention, ADR-007)
- Both learn continuously during forward passes
- Both maintain temporal state
- Both need growth for curriculum learning
- Both need diagnostics and health checks
- Both need checkpoint compatibility

By having a unified protocol, we ensure:
1. Feature parity: New features added to regions MUST be added to pathways
2. Consistent API: Code that works with regions works with pathways
3. Type safety: Static checkers catch missing implementations
4. Documentation: Explicit that pathways = first-class citizens

Author: Thalia Team
Date: December 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, runtime_checkable, Optional

import torch
import torch.nn as nn

from thalia.core.component_state import NeuralComponentState
from thalia.neuromodulation.mixin import NeuromodulatorMixin
from thalia.learning.strategy_mixin import LearningStrategyMixin
from thalia.mixins.diagnostics_mixin import DiagnosticsMixin
from thalia.mixins.growth_mixin import GrowthMixin
from thalia.mixins.resettable_mixin import ResettableMixin


@runtime_checkable
class BrainComponent(Protocol):
    """
    Unified protocol for all brain components (regions AND pathways).

    **CRITICAL**: Pathways are just as important as regions!

    Both regions and pathways implement this interface to ensure feature parity.
    When adding new functionality:
    1. Add it to this protocol first
    2. Implement for NeuralRegion (regions)
    3. Implement for BaseNeuralPathway (pathways)
    4. Update tests for both

    This prevents accidentally forgetting pathways when adding features.

    Core Capabilities
    =================
    All brain components must support:

    1. **Information Processing** (ADR-007):
       - forward(): Transform inputs to outputs (standard PyTorch convention)
       - Enables callable syntax: component(input)
       - Continuous learning during forward passes (no separate learn())

    2. **State Management**:
       - reset_state(): Clear temporal dynamics
       - Maintain membrane potentials, spike history, traces

    3. **Growth** (Curriculum Learning):
       - grow_input(): Expand input dimension (pathways only)
       - grow_output(): Expand output dimension (regions and pathways)
       - get_capacity_metrics(): Report utilization for growth decisions

    4. **Diagnostics**:
       - get_diagnostics(): Report activity, health, learning metrics
       - check_health(): Detect pathologies (silence, runaway activity, etc.)

    5. **Checkpointing**:
       - get_full_state(): Serialize all state (weights, config, history)
       - load_full_state(): Restore from checkpoint

    Why Pathways Matter
    ===================
    Pathways are NOT just "glue" between regions. They are active components that:
    - Learn continuously via STDP, BCM, or other plasticity rules
    - Transform and filter information between regions
    - Implement attention, gating, and routing
    - Need to grow when connected regions grow
    - Can become pathological (silent, saturated, etc.)

    Forgetting to implement features for pathways breaks curriculum learning
    and prevents coordinated system-wide adaptation.

    Usage Example
    =============
    The protocol enables uniform handling of both regions and pathways:

    .. code-block:: python

        def analyze_and_adapt_component(component: BrainComponent) -> Dict[str, Any]:
            '''Analyze any brain component and adapt if needed.'''
            # Check for pathologies
            health = component.check_health()
            if not health.is_healthy:
                print(f"Component issues: {health.issues}")

            # Check if capacity is saturated
            metrics = component.get_capacity_metrics()
            if metrics.growth_recommended:
                print(f"Growing: {metrics.growth_reason}")
                component.grow_output(n_new=100)

            # Return diagnostics
            return component.get_diagnostics()

        # Works identically for regions AND pathways
        cortex_diag = analyze_and_adapt_component(cortex)
        pathway_diag = analyze_and_adapt_component(cortex_to_hippo_pathway)
    """

    # =========================================================================
    # Core Processing
    # =========================================================================

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Process input and update state (standard PyTorch convention).

        **Standard PyTorch Method** (ADR-007):
        All components use forward() to enable callable syntax:
        >>> output = component(input)  # Calls forward() automatically

        Signatures:
        - **Regions**: forward(spikes: Tensor) -> Tensor
        - **Pathways**: forward(spikes: Tensor) -> Tensor
        - **Sensory pathways**: forward(raw_input) -> (Tensor, Dict[str, Any])

        **Learning happens automatically during forward passes** - there is no
        separate learn() method. Plasticity is always active, modulated by
        neuromodulators (dopamine, acetylcholine, etc.).

        Returns:
            Output spikes or activations (region/pathway specific)
        """

    # =========================================================================
    # State Management
    # =========================================================================

    @abstractmethod
    def reset_state(self) -> None:
        """
        Reset temporal state to initial conditions.

        Clears:
        - Membrane potentials
        - Spike history
        - Eligibility traces
        - Short-term plasticity variables
        - Timestep counter

        Does NOT reset:
        - Learned weights
        - Long-term configuration
        - Growth history
        """

    # =========================================================================
    # Neuromodulation & Oscillators
    # =========================================================================

    @abstractmethod
    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Dict[str, float] | None = None,
        theta_slot: int = 0,
        coupled_amplitudes: Dict[str, float] | None = None,
    ) -> None:
        """
        Receive oscillator phases and amplitudes from brain broadcast.

        Brain oscillations coordinate activity across regions and pathways:
        - **Delta (0.5-4 Hz)**: Deep sleep, large-scale synchronization
        - **Theta (4-10 Hz)**: Memory encoding, spatial navigation, sequence learning
        - **Alpha (8-13 Hz)**: Attention gating, inhibitory control
        - **Beta (13-30 Hz)**: Motor control, active cognitive processing
        - **Gamma (30-100 Hz)**: Feature binding, local circuit processing

        Components use oscillators for:
        - **Phase-dependent gating**: Theta phase determines encode vs retrieve
        - **Attention modulation**: Alpha power controls sensory suppression
        - **Motor preparation**: Beta synchrony coordinates motor planning
        - **Feature binding**: Gamma synchrony links distributed representations
        - **Transmission efficiency**: Pathways gate information flow by oscillatory state

        Called every timestep by Brain (centralized broadcast, like neuromodulators).
        Default implementation stores phases; components override for specialized behavior.

        Args:
            phases: Oscillator phases in radians [0, 2π)
                   Example: {'delta': 1.2, 'theta': 3.4, 'alpha': 0.5}
            signals: Oscillator signal values in [-1, 1] (sin/cos waveforms)
                    Example: {'delta': 0.8, 'theta': -0.3}
            theta_slot: Current theta slot index [0, n_slots-1] for sequence encoding.
                       Theta cycle is divided into slots for temporal coding.
            coupled_amplitudes: Effective amplitudes per oscillator (pre-computed)
                               Example: {'delta': 1.0, 'theta': 0.73, 'gamma': 0.48}
                               Values incorporate multiplicative coupling between bands.

        Example - Hippocampus theta-phase encoding:

        .. code-block:: python

            theta_phase = phases['theta']
            is_encoding = 0 <= theta_phase < np.pi  # First half = encoding
            gamma_amp = coupled_amplitudes.get('gamma', 1.0)
            learning_rate = gamma_amp * self.base_learning_rate

        Example - Attention pathway beta modulation:

        .. code-block:: python

            beta_amp = coupled_amplitudes.get('beta', 1.0)
            attention_strength = self.attention_level  # From PFC
            transmission_gain = 1.0 + (beta_amp - 1.0) * attention_strength
        """

    # =========================================================================
    # Growth (Curriculum Learning) - Unified API
    # =========================================================================

    @abstractmethod
    def grow_input(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """
        Grow component's input dimension without disrupting existing circuits.

        **CRITICAL for curriculum learning**: As upstream components grow, this
        component must adapt to accept more input connections.

        For regions:
        - Expands input weights [n_output, n_input] → [n_output, n_input + n_new]
        - New synapses start with sparse random connections
        - Existing weights unchanged

        For pathways:
        - Expands source dimension to match upstream region growth
        - Maintains connectivity when source region grows
        - Preserves existing connection strengths

        Args:
            n_new: Number of input neurons/units to add
            initialization: Weight init strategy ('sparse_random', 'xavier', etc.)
            sparsity: Connection sparsity for new neurons (0.0 = none, 1.0 = all)

        Example:
            >>> # Upstream region grows
            >>> cortex.grow_output(20)
            >>> # Downstream components must adapt
            >>> hippocampus.grow_input(20)
            >>> cortex_to_hippo_pathway.grow_input(20)

        Raises:
            NotImplementedError: If component doesn't support input growth yet
        """

    # === GROWTH METHODS ===
    @abstractmethod
    def grow_output(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Grow output dimension by adding neurons.

        Called when this component needs to produce more outputs.
        This adds neurons to the component's output population.

        Args:
            n_new: Number of output neurons/dimensions to add
            initialization: Weight initialization strategy for new connections
            sparsity: Connection sparsity for sparse random initialization

        Effects:
            - Expands output-related weight matrices (adds rows)
            - Adds new neurons to neuron population
            - Expands output-side state tensors (membrane, traces, etc.)
            - Updates config.n_output

        **CRITICAL for curriculum learning**: Growing must preserve existing knowledge.
        Existing neurons and weights remain unchanged

        Args:
            n_new: Number of output neurons/units to add
            initialization: Weight init strategy ('sparse_random', 'xavier', etc.')
            sparsity: Connection sparsity for new neurons (0.0 = none, 1.0 = all)

        Example:
            >>> # Region grows its neuron population
            >>> hippocampus.grow_output(15)
            >>> # Downstream components must adapt
            >>> hippo_to_cortex_pathway.grow_output(15)

        Raises:
            NotImplementedError: If component doesn't support output growth yet
        """

    @abstractmethod
    def get_capacity_metrics(self) -> Any:  # Returns CapacityMetrics
        """
        Compute utilization metrics to guide growth decisions.

        Analyzes:
        - Firing rate: Are neurons saturating? (>90% bad)
        - Weight saturation: Are weights hitting limits? (>85% bad)
        - Synapse usage: Are connections active? (<5% bad)

        Returns:
            CapacityMetrics with growth_recommended flag and reason

        Example:
            metrics = component.get_capacity_metrics()
            if metrics.growth_recommended:
                print(f"Growing: {metrics.growth_reason}")
                component.grow_output(n_new=100)
        """

    # =========================================================================
    # Diagnostics
    # =========================================================================

    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get current activity and health metrics.

        Returns dictionary with component-specific diagnostics:
        - Firing rates
        - Weight statistics
        - Learning activity (LTP/LTD)
        - Neuromodulator levels (regions)
        - Input/output statistics

        Used for:
        - Monitoring training progress
        - Detecting pathologies
        - Logging to tensorboard/wandb
        """

    @abstractmethod
    def check_health(self) -> Any:  # Returns HealthReport
        """
        Check for pathological states.

        Detects:
        - Silence: Firing rate too low (<1%)
        - Runaway activity: Firing rate too high (>90%)
        - Dead neurons: No activity for extended period
        - Weight saturation: Weights stuck at limits
        - NaN/Inf values: Numerical instability

        Returns:
            HealthReport with is_healthy flag and list of issues

        Example:
            health = component.check_health()
            if not health.is_healthy:
                print(f"Issues: {health.issues}")
        """

    # =========================================================================
    # Checkpointing
    # =========================================================================

    @abstractmethod
    def get_full_state(self) -> Dict[str, Any]:
        """
        Serialize complete component state for checkpointing.

        Includes:
        - Learned weights (all matrices)
        - Configuration
        - Growth history
        - Component metadata (type, version, etc.)

        Does NOT include:
        - Temporary simulation state (membrane potentials, etc.)
        - Optimizer state (handled separately)

        Returns:
            Dictionary that can be saved to disk and restored later
        """

    @abstractmethod
    def load_full_state(self, state: Dict[str, Any]) -> None:
        """
        Restore component from checkpoint.

        Args:
            state: Dictionary from get_full_state()

        Validates:
        - State version compatibility
        - Tensor shapes match current architecture
        - Required keys present

        After loading, component should be functionally identical to
        when checkpoint was saved.
        """

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device where tensors are stored (CPU or CUDA)."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Data type for floating point tensors."""


# =============================================================================
# Abstract Base Class (Enforces Protocol)
# =============================================================================

class BrainComponentBase(ABC):
    """
    Abstract base class enforcing the BrainComponent protocol.

    **ALL regions and pathways MUST inherit from this class.**

    This enforces implementation of the complete BrainComponent interface at
    compile time, preventing missing methods and ensuring component parity.

    Why Enforce with ABC?
    =====================
    The BrainComponent Protocol (above) defines the interface but doesn't enforce
    it until runtime. This ABC provides:
    - **Static checking**: Missing methods caught by IDEs and type checkers
    - **Clear errors**: Python raises TypeError if abstract methods not implemented
    - **Documentation**: Explicit which methods are required vs optional

    Migration Path
    ==============
    For existing components that don't implement all methods yet:
    1. Inherit from BrainComponentBase
    2. Python will raise TypeError listing missing methods
    3. Implement missing methods (can use defaults from BrainComponentMixin)
    4. Ensure all tests pass

    See: docs/patterns/component-interface-enforcement.md for migration guide
    """

    # =========================================================================
    # Core Processing (REQUIRED)
    # =========================================================================

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Process input and update state (standard PyTorch convention).

        REQUIRED for all components. Must be implemented by all subclasses.
        See BrainComponent protocol for full documentation.
        """

    # =========================================================================
    # State Management (REQUIRED)
    # =========================================================================

    @abstractmethod
    def reset_state(self) -> None:
        """
        Reset temporal state to initial conditions.

        REQUIRED for all components. Must be implemented by all subclasses.
        See BrainComponent protocol for full documentation.
        """

    # =========================================================================
    # Neuromodulation & Oscillators (REQUIRED)
    # =========================================================================

    @abstractmethod
    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Dict[str, float] | None = None,
        theta_slot: int = 0,
        coupled_amplitudes: Dict[str, float] | None = None,
    ) -> None:
        """
        Receive oscillator phases and amplitudes from brain broadcast.

        REQUIRED for all components. Default implementation available in
        BrainComponentMixin. See BrainComponent protocol for full documentation.
        """

    # =========================================================================
    # Growth (REQUIRED for curriculum learning) - Unified API
    # =========================================================================

    @abstractmethod
    def grow_input(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """
        Grow component's input dimension without disrupting existing circuits.

        REQUIRED for all components. Must be implemented by all subclasses.
        See BrainComponent protocol for full documentation.
        """

    @abstractmethod
    def grow_output(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """
        Grow component's output dimension without disrupting existing circuits.

        REQUIRED for all components. Must be implemented by all subclasses.
        See BrainComponent protocol for full documentation.
        """

    @abstractmethod
    def get_capacity_metrics(self) -> Any:
        """
        Compute utilization metrics to guide growth decisions.

        REQUIRED for all components. Default implementation available in
        BrainComponentMixin. See BrainComponent protocol for full documentation.
        """

    # =========================================================================
    # Diagnostics (REQUIRED)
    # =========================================================================

    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get current activity and health metrics.

        REQUIRED for all components. Must be implemented by all subclasses.
        See BrainComponent protocol for full documentation.
        """

    @abstractmethod
    def check_health(self) -> Any:
        """
        Check for pathological states.

        REQUIRED for all components. Default implementation available in
        BrainComponentMixin. See BrainComponent protocol for full documentation.
        """

    # =========================================================================
    # Checkpointing (REQUIRED)
    # =========================================================================

    @abstractmethod
    def get_full_state(self) -> Dict[str, Any]:
        """
        Serialize complete component state for checkpointing.

        REQUIRED for all components. Must be implemented by all subclasses.
        See BrainComponent protocol for full documentation.
        """

    @abstractmethod
    def load_full_state(self, state: Dict[str, Any]) -> None:
        """
        Restore component from checkpoint.

        REQUIRED for all components. Must be implemented by all subclasses.
        See BrainComponent protocol for full documentation.
        """

    # =========================================================================
    # Properties (REQUIRED)
    # =========================================================================

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device where tensors are stored (CPU or CUDA). REQUIRED."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Data type for floating point tensors. REQUIRED."""


# =============================================================================
# Mixin with Default Implementations
# =============================================================================

class BrainComponentMixin:
    """
    Mixin providing default implementations of BrainComponent methods.

    Regions and pathways can inherit this to get:
    - Standardized error messages
    - Common utility methods
    - Consistent behavior

    Subclasses override specific methods as needed.

    Usage:
        class MyRegion(BrainComponentBase, nn.Module, BrainComponentMixin):
            # BrainComponentBase enforces interface
            # nn.Module provides PyTorch functionality
            # BrainComponentMixin provides default implementations
            pass
    """

    def grow_input(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Default: raise NotImplementedError with helpful message."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.grow_input() not yet implemented. "
            f"Input growth is essential for curriculum learning when upstream components grow. "
            f"See docs/architecture/UNIFIED_GROWTH_API.md for implementation guide."
        )

    def grow_output(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Default: raise NotImplementedError with helpful message."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.grow_output() not yet implemented. "
            f"Output growth is essential for curriculum learning to expand capacity. "
            f"See docs/architecture/UNIFIED_GROWTH_API.md for implementation guide."
        )

    def get_capacity_metrics(self) -> Any:
        """Default: use GrowthManager to compute metrics."""
        from thalia.coordination.growth import GrowthManager

        # Use component name if available, otherwise class name
        name = getattr(self, 'name', self.__class__.__name__)
        manager = GrowthManager(region_name=name)
        return manager.get_capacity_metrics(self)

    def check_health(self) -> Any:
        """Default: return healthy status with no issues."""
        from thalia.diagnostics.health_monitor import HealthReport
        return HealthReport(
            is_healthy=True,
            overall_severity=0.0,
            issues=[],
            summary="Component is healthy",
            metrics={},
        )

    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Dict[str, float] | None = None,
        theta_slot: int = 0,
        coupled_amplitudes: Dict[str, float] | None = None,
    ) -> None:
        """Default: store oscillator info but don't require usage."""
        # Store in a standard location that subclasses can access
        if not hasattr(self, '_oscillator_phases'):
            self._oscillator_phases: Dict[str, float] = {}
            self._oscillator_signals: Dict[str, float] = {}
            self._oscillator_theta_slot: int = 0
            self._coupled_amplitudes: Dict[str, float] = {}

        self._oscillator_phases = phases
        self._oscillator_signals = signals or {}
        self._oscillator_theta_slot = theta_slot
        self._coupled_amplitudes = coupled_amplitudes or {}

    # =========================================================================
    # Convenience properties for accessing individual oscillator phases
    # =========================================================================
    # These eliminate the need for regions to repeatedly use .get() with defaults

    @property
    def _theta_phase(self) -> float:
        """Current theta phase in radians [0, 2π)."""
        return getattr(self, '_oscillator_phases', {}).get('theta', 0.0)

    @_theta_phase.setter
    def _theta_phase(self, value: float) -> None:
        """Allow regions to set theta phase directly (backward compatibility)."""
        if not hasattr(self, '_oscillator_phases'):
            self._oscillator_phases = {}
        self._oscillator_phases['theta'] = value

    @property
    def _gamma_phase(self) -> float:
        """Current gamma phase in radians [0, 2π)."""
        return getattr(self, '_oscillator_phases', {}).get('gamma', 0.0)

    @_gamma_phase.setter
    def _gamma_phase(self, value: float) -> None:
        """Allow regions to set gamma phase directly (backward compatibility)."""
        if not hasattr(self, '_oscillator_phases'):
            self._oscillator_phases = {}
        self._oscillator_phases['gamma'] = value

    @property
    def _alpha_phase(self) -> float:
        """Current alpha phase in radians [0, 2π)."""
        return getattr(self, '_oscillator_phases', {}).get('alpha', 0.0)

    @_alpha_phase.setter
    def _alpha_phase(self, value: float) -> None:
        """Allow regions to set alpha phase directly (backward compatibility)."""
        if not hasattr(self, '_oscillator_phases'):
            self._oscillator_phases = {}
        self._oscillator_phases['alpha'] = value

    @property
    def _beta_phase(self) -> float:
        """Current beta phase in radians [0, 2π)."""
        return getattr(self, '_oscillator_phases', {}).get('beta', 0.0)

    @_beta_phase.setter
    def _beta_phase(self, value: float) -> None:
        """Allow regions to set beta phase directly (backward compatibility)."""
        if not hasattr(self, '_oscillator_phases'):
            self._oscillator_phases = {}
        self._oscillator_phases['beta'] = value

    @property
    def _delta_phase(self) -> float:
        """Current delta phase in radians [0, 2π)."""
        return getattr(self, '_oscillator_phases', {}).get('delta', 0.0)

    @_delta_phase.setter
    def _delta_phase(self, value: float) -> None:
        """Allow regions to set delta phase directly (backward compatibility)."""
        if not hasattr(self, '_oscillator_phases'):
            self._oscillator_phases = {}
        self._oscillator_phases['delta'] = value

    @property
    def _gamma_amplitude_effective(self) -> float:
        """Effective gamma amplitude (with cross-frequency coupling)."""
        return getattr(self, '_coupled_amplitudes', {}).get('gamma', 1.0)

    @property
    def _beta_amplitude_effective(self) -> float:
        """Effective beta amplitude (with cross-frequency coupling)."""
        return getattr(self, '_coupled_amplitudes', {}).get('beta', 1.0)


# =============================================================================
# Component Type Hierarchy (v2.0 Architecture)
# =============================================================================

class LearnableComponent(BrainComponentBase, nn.Module, NeuromodulatorMixin, LearningStrategyMixin, DiagnosticsMixin, GrowthMixin, ResettableMixin):
    """
    Complete base class for all learnable neural components (regions and weighted pathways).

    LearnableComponent provides EVERYTHING needed for biologically-plausible learning:
    - **Synaptic weights**: Learnable connection strengths
    - **Neurons**: Spiking neuron models with membrane dynamics
    - **Plasticity**: Local learning rules (STDP, BCM, three-factor, etc.)
    - **Neuromodulation**: Dopamine, acetylcholine, norepinephrine modulation
    - **State management**: Membrane potentials, spikes, eligibility traces
    - **Growth**: Dynamic expansion during curriculum learning
    - **Diagnostics**: Health monitoring and activity metrics

    This is the base class for custom pathways and legacy components.
    Brain regions now use NeuralRegion (v3.0 architecture).

    Examples:
    - Brain regions (v3.0): `class Striatum(NeuralRegion)`  [PREFERRED]
    - Standard pathways: AxonalProjection (no learnable params)  [V3.0 DEFAULT]
    - Custom weighted pathways: `class MyPathway(LearnableComponent)`  [ADVANCED USE]

    **Note**: Weighted pathways are no longer the default. V3.0 uses AxonalProjection
    (spike routing) with synaptic weights stored at target regions. LearnableComponent
    remains available for custom pathway implementations with learned routing.

    Key Principle: Regions use NeuralRegion, custom pathways can use LearnableComponent.

    Usage (for custom pathways only):
        class MyPathway(LearnableComponent):
            def __init__(self, config):
                super().__init__(config)
                # Weights and neurons already initialized via _initialize_weights/_create_neurons

            def _initialize_weights(self) -> nn.Parameter:
                return nn.Parameter(torch.randn(self.n_output, self.n_input))

            def _create_neurons(self) -> Any:
                from thalia.neurons import ConductanceLIF
                return ConductanceLIF(self.n_output, ...)

            def forward(self, input_spikes):
                # Process + learn in one pass (continuous plasticity)
                current = input_spikes @ self.weights.T
                output_spikes = self.neurons(current)
                if self.plasticity_enabled:
                    self._apply_learning(input_spikes, output_spikes)
                return output_spikes

            def reset_state(self):
                self.neurons.reset()
                self.state = NeuralComponentState()
    """

    def __init__(self, config: Any):
        """Initialize learnable neural component.

        Args:
            config: NeuralComponentConfig with device, dtype, learning_rate, etc.
        """
        # Initialize nn.Module first (MUST be before any attribute assignment)
        nn.Module.__init__(self)

        # Store configuration
        self.config = config
        self._device = torch.device(config.device)
        self._dtype = config.get_torch_dtype()

        # Initialize weights (subclasses implement _initialize_weights)
        weights = self._initialize_weights()
        if weights is not None:
            self.weights = weights

            # =============================================================
            # SPILLOVER TRANSMISSION (optional weight augmentation)
            # =============================================================
            # Apply spillover if enabled - augments weights with volume transmission
            # W_effective = W_direct + W_spillover
            # Computed once at init, zero cost during forward pass
            if getattr(config, 'enable_spillover', False):
                from thalia.synapses.spillover import SpilloverTransmission, SpilloverConfig

                spillover_config = SpilloverConfig(
                    enabled=True,
                    strength=getattr(config, 'spillover_strength', 0.15),
                    mode=getattr(config, 'spillover_mode', 'connectivity'),
                    lateral_radius=getattr(config, 'spillover_lateral_radius', 3),
                    similarity_threshold=getattr(config, 'spillover_similarity_threshold', 0.5),
                    normalize=getattr(config, 'spillover_normalize', True),
                )

                self.spillover = SpilloverTransmission(
                    self.weights.data,
                    spillover_config,
                    device=self._device,
                )

                # Replace direct weights with effective weights (direct + spillover)
                self.weights.data = self.spillover.get_effective_weights()
            else:
                self.spillover = None

        # Initialize neurons (subclasses implement _create_neurons)
        neurons = self._create_neurons()
        if neurons is not None:
            self.neurons = neurons

        # =================================================================
        # NEURAL COMPONENT STATE
        # =================================================================
        self.state = NeuralComponentState()

        # =================================================================
        # AFFERENT SYNAPSES (v2.0 Architecture - Optional)
        # =================================================================
        # In v2.0 architecture, regions can own their afferent synapses
        # (weights + learning) instead of having pathways with weights.
        self.afferent_synapses: Optional[Any] = None
        if getattr(config, 'use_afferent_synapses', False):
            self._init_afferent_synapses()

        # =================================================================
        # CONTINUOUS PLASTICITY SETTINGS
        # =================================================================
        self.plasticity_enabled: bool = True       # Can be disabled for eval
        self.base_learning_rate: float = config.learning_rate

        # =================================================================
        # AXONAL DELAY BUFFER (ALL neural components have conduction delays)
        # =================================================================
        # Delay buffer initialized on first forward() to avoid conflicts
        # with subclasses that use register_buffer()
        self.axonal_delay_ms = config.axonal_delay_ms
        self.avg_delay_steps = int(self.axonal_delay_ms / config.dt_ms)
        self.max_delay_steps = max(1, int(self.axonal_delay_ms * 2 / config.dt_ms) + 1)

    @abstractmethod
    def _initialize_weights(self) -> Optional[nn.Parameter]:
        """Initialize synaptic weights.

        Returns:
            nn.Parameter with weights, or None if composition-based (weights set later)
        """

    @abstractmethod
    def _create_neurons(self) -> Optional[Any]:
        """Create neuron models.

        Returns:
            Neuron object (e.g., ConductanceLIF), or None if composition-based
        """

    def _init_afferent_synapses(self) -> None:
        """Initialize afferent synapses layer (v2.0 architecture).

        Subclasses can override to customize synaptic configuration.
        """
        from thalia.synapses import AfferentSynapses, AfferentSynapsesConfig

        synapses_config = AfferentSynapsesConfig(
            n_neurons=self.config.n_neurons,
            n_inputs=self.config.n_input,
            learning_rule=getattr(self.config, 'learning_rule', 'hebbian'),
            learning_rate=self.config.learning_rate,
            device=self.config.device,
        )
        self.afferent_synapses = AfferentSynapses(synapses_config)

    @property
    def device(self) -> torch.device:
        """Device where tensors are stored."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Data type for floating point tensors."""
        return self._dtype

    @property
    def n_input(self) -> int:
        """Number of input connections (convenience accessor)."""
        if hasattr(self.config, 'n_input'):
            return self.config.n_input
        else:
            raise AttributeError(f"{self.__class__.__name__} config has no n_input attribute")

    @property
    def n_output(self) -> int:
        """Number of output neurons (convenience accessor)."""
        if hasattr(self.config, 'n_output'):
            return self.config.n_output
        elif hasattr(self.config, 'n_neurons'):
            return self.config.n_neurons
        else:
            raise AttributeError(f"{self.__class__.__name__} config has no n_output or n_neurons attribute")

    # =========================================================================
    # CONCRETE IMPLEMENTATIONS (shared by all components)
    # =========================================================================

    def check_health(self) -> Any:  # Returns HealthReport
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
        from thalia.components.coding.spike_utils import compute_firing_rate

        issues = []

        # Check firing rate
        if self.state.spikes is not None:
            firing_rate = compute_firing_rate(self.state.spikes)

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
        if hasattr(self, 'weights') and self.weights is not None:
            w = self.weights.detach()
            w_max = getattr(self.config, 'w_max', 1.0)
            w_min = getattr(self.config, 'w_min', 0.0)
            near_max = (w > w_max * 0.95).float().mean().item()
            near_min = (w < w_min + 0.05).float().mean().item()

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

    def get_capacity_metrics(self) -> Any:  # Returns CapacityMetrics
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
        from thalia.coordination.growth import GrowthManager

        # Use GrowthManager for standard metrics computation
        manager = GrowthManager(region_name=self.__class__.__name__)
        metrics = manager.get_capacity_metrics(self)
        return metrics  # Return CapacityMetrics object, not dict

    # =========================================================================
    # AXONAL DELAY HELPERS (ADR-010)
    # =========================================================================

    def _initialize_delay_buffer(self, n_neurons: int) -> None:
        """Initialize circular buffer for axonal delays.

        Called lazily on first forward() to avoid conflicts with subclasses
        that use register_buffer().

        Args:
            n_neurons: Number of neurons (output dimension)
        """
        self.delay_buffer = torch.zeros(
            (self.max_delay_steps, n_neurons),
            dtype=torch.bool,
            device=self._device
        )
        self.delay_buffer_idx = 0

    def _apply_axonal_delay(
        self,
        output_spikes: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Apply axonal delay to output spikes using circular buffer.

        Args:
            output_spikes: Binary spike tensor [n_neurons]
            dt: Timestep in milliseconds

        Returns:
            Delayed spikes from avg_delay_steps ago
        """
        if not hasattr(self, 'delay_buffer') or self.delay_buffer is None:
            self._initialize_delay_buffer(output_spikes.shape[0])

        self.delay_buffer[self.delay_buffer_idx] = output_spikes
        delayed_idx = (self.delay_buffer_idx - self.avg_delay_steps) % self.delay_buffer.shape[0]
        delayed_spikes = self.delay_buffer[delayed_idx]
        self.delay_buffer_idx = (self.delay_buffer_idx + 1) % self.delay_buffer.shape[0]
        return delayed_spikes

    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Dict[str, float] | None = None,
        theta_slot: int = 0,
        coupled_amplitudes: Dict[str, float] | None = None,
    ) -> None:
        """Set oscillator phases for neural oscillations (default: no-op).

        Subclasses with oscillators should override to update phase state.

        Args:
            phases: Dictionary of oscillator phases (e.g., {'theta': 0.5, 'gamma': 0.25})
            signals: Optional oscillator signals (amplitudes)
            theta_slot: Current theta slot for sequence learning
            coupled_amplitudes: Optional coupled oscillator amplitudes
        """
        pass  # Default: no oscillators, do nothing

    # =========================================================================
    # RESET HELPERS (for subclasses)
    # =========================================================================

    def _reset_subsystems(self, *subsystem_names: str) -> None:
        """Reset multiple subsystems by calling their reset_state() methods.

        Convenience helper to avoid repetitive code in reset_state() implementations.

        Args:
            *subsystem_names: Names of attributes to reset (must have reset_state())

        Example:
            >>> def reset_state(self):
            >>>     super().reset_state()
            >>>     self._reset_subsystems('neurons', 'stp', 'trace_manager')
        """
        for name in subsystem_names:
            if hasattr(self, name):
                subsystem = getattr(self, name)
                if subsystem is not None and hasattr(subsystem, 'reset_state'):
                    subsystem.reset_state()

    def _reset_scalars(self, **scalar_values: Any) -> None:
        """Reset scalar attributes to specified values.

        Convenience helper for resetting counters, accumulators, etc.

        Args:
            **scalar_values: Attribute names and their reset values

        Example:
            >>> def reset_state(self):
            >>>     super().reset_state()
            >>>     self._reset_scalars(
            >>>         _cumulative_spikes=0,
            >>>         _timestep=0,
            >>>         _episode_reward=0.0
            >>>     )
        """
        for name, value in scalar_values.items():
            setattr(self, name, value)


class RoutingComponent(nn.Module):
    """
    Abstract base for non-learnable routing components.

    **Note**: RoutingComponent does NOT inherit from BrainComponentBase, similar to
    NeuralRegion. It's a simpler architecture that informally implements the
    BrainComponent protocol without strict enforcement. This provides flexibility
    for lightweight routing components that don't need all the machinery.

    RoutingComponent provides routing WITHOUT learning:
    - NO synaptic weights (pure spike routing)
    - NO neurons (transmission only, no computation)
    - NO plasticity (fixed connectivity)

    What RoutingComponents DO have:
    - Spike transmission with delays
    - Multi-source concatenation
    - Dynamic growth (routing table updates)

    Examples:
    - Axonal projections (v2.0 architecture)
    - Sensory encoders (convert external input to spikes)
    - Motor decoders (convert spikes to actions)

    Key Principle: If it only routes/transforms without learning, it's a RoutingComponent.

    Usage:
        class AxonalProjection(RoutingComponent):
            def __init__(self, sources, device, dt_ms):
                config = SimpleNamespace(device=device)
                super().__init__(config)
                self.sources = sources
                self._init_delay_buffers()

            def forward(self, source_outputs):
                # Route spikes with delays, NO learning
                return self._apply_delays_and_concatenate(source_outputs)

            def grow_input(self, n_new, **kwargs):
                pass  # Routing components don't have input dimension

            def grow_output(self, n_new, **kwargs):
                raise NotImplementedError("Use grow_source() instead")
    """

    def __init__(self, config: Any):
        """Initialize routing component.

        Args:
            config: Minimal config (at minimum: device)
        """
        # Initialize nn.Module first
        nn.Module.__init__(self)

        # Store configuration
        self.config = config
        self._device = torch.device(config.device)
        # Routing components don't need dtype (no learnable parameters)
        self._dtype = torch.float32

    @property
    def device(self) -> torch.device:
        """Device where tensors are stored."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Data type (routing components use default float32)."""
        return self._dtype
