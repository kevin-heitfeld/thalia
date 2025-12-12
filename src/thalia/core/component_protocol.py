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
Date: December 7, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, runtime_checkable

import torch


@runtime_checkable
class BrainComponent(Protocol):
    """
    Unified protocol for all brain components (regions AND pathways).

    **CRITICAL**: Pathways are just as important as regions!

    Both regions and pathways implement this interface to ensure feature parity.
    When adding new functionality:
    1. Add it to this protocol first
    2. Implement for NeuralComponent
    3. Implement for BaseNeuralPathway
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
       - add_neurons(): Expand capacity without disrupting existing circuits
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
    ```python
    def analyze_component(component: BrainComponent) -> Dict[str, Any]:
        '''Analyze any brain component (region or pathway).'''
        # Check health
        health = component.check_health()

        # Check capacity
        metrics = component.get_capacity_metrics()

        # Grow if needed
        if metrics.growth_recommended:
            component.add_neurons(n_new=100)

        # Get diagnostics
        return component.get_diagnostics()

    # Works uniformly for regions AND pathways
    cortex_health = analyze_component(cortex)
    pathway_health = analyze_component(cortex_to_hippo_pathway)
    ```
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
        - **Delta (0.5-4 Hz)**: Deep sleep, attention
        - **Theta (4-10 Hz)**: Memory encoding, spatial navigation
        - **Alpha (8-13 Hz)**: Attention, inhibitory control
        - **Beta (13-30 Hz)**: Motor control, cognitive processing
        - **Gamma (30-100 Hz)**: Binding, local processing

        Components use oscillators for:
        - **Phase-dependent gating**: Theta encoding vs retrieval
        - **Attention modulation**: Alpha suppression
        - **Motor preparation**: Beta synchrony
        - **Feature binding**: Gamma synchrony
        - **Transmission efficiency**: Pathways can gate by oscillatory state

        Called every timestep by Brain (similar to dopamine broadcast).
        Default implementation stores phases but doesn't require usage.
        Components can override to implement oscillator-dependent behavior.

        Args:
            phases: Oscillator phases in radians [0, 2π)
                   {'delta': 1.2, 'theta': 3.4, 'alpha': 0.5, ...}
            signals: Oscillator signal values [-1, 1] (sin/cos waveforms)
                    {'delta': 0.8, 'theta': -0.3, ...}
            theta_slot: Current theta slot [0, n_slots-1] for sequence encoding
            coupled_amplitudes: Effective amplitudes per oscillator (pre-computed)
                               {'delta': 1.0, 'theta': 0.73, 'gamma': 0.48}
                               Values reflect automatic multiplicative coupling.

        Example (regions):
            >>> # Hippocampus uses theta for encoding/retrieval
            >>> theta_phase = phases['theta']
            >>> is_encoding = 0 <= theta_phase < np.pi
            >>> learning_rate = gamma_amplitude * base_lr

        Example (pathways):
            >>> # Attention pathway uses beta for gain modulation
            >>> beta_amp = coupled_amplitudes.get('beta', 1.0)
            >>> transmission_gain = 1.0 + (beta_amp - 1.0) * attention_strength
        """

    # =========================================================================
    # Growth (Curriculum Learning)
    # =========================================================================

    @abstractmethod
    def add_neurons(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """
        Add neurons/capacity to component without disrupting existing circuits.

        **CRITICAL for curriculum learning**: As the system learns harder tasks,
        it needs more capacity. Growing must preserve existing knowledge.

        For regions:
        - Expands weight matrices [n_output, n_input] → [n_output + n_new, n_input]
        - New neurons start with sparse random connections
        - Existing weights unchanged

        For pathways:
        - Expands transformation matrices to match connected region sizes
        - Maintains connectivity when upstream/downstream regions grow
        - Preserves existing connection strengths

        Args:
            n_new: Number of neurons/units to add
            initialization: Weight init strategy ('sparse_random', 'xavier', etc.)
            sparsity: Connection sparsity for new neurons (0.0 = none, 1.0 = all)

        Raises:
            NotImplementedError: If component doesn't support growth yet
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
                component.add_neurons(n_new=100)
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
    def device(self) -> torch.device:
        """Device where tensors are stored (CPU or CUDA)."""

    @property
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
    # Growth (REQUIRED for curriculum learning)
    # =========================================================================

    # NOTE: add_neurons() is NOT defined here to allow mixin-based implementations.
    # GrowthMixin provides the template method, BrainComponentMixin provides default.

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

    def add_neurons(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Default: raise NotImplementedError with helpful message."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.add_neurons() not yet implemented. "
            f"Growth is essential for curriculum learning. "
            f"See src/thalia/core/growth.py for implementation guide."
        )

    def get_capacity_metrics(self) -> Any:
        """Default: use GrowthManager to compute metrics."""
        from thalia.core.growth import GrowthManager

        # Use component name if available, otherwise class name
        name = getattr(self, 'name', self.__class__.__name__)
        manager = GrowthManager(component_name=name)
        return manager.get_capacity_metrics(self)

    def check_health(self) -> Any:
        """Default: return healthy status with no issues."""
        from thalia.diagnostics.health import HealthReport
        return HealthReport(
            component_name=getattr(self, 'name', self.__class__.__name__),
            is_healthy=True,
            issues=[],
            warnings=[],
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
