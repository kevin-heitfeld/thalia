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

from abc import abstractmethod
from typing import Any, Dict, Protocol, runtime_checkable

import torch
import torch.nn as nn


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
        initialization: str = "sparse_random",
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
        initialization: str = "sparse_random",
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
    """

    def grow_input(
        self,
        n_new: int,
        initialization: str = "sparse_random",
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
        initialization: str = "sparse_random",
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
        name = getattr(self, "name", self.__class__.__name__)
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
        if not hasattr(self, "_oscillator_phases"):
            self._oscillator_phases: Dict[str, float] = {}
            self._oscillator_signals: Dict[str, float] = {}
            self._oscillator_theta_slot: int = 0
            self._coupled_amplitudes: Dict[str, float] = {}

        self._oscillator_phases = phases
        self._oscillator_signals = signals or {}
        self._oscillator_theta_slot = theta_slot
        self._coupled_amplitudes = coupled_amplitudes or {}

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Default: propagate to neurons, STP, and learning strategies if they exist.

        This provides a sensible default that works for most components.
        Override if custom temporal update logic is needed.

        Args:
            dt_ms: New timestep in milliseconds
        """
        # Update neurons if present
        if hasattr(self, "neurons") and hasattr(self.neurons, "update_temporal_parameters"):
            self.neurons.update_temporal_parameters(dt_ms)

        # Update STP if present
        if hasattr(self, "stp") and hasattr(self.stp, "update_temporal_parameters"):
            self.stp.update_temporal_parameters(dt_ms)

        # Update learning strategies if present
        if hasattr(self, "strategies"):
            for strategy in self.strategies.values():
                if hasattr(strategy, "update_temporal_parameters"):
                    strategy.update_temporal_parameters(dt_ms)

        # Update single learning strategy if present
        if hasattr(self, "learning_strategy") and hasattr(
            self.learning_strategy, "update_temporal_parameters"
        ):
            self.learning_strategy.update_temporal_parameters(dt_ms)

    # =========================================================================
    # Convenience properties for accessing individual oscillator phases
    # =========================================================================
    # These eliminate the need for regions to repeatedly use .get() with defaults

    @property
    def _theta_phase(self) -> float:
        """Current theta phase in radians [0, 2π)."""
        return float(getattr(self, "_oscillator_phases", {}).get("theta", 0.0))

    @_theta_phase.setter
    def _theta_phase(self, value: float) -> None:
        """Allow regions to set theta phase directly (backward compatibility)."""
        if not hasattr(self, "_oscillator_phases"):
            self._oscillator_phases = {}
        self._oscillator_phases["theta"] = value

    @property
    def _gamma_phase(self) -> float:
        """Current gamma phase in radians [0, 2π)."""
        return float(getattr(self, "_oscillator_phases", {}).get("gamma", 0.0))

    @_gamma_phase.setter
    def _gamma_phase(self, value: float) -> None:
        """Allow regions to set gamma phase directly (backward compatibility)."""
        if not hasattr(self, "_oscillator_phases"):
            self._oscillator_phases = {}
        self._oscillator_phases["gamma"] = value

    @property
    def _alpha_phase(self) -> float:
        """Current alpha phase in radians [0, 2π)."""
        return float(getattr(self, "_oscillator_phases", {}).get("alpha", 0.0))

    @_alpha_phase.setter
    def _alpha_phase(self, value: float) -> None:
        """Allow regions to set alpha phase directly (backward compatibility)."""
        if not hasattr(self, "_oscillator_phases"):
            self._oscillator_phases = {}
        self._oscillator_phases["alpha"] = value

    @property
    def _beta_phase(self) -> float:
        """Current beta phase in radians [0, 2π)."""
        return float(getattr(self, "_oscillator_phases", {}).get("beta", 0.0))

    @_beta_phase.setter
    def _beta_phase(self, value: float) -> None:
        """Allow regions to set beta phase directly (backward compatibility)."""
        if not hasattr(self, "_oscillator_phases"):
            self._oscillator_phases = {}
        self._oscillator_phases["beta"] = value

    @property
    def _delta_phase(self) -> float:
        """Current delta phase in radians [0, 2π)."""
        return float(getattr(self, "_oscillator_phases", {}).get("delta", 0.0))

    @_delta_phase.setter
    def _delta_phase(self, value: float) -> None:
        """Allow regions to set delta phase directly (backward compatibility)."""
        if not hasattr(self, "_oscillator_phases"):
            self._oscillator_phases = {}
        self._oscillator_phases["delta"] = value

    @property
    def _gamma_amplitude_effective(self) -> float:
        """Effective gamma amplitude (with cross-frequency coupling)."""
        return float(getattr(self, "_coupled_amplitudes", {}).get("gamma", 1.0))

    @property
    def _beta_amplitude_effective(self) -> float:
        """Effective beta amplitude (with cross-frequency coupling)."""
        return float(getattr(self, "_coupled_amplitudes", {}).get("beta", 1.0))


# =============================================================================
# Component Type Hierarchy
# =============================================================================


class RoutingComponent(nn.Module):
    """
    Abstract base for non-learnable routing components.

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
