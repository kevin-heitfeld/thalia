"""
Unified protocol for brain components (regions and pathways).

This module defines the common interface that both brain regions and neural
pathways implement, ensuring feature parity and preventing oversight.

Design Philosophy
=================
Brain regions and pathways are EQUALLY IMPORTANT:
- Both process information (forward/encode)
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

from abc import abstractmethod
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
    2. Implement for BrainRegion
    3. Implement for BaseNeuralPathway
    4. Update tests for both

    This prevents accidentally forgetting pathways when adding features.

    Core Capabilities
    =================
    All brain components must support:

    1. **Information Processing**:
       - forward() or encode(): Transform inputs to outputs
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

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Process input and update state.

        For regions: forward(spikes: Tensor) -> Tensor
        For pathways: forward(spikes: Tensor) -> Tensor
        For sensory pathways: May also implement encode(raw_input) -> (Tensor, Dict)

        **Learning happens automatically during forward passes** - there is no
        separate learn() method. Plasticity is always active, modulated by
        neuromodulators (dopamine, acetylcholine, etc.).

        Returns:
            Output spikes or activations (region/pathway specific)
        """
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def device(self) -> torch.device:
        """Device where tensors are stored (CPU or CUDA)."""
        ...

    @property
    def dtype(self) -> torch.dtype:
        """Data type for floating point tensors."""
        ...


# =============================================================================
# Concrete Base Classes
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
