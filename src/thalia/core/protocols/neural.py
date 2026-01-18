"""
Protocol Definitions for THALIA Brain Regions.

This module defines structural typing protocols that describe capabilities
of brain region components. Using protocols instead of inheritance allows:

1. **Duck Typing**: Classes automatically satisfy protocols if they have
   the right methods - no inheritance required
2. **Gradual Migration**: Existing classes work without changes
3. **Type Safety**: Pylance can verify protocol compliance statically
4. **Flexibility**: Multiple inheritance hierarchies can share protocols

Design Philosophy:
==================
The THALIA codebase has two parallel hierarchies:
- `LearnableComponent` (core/protocols/component.py) - Domain model for neural components
- `EventDrivenRegionBase` (event_regions/base.py) - Event system adapter

Rather than force these into a single hierarchy, we define protocols
that describe what components CAN DO, not what they ARE.

Protocols Defined:
==================
- `Learnable`: Has learn() method for synaptic plasticity
- `Resettable`: Has reset()/reset_state() methods to clear state
- `Diagnosable`: Has get_diagnostics() method for monitoring
- `Forwardable`: Has forward() method for processing
- `WeightContainer`: Has weights that can be get/set

Usage:
======
    from thalia.core.protocols.neural import Learnable, Diagnosable

    def train_region(region: Learnable, data: torch.Tensor) -> Dict:
        return region.learn(data, data)

    def monitor(component: Diagnosable) -> None:
        print(component.get_diagnostics())

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

import torch


@runtime_checkable
class Resettable(Protocol):
    """Protocol for components that can reset their state.

    Implementing classes should clear all transient state while preserving
    learned parameters (weights, thresholds).

    THALIA enforces single-instance architecture (batch_size=1) to maintain
    continuous temporal dynamics. For parallel simulations, create multiple
    component instances rather than batching.

    Use for:
    - LIF neurons (membrane potentials, refractory periods)
    - Brain regions (spike traces, working memory)
    - Synapses with STP (facilitation/depression)
    - Neuromodulators (reset to baseline)
    - Event schedulers (clear queue)
    - All stateful components
    """

    def reset_state(self) -> None:
        """Reset component state to initial conditions.

        Resets dynamic state (membrane potentials, traces, working memory)
        while preserving learned parameters (weights, thresholds).

        Always initializes to batch_size=1 per THALIA's single-instance
        architecture.
        """
        ...


@runtime_checkable
class Learnable(Protocol):
    """Protocol for components with synaptic plasticity.

    Implementing classes should update weights based on pre/post activity
    and optional learning signals (reward, error, target).

    Returns a dict with learning metrics (ltp, ltd, weight changes, etc.)
    """

    def learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Apply learning rule based on activity.

        Args:
            input_spikes: Presynaptic activity
            output_spikes: Postsynaptic activity
            **kwargs: Optional learning signals (reward, target, error, etc.)

        Returns:
            Dict with learning metrics
        """
        ...


@runtime_checkable
class Forwardable(Protocol):
    """Protocol for components that process input through forward pass.

    The standard neural network forward pass: input → processing → output.
    """

    def forward(
        self,
        input_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Process input and produce output.

        Args:
            input_spikes: Input activity tensor
            **kwargs: Additional inputs

        Returns:
            Output activity tensor
        """
        ...


@runtime_checkable
class Diagnosable(Protocol):
    """Protocol for components that provide diagnostic information.

    Implementing classes should return a dict with relevant metrics
    for monitoring, debugging, and analysis.

    Convention for metric names:
    - Use snake_case
    - Prefix with component name for disambiguation
    - Include units in name where helpful (e.g., `firing_rate_hz`)
    """

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information about component state.

        Returns:
            Dict with component-specific metrics. May include:
            - Weight statistics (mean, std, sparsity)
            - Activity levels (spike rates, trace values)
            - Learning metrics (recent LTP/LTD)
            - State indicators (threshold, phase, etc.)
        """
        ...


@runtime_checkable
class WeightContainer(Protocol):
    """Protocol for components that have learnable weights.

    Provides standard interface for weight access and modification.
    """

    def get_weights(self) -> torch.Tensor:
        """Return the current weight matrix (detached copy)."""
        ...

    def set_weights(self, weights: torch.Tensor) -> None:
        """Set the weight matrix."""
        ...


@runtime_checkable
class Configurable(Protocol):
    """Protocol for components initialized from configuration.

    Provides a standard factory method for creating instances
    from unified ThaliaConfig.
    """

    @classmethod
    def from_thalia_config(cls, config: Any) -> Any:
        """Create instance from ThaliaConfig.

        Args:
            config: ThaliaConfig or appropriate sub-config

        Returns:
            Configured instance
        """
        ...


# =============================================================================
# Compound Protocols
# =============================================================================


@runtime_checkable
class NeuralComponentProtocol(Forwardable, Learnable, Resettable, Diagnosable, Protocol):
    """Full protocol for neural components (regions, pathways, populations).

    Neural components should be able to:
    - Process inputs (forward)
    - Learn from experience (learn)
    - Reset state (reset/reset_state)
    - Provide diagnostics (get_diagnostics)
    """

    pass


# =============================================================================
# Type aliases for common patterns
# =============================================================================

# Any component that can learn
LearnableComponent = Learnable

# Any component that can be monitored
MonitorableComponent = Diagnosable

# Any component that can be reset
ResettableComponent = Resettable
