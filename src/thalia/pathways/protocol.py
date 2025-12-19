"""Neural Pathway Protocol - Unified interface for all pathway types.

This module defines the NeuralPathway protocol that standardizes the interface
across sensory pathways, inter-region pathways, and specialized pathways.

IMPORTANT: All pathways implement the BrainComponent protocol!
===============================================================
As of ADR-008 (Neural Component Consolidation), both brain regions and pathways
inherit from NeuralComponent base class and implement the BrainComponent protocol
(defined in component_protocol.py).

This unified architecture ensures component parity:
- Both regions and pathways process information via forward()
- Both learn continuously during forward passes (no separate learn())
- Both maintain temporal state and support growth
- Both provide diagnostics and health monitoring

This file provides the NeuralPathway Protocol for additional pathway-specific
type checking, while BrainComponent protocol defines the shared interface.

Biological Motivation
=====================
Neural pathways in the brain are not passive "wires" - they are active
processing units with their own dynamics and plasticity:

1. **Transform Information**: Pathways actively filter and route signals
2. **Adapt Through Experience**: Learn via STDP/BCM during forward passes
3. **Maintain State**: Track temporal dynamics (membrane potentials, traces)
4. **Provide Diagnostics**: Monitor pathway health and activity metrics

Pathways ARE mini-regions with neurons, synapses, and learning rules.

Types of Pathways
==================

1. **Sensory Pathways** (SensoryPathway):
   - Transform raw sensory input → spike patterns
   - Examples: VisualPathway (images→V1), AudioPathway (sounds→A1),
               LanguagePathway (tokens→spikes)
   - Method: forward(raw_input) → (spikes, metadata)
   - Standard PyTorch convention (ADR-007)

2. **Inter-Region Pathways** (AxonalProjection):
   - Route spikes between brain regions (pure axonal transmission)
   - Examples: Cortex→Hippocampus, Hippocampus→Cortex, Cortex→Striatum
   - Method: forward(source_dict) → spikes
   - Learning: Happens at target region dendrites (NeuralRegion pattern)

3. **Specialized Pathways**:
   - Sensory pathways transform raw input → spikes
   - Custom routing logic for complex architectures
   - Inherit from NeuralComponent with specialized forward() behavior

Protocol Design:
================

The protocol allows for flexibility while ensuring consistency:
- All pathways must implement forward() (standard PyTorch convention, ADR-007)
- Learning is automatic during forward passes
- State management and diagnostics are required
- Type hints enable static checking

Usage Example
==============
All pathways (regions too) inherit from NeuralComponent:

.. code-block:: python

    from thalia.regions.base import NeuralComponent

    # Define an inter-region pathway
    class MyPathway(NeuralComponent):
        def forward(self, spikes):
            # Transform spikes AND apply STDP (automatically)
            return self.transform(spikes)

        def reset_state(self) -> None:
            self.membrane.zero_()
            self.input_trace.zero_()

        def get_diagnostics(self):
            return {
                "spike_rate": compute_firing_rate(self.spikes),
                "weight_mean": self.weights.mean().item(),
            }

    # Define a region (same interface!)
    class MyCortex(NeuralComponent):
        def forward(self, input_spikes):
            return self.process_layers(input_spikes)

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Protocol, Tuple, Union, runtime_checkable

import torch

if TYPE_CHECKING:
    pass


@runtime_checkable
class NeuralPathway(Protocol):
    """
    Protocol defining the unified interface for all neural pathways.

    All pathways (sensory, inter-region, specialized) implement this interface
    to ensure consistent usage patterns across the codebase.

    Core Methods:
    -------------
    - forward(): Transform input to output (standard PyTorch, ADR-007)
    - reset_state(): Reset temporal state
    - get_diagnostics(): Report pathway metrics

    Design Rationale:
    -----------------
    1. **All pathways** use forward() (standard PyTorch convention, ADR-007)
    2. **Sensory pathways**: forward(raw_input) → (spikes, metadata)
    3. **Inter-region pathways**: forward(spikes) → spikes
    4. **Pathways always learn** during forward passes (like regions)
    5. Learning is via STDP, BCM, or other plasticity rules applied automatically
    """

    def forward(
        self,
        input_data: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Transform input to output (standard PyTorch convention, ADR-007).

        All pathways use forward() to enable callable syntax:
        >>> output = pathway(input_data)  # Calls forward() automatically

        Args:
            input_data: Input tensor
                - Inter-region pathways: spikes [n_neurons]
                - Sensory pathways: raw input (image, audio, tokens)
            **kwargs: Additional arguments (dt, time_ms, etc.)

        Returns:
            output: Output tensor, or (output, metadata) tuple
                - Inter-region: spikes [n_neurons]
                - Sensory: (spikes [n_timesteps, n_neurons], metadata)
        """
        ...

    def reset_state(self) -> None:
        """
        Reset pathway temporal state.

        Clears:
        - Synaptic traces
        - Membrane potentials
        - Delay buffers
        - Adaptation state
        - Any other temporal dynamics

        Call this between trials/sequences to ensure clean state.
        """
        ...

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get pathway diagnostics and metrics.

        Returns:
            Dictionary containing:
            - Activity metrics (spike rates, membrane stats)
            - Learning metrics (weight changes, STDP traces)
            - Health indicators (weights in bounds, no NaNs)
            - Pathway-specific metrics
        """
        ...


# Note: SensoryPathwayProtocol removed (ADR-007)
# All pathways (sensory and inter-region) use forward() for consistency.
# Sensory pathways return (spikes, metadata) tuple from forward().
# Inter-region pathways return spikes tensor from forward().

# Note: LearnablePathway protocol removed - pathways ALWAYS learn during forward passes.
# Learning happens automatically via STDP, BCM, or other plasticity rules,
# just like regions (Prefrontal, Hippocampus, etc.) always learn.
# No separate learn() method needed.


# =============================================================================
# Unified Base: NeuralComponent is imported via TYPE_CHECKING
# =============================================================================

# All neural populations inherit from NeuralComponent (defined in thalia.regions.base)
# This reflects the biological reality that regions and pathways are both
# just populations of neurons with weights, dynamics, and learning rules.
# To avoid circular import, we use TYPE_CHECKING above.


# Type alias for convenience (ADR-007: all pathways use NeuralPathway protocol)
Pathway = NeuralPathway


__all__ = [
    "NeuralPathway",
    "Pathway",
]
