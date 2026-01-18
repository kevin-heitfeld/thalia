"""Neural Pathway Protocol - Unified interface for all pathway types.

This module defines the NeuralPathway protocol that standardizes the interface
across sensory pathways, inter-region pathways, and specialized pathways.

IMPORTANT: Architecture Evolution
==================================
- Brain regions now inherit from NeuralRegion (synaptic weights at dendrites)
- Pathways can use LearnableComponent for custom implementations
- AxonalProjection is the standard inter-region pathway (pure routing, no weights)

This unified architecture ensures component parity:
- Both regions and pathways process information via forward()
- Regions learn at dendrites (synaptic_weights dict)
- Pathways route spikes (AxonalProjection) or provide custom logic
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
   - Can inherit from LearnableComponent for custom implementations

Protocol Design:
================

The protocol allows for flexibility while ensuring consistency:
- All pathways must implement forward() (standard PyTorch convention, ADR-007)
- Regions learn at dendrites (NeuralRegion pattern)
- State management and diagnostics are required
- Type hints enable static checking

Usage Example
==============
Custom pathways can inherit from LearnableComponent:

.. code-block:: python

    from thalia.core.protocols.component import LearnableComponent

    # Define a custom pathway (LearnableComponent for custom implementations)
    class MyPathway(LearnableComponent):
        def forward(self, spikes):
            # Transform spikes with custom logic
            return self.transform(spikes)

        def reset_state(self) -> None:
            self.membrane.zero_()
            self.input_trace.zero_()

        def get_diagnostics(self):
            return {
                "spike_rate": compute_firing_rate(self.spikes),
                "weight_mean": self.weights.mean().item(),
            }

    # Define a region (use NeuralRegion for brain regions)
    from thalia.core.neural_region import NeuralRegion
    class MyCortex(NeuralRegion):
        def forward(self, inputs: Dict[str, Tensor]):
            return self.process_layers(inputs)

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
            **kwargs: Additional arguments

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


# Type alias for convenience (ADR-007: all pathways use NeuralPathway protocol)
Pathway = NeuralPathway


__all__ = [
    "NeuralPathway",
    "Pathway",
]
