"""
Resettable State Mixin for Thalia Components.

Provides a standard interface for resetting component state, reducing
inconsistencies in reset behavior across the codebase.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from typing import List, Optional


class ResettableMixin:
    """Mixin for components with resettable state.

    Provides a standard interface for resetting component state, reducing
    inconsistencies in reset behavior across the codebase.

    Usage:
        class MyComponent(ResettableMixin, nn.Module):
            def __init__(self):
                super().__init__()
                self.state = None

            def reset_state(self) -> None:
                '''Reset internal state for new sequence.'''
                self.state = torch.zeros(1, self.n_neurons, device=self.device)
    """

    def reset_state(self) -> None:
        """Reset internal state for new sequence/episode.

        Standard signature: `reset_state(self) -> None`

        Components should reset:
        - Neuron membrane potentials and refractory states
        - Synaptic conductances (AMPA, NMDA, GABA)
        - Learning traces (eligibility, STDP, BCM)
        - Activity history and homeostatic variables
        - Working memory and gating states

        Do NOT reset:
        - Synaptic weights (learned knowledge)
        - Structural parameters (neuron counts, connectivity)
        - Configuration settings

        Always initializes to batch_size=1 per THALIA's single-instance architecture.

        Note:
            Subclasses should override this method to reset their
            specific state variables. The signature must be exactly
            `reset_state(self) -> None` with no optional parameters
            for consistency across all components.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement reset_state()")

    def reset_standard_state(self, state_attrs: Optional[List[str]] = None) -> None:
        """Helper to reset common state attributes to None.

        This helper reduces boilerplate in region reset() methods by
        handling the common pattern of resetting state attributes to None.

        Args:
            state_attrs: List of state attribute names to reset.
                        If None, uses default list: ["membrane", "spikes",
                        "eligibility", "spike_trace", "spike_history"]

        Example:
            >>> class MyRegion(NeuralRegion, ResettableMixin):
            ...     def reset(self) -> None:
            ...         # Reset standard attributes
            ...         self.reset_standard_state()
            ...
            ...         # Reset region-specific attributes
            ...         self.state.working_memory = None
            ...         self.state.update_gate = None

        Note:
            Only resets attributes that exist on self.state object.
            Silently skips missing attributes for flexibility.
        """
        if state_attrs is None:
            state_attrs = [
                "membrane",
                "spikes",
                "eligibility",
                "spike_trace",
                "spike_history",
            ]

        if not hasattr(self, "state"):
            return

        for attr in state_attrs:
            if hasattr(self.state, attr):
                setattr(self.state, attr, None)


__all__ = ["ResettableMixin"]
