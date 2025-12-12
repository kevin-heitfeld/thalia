"""
Resettable State Mixin for Thalia Components.

Provides a standard interface for resetting component state, reducing
inconsistencies in reset behavior across the codebase.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations


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

        Resets dynamic state (membrane potentials, traces, working memory)
        while preserving learned parameters. Always initializes to batch_size=1
        per THALIA's single-instance architecture.

        Note:
            Subclasses should override this method to reset their
            specific state variables.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement reset_state()"
        )


__all__ = ["ResettableMixin"]
