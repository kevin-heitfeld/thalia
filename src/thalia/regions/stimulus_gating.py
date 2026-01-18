"""
Stimulus gating and temporal integration for cortical processing.

This module provides utilities for transient inhibition and temporal buffering:

1. **StimulusGating**: Computes transient inhibition at stimulus changes
   - Clears ongoing activity when new stimulus arrives
   - Sharpens temporal precision of responses
   - Enables clean separation between stimuli

Note:
    Previously named "FeedforwardInhibition", which was confusing since
    canonical feedforward inhibition refers to interneuron-mediated lateral
    inhibition (e.g., basket cells). This module implements stimulus-onset
    inhibition (clearing residual activity), not lateral inhibition.

References:
- Hasselmo et al. (2002): Theta rhythm and encoding/retrieval
- Colgin (2013): Theta-gamma coupling in hippocampus
- Siegle & Wilson (2014): Enhancement of encoding and retrieval
"""

from __future__ import annotations

from typing import Optional

import torch


class StimulusGating:
    """
    Computes stimulus-onset inhibition triggered by stimulus changes.

    In biological circuits, the arrival of a new stimulus triggers
    strong transient inhibition via fast-spiking interneurons. This:

    1. Clears ongoing activity (no explicit reset needed!)
    2. Sharpens temporal precision of responses
    3. Enables clean separation between stimuli

    The inhibition strength is proportional to how much the input changed.

    Note:
        This is NOT canonical "feedforward inhibition" (lateral inhibition
        via interneurons). This is stimulus-onset gating that clears
        residual activity when a new stimulus arrives.

    Example:
        gating = StimulusGating(threshold=0.5, decay=0.9)

        for stimulus in stimuli:
            # Compute inhibition based on stimulus change
            inhibition = gating.compute(stimulus)

            # Apply to membrane potentials
            membrane = membrane - inhibition * max_inhibition
    """

    def __init__(
        self,
        threshold: float = 0.3,
        max_inhibition: float = 5.0,
        decay_rate: float = 0.8,
        steepness: float = 10.0,
    ):
        """
        Initialize stimulus gating.

        Args:
            threshold: Input change threshold to trigger inhibition
            max_inhibition: Maximum inhibition strength
            decay_rate: How fast the previous input trace decays
            steepness: Steepness of sigmoid for change detection
        """
        self.threshold = threshold
        self.max_inhibition = max_inhibition
        self.decay_rate = decay_rate
        self.steepness = steepness

        # Track previous input for change detection
        self._prev_input: Optional[torch.Tensor] = None
        self._current_inhibition: float = 0.0

    def compute(
        self,
        current_input: torch.Tensor,
        return_tensor: bool = True,
    ) -> torch.Tensor:
        """
        Compute stimulus-onset inhibition for current input.

        Args:
            current_input: Current input tensor
            return_tensor: If True, return tensor matching input shape

        Returns:
            Inhibition strength (scalar or tensor matching input shape)
        """
        if self._prev_input is None:
            # First input - no inhibition
            self._prev_input = current_input.detach().clone().float()
            if return_tensor:
                return torch.zeros_like(current_input, dtype=torch.float32)
            return torch.tensor(0.0)

        # Convert bool spikes to float for arithmetic (ADR-004)
        current_float = current_input.float()
        prev_float = self._prev_input.float()

        # Compute input change magnitude (normalized by input size)
        input_diff = (current_float - prev_float).abs()
        change_magnitude = input_diff.sum() / (current_input.numel() + 1e-6)

        # Sigmoid activation based on change magnitude
        # High change → high inhibition
        inhibition = (
            torch.sigmoid((change_magnitude - self.threshold) * self.steepness)
            * self.max_inhibition
        )

        # Update previous input (with decay for smooth tracking)
        self._prev_input = (
            self.decay_rate * self._prev_input
            + (1 - self.decay_rate) * current_float.detach().clone()
        )

        self._current_inhibition = inhibition.item()

        if return_tensor:
            # Return per-neuron inhibition proportional to local change
            return input_diff * inhibition / (input_diff.max() + 1e-6)
        return inhibition

    @property
    def current_inhibition(self) -> float:
        """Current inhibition level."""
        return self._current_inhibition


__all__ = [
    "StimulusGating",
]
