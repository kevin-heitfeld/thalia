"""
Stimulus gating and temporal integration for cortical processing.

This module provides utilities for transient inhibition and temporal buffering:

**StimulusGating**: Computes transient inhibition at stimulus changes
   - Clears ongoing activity when new stimulus arrives
   - Sharpens temporal precision of responses
   - Enables clean separation between stimuli

Note:
    Previously named "FeedforwardInhibition", which was confusing since
    canonical feedforward inhibition refers to interneuron-mediated lateral
    inhibition (e.g., basket cells). This module implements stimulus-onset
    inhibition (clearing residual activity), not lateral inhibition.
"""

from __future__ import annotations

from typing import Optional

import torch

from thalia.utils import CircularDelayBuffer


class StimulusGating:
    """
    Computes stimulus-onset inhibition triggered by stimulus changes.

    In biological circuits, the arrival of a new stimulus triggers
    strong transient inhibition via fast-spiking interneurons. This:

    1. Clears ongoing activity
    2. Sharpens temporal precision of responses
    3. Enables clean separation between stimuli

    The inhibition strength is proportional to how much the input changed.

    Note:
        This is NOT canonical "feedforward inhibition" (lateral inhibition
        via interneurons). This is stimulus-onset gating that clears
        residual activity when a new stimulus arrives.
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

        # Track previous input for change detection using CircularDelayBuffer
        # Note: Buffer will be initialized on first call with proper shape/device
        self._input_buffer: Optional[CircularDelayBuffer] = None
        self._smoothed_input: Optional[torch.Tensor] = None  # Exponentially-smoothed input
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
        # Initialize buffer on first call
        if self._input_buffer is None:
            self._input_buffer = CircularDelayBuffer(
                max_delay=1,
                size=current_input.shape[0],
                dtype=torch.float32,
                device=current_input.device,
            )
            self._smoothed_input = current_input.detach().clone().float()
            if return_tensor:
                return torch.zeros_like(current_input, dtype=torch.float32)
            return torch.tensor(0.0)

        # Convert bool spikes to float for arithmetic (ADR-004)
        current_float = current_input.float()

        # Compute input change magnitude (normalized by input size)
        input_diff = (current_float - self._smoothed_input).abs()
        change_magnitude = input_diff.sum() / (current_input.numel() + 1e-6)

        # Sigmoid activation based on change magnitude
        # High change → high inhibition
        inhibition = (
            torch.sigmoid((change_magnitude - self.threshold) * self.steepness)
            * self.max_inhibition
        )

        # Update smoothed input (exponential moving average for smooth tracking)
        self._smoothed_input = (
            self.decay_rate * self._smoothed_input
            + (1 - self.decay_rate) * current_float.detach().clone()
        )

        # Write current smoothed input to buffer for next timestep
        self._input_buffer.write(self._smoothed_input)

        self._current_inhibition = inhibition.item()

        if return_tensor:
            # Return per-neuron inhibition proportional to local change
            return input_diff * inhibition / (input_diff.max() + 1e-6)
        return inhibition

    @property
    def current_inhibition(self) -> float:
        """Current inhibition level."""
        return self._current_inhibition
