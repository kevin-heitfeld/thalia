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

from typing import Union

import torch
import torch.nn as nn


class StimulusGating(nn.Module):
    """
    Computes stimulus-onset inhibition triggered by stimulus changes.

    In biological circuits, the arrival of a new stimulus triggers
    strong transient inhibition via fast-spiking interneurons. This:

    1. Clears ongoing activity
    2. Sharpens temporal precision of responses
    3. Enables clean separation between stimuli

    The inhibition strength is proportional to how much the input changed.

    As an ``nn.Module``, all internal state (specifically ``_smoothed_input``)
    is registered as a buffer so it:

    - Moves correctly with ``.to(device)``
    - Is included in ``state_dict()`` and survives checkpoint save/load
    - Is initialised on the correct device at construction time

    Note:
        This is NOT canonical "feedforward inhibition" (lateral inhibition
        via interneurons). This is stimulus-onset gating that clears
        residual activity when a new stimulus arrives.
    """

    def __init__(
        self,
        n_neurons: int,
        device: Union[str, torch.device] = "cpu",
        threshold: float = 0.3,
        max_inhibition: float = 5.0,
        decay_rate: float = 0.8,
        steepness: float = 10.0,
    ):
        """
        Initialize stimulus gating.

        Args:
            n_neurons: Number of neurons in the gated population.  The
                smoothed-input buffer is pre-allocated with this size so the
                module is device-safe from construction.
            device: Device on which the internal buffer should reside.
            threshold: Input change threshold to trigger inhibition.
            max_inhibition: Maximum inhibition strength.
            decay_rate: How fast the previous input trace decays.
            steepness: Steepness of sigmoid for change detection.
        """
        super().__init__()

        self.threshold = threshold
        self.max_inhibition = max_inhibition
        self.decay_rate = decay_rate
        self.steepness = steepness
        self._current_inhibition: float = 0.0

        # Exponentially-smoothed input trace used for change detection.
        # Registered as a buffer so it is included in state_dict and moved
        # automatically with .to(device).
        self.register_buffer(
            "_smoothed_input",
            torch.zeros(n_neurons, dtype=torch.float32, device=device),
        )

    def compute(
        self,
        current_input: torch.Tensor,
        return_tensor: bool = True,
    ) -> torch.Tensor:
        """
        Compute stimulus-onset inhibition for current input.

        Args:
            current_input: Current input tensor (shape ``[n_neurons]``).
            return_tensor: If True, return a per-neuron inhibition tensor
                matching the input shape; otherwise return a scalar tensor.

        Returns:
            Inhibition strength (scalar tensor, or per-neuron tensor when
            ``return_tensor=True``).
        """
        # Convert bool spikes to float for arithmetic (ADR-004)
        current_float = current_input.float()

        # Compute input change magnitude (normalized by input size)
        input_diff = (current_float - self._smoothed_input).abs()
        change_magnitude = input_diff.sum() / (current_input.numel() + 1e-6)

        # Sigmoid activation: high change → high inhibition
        inhibition = (
            torch.sigmoid((change_magnitude - self.threshold) * self.steepness)
            * self.max_inhibition
        )

        # Update smoothed input in-place (exponential moving average).
        # In-place ops on a registered buffer are safe because the buffer is
        # never part of a computation graph that requires grad.
        self._smoothed_input.mul_(self.decay_rate).add_(
            (1.0 - self.decay_rate) * current_float.detach()
        )

        self._current_inhibition = inhibition.item()

        if return_tensor:
            # Return per-neuron inhibition proportional to local change
            return input_diff * inhibition / (input_diff.max() + 1e-6)
        return inhibition

    @property
    def current_inhibition(self) -> float:
        """Current inhibition level (scalar, updated each ``compute()`` call)."""
        return self._current_inhibition
