"""Utility functions for weight management in neural networks."""

from __future__ import annotations

import torch


def clamp_weights(
    weights: torch.Tensor,
    w_min: float = 0.0,
    w_max: float = 1.0,
    inplace: bool = True,
) -> torch.Tensor:
    """Clamp weight tensor to valid range.

    Standard pattern for enforcing weight bounds after learning updates.
    Operates in-place by default for efficiency.

    Args:
        weights: Weight tensor to clamp
        w_min: Minimum weight value (default: 0.0)
        w_max: Maximum weight value (default: 1.0)
        inplace: If True, modify weights in place (default: True)

    Returns:
        Clamped weight tensor
    """
    if inplace:
        return weights.clamp_(w_min, w_max)
    return weights.clamp(w_min, w_max)
