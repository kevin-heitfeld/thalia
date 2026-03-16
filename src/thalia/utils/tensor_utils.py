"""Utility functions for tensor operations in Thalia."""

from __future__ import annotations

from typing import Union
import math

import torch


def decay_float(dt_ms: float, tau: float) -> float:
    """Calculate the decay factor for a given time difference and time constant."""
    return math.exp(-dt_ms / tau)


def decay_tensor(dt_ms: float, tau: Union[float, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Calculate the decay factor for a given time difference and time constant, supporting both float and tensor tau."""
    if isinstance(tau, torch.Tensor):
        return torch.exp(-dt_ms / tau)
    else:
        return torch.tensor(math.exp(-dt_ms / tau), device=device)
