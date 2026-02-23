"""Random number generation utilities for Thalia, including Philox-based RNG for reproducibility and GPU efficiency."""

import math

import torch


def philox_uniform(counters: torch.Tensor) -> torch.Tensor:
    """Philox 4x32 uniform [0, 1) — identical to ConductanceLIF."""
    W0, W1 = 0x9E3779B9, 0xBB67AE85
    x = counters.clone().to(torch.int64)
    for r in range(10):
        lo = x & 0xFFFFFFFF
        hi = (x >> 32) & 0xFFFFFFFF
        lo = (lo * W0) & 0xFFFFFFFF
        hi = (hi * W1) & 0xFFFFFFFF
        x = ((hi << 32) | lo) ^ (W0 * (r + 1))
    return ((x & 0xFFFFFFFF) + 1).float() / (2**32 + 2)


def philox_gaussian(counters: torch.Tensor) -> torch.Tensor:
    """Per-neuron independent Gaussian(0,1) noise via Philox + Box-Muller."""
    u1 = philox_uniform(counters)
    u2 = philox_uniform(counters + 1)
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2 * math.pi * u2)
