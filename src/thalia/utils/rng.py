"""Random number generation utilities for Thalia, including Philox-based RNG for reproducibility and GPU efficiency."""

import math

import torch


def gaussian_from_uniform(u1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
    """Convert two independent uniform(0,1) tensors to a standard Gaussian(0,1) tensor using Box-Muller."""
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)


def philox_uniform(counters: torch.Tensor) -> torch.Tensor:
    """Philox 4x32 -> uniform [0,1) using full int64 math."""
    W0, W1 = 0x9E3779B9, 0xBB67AE85
    x = counters.clone().to(torch.int64)
    for r in range(10):
        lo = x & 0xffffffff
        hi = (x >> 32) & 0xffffffff
        lo = (lo * W0) & 0xffffffff
        hi = (hi * W1) & 0xffffffff
        # mix hi/lo without truncating the original x
        x = ((hi << 32) | lo) ^ (W0 * (r + 1))

    # normalize
    u = ((x & 0xffffffff) + 1).float() / (2**32 + 2)  # avoid exact 0 or 1
    return u


def philox_gaussian(counters: torch.Tensor) -> torch.Tensor:
    """Per-neuron independent Gaussian(0,1) noise via Philox + Box-Muller."""
    u1 = philox_uniform(counters)
    u2 = philox_uniform(counters + 1)
    return gaussian_from_uniform(u1, u2)
