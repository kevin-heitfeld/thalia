"""
Diagnostics Mixin for Brain Components.

This module provides a reusable mixin that implements common diagnostic
patterns for brain regions and other components.

The mixin provides:
1. Standard weight statistics (mean, std, min, max, sparsity)
2. Spike/activity statistics (rate, sparsity)
3. Trace statistics (norm, mean)
4. Formatted output helpers

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from typing import Dict

import torch

from thalia.constants.time import MS_PER_SECOND


class DiagnosticsMixin:
    """Mixin providing common diagnostic computation patterns.

    Add this mixin to any class that implements get_diagnostics() to
    reuse standard metric calculations.

    All methods are static or use only the provided arguments, so they
    work regardless of the class structure.
    """

    @staticmethod
    def weight_diagnostics(
        weights: torch.Tensor,
        prefix: str = "",
        include_histogram: bool = False,
    ) -> Dict[str, float]:
        """Compute standard weight statistics.

        Args:
            weights: Weight tensor (any shape)
            prefix: Prefix for metric names (e.g., "d1" → "d1_weight_mean")
            include_histogram: Include histogram bins (more expensive)

        Returns:
            Dict with weight statistics
        """
        prefix = f"{prefix}_" if prefix else ""

        w = weights.detach()

        # Handle empty tensors
        if w.numel() == 0:
            return {
                f"{prefix}weight_mean": 0.0,
                f"{prefix}weight_std": 0.0,
                f"{prefix}weight_min": 0.0,
                f"{prefix}weight_max": 0.0,
                f"{prefix}weight_sparsity": 1.0,
            }

        stats = {
            f"{prefix}weight_mean": w.mean().item(),
            f"{prefix}weight_std": w.std().item(),
            f"{prefix}weight_min": w.min().item(),
            f"{prefix}weight_max": w.max().item(),
            f"{prefix}weight_sparsity": (w.abs() < 1e-6).float().mean().item(),
        }

        # Non-zero weights (for sparse analysis)
        nonzero_mask = w.abs() >= 1e-6
        if nonzero_mask.any():
            stats[f"{prefix}weight_nonzero_mean"] = w[nonzero_mask].mean().item()

        if include_histogram:
            # 10-bin histogram
            hist = torch.histc(w.float(), bins=10)
            for i, count in enumerate(hist.tolist()):
                stats[f"{prefix}weight_hist_{i}"] = count

        return stats

    @staticmethod
    def spike_diagnostics(
        spikes: torch.Tensor,
        prefix: str = "",
        dt_ms: float = 1.0,
    ) -> Dict[str, float]:
        """Compute spike/activity statistics.

        Args:
            spikes: Spike tensor (binary or rate)
            prefix: Prefix for metric names
            dt_ms: Timestep in milliseconds (for rate calculation)

        Returns:
            Dict with spike statistics
        """
        prefix = f"{prefix}_" if prefix else ""

        s = spikes.detach().float()

        # Sparsity: fraction of neurons NOT spiking
        sparsity = 1.0 - s.mean().item()

        # Active count
        active_count = (s > 0.5).sum().item()
        total_neurons = s.numel()

        # Firing rate (if binary spikes)
        rate_hz = s.mean().item() * (MS_PER_SECOND / dt_ms)

        return {
            f"{prefix}sparsity": sparsity,
            f"{prefix}active_count": active_count,
            f"{prefix}total_neurons": total_neurons,
            f"{prefix}firing_rate_hz": rate_hz,
            f"{prefix}mean_activity": s.mean().item(),
        }

    @staticmethod
    def trace_diagnostics(
        trace: torch.Tensor,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute eligibility/activity trace statistics.

        Args:
            trace: Trace tensor (typically exponentially decaying)
            prefix: Prefix for metric names

        Returns:
            Dict with trace statistics
        """
        prefix = f"{prefix}_" if prefix else ""

        t = trace.detach()

        return {
            f"{prefix}trace_mean": t.mean().item(),
            f"{prefix}trace_max": t.max().item(),
            f"{prefix}trace_norm": t.norm().item(),
            f"{prefix}trace_nonzero": (t.abs() > 1e-6).sum().item(),
        }
