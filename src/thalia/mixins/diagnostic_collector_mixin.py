"""
Diagnostic Collection Mixin for Thalia Components.

Provides helper methods to collect and format diagnostics in a
consistent way across components.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from typing import Optional

import torch

from thalia.constants.time import MS_PER_SECOND


class DiagnosticCollectorMixin:
    """Mixin for standardized diagnostic collection.

    Provides helper methods to collect and format diagnostics in a
    consistent way across components.

    Usage:
        class MyComponent(DiagnosticCollectorMixin, nn.Module):
            def get_diagnostics(self) -> Dict[str, Any]:
                diag = self.init_diagnostics()
                diag.update(self.collect_tensor_stats(self.weights, "weights"))
                diag.update(self.collect_scalar("spike_rate", self.rate))
                return diag
    """

    def init_diagnostics(self, **base_fields) -> dict:
        """Initialize diagnostics dict with base fields.

        Args:
            **base_fields: Base fields to include in diagnostics

        Returns:
            Dictionary with base fields and empty diagnostics
        """
        return dict(**base_fields)

    def collect_tensor_stats(
        self,
        tensor: torch.Tensor,
        name: str,
        percentiles: Optional[list[int]] = None,
    ) -> dict:
        """Collect statistics from a tensor.

        Args:
            tensor: Tensor to analyze
            name: Prefix for stat names
            percentiles: List of percentiles to compute (e.g., [25, 50, 75])

        Returns:
            Dictionary with tensor statistics
        """
        if percentiles is None:
            percentiles = [50]

        stats = {
            f"{name}_mean": tensor.mean().item(),
            f"{name}_std": tensor.std().item(),
            f"{name}_min": tensor.min().item(),
            f"{name}_max": tensor.max().item(),
        }

        for p in percentiles:
            pct_val = torch.quantile(tensor.float(), p / 100.0)
            stats[f"{name}_p{p}"] = pct_val.item()

        return stats

    def collect_scalar(self, name: str, value: float) -> dict:
        """Collect a scalar value.

        Args:
            name: Name of the scalar
            value: Value to collect

        Returns:
            Dictionary with single scalar entry
        """
        return {name: float(value)}

    def collect_rate(self, spikes: torch.Tensor, name: str, dt: float = 1.0) -> dict:
        """Collect spike rate statistics.

        Args:
            spikes: Binary spike tensor
            name: Prefix for rate name
            dt: Time step in ms

        Returns:
            Dictionary with rate statistics
        """
        rate_hz = (spikes.float().mean() / dt) * MS_PER_SECOND
        return {f"{name}_rate_hz": rate_hz.item()}

    def weight_diagnostics(
        self,
        weights: torch.Tensor,
        prefix: str,
        sparsity_threshold: float = 0.01,
    ) -> dict:
        """Collect comprehensive weight statistics.

        Args:
            weights: Weight tensor
            prefix: Prefix for stat names (e.g., "w_input")
            sparsity_threshold: Threshold below which weights are considered sparse

        Returns:
            Dictionary with weight statistics including sparsity
        """
        stats = {
            f"{prefix}_mean": weights.mean().item(),
            f"{prefix}_std": weights.std().item(),
            f"{prefix}_min": weights.min().item(),
            f"{prefix}_max": weights.max().item(),
            f"{prefix}_sparsity": (weights.abs() < sparsity_threshold).float().mean().item(),
        }
        return stats

    def spike_diagnostics(
        self,
        spikes: torch.Tensor,
        prefix: str,
        dt: float = 1.0,
    ) -> dict:
        """Collect comprehensive spike statistics.

        Args:
            spikes: Binary spike tensor
            prefix: Prefix for stat names (e.g., "l4")
            dt: Time step in ms for rate calculation

        Returns:
            Dictionary with spike statistics
        """
        n_active = spikes.sum().item()
        n_total = spikes.numel()
        rate_hz = (spikes.float().mean() / dt) * MS_PER_SECOND

        stats = {
            f"{prefix}_active_count": int(n_active),
            f"{prefix}_rate_hz": rate_hz.item(),
            f"{prefix}_active_fraction": n_active / max(1, n_total),
        }
        return stats

    def trace_diagnostics(
        self,
        trace: torch.Tensor,
        prefix: str,
        active_threshold: float = 0.01,
    ) -> dict:
        """Collect eligibility trace or activity trace statistics.

        Args:
            trace: Trace tensor (eligibility, NMDA, etc.)
            prefix: Prefix for stat names (e.g., "eligibility")
            active_threshold: Threshold above which trace is considered active

        Returns:
            Dictionary with trace statistics
        """
        stats = {
            f"{prefix}_mean": trace.mean().item(),
            f"{prefix}_max": trace.max().item(),
            f"{prefix}_active_fraction": (trace > active_threshold).float().mean().item(),
        }
        return stats

    def auto_collect_diagnostics(
        self,
        weights: Optional[dict] = None,
        spikes: Optional[dict] = None,
        traces: Optional[dict] = None,
        scalars: Optional[dict] = None,
    ) -> dict:
        """Auto-collect diagnostics from multiple sources.

        This is a convenience method that combines multiple diagnostic collection
        methods. Use this in get_diagnostics() implementations to reduce boilerplate.

        Args:
            weights: Dict mapping weight tensor names to tensors (e.g., {"w_input": tensor})
            spikes: Dict mapping spike tensor names to tensors (e.g., {"l4_spikes": tensor})
            traces: Dict mapping trace tensor names to tensors (e.g., {"eligibility": tensor})
            scalars: Dict mapping scalar names to values (e.g., {"dopamine": 0.5})

        Returns:
            Combined diagnostics dictionary

        Example:
            >>> def get_diagnostics(self) -> Dict[str, Any]:
            >>>     return self.auto_collect_diagnostics(
            >>>         weights={"w_input": self.w_input, "w_recurrent": self.w_rec},
            >>>         spikes={"output": self.output_spikes},
            >>>         scalars={"threshold": self.threshold},
            >>>     )
        """
        diag = {}

        # Collect weight statistics
        if weights:
            for name, tensor in weights.items():
                if tensor is not None:
                    diag.update(self.weight_diagnostics(tensor, name))

        # Collect spike statistics
        if spikes:
            for name, tensor in spikes.items():
                if tensor is not None:
                    diag.update(self.spike_diagnostics(tensor, name))

        # Collect trace statistics
        if traces:
            for name, tensor in traces.items():
                if tensor is not None:
                    diag.update(self.trace_diagnostics(tensor, name))

        # Collect scalar values
        if scalars:
            for name, value in scalars.items():
                diag[name] = float(value) if value is not None else None

        return diag


__all__ = ["DiagnosticCollectorMixin"]
