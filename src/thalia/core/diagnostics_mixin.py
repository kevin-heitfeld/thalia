"""
Diagnostics Mixin for Brain Components.

This module provides a reusable mixin that implements common diagnostic
patterns for brain regions and other components.

The mixin provides:
1. Standard weight statistics (mean, std, min, max, sparsity)
2. Spike/activity statistics (rate, sparsity)
3. Trace statistics (norm, mean)
4. Formatted output helpers

Usage:
======
    class MyRegion(DiagnosticsMixin, nn.Module):
        def __init__(self):
            self.weights = nn.Parameter(torch.randn(100, 50))
            self.spike_trace = None
            
        def get_diagnostics(self) -> Dict[str, Any]:
            diag = {}
            diag.update(self.weight_diagnostics(self.weights, "my_region"))
            if self.spike_trace is not None:
                diag.update(self.trace_diagnostics(self.spike_trace, "spike"))
            return diag

Author: Thalia Project  
Date: December 2025
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch


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
        rate_hz = s.mean().item() * (1000.0 / dt_ms)
        
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
    
    @staticmethod
    def learning_diagnostics(
        ltp: float,
        ltd: float,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute learning update statistics.
        
        Args:
            ltp: Total long-term potentiation (weight increases)
            ltd: Total long-term depression (weight decreases)
            prefix: Prefix for metric names
            
        Returns:
            Dict with learning statistics
        """
        prefix = f"{prefix}_" if prefix else ""
        
        net = ltp + ltd  # ltd is typically negative
        
        return {
            f"{prefix}ltp": ltp,
            f"{prefix}ltd": ltd,
            f"{prefix}net_change": net,
            f"{prefix}plasticity_ratio": ltp / (abs(ltd) + 1e-8),
        }
    
    @staticmethod
    def membrane_diagnostics(
        membrane: torch.Tensor,
        threshold: float = 1.0,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute membrane potential statistics.
        
        Args:
            membrane: Membrane potential tensor
            threshold: Spike threshold for context
            prefix: Prefix for metric names
            
        Returns:
            Dict with membrane statistics
        """
        prefix = f"{prefix}_" if prefix else ""
        
        v = membrane.detach()
        
        return {
            f"{prefix}membrane_mean": v.mean().item(),
            f"{prefix}membrane_std": v.std().item(),
            f"{prefix}membrane_min": v.min().item(),
            f"{prefix}membrane_max": v.max().item(),
            f"{prefix}membrane_near_threshold": (v > threshold * 0.8).float().mean().item(),
        }
    
    @staticmethod
    def similarity_diagnostics(
        pattern_a: torch.Tensor,
        pattern_b: torch.Tensor,
        prefix: str = "",
        eps: float = 1e-8,
    ) -> Dict[str, float]:
        """Compute similarity statistics between two patterns.
        
        Args:
            pattern_a: First pattern tensor
            pattern_b: Second pattern tensor
            prefix: Prefix for metric names
            eps: Epsilon for numerical stability
            
        Returns:
            Dict with similarity statistics
        """
        prefix = f"{prefix}_" if prefix else ""
        
        a = pattern_a.detach().float().flatten()
        b = pattern_b.detach().float().flatten()
        
        # Cosine similarity
        norm_a = a.norm() + eps
        norm_b = b.norm() + eps
        cosine = (a @ b) / (norm_a * norm_b)
        
        # Overlap (for binary patterns)
        if a.max() <= 1 and b.max() <= 1:
            overlap = ((a > 0.5) & (b > 0.5)).sum().item()
            union = ((a > 0.5) | (b > 0.5)).sum().item()
            jaccard = overlap / (union + 1e-8)
        else:
            overlap = 0.0
            jaccard = 0.0
        
        return {
            f"{prefix}cosine_similarity": cosine.item(),
            f"{prefix}overlap": overlap,
            f"{prefix}jaccard": jaccard,
        }
    
    def collect_all_diagnostics(
        self,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        spikes: Optional[Dict[str, torch.Tensor]] = None,
        traces: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Collect diagnostics from multiple sources.
        
        Convenience method to gather all diagnostics at once.
        
        Args:
            weights: Dict mapping name → weight tensor
            spikes: Dict mapping name → spike tensor
            traces: Dict mapping name → trace tensor
            
        Returns:
            Combined diagnostics dict
        """
        diag: Dict[str, Any] = {}
        
        if weights:
            for name, w in weights.items():
                diag.update(self.weight_diagnostics(w, name))
                
        if spikes:
            for name, s in spikes.items():
                diag.update(self.spike_diagnostics(s, name))
                
        if traces:
            for name, t in traces.items():
                diag.update(self.trace_diagnostics(t, name))
                
        return diag
