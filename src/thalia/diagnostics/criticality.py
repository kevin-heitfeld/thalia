"""
Criticality Monitoring: Self-Organized Criticality for Neural Networks.

Healthy neural systems operate near a "critical point" — the boundary between
ordered and chaotic dynamics. At criticality, networks exhibit:

1. OPTIMAL INFORMATION PROCESSING: Maximum dynamic range and sensitivity
2. OPTIMAL TRANSMISSION: Information propagates efficiently without dying out
3. LONG-RANGE CORRELATIONS: Coordinated activity across the network
4. POWER-LAW DISTRIBUTIONS: Avalanches of all sizes (scale-free)

The Branching Ratio:
====================
The key metric for criticality is the branching ratio σ:

    σ = E[spikes(t+1)] / E[spikes(t)]

- σ < 1: SUBCRITICAL — Activity dies out exponentially (boring/silent)
- σ = 1: CRITICAL — Activity is self-sustaining (optimal)
- σ > 1: SUPERCRITICAL — Activity explodes exponentially (seizure/chaos)

This module monitors the branching ratio and can optionally apply corrections
to keep the network near criticality.

Biological Basis:
=================
- Cortical networks in vivo operate near criticality
- Sleep transitions networks between subcritical and supercritical states
- Criticality may be maintained by synaptic homeostasis
- Disrupted criticality is associated with epilepsy, autism, disorders

Key References:
- Beggs & Plenz (2003): Neuronal avalanches in neocortical circuits
- Shew & Plenz (2013): The functional benefits of criticality
- Wilting & Priesemann (2019): 25 years of criticality in neuroscience

Usage:
======
    from thalia.diagnostics.criticality import CriticalityMonitor, CriticalityConfig
    
    monitor = CriticalityMonitor(CriticalityConfig(target_branching=1.0))
    
    # Each timestep:
    monitor.update(current_spikes)
    
    # Check network state:
    print(monitor.get_state())  # "subcritical", "critical", or "supercritical"
    
    # Get weight scaling to move toward criticality:
    scale = monitor.get_weight_scaling()
    weights = weights * scale
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, Any, List, Optional

import torch
import torch.nn as nn


class CriticalityState(Enum):
    """Network criticality state."""
    SUBCRITICAL = "subcritical"
    CRITICAL = "critical"
    SUPERCRITICAL = "supercritical"


@dataclass
class CriticalityConfig:
    """Configuration for criticality monitoring.
    
    Attributes:
        target_branching: Target branching ratio (default: 1.0 = critical)
            Slight subcriticality (0.95-0.99) is often observed in healthy cortex.
            
        window_size: Number of timesteps for branching estimation
            Larger = more stable estimate, slower response.
            
        critical_tolerance: Tolerance for "critical" classification
            Network is "critical" if |σ - target| < tolerance.
            
        correction_enabled: Whether to compute correction scaling
            If True, get_weight_scaling() returns a correction factor.
            
        correction_rate: How fast to correct toward criticality
            Higher = faster correction, risk of oscillation.
            Must be VERY slow to avoid destabilizing the network.
            
        scale_min: Minimum weight scaling factor
            Prevents weights from being scaled too low.
            
        scale_max: Maximum weight scaling factor
            Prevents weights from being scaled too high.
            
        min_spikes_for_estimate: Minimum total spikes for valid estimate
            Below this, branching ratio is unreliable.
    """
    target_branching: float = 1.0
    window_size: int = 100
    critical_tolerance: float = 0.1
    correction_enabled: bool = True
    correction_rate: float = 0.0001  # Very slow!
    scale_min: float = 0.5
    scale_max: float = 2.0
    min_spikes_for_estimate: int = 10


class CriticalityMonitor(nn.Module):
    """Monitor network criticality via branching ratio.
    
    Tracks spike counts over time and estimates the branching ratio,
    which indicates whether the network is subcritical, critical, or
    supercritical.
    
    Can optionally provide weight scaling factors to move the network
    toward criticality.
    """
    
    def __init__(
        self,
        config: Optional[CriticalityConfig] = None,
    ):
        super().__init__()
        self.config = config or CriticalityConfig()
        
        # History of spike counts
        self._spike_counts: Deque[float] = deque(maxlen=self.config.window_size)
        
        # Current branching ratio estimate
        self._branching_ratio: float = 1.0
        
        # Current weight scaling factor
        self._weight_scale: float = 1.0
        
        # History for diagnostics
        self._branching_history: List[float] = []
        
    def reset(self):
        """Reset all state."""
        self._spike_counts.clear()
        self._branching_ratio = 1.0
        self._weight_scale = 1.0
        self._branching_history.clear()
    
    def update(self, spikes: torch.Tensor) -> float:
        """Update with current spike tensor.
        
        Args:
            spikes: Spike tensor (any shape, will be summed)
            
        Returns:
            Current branching ratio estimate
        """
        cfg = self.config
        
        # Count total spikes
        spike_count = spikes.float().sum().item()
        self._spike_counts.append(spike_count)
        
        # Need at least 2 timesteps for branching ratio
        if len(self._spike_counts) < 2:
            return self._branching_ratio
        
        # Compute branching ratio from history
        self._branching_ratio = self._compute_branching_ratio()
        
        # Update weight scaling if correction is enabled
        if cfg.correction_enabled:
            self._update_weight_scaling()
        
        # Store history
        self._branching_history.append(self._branching_ratio)
        if len(self._branching_history) > 10000:
            self._branching_history = self._branching_history[-5000:]
        
        return self._branching_ratio
    
    def _compute_branching_ratio(self) -> float:
        """Compute branching ratio from spike history.
        
        Uses the slope method: fit a line to log(spikes(t)) and extract
        the branching ratio as exp(slope).
        
        For simplicity, we use the ratio of consecutive averages.
        """
        cfg = self.config
        counts = list(self._spike_counts)
        
        # Check if we have enough spikes
        total_spikes = sum(counts)
        if total_spikes < cfg.min_spikes_for_estimate:
            return self._branching_ratio  # Keep previous estimate
        
        # Simple method: ratio of "second half" to "first half"
        # This is more stable than consecutive timestep ratios
        n = len(counts)
        if n < 4:
            # Use consecutive ratio
            prev = counts[-2] if counts[-2] > 0 else 1e-8
            curr = counts[-1]
            return curr / prev
        
        # Split into first and second half
        mid = n // 2
        first_half = sum(counts[:mid]) / mid
        second_half = sum(counts[mid:]) / (n - mid)
        
        # Compute per-timestep ratio
        if first_half < 1e-8:
            return 1.0  # No activity, assume critical
        
        # Branching ratio per timestep
        # If first_half → second_half over (n-mid) timesteps,
        # then σ^(n-mid) ≈ second_half / first_half
        # So σ ≈ (second_half / first_half)^(1/(n-mid))
        ratio = second_half / first_half
        timesteps = (n - mid)
        
        # Handle edge cases
        if ratio <= 0:
            return 0.5  # Subcritical
        
        branching = ratio ** (1.0 / timesteps)
        
        # Clamp to reasonable range
        return max(0.1, min(10.0, branching))
    
    def _update_weight_scaling(self):
        """Update weight scaling to move toward criticality."""
        cfg = self.config
        
        # Error from target
        error = self._branching_ratio - cfg.target_branching
        
        # If supercritical (σ > target), reduce weights
        # If subcritical (σ < target), increase weights
        adjustment = -cfg.correction_rate * error
        
        # Apply adjustment
        self._weight_scale = self._weight_scale * (1 + adjustment)
        
        # Clamp to safe range
        self._weight_scale = max(cfg.scale_min, 
                                  min(cfg.scale_max, self._weight_scale))
    
    def get_branching_ratio(self) -> float:
        """Get current branching ratio estimate."""
        return self._branching_ratio
    
    def get_weight_scaling(self) -> float:
        """Get weight scaling factor to move toward criticality.
        
        Multiply all weights by this factor.
        
        Returns:
            Scaling factor (>1 = boost activity, <1 = reduce activity)
        """
        return self._weight_scale
    
    def get_state(self) -> CriticalityState:
        """Get current criticality state classification."""
        cfg = self.config
        
        error = abs(self._branching_ratio - cfg.target_branching)
        
        if error <= cfg.critical_tolerance:
            return CriticalityState.CRITICAL
        elif self._branching_ratio < cfg.target_branching:
            return CriticalityState.SUBCRITICAL
        else:
            return CriticalityState.SUPERCRITICAL
    
    def is_critical(self) -> bool:
        """Check if network is currently in critical state."""
        return self.get_state() == CriticalityState.CRITICAL
    
    def get_error(self) -> float:
        """Get error from target branching ratio."""
        return self._branching_ratio - self.config.target_branching
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        return {
            "branching_ratio": self._branching_ratio,
            "target_branching": self.config.target_branching,
            "error": self.get_error(),
            "state": self.get_state().value,
            "weight_scale": self._weight_scale,
            "window_length": len(self._spike_counts),
            "total_spikes_in_window": sum(self._spike_counts),
            "history_length": len(self._branching_history),
        }
    
    def forward(self, spikes: torch.Tensor) -> float:
        """Forward pass: same as update."""
        return self.update(spikes)


class AvalancheAnalyzer:
    """Analyze neural avalanches for criticality signatures.
    
    At criticality, avalanche sizes follow a power-law distribution:
    P(size) ∝ size^(-τ) with τ ≈ 1.5 for 2D networks.
    
    This class collects avalanche statistics and can test for
    power-law distributions.
    """
    
    def __init__(
        self,
        silence_threshold: int = 0,
        max_avalanches: int = 10000,
    ):
        """Initialize avalanche analyzer.
        
        Args:
            silence_threshold: Spike count below which network is "silent"
            max_avalanches: Maximum avalanches to store
        """
        self.silence_threshold = silence_threshold
        self.max_avalanches = max_avalanches
        
        # Avalanche size history
        self._avalanche_sizes: List[int] = []
        
        # Current avalanche tracking
        self._in_avalanche: bool = False
        self._current_size: int = 0
    
    def reset(self):
        """Reset avalanche history."""
        self._avalanche_sizes.clear()
        self._in_avalanche = False
        self._current_size = 0
    
    def update(self, spikes: torch.Tensor):
        """Update with current spikes.
        
        Args:
            spikes: Spike tensor
        """
        spike_count = int(spikes.float().sum().item())
        
        if spike_count > self.silence_threshold:
            # Network is active
            if not self._in_avalanche:
                # Starting new avalanche
                self._in_avalanche = True
                self._current_size = spike_count
            else:
                # Continuing avalanche
                self._current_size += spike_count
        else:
            # Network is silent
            if self._in_avalanche:
                # Avalanche just ended
                self._avalanche_sizes.append(self._current_size)
                self._in_avalanche = False
                self._current_size = 0
                
                # Limit history size
                if len(self._avalanche_sizes) > self.max_avalanches:
                    self._avalanche_sizes = self._avalanche_sizes[-self.max_avalanches // 2:]
    
    def get_avalanche_sizes(self) -> List[int]:
        """Get list of avalanche sizes."""
        return self._avalanche_sizes.copy()
    
    def get_mean_size(self) -> float:
        """Get mean avalanche size."""
        if not self._avalanche_sizes:
            return 0.0
        return sum(self._avalanche_sizes) / len(self._avalanche_sizes)
    
    def get_size_distribution(self, n_bins: int = 20) -> Dict[str, List[float]]:
        """Get avalanche size distribution.
        
        Args:
            n_bins: Number of bins for histogram
            
        Returns:
            Dict with 'bin_edges' and 'counts'
        """
        if not self._avalanche_sizes:
            return {"bin_edges": [], "counts": []}
        
        # Log-spaced bins for power-law
        min_size = max(1, min(self._avalanche_sizes))
        max_size = max(self._avalanche_sizes)
        
        if min_size == max_size:
            return {
                "bin_edges": [float(min_size)],
                "counts": [float(len(self._avalanche_sizes))],
            }
        
        import math
        log_min = math.log10(min_size)
        log_max = math.log10(max_size)
        log_edges = [log_min + i * (log_max - log_min) / n_bins 
                     for i in range(n_bins + 1)]
        edges = [10 ** e for e in log_edges]
        
        # Count avalanches in each bin
        counts = [0.0] * n_bins
        for size in self._avalanche_sizes:
            for i in range(n_bins):
                if edges[i] <= size < edges[i + 1]:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1  # Last bin includes max
        
        return {"bin_edges": edges, "counts": counts}
    
    def estimate_power_law_exponent(self) -> Optional[float]:
        """Estimate power-law exponent τ from avalanche sizes.
        
        Uses maximum likelihood estimation for power-law:
        τ = 1 + n / Σ log(size / size_min)
        
        Returns:
            Estimated τ, or None if insufficient data
        """
        if len(self._avalanche_sizes) < 50:
            return None  # Need more data
        
        import math
        
        sizes = [s for s in self._avalanche_sizes if s > 0]
        if not sizes:
            return None
        
        size_min = min(sizes)
        n = len(sizes)
        
        log_sum = sum(math.log(s / size_min) for s in sizes)
        
        if log_sum < 1e-8:
            return None
        
        tau = 1 + n / log_sum
        return tau
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        return {
            "n_avalanches": len(self._avalanche_sizes),
            "mean_size": self.get_mean_size(),
            "max_size": max(self._avalanche_sizes) if self._avalanche_sizes else 0,
            "power_law_exponent": self.estimate_power_law_exponent(),
            "in_avalanche": self._in_avalanche,
            "current_size": self._current_size,
        }
