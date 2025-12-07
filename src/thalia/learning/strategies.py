"""
Learning Rule Strategies: Pluggable Learning Algorithms for Brain Regions.

This module implements a Strategy pattern for learning rules, allowing regions
to compose and switch between different learning algorithms without code duplication.

Design Philosophy:
==================
Instead of each region implementing its own learn() method with duplicated
STDP/BCM/three-factor logic, regions can compose strategies:

    region.learning_strategy = STDPStrategy(config)
    # or compose:
    region.learning_strategy = CompositeStrategy([
        STDPStrategy(stdp_config),
        BCMModulationStrategy(bcm_config),
    ])

Each strategy handles:
- Weight update computation
- Trace management
- Bounds enforcement
- Metrics collection

Supported Strategies:
=====================
- HebbianStrategy: Basic Hebbian learning (Δw ∝ pre × post)
- STDPStrategy: Spike-timing dependent plasticity
- BCMStrategy: Bienenstock-Cooper-Munro with sliding threshold
- ThreeFactorStrategy: RL eligibility × neuromodulator
- ErrorCorrectiveStrategy: Supervised delta rule
- CompositeStrategy: Compose multiple strategies

Usage:
======
    # Create strategy
    stdp = STDPStrategy(STDPConfig(a_plus=0.01, a_minus=0.012))
    
    # Apply in region's learn() method:
    def learn(self, pre, post, **kwargs):
        return stdp.apply(self.weights, pre, post)
    
    # Or use composable modulation:
    composite = CompositeStrategy([
        STDPStrategy(stdp_config),
        BCMModulationStrategy(bcm_config),  # Modulates STDP output
    ])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Protocol

import torch
import torch.nn as nn

from thalia.config.base import LearningComponentConfig


# =============================================================================
# Strategy Configuration Dataclasses
# =============================================================================

@dataclass
class LearningConfig(LearningComponentConfig):
    """Base configuration for all learning strategies.
    
    Inherits learning_rate, enabled from LearningComponentConfig.
    Inherits device, dtype, seed from BaseConfig.
    """
    w_min: float = 0.0
    w_max: float = 1.0
    soft_bounds: bool = True  # Use soft bounds (multiplicative) vs hard clamp


@dataclass
class HebbianConfig(LearningConfig):
    """Configuration for basic Hebbian learning."""
    normalize: bool = False  # Normalize weight updates
    decay_rate: float = 0.0  # Weight decay per timestep


@dataclass
class STDPConfig(LearningConfig):
    """Configuration for STDP learning.
    
    Classic STDP window:
        Δw = A+ × exp(-Δt/τ+) if post after pre (Δt > 0, LTP)
        Δw = -A- × exp(Δt/τ-) if pre after post (Δt < 0, LTD)
    
    In trace-based form:
        LTP: pre_trace × post_spike
        LTD: post_trace × pre_spike
    """
    a_plus: float = 0.01      # LTP amplitude
    a_minus: float = 0.012    # LTD amplitude (slightly larger for stability)
    tau_plus: float = 20.0    # LTP time constant (ms)
    tau_minus: float = 20.0   # LTD time constant (ms)
    dt: float = 1.0           # Simulation timestep (ms)


@dataclass
class BCMConfig(LearningConfig):
    """Configuration for BCM learning.
    
    BCM function: φ(c, θ) = c(c - θ)
    - c > θ: LTP
    - c < θ: LTD
    
    Threshold adapts: θ → E[c²]
    """
    tau_theta: float = 5000.0   # Threshold time constant (ms)
    theta_init: float = 0.01    # Initial threshold
    theta_min: float = 1e-6     # Minimum threshold
    theta_max: float = 1.0      # Maximum threshold
    power: float = 2.0          # Power for threshold (c^p)
    dt: float = 1.0             # Simulation timestep


@dataclass
class ThreeFactorConfig(LearningConfig):
    """Configuration for three-factor RL learning.
    
    Three-factor rule: Δw = lr × eligibility × modulator
    - Eligibility: accumulated spike timing correlations
    - Modulator: dopamine, reward signal, or error
    """
    eligibility_tau: float = 100.0    # Eligibility trace decay (ms)
    modulator_tau: float = 50.0       # Modulator decay (ms)
    dt: float = 1.0


@dataclass
class ErrorCorrectiveConfig(LearningConfig):
    """Configuration for supervised error-corrective learning.
    
    Delta rule: Δw = lr × pre × (target - actual)
    """
    error_threshold: float = 0.01  # Minimum error to trigger learning


# =============================================================================
# Strategy Protocol
# =============================================================================

class LearningStrategy(Protocol):
    """Protocol defining the interface for all learning strategies."""
    
    def compute_update(
        self,
        weights: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weight update using this learning rule.
        
        Args:
            weights: Current weight matrix [n_post, n_pre]
            pre: Presynaptic activity [batch, n_pre]
            post: Postsynaptic activity [batch, n_post]
            **kwargs: Strategy-specific inputs (target, reward, etc.)
            
        Returns:
            Tuple of:
                - Updated weight matrix
                - Dict of learning metrics
        """
        ...
    
    def reset_state(self) -> None:
        """Reset strategy state (traces, thresholds, etc.)."""
        ...


# =============================================================================
# Strategy Implementations
# =============================================================================

class BaseStrategy(nn.Module, ABC):
    """Base class for learning strategies with common functionality."""
    
    def __init__(self, config: LearningConfig):
        super().__init__()
        self.config = config
    
    def _apply_bounds(
        self,
        weights: torch.Tensor,
        dw: torch.Tensor,
    ) -> torch.Tensor:
        """Apply weight bounds using soft or hard constraints."""
        cfg = self.config
        
        if cfg.soft_bounds:
            # Soft bounds: scale update by headroom/footroom
            headroom = (cfg.w_max - weights) / (cfg.w_max - cfg.w_min + 1e-8)
            footroom = (weights - cfg.w_min) / (cfg.w_max - cfg.w_min + 1e-8)
            dw = torch.where(
                dw > 0,
                dw * headroom.clamp(0, 1),
                dw * footroom.clamp(0, 1),
            )
        
        # Apply update and clamp
        new_weights = (weights + dw).clamp(cfg.w_min, cfg.w_max)
        return new_weights
    
    def _compute_metrics(
        self,
        old_weights: torch.Tensor,
        new_weights: torch.Tensor,
        dw: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute standard learning metrics."""
        actual_dw = new_weights - old_weights
        
        ltp_mask = actual_dw > 0
        ltd_mask = actual_dw < 0
        
        return {
            "ltp": actual_dw[ltp_mask].sum().item() if ltp_mask.any() else 0.0,
            "ltd": actual_dw[ltd_mask].sum().item() if ltd_mask.any() else 0.0,
            "net_change": actual_dw.sum().item(),
            "mean_change": actual_dw.abs().mean().item(),
            "weight_mean": new_weights.mean().item(),
        }
    
    @abstractmethod
    def compute_update(
        self,
        weights: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply learning rule."""
        pass
    
    def reset_state(self) -> None:
        """Reset strategy state. Override in subclasses with state."""
        pass


class HebbianStrategy(BaseStrategy):
    """Basic Hebbian learning: Δw ∝ pre × post.
    
    The simplest correlation-based learning rule. Strengthens connections
    where pre and post are co-active.
    """
    
    def __init__(self, config: Optional[HebbianConfig] = None):
        super().__init__(config or HebbianConfig())
        self.hebbian_config = self.config  # type: HebbianConfig
    
    def compute_update(
        self,
        weights: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply Hebbian learning.
        
        Δw[j,i] = lr × pre[i] × post[j]
        """
        cfg = self.hebbian_config
        
        # Ensure 2D
        if pre.dim() == 1:
            pre = pre.unsqueeze(0)
        if post.dim() == 1:
            post = post.unsqueeze(0)
        
        # Hebbian outer product, averaged over batch
        # dw[j,i] = mean(post[b,j] × pre[b,i])
        dw = torch.einsum('bj,bi->ji', post.float(), pre.float()) / pre.shape[0]
        
        # Scale by learning rate
        dw = cfg.learning_rate * dw
        
        # Optional normalization
        if cfg.normalize and dw.abs().max() > 0:
            dw = dw / dw.abs().max()
            dw = cfg.learning_rate * dw
        
        # Optional weight decay
        if cfg.decay_rate > 0:
            dw = dw - cfg.decay_rate * weights
        
        # Apply bounds and compute new weights
        old_weights = weights.clone()
        new_weights = self._apply_bounds(weights, dw)
        
        metrics = self._compute_metrics(old_weights, new_weights, dw)
        return new_weights, metrics


class STDPStrategy(BaseStrategy):
    """Spike-Timing Dependent Plasticity.
    
    Trace-based STDP:
        - LTP: pre_trace × post_spike (pre before post)
        - LTD: post_trace × pre_spike (post before pre)
    """
    
    def __init__(self, config: Optional[STDPConfig] = None):
        super().__init__(config or STDPConfig())
        self.stdp_config: STDPConfig = self.config  # type: ignore
        
        # Compute decay factors
        dt = self.stdp_config.dt
        self.register_buffer(
            "decay_pre",
            torch.tensor(1.0 - dt / self.stdp_config.tau_plus, dtype=torch.float32),
        )
        self.register_buffer(
            "decay_post",
            torch.tensor(1.0 - dt / self.stdp_config.tau_minus, dtype=torch.float32),
        )
        
        # Traces (initialized lazily)
        self.pre_trace: Optional[torch.Tensor] = None
        self.post_trace: Optional[torch.Tensor] = None
    
    def reset(self) -> None:
        """Reset traces."""
        self.pre_trace = None
        self.post_trace = None
    
    def _update_traces(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update and return eligibility traces."""
        batch_size = pre.shape[0]
        n_pre = pre.shape[-1]
        n_post = post.shape[-1]
        device = pre.device
        
        # Initialize traces if needed
        if self.pre_trace is None or self.pre_trace.shape[-1] != n_pre:
            self.pre_trace = torch.zeros(batch_size, n_pre, device=device)
        if self.post_trace is None or self.post_trace.shape[-1] != n_post:
            self.post_trace = torch.zeros(batch_size, n_post, device=device)
        
        # Decay traces
        self.pre_trace = self.decay_pre * self.pre_trace
        self.post_trace = self.decay_post * self.post_trace
        
        # Add new spikes
        self.pre_trace = self.pre_trace + pre.float()
        self.post_trace = self.post_trace + post.float()
        
        return self.pre_trace, self.post_trace
    
    def compute_update(
        self,
        weights: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply STDP learning.
        
        LTP: A+ × pre_trace × post_spike
        LTD: A- × post_trace × pre_spike
        """
        cfg = self.stdp_config
        
        # Ensure 2D
        if pre.dim() == 1:
            pre = pre.unsqueeze(0)
        if post.dim() == 1:
            post = post.unsqueeze(0)
        
        # Update traces
        pre_trace, post_trace = self._update_traces(pre, post)
        
        # LTP: pre was active before post fired now
        # dw_ltp[j,i] = A+ × pre_trace[i] × post[j]
        ltp = cfg.a_plus * torch.einsum('bj,bi->ji', post.float(), pre_trace) / pre.shape[0]
        
        # LTD: post was active before pre fired now
        # dw_ltd[j,i] = -A- × post_trace[j] × pre[i]
        ltd = cfg.a_minus * torch.einsum('bj,bi->ji', post_trace, pre.float()) / pre.shape[0]
        
        dw = ltp - ltd
        
        # Apply bounds
        old_weights = weights.clone()
        new_weights = self._apply_bounds(weights, dw)
        
        metrics = self._compute_metrics(old_weights, new_weights, dw)
        metrics["pre_trace_mean"] = pre_trace.mean().item()
        metrics["post_trace_mean"] = post_trace.mean().item()
        
        return new_weights, metrics


class BCMStrategy(BaseStrategy):
    """Bienenstock-Cooper-Munro learning with sliding threshold.
    
    BCM function: φ(c, θ) = c(c - θ)
    Weight update: Δw = lr × pre × φ(post, θ)
    
    The threshold θ adapts to track E[post²], providing automatic
    metaplasticity that stabilizes learning.
    """
    
    def __init__(self, config: Optional[BCMConfig] = None):
        super().__init__(config or BCMConfig())
        self.bcm_config: BCMConfig = self.config  # type: ignore
        
        # Compute decay factor
        dt = self.bcm_config.dt
        tau = self.bcm_config.tau_theta
        self.register_buffer(
            "decay_theta",
            torch.tensor(1.0 - dt / tau, dtype=torch.float32),
        )
        
        # Sliding threshold (per-neuron)
        self.theta: Optional[torch.Tensor] = None
    
    def reset(self) -> None:
        """Reset threshold."""
        self.theta = None
    
    def _init_theta(self, n_post: int, device: torch.device) -> None:
        """Initialize threshold if needed."""
        if self.theta is None or self.theta.shape[0] != n_post:
            self.theta = torch.full(
                (n_post,),
                self.bcm_config.theta_init,
                device=device,
            )
    
    def compute_phi(self, post: torch.Tensor) -> torch.Tensor:
        """Compute BCM modulation factor.
        
        φ(c, θ) = c(c - θ) / θ (normalized for stability)
        
        Returns per-neuron modulation factor.
        """
        if self.theta is None:
            return torch.ones_like(post)
        
        # Average over batch if needed
        c = post.float()
        if c.dim() > 1:
            c = c.mean(dim=0)
        
        phi = c * (c - self.theta) / (self.theta + 1e-8)
        return phi
    
    def _update_theta(self, post: torch.Tensor) -> None:
        """Update sliding threshold."""
        cfg = self.bcm_config
        
        # Compute post^p averaged over batch
        c = post.float()
        if c.dim() > 1:
            c = c.mean(dim=0)
        
        c_p = c.pow(cfg.power)
        
        # EMA update
        self.theta = self.decay_theta * self.theta + (1 - self.decay_theta) * c_p
        
        # Clamp
        self.theta = self.theta.clamp(cfg.theta_min, cfg.theta_max)
    
    def compute_update(
        self,
        weights: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply BCM learning.
        
        Δw[j,i] = lr × pre[i] × φ(post[j], θ[j])
        """
        cfg = self.bcm_config
        
        # Ensure 2D
        if pre.dim() == 1:
            pre = pre.unsqueeze(0)
        if post.dim() == 1:
            post = post.unsqueeze(0)
        
        n_post = post.shape[-1]
        self._init_theta(n_post, post.device)
        
        # Compute BCM modulation
        phi = self.compute_phi(post)  # (n_post,)
        
        # Weight update: dw[j,i] = lr × pre[i] × φ[j]
        # Expand phi for broadcasting: (n_post,) -> (n_post, 1)
        dw = cfg.learning_rate * torch.outer(phi, pre.float().mean(dim=0))
        
        # Update threshold
        self._update_theta(post)
        
        # Apply bounds
        old_weights = weights.clone()
        new_weights = self._apply_bounds(weights, dw)
        
        metrics = self._compute_metrics(old_weights, new_weights, dw)
        metrics["theta_mean"] = self.theta.mean().item() if self.theta is not None else 0.0
        metrics["phi_mean"] = phi.mean().item()
        
        return new_weights, metrics


class ThreeFactorStrategy(BaseStrategy):
    """Three-factor reinforcement learning rule.
    
    Δw = lr × eligibility × modulator
    
    - Eligibility: accumulated spike-timing correlations (Hebbian)
    - Modulator: dopamine, reward signal, or TD error
    
    Key insight: Without modulator, NO learning occurs (not reduced, NONE).
    """
    
    def __init__(self, config: Optional[ThreeFactorConfig] = None):
        super().__init__(config or ThreeFactorConfig())
        self.tf_config: ThreeFactorConfig = self.config  # type: ignore
        
        # Decay factors
        dt = self.tf_config.dt
        self.register_buffer(
            "decay_elig",
            torch.tensor(1.0 - dt / self.tf_config.eligibility_tau, dtype=torch.float32),
        )
        
        # Eligibility trace (Hebbian correlation)
        self.eligibility: Optional[torch.Tensor] = None
    
    def reset(self) -> None:
        """Reset eligibility trace."""
        self.eligibility = None
    
    def update_eligibility(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:
        """Update eligibility trace with current activity.
        
        Eligibility accumulates Hebbian correlations until modulator arrives.
        """
        # Ensure 2D
        if pre.dim() == 1:
            pre = pre.unsqueeze(0)
        if post.dim() == 1:
            post = post.unsqueeze(0)
        
        n_post = post.shape[-1]
        n_pre = pre.shape[-1]
        device = pre.device
        
        # Initialize if needed
        if self.eligibility is None or self.eligibility.shape != (n_post, n_pre):
            self.eligibility = torch.zeros(n_post, n_pre, device=device)
        
        # Decay existing eligibility
        self.eligibility = self.decay_elig * self.eligibility
        
        # Add new Hebbian correlation
        hebbian = torch.einsum('bj,bi->ji', post.float(), pre.float()) / pre.shape[0]
        self.eligibility = self.eligibility + hebbian
        
        return self.eligibility
    
    def compute_update(
        self,
        weights: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        modulator: float = 0.0,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply three-factor learning.
        
        Args:
            weights: Current weights
            pre: Presynaptic activity
            post: Postsynaptic activity
            modulator: Neuromodulatory signal (dopamine, reward, etc.)
        
        Returns:
            Updated weights and metrics
        """
        cfg = self.tf_config
        
        # Update eligibility
        self.update_eligibility(pre, post)
        
        # No learning without modulator
        if abs(modulator) < 0.01:
            return weights, {
                "modulator": modulator,
                "eligibility_mean": self.eligibility.mean().item() if self.eligibility is not None else 0.0,
                "ltp": 0.0,
                "ltd": 0.0,
                "net_change": 0.0,
            }
        
        # Three-factor update
        dw = cfg.learning_rate * self.eligibility * modulator
        
        # Apply bounds
        old_weights = weights.clone()
        new_weights = self._apply_bounds(weights, dw)
        
        metrics = self._compute_metrics(old_weights, new_weights, dw)
        metrics["modulator"] = modulator
        metrics["eligibility_mean"] = self.eligibility.mean().item() if self.eligibility is not None else 0.0
        
        return new_weights, metrics


class ErrorCorrectiveStrategy(BaseStrategy):
    """Supervised error-corrective learning (delta rule).
    
    Δw = lr × pre × (target - actual)
    
    Used in cerebellum-like circuits for supervised motor learning.
    """
    
    def __init__(self, config: Optional[ErrorCorrectiveConfig] = None):
        super().__init__(config or ErrorCorrectiveConfig())
        self.ec_config: ErrorCorrectiveConfig = self.config  # type: ignore
    
    def compute_update(
        self,
        weights: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply error-corrective learning.
        
        Args:
            weights: Current weights
            pre: Presynaptic activity (input)
            post: Postsynaptic activity (actual output)
            target: Target output
        
        Returns:
            Updated weights and metrics
        """
        cfg = self.ec_config
        
        if target is None:
            return weights, {"error": 0.0, "ltp": 0.0, "ltd": 0.0, "net_change": 0.0}
        
        # Ensure 2D
        if pre.dim() == 1:
            pre = pre.unsqueeze(0)
        if post.dim() == 1:
            post = post.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        
        # Compute error
        error = target.float() - post.float()
        
        # Check threshold
        if error.abs().max() < cfg.error_threshold:
            return weights, {"error": 0.0, "ltp": 0.0, "ltd": 0.0, "net_change": 0.0}
        
        # Delta rule: dw[j,i] = lr × error[j] × pre[i]
        dw = cfg.learning_rate * torch.einsum('bj,bi->ji', error, pre.float()) / pre.shape[0]
        
        # Apply bounds
        old_weights = weights.clone()
        new_weights = self._apply_bounds(weights, dw)
        
        metrics = self._compute_metrics(old_weights, new_weights, dw)
        metrics["error"] = error.abs().mean().item()
        
        return new_weights, metrics


class CompositeStrategy(BaseStrategy):
    """Compose multiple learning strategies.
    
    Strategies are applied sequentially, with each one potentially
    modulating the output of the previous.
    
    Example:
        # STDP modulated by BCM
        composite = CompositeStrategy([
            STDPStrategy(stdp_config),
            BCMModulationStrategy(bcm_config),  # Scales STDP output
        ])
    """
    
    def __init__(
        self,
        strategies: List[BaseStrategy],
        config: Optional[LearningConfig] = None,
    ):
        super().__init__(config or LearningConfig())
        self.strategies = nn.ModuleList(strategies)
    
    def reset_state(self) -> None:
        """Reset all sub-strategies."""
        for strategy in self.strategies:
            if isinstance(strategy, BaseStrategy):
                strategy.reset_state()
    
    def compute_update(
        self,
        weights: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply all strategies sequentially.
        
        Each strategy receives the weights output by the previous one.
        Metrics are merged with prefixes.
        """
        current_weights = weights
        all_metrics: Dict[str, float] = {}
        
        for i, strategy in enumerate(self.strategies):
            if isinstance(strategy, BaseStrategy):
                current_weights, metrics = strategy.compute_update(
                    current_weights, pre, post, **kwargs
                )
                # Prefix metrics with strategy index
                for key, value in metrics.items():
                    all_metrics[f"s{i}_{key}"] = value
        
        return current_weights, all_metrics


# =============================================================================
# Convenience: Strategy Factory
# =============================================================================

def create_strategy(
    rule_name: str,
    **config_kwargs: Any,
) -> BaseStrategy:
    """Factory function to create learning strategies by name.
    
    Args:
        rule_name: One of 'hebbian', 'stdp', 'bcm', 'three_factor', 'error_corrective'
        **config_kwargs: Configuration parameters for the strategy
    
    Returns:
        Configured learning strategy
    
    Example:
        strategy = create_strategy('stdp', a_plus=0.02, a_minus=0.02)
    """
    name = rule_name.lower().replace('-', '_').replace(' ', '_')
    
    if name == 'hebbian':
        return HebbianStrategy(HebbianConfig(**config_kwargs))
    elif name == 'stdp':
        return STDPStrategy(STDPConfig(**config_kwargs))
    elif name == 'bcm':
        return BCMStrategy(BCMConfig(**config_kwargs))
    elif name in ('three_factor', 'threefactor', 'rl'):
        return ThreeFactorStrategy(ThreeFactorConfig(**config_kwargs))
    elif name in ('error_corrective', 'delta', 'supervised'):
        return ErrorCorrectiveStrategy(ErrorCorrectiveConfig(**config_kwargs))
    else:
        raise ValueError(f"Unknown learning rule: {rule_name}")
