"""
Learning Rule Strategies: Pluggable Learning Algorithms for Brain Components.

This module implements a Strategy pattern for learning rules, allowing regions
and pathways to compose and switch between different learning algorithms without
code duplication.

Design Philosophy
==================
Instead of each component implementing its own learning logic with duplicated
STDP/BCM/three-factor code, components can compose strategies:

.. code-block:: python

    # Simple usage
    region.learning_strategy = STDPStrategy(config)

    # Or compose multiple strategies
    region.learning_strategy = CompositeStrategy([
        STDPStrategy(stdp_config),
        BCMModulationStrategy(bcm_config),  # Modulates STDP output
    ])

Each strategy encapsulates:
- Weight update computation
- Trace management
- Bounds enforcement
- Metrics collection

Supported Strategies
=====================
- **HebbianStrategy**: Basic Hebbian learning (Δw ∝ pre × post)
- **STDPStrategy**: Spike-timing dependent plasticity (causal vs anti-causal)
- **BCMStrategy**: Bienenstock-Cooper-Munro with sliding threshold
- **ThreeFactorStrategy**: RL eligibility × neuromodulator (dopamine)
- **ErrorCorrectiveStrategy**: Supervised delta rule (cerebellum-style)
- **CompositeStrategy**: Compose multiple strategies

Strategy Interface
==================
All strategies implement:

.. code-block:: python

    class LearningStrategy(Protocol):
        def apply(
            self,
            weights: Tensor,
            pre_spikes: Tensor,
            post_spikes: Tensor,
            **kwargs
        ) -> Dict[str, Any]:
            '''Apply learning rule and return metrics.'''

Usage in Components
===================
Components apply strategies during forward passes:

.. code-block:: python

    class MyRegion(NeuralComponent):
        def __init__(self, config):
            super().__init__(config)
            self.learning_strategy = STDPStrategy(
                STDPConfig(a_plus=0.01, a_minus=0.012)
            )

        def forward(self, input_spikes):
            output = self._compute_output(input_spikes)

            # Apply learning automatically during forward
            if self.plasticity_enabled:
                metrics = self.learning_strategy.apply(
                    self.weights, input_spikes, output
                )

            return output

Benefits
========
1. **Modularity**: Learning rules are independent, testable modules
2. **Reusability**: Same strategy works for regions AND pathways
3. **Composition**: Combine multiple rules (STDP + BCM + DA modulation)
4. **Experimentation**: Easy to swap learning rules for ablation studies

Sparse Updates for Large-Scale Regions
========================================
For regions with large populations (>10k neurons) and sparse activity (<5%),
sparse weight updates can significantly improve performance:

.. code-block:: python

    # Enable sparse updates for large cortex
    config = HebbianConfig(
        learning_rate=0.01,
        use_sparse_updates=True,  # Automatically uses sparse ops when beneficial
    )
    strategy = HebbianStrategy(config)

**When to Use Sparse Updates:**
- Large regions (n_neurons > 10,000)
- Low firing rates (<5% active neurons per timestep)
- Typical cortical activity patterns

**Biological Justification:**
Cortical neurons fire at ~1-10Hz with millisecond precision. In a 1ms timestep,
only ~0.1-1% of neurons are active. Dense outer products waste computation on
zeros. Sparse operations compute only for spiking neurons, matching biological
sparsity.

**Performance:**
- At 2% activity: ~10-50x speedup for large populations
- Automatically falls back to dense for >5% activity
- Zero accuracy loss (numerically identical results)

**Supported Strategies:**
- HebbianStrategy: Fully supported
- STDPStrategy: Partial (trace manager updates in future work)
- BCMStrategy: Planned for future implementation

See tests/unit/test_sparse_learning.py for validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Protocol

import numpy as np
import torch
import torch.nn as nn

from thalia.core.base.component_config import LearningComponentConfig
from thalia.utils.core_utils import clamp_weights
from thalia.learning.eligibility.trace_manager import EligibilityTraceManager, STDPConfig as CoreSTDPConfig


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
    use_sparse_updates: bool = False  # Enable sparse operations for large/sparse regions


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
    dt_ms: float = 1.0        # Simulation timestep (ms)


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
        """Apply weight bounds using hard clamp.

        Weight bounds are enforced via hard clamping. Biological regulation
        is provided by UnifiedHomeostasis (weight normalization) and BCM
        (sliding threshold), not by soft bounds.
        """
        cfg = self.config
        # Apply update and clamp to hard bounds
        new_weights = clamp_weights(weights + dw, cfg.w_min, cfg.w_max, inplace=False)
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

    def _compute_sparse_outer(
        self,
        post: torch.Tensor,
        pre: torch.Tensor,
        sparsity_threshold: float = 0.05,
    ) -> torch.Tensor:
        """Compute outer product, using sparse ops if beneficial.

        For sparse spike patterns (<5% active), sparse operations are more
        efficient than dense outer products. This is biologically realistic:
        cortical neurons fire at ~1-10Hz with millisecond precision, so most
        timesteps have <5% active neurons.

        Args:
            post: Postsynaptic spikes [n_post]
            pre: Presynaptic spikes [n_pre]
            sparsity_threshold: Threshold for using sparse ops (default 5%)

        Returns:
            Outer product [n_post, n_pre]
        """
        if not self.config.use_sparse_updates:
            # Dense path (standard)
            return torch.outer(post, pre)

        # Check if sparsity justifies sparse ops
        pre_sparsity = (pre != 0).float().mean().item()
        post_sparsity = (post != 0).float().mean().item()
        avg_sparsity = (pre_sparsity + post_sparsity) / 2

        if avg_sparsity > sparsity_threshold:
            # Too dense, use standard outer product
            return torch.outer(post, pre)

        # Sparse path: only compute non-zero entries
        post_indices = torch.nonzero(post, as_tuple=True)[0]
        pre_indices = torch.nonzero(pre, as_tuple=True)[0]

        if len(post_indices) == 0 or len(pre_indices) == 0:
            # No spikes, return zeros
            return torch.zeros(post.shape[0], pre.shape[0], device=post.device)

        # Build sparse outer product
        # For each (post_idx, pre_idx) pair, compute post[i] * pre[j]
        n_post, n_pre = post.shape[0], pre.shape[0]
        post_vals = post[post_indices]
        pre_vals = pre[pre_indices]

        # Create index tensors for all combinations
        post_idx = post_indices.unsqueeze(1).expand(-1, len(pre_indices)).flatten()
        pre_idx = pre_indices.unsqueeze(0).expand(len(post_indices), -1).flatten()
        values = (post_vals.unsqueeze(1) * pre_vals.unsqueeze(0)).flatten()

        # Build sparse COO tensor
        indices = torch.stack([post_idx, pre_idx])
        sparse_outer = torch.sparse_coo_tensor(
            indices,
            values,
            size=(n_post, n_pre),
            device=post.device,
        )

        # Convert to dense for compatibility with existing code
        # (Future optimization: keep sparse throughout pipeline)
        return sparse_outer.to_dense()

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

        Args:
            weights: Current weights [n_post, n_pre]
            pre: Presynaptic activity [n_pre] (1D)
            post: Postsynaptic activity [n_post] (1D)
        """
        cfg = self.hebbian_config

        # Ensure 1D inputs
        if pre.dim() != 1:
            pre = pre.squeeze()
        if post.dim() != 1:
            post = post.squeeze()

        assert pre.dim() == 1 and post.dim() == 1, "HebbianStrategy expects 1D inputs"

        # Hebbian outer product: dw[j,i] = post[j] × pre[i]
        # Use sparse computation if configured
        dw = self._compute_sparse_outer(post.float(), pre.float())

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

        # Trace manager (initialized lazily when we know dimensions)
        self._trace_manager: Optional[EligibilityTraceManager] = None

    def reset_state(self) -> None:
        """Reset traces."""
        if self._trace_manager is not None:
            self._trace_manager.reset_traces()

    def _ensure_trace_manager(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> None:
        """Initialize trace manager if needed or reinitialize if dimensions changed.

        Args:
            pre: Presynaptic spikes [n_pre] (1D)
            post: Postsynaptic spikes [n_post] (1D)
        """
        n_pre = pre.shape[0]
        n_post = post.shape[0]
        device = pre.device

        # Check if we need to (re)initialize
        needs_init = (
            self._trace_manager is None
            or self._trace_manager.n_input != n_pre
            or self._trace_manager.n_output != n_post
            or self._trace_manager.input_trace.device != device
        )

        if needs_init:
            cfg = self.stdp_config
            self._trace_manager = EligibilityTraceManager(
                n_input=n_pre,
                n_output=n_post,
                config=CoreSTDPConfig(
                    stdp_tau_ms=cfg.tau_plus,  # Use tau_plus for trace decay
                    eligibility_tau_ms=1000.0,  # Not used in this context
                    a_plus=cfg.a_plus,
                    a_minus=cfg.a_minus,
                    stdp_lr=1.0,  # We handle learning rate separately
                ),
                device=device,
            )

    def compute_update(
        self,
        weights: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply STDP learning with optional neuromodulator and BCM modulation.

        LTP: A+ × pre_trace × post_spike
        LTD: A- × post_trace × pre_spike

        Args:
            weights: Current weights [n_post, n_pre]
            pre: Presynaptic spikes [n_pre] (1D)
            post: Postsynaptic spikes [n_post] (1D)
            **kwargs: Optional modulations:
                - dopamine (float): Dopamine level for DA-modulated STDP
                - acetylcholine (float): ACh level (favors LTP/encoding)
                - norepinephrine (float): NE level (inverted-U modulation)
                - bcm_modulation (float|Tensor): BCM metaplasticity factor
                - oscillation_phase (float): Phase for phase-locked STDP
                - learning_rule (SpikingLearningRule): Rule type for conditional modulation
        """
        cfg = self.stdp_config

        # Ensure 1D inputs
        if pre.dim() != 1:
            pre = pre.squeeze()
        if post.dim() != 1:
            post = post.squeeze()

        assert pre.dim() == 1 and post.dim() == 1, "STDPStrategy expects 1D inputs"

        # Initialize trace manager if needed
        self._ensure_trace_manager(pre, post)

        # Update traces and compute LTP/LTD
        self._trace_manager.update_traces(pre, post, cfg.dt_ms)
        ltp, ltd = self._trace_manager.compute_ltp_ltd_separate(pre, post)

        # Extract modulation kwargs
        dopamine = kwargs.get('dopamine', 0.0)
        acetylcholine = kwargs.get('acetylcholine', 0.0)
        norepinephrine = kwargs.get('norepinephrine', 0.0)
        bcm_modulation = kwargs.get('bcm_modulation', 1.0)
        oscillation_phase = kwargs.get('oscillation_phase', 0.0)
        learning_rule = kwargs.get('learning_rule', None)

        # Apply neuromodulator modulations to LTP
        if isinstance(ltp, torch.Tensor):
            # Dopamine modulation (for dopamine-STDP)
            if learning_rule is not None and hasattr(learning_rule, 'name'):
                # Check if this is dopamine-STDP rule
                if 'DOPAMINE' in learning_rule.name:
                    ltp = ltp * (1.0 + dopamine)

            # Acetylcholine modulation (high ACh = favor LTP/encoding)
            ach_modulation = 0.5 + 0.5 * acetylcholine  # Range: [0.5, 1.5]
            ltp = ltp * ach_modulation

            # Norepinephrine modulation (inverted-U: moderate NE optimal)
            ne_modulation = 1.0 - 0.5 * abs(norepinephrine - 0.5)  # Peak at 0.5
            ltp = ltp * ne_modulation

            # Phase modulation (for phase-STDP)
            if learning_rule is not None and hasattr(learning_rule, 'name'):
                if 'PHASE' in learning_rule.name:
                    phase_mod = 0.5 + 0.5 * np.cos(oscillation_phase)
                    ltp = ltp * phase_mod

            # BCM modulation
            if isinstance(bcm_modulation, torch.Tensor):
                ltp = ltp * bcm_modulation.unsqueeze(1)
            else:
                ltp = ltp * bcm_modulation

        # Apply neuromodulator modulations to LTD
        if isinstance(ltd, torch.Tensor):
            # Dopamine modulation (reduces LTD, protects good synapses)
            if learning_rule is not None and hasattr(learning_rule, 'name'):
                if 'DOPAMINE' in learning_rule.name:
                    ltd = ltd * (1.0 - 0.5 * max(0.0, dopamine))

            # Acetylcholine modulation (high ACh = reduce LTD/favor encoding)
            ach_ltd_suppression = 1.0 - 0.3 * acetylcholine  # Range: [0.7, 1.0]
            ltd = ltd * ach_ltd_suppression

            # Norepinephrine modulation (inverted-U)
            ne_modulation = 1.0 - 0.5 * abs(norepinephrine - 0.5)
            ltd = ltd * ne_modulation

            # Phase modulation (for phase-STDP)
            if learning_rule is not None and hasattr(learning_rule, 'name'):
                if 'PHASE' in learning_rule.name:
                    phase_mod = 0.5 - 0.5 * np.cos(oscillation_phase)
                    ltd = ltd * phase_mod

        # Compute weight change
        dw = ltp - ltd if isinstance(ltp, torch.Tensor) or isinstance(ltd, torch.Tensor) else 0

        # Apply bounds
        old_weights = weights.clone()
        new_weights = self._apply_bounds(weights, dw) if isinstance(dw, torch.Tensor) else weights

        metrics = self._compute_metrics(old_weights, new_weights, dw if isinstance(dw, torch.Tensor) else torch.zeros_like(weights))
        if self._trace_manager is not None:
            metrics["pre_trace_mean"] = self._trace_manager.input_trace.mean().item()
            metrics["post_trace_mean"] = self._trace_manager.output_trace.mean().item()

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

    def reset_state(self) -> None:
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
        """Compute BCM modulation function.

        φ(c, θ) = c × (c - θ) / θ

        Args:
            post: Postsynaptic activity [n_post] (1D)

        Returns:
            BCM modulation [n_post] (1D)
        """
        if self.theta is None:
            return torch.ones_like(post)

        c = post.float()
        phi = c * (c - self.theta) / (self.theta + 1e-8)
        return phi

    def _update_theta(self, post: torch.Tensor) -> None:
        """Update sliding threshold.

        Args:
            post: Postsynaptic activity [n_post] (1D)
        """
        cfg = self.bcm_config

        c = post.float()
        c_p = c.pow(cfg.power)

        # EMA update
        self.theta = self.decay_theta * self.theta + (1 - self.decay_theta) * c_p

        # Clamp
        self.theta = self.theta.clamp(cfg.theta_min, cfg.theta_max)

    def update_threshold(self, post: torch.Tensor) -> None:
        """Update sliding threshold (public API).

        This is the standard public interface for updating BCM thresholds,
        matching the API in BCMRule for consistency.

        Args:
            post: Postsynaptic activity [n_post] (1D)
        """
        # Initialize theta if needed
        if self.theta is None:
            self._init_theta(post.shape[0], post.device)
        self._update_theta(post)

    def compute_update(
        self,
        weights: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply BCM learning.

        Δw[j,i] = lr × pre[i] × φ(post[j], θ[j])

        Args:
            weights: Current weights [n_post, n_pre]
            pre: Presynaptic activity [n_pre] (1D)
            post: Postsynaptic activity [n_post] (1D)
        """
        cfg = self.bcm_config

        # Ensure 1D inputs
        if pre.dim() != 1:
            pre = pre.squeeze()
        if post.dim() != 1:
            post = post.squeeze()

        assert pre.dim() == 1 and post.dim() == 1, "BCMStrategy expects 1D inputs"

        n_post = post.shape[0]
        self._init_theta(n_post, post.device)

        # Compute BCM modulation
        phi = self.compute_phi(post)  # [n_post]

        # Weight update: dw[j,i] = lr × pre[i] × φ[j]
        dw = cfg.learning_rate * torch.outer(phi, pre.float())

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

    def reset_state(self) -> None:
        """Reset eligibility trace."""
        self.eligibility = None

    def update_eligibility(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:
        """Update eligibility trace with current activity.

        Eligibility accumulates Hebbian correlations until modulator arrives.

        Args:
            pre: Presynaptic spikes [n_pre] (1D)
            post: Postsynaptic spikes [n_post] (1D)
        """
        # Ensure 1D inputs
        if pre.dim() != 1:
            pre = pre.squeeze()
        if post.dim() != 1:
            post = post.squeeze()

        n_post = post.shape[0]
        n_pre = pre.shape[0]
        device = pre.device

        # Initialize if needed
        if self.eligibility is None or self.eligibility.shape != (n_post, n_pre):
            self.eligibility = torch.zeros(n_post, n_pre, device=device)

        # Decay existing eligibility
        self.eligibility = self.decay_elig * self.eligibility

        # Add new Hebbian correlation: outer product [n_post, n_pre]
        hebbian = torch.outer(post.float(), pre.float())
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
            weights: Current weights [n_post, n_pre]
            pre: Presynaptic activity (input) [n_pre] (1D)
            post: Postsynaptic activity (actual output) [n_post] (1D)
            target: Target output [n_post] (1D)

        Returns:
            Updated weights and metrics
        """
        cfg = self.ec_config

        if target is None:
            return weights, {"error": 0.0, "ltp": 0.0, "ltd": 0.0, "net_change": 0.0}

        # Ensure 1D inputs
        if pre.dim() != 1:
            pre = pre.squeeze()
        if post.dim() != 1:
            post = post.squeeze()
        if target.dim() != 1:
            target = target.squeeze()

        assert pre.dim() == 1 and post.dim() == 1 and target.dim() == 1, (
            "ErrorCorrectiveStrategy expects 1D inputs"
        )

        # Compute error
        error = target.float() - post.float()

        # Check threshold
        if error.abs().max() < cfg.error_threshold:
            return weights, {"error": 0.0, "ltp": 0.0, "ltd": 0.0, "net_change": 0.0}

        # Delta rule: dw[j,i] = lr × error[j] × pre[i]
        dw = cfg.learning_rate * torch.outer(error, pre.float())

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


# =============================================================================
# Strategy Registration (at module load time)
# =============================================================================

def _register_builtin_strategies() -> None:
    """Register all built-in learning strategies with the registry.

    This function is called at module import time to populate the
    LearningStrategyRegistry with all standard learning strategies.

    Note: Import is done here to avoid circular dependency since
    strategy_registry.py imports this module.
    """
    from thalia.learning.strategy_registry import LearningStrategyRegistry

    # Register Hebbian
    LearningStrategyRegistry.register(
        "hebbian",
        config_class=HebbianConfig,
        description="Basic Hebbian learning: Δw ∝ pre × post",
        version="1.0",
    )(HebbianStrategy)

    # Register STDP
    LearningStrategyRegistry.register(
        "stdp",
        config_class=STDPConfig,
        aliases=["spike_timing"],
        description="Spike-timing dependent plasticity with LTP/LTD windows",
        version="1.0",
    )(STDPStrategy)

    # Register BCM
    LearningStrategyRegistry.register(
        "bcm",
        config_class=BCMConfig,
        description="Bienenstock-Cooper-Munro with sliding threshold",
        version="1.0",
    )(BCMStrategy)

    # Register Three-Factor
    LearningStrategyRegistry.register(
        "three_factor",
        config_class=ThreeFactorConfig,
        aliases=["rl", "dopamine", "threefactor"],
        description="Three-factor learning: eligibility × neuromodulator",
        version="1.0",
    )(ThreeFactorStrategy)

    # Register Error-Corrective
    LearningStrategyRegistry.register(
        "error_corrective",
        config_class=ErrorCorrectiveConfig,
        aliases=["delta", "supervised", "error"],
        description="Supervised error-corrective learning (delta rule)",
        version="1.0",
    )(ErrorCorrectiveStrategy)

    # Register Composite
    LearningStrategyRegistry.register(
        "composite",
        description="Compose multiple learning strategies",
        version="1.0",
    )(CompositeStrategy)


# Register strategies when module is imported
_register_builtin_strategies()
