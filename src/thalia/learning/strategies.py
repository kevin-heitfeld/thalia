"""
Learning Rule Strategies: Pluggable Learning Algorithms for Brain Components.

This module implements a Strategy pattern for learning rules, allowing regions
and pathways to compose and switch between different learning algorithms without
code duplication.

Design Philosophy
==================
Instead of each component implementing its own learning logic with duplicated
STDP/BCM/three-factor code, components can compose strategies.

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

Benefits
========
1. **Modularity**: Learning rules are independent, testable modules
2. **Reusability**: Same strategy works for regions AND pathways
3. **Composition**: Combine multiple rules (STDP + BCM + DA modulation)
4. **Experimentation**: Easy to swap learning rules for ablation studies
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn

from thalia.errors import ConfigurationError
from thalia.utils import clamp_weights, validate_spike_tensor

from .eligibility_trace_manager import EligibilityTraceManager, EligibilitySTDPConfig


# =============================================================================
# Strategy Configuration Dataclasses
# =============================================================================

@dataclass
class LearningConfig:
    """Base configuration for all learning strategies."""

    device: str = "cpu"  # Device to run on: 'cpu', 'cuda', 'cuda:0', etc.
    seed: Optional[int] = None  # Random seed for reproducibility. None = no seeding.

    learning_rate: float = 0.01
    """Base learning rate."""

    w_min: float = 0.0
    """Minimum synaptic weight."""

    w_max: float = 1.0
    """Maximum synaptic weight."""


@dataclass
class BCMConfig(LearningConfig):
    """Configuration for BCM learning.

    BCM function: φ(c, θ) = c(c - θ)
    - c > θ: LTP
    - c < θ: LTD

    Threshold adapts: θ → E[c²]

    CRITICAL: BCM learning rate must be much slower than STDP to prevent
    runaway potentiation. Typical ratio: BCM_LR = 0.01 * STDP_LR
    """

    tau_theta: float = 100000.0  # Threshold time constant (100 seconds) - VERY slow adaptation
    theta_init: float = 0.01  # Initial threshold - higher to prevent immediate potentiation
    theta_min: float = 1e-6  # Minimum threshold
    theta_max: float = 0.5  # Maximum threshold - REDUCED from 1.0 to prevent saturation
    p: float = 2.0  # Power for threshold (c^p)
    weight_decay: float = 0.0001  # L2 regularization toward ZERO (proper weight decay)
    min_activity_for_decay: float = 0.005  # Minimum post-synaptic activity (0.5%) to apply full decay
    silent_decay_factor: float = 0.1  # Fraction of weight_decay applied when below min_activity (10%)
    activity_threshold: float = 0.001  # Minimum activity (0.1%) - allows LTD in sparse networks


@dataclass
class ErrorCorrectiveConfig(LearningConfig):
    """Configuration for supervised error-corrective learning.

    Delta rule: Δw = lr × pre × (target - actual)
    """

    error_threshold: float = 0.01  # Minimum error to trigger learning


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

    a_plus: float = 0.01  # LTP amplitude
    a_minus: float = 0.002  # LTD amplitude (reduced 6x from original 0.012)
    tau_plus: float = 20.0  # LTP time constant (ms)
    tau_minus: float = 20.0  # LTD time constant (ms)
    activity_threshold: float = 0.01  # Minimum postsynaptic activity (1%) to enable LTD

    # Retrograde signaling (endocannabinoid-like)
    retrograde_enabled: bool = True  # Enable retrograde signaling
    retrograde_threshold: float = 0.05  # Minimum postsynaptic activity (5%) to trigger retrograde signal
    retrograde_tau_ms: float = 1000.0  # Retrograde signal decay timescale (~1 second)
    retrograde_ltp_gate: float = 0.1  # Minimum retrograde signal to allow full LTP (0-1)
    retrograde_ltd_enhance: float = 2.0  # LTD enhancement factor when retrograde signal is weak


@dataclass
class ThreeFactorConfig(LearningConfig):
    """Configuration for three-factor RL learning.

    Three-factor rule: Δw = lr × eligibility × modulator
    - Eligibility: accumulated spike timing correlations
    - Modulator: dopamine, reward signal, or error
    """

    eligibility_tau: float = 100.0  # Eligibility trace decay (ms)
    modulator_tau: float = 50.0  # Modulator decay (ms)


# =============================================================================
# Base Learning Strategy
# =============================================================================


class LearningStrategy(nn.Module, ABC):
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
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        sparsity_threshold: float = 0.05,
    ) -> torch.Tensor:
        """Compute outer product, using sparse ops if beneficial.

        For sparse spike patterns (<5% active), sparse operations are more
        efficient than dense outer products. This is biologically realistic:
        cortical neurons fire at ~1-10Hz with millisecond precision, so most
        timesteps have <5% active neurons.

        Args:
            post_spikes: Postsynaptic spikes [n_post]
            pre_spikes: Presynaptic spikes [n_pre]
            sparsity_threshold: Threshold for using sparse ops (default 5%)

        Returns:
            Outer product [n_post, n_pre]
        """
        validate_spike_tensor(pre_spikes, "pre_spikes")
        validate_spike_tensor(post_spikes, "post_spikes")

        pre_float = pre_spikes.float()
        post_float = post_spikes.float()

        return torch.outer(post_float, pre_float)

    @abstractmethod
    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weight update using this learning rule.

        Args:
            weights: Current weight matrix [n_post, n_pre]
            pre_spikes: Presynaptic activity [n_pre]
            post_spikes: Postsynaptic activity [n_post]
            **kwargs: Strategy-specific inputs (target, reward, etc.)

        Returns:
            Tuple of:
                - Updated weight matrix
                - Dict of learning metrics
        """
        pass


# =============================================================================
# Learning Strategy Registry
# =============================================================================


class LearningStrategyRegistry:
    """Registry for all learning strategies.

    Maintains a registry of learning strategy classes with their configurations,
    enabling dynamic strategy creation and discovery.

    Registry Structure:
        _registry = {
            "hebbian": HebbianStrategy,
            "stdp": STDPStrategy,
            "bcm": BCMStrategy,
            ...
        }

    Attributes:
        _registry: Dict mapping strategy name to strategy class
        _configs: Dict mapping strategy name to config class
        _aliases: Dict mapping alias to canonical name
        _metadata: Strategy metadata (description, version, author, etc.)
    """

    _registry: Dict[str, Type[LearningStrategy]] = {}
    _configs: Dict[str, Type[LearningConfig]] = {}
    _aliases: Dict[str, str] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        config_class: Optional[Type[LearningConfig]] = None,
        aliases: Optional[List[str]] = None,
        description: str = "",
        version: str = "1.0",
        author: str = "",
    ) -> Callable[[Type[LearningStrategy]], Type[LearningStrategy]]:
        """Decorator to register a learning strategy.

        Args:
            name: Primary name for the strategy
            config_class: Configuration class for the strategy (optional)
            aliases: Optional list of alternative names
            description: Human-readable description
            version: Strategy version string
            author: Strategy author/maintainer

        Returns:
            Decorator function

        Raises:
            ValueError: If name already registered or strategy invalid
        """

        def decorator(strategy_class: Type[LearningStrategy]) -> Type[LearningStrategy]:
            # Validate strategy class
            if not inspect.isclass(strategy_class):
                raise ConfigurationError(f"Strategy must be a class, got {type(strategy_class)}")

            # Check if name already registered
            if name in cls._registry:
                raise ConfigurationError(
                    f"Strategy '{name}' already registered as {cls._registry[name].__name__}"
                )

            # Register strategy
            cls._registry[name] = strategy_class

            # Register config if provided
            if config_class is not None:
                cls._configs[name] = config_class

            # Register aliases
            if aliases:
                for alias in aliases:
                    if alias in cls._aliases:
                        raise ConfigurationError(
                            f"Alias '{alias}' already registered for '{cls._aliases[alias]}'"
                        )
                    cls._aliases[alias] = name

            # Store metadata
            cls._metadata[name] = {
                "class": strategy_class.__name__,
                "description": description or strategy_class.__doc__ or "",
                "version": version,
                "author": author,
                "aliases": aliases or [],
                "config_class": config_class.__name__ if config_class else None,
            }

            return strategy_class

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        config: LearningConfig,
        **kwargs: Any,
    ) -> LearningStrategy:
        """Create a learning strategy instance.

        Args:
            name: Strategy name (or alias)
            config: Strategy configuration object
            **kwargs: Additional arguments passed to strategy constructor

        Returns:
            Configured learning strategy instance

        Raises:
            ValueError: If strategy not found or creation fails
        """
        # Resolve alias
        canonical_name = cls._aliases.get(name, name)

        # Check if strategy exists
        if canonical_name not in cls._registry:
            available = cls.list_strategies(include_aliases=True)
            raise ConfigurationError(
                f"Unknown learning strategy: '{name}'. "
                f"Available strategies: {', '.join(available)}"
            )

        # Get strategy class
        strategy_class = cls._registry[canonical_name]

        # Validate config type if registered
        if canonical_name in cls._configs:
            expected_config = cls._configs[canonical_name]
            if not isinstance(config, expected_config):
                raise ConfigurationError(
                    f"Strategy '{canonical_name}' expects config type {expected_config.__name__}, "
                    f"got {type(config).__name__}"
                )

        # Create strategy instance
        try:
            return strategy_class(config, **kwargs)
        except Exception as e:
            raise ConfigurationError(f"Failed to create strategy '{canonical_name}': {e}") from e

    @classmethod
    def list_strategies(cls, include_aliases: bool = False) -> List[str]:
        """List all registered strategies.

        Args:
            include_aliases: Whether to include aliases in the list

        Returns:
            List of strategy names (and aliases if requested)
        """
        strategies = list(cls._registry.keys())

        if include_aliases:
            strategies.extend(cls._aliases.keys())
            strategies = sorted(set(strategies))

        return sorted(strategies)

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """Get metadata for a strategy.

        Args:
            name: Strategy name (or alias)

        Returns:
            Dictionary containing strategy metadata

        Raises:
            ValueError: If strategy not found
        """
        # Resolve alias
        canonical_name = cls._aliases.get(name, name)

        if canonical_name not in cls._metadata:
            raise ConfigurationError(f"Unknown strategy: '{name}'")

        return cls._metadata[canonical_name]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a strategy is registered.

        Args:
            name: Strategy name (or alias)

        Returns:
            True if strategy is registered, False otherwise
        """
        canonical_name = cls._aliases.get(name, name)
        return canonical_name in cls._registry

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a strategy (primarily for testing).

        Args:
            name: Strategy name to unregister
        """
        if name in cls._registry:
            del cls._registry[name]

        if name in cls._configs:
            del cls._configs[name]

        if name in cls._metadata:
            del cls._metadata[name]

        # Remove aliases pointing to this strategy
        aliases_to_remove = [alias for alias, target in cls._aliases.items() if target == name]
        for alias in aliases_to_remove:
            del cls._aliases[alias]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered strategies."""
        cls._registry.clear()
        cls._configs.clear()
        cls._aliases.clear()
        cls._metadata.clear()


# =============================================================================
# Strategy Implementations
# =============================================================================


@LearningStrategyRegistry.register(
    "hebbian",
    config_class=HebbianConfig,
    aliases=["basic_hebbian"],
    description="Basic Hebbian learning: Δw ∝ pre × post",
    version="1.0",
)
class HebbianStrategy(LearningStrategy):
    """Basic Hebbian learning: Δw ∝ pre × post.

    The simplest correlation-based learning rule. Strengthens connections
    where pre and post are co-active.
    """

    def __init__(self, config: Optional[HebbianConfig] = None):
        super().__init__(config or HebbianConfig())
        self.hebbian_config: HebbianConfig = self.config

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply Hebbian learning.

        Δw[j,i] = lr × pre[i] × post[j]

        Args:
            weights: Current weights [n_post, n_pre]
            pre_spikes: Presynaptic activity [n_pre] (1D)
            post_spikes: Postsynaptic activity [n_post] (1D)
        """
        cfg = self.hebbian_config

        # Ensure 1D inputs
        if pre_spikes.dim() != 1:
            pre_spikes = pre_spikes.squeeze()
        if post_spikes.dim() != 1:
            post_spikes = post_spikes.squeeze()

        assert pre_spikes.dim() == 1 and post_spikes.dim() == 1, "HebbianStrategy expects 1D inputs"

        # Hebbian outer product: dw[j,i] = post[j] × pre[i]
        # Use sparse computation if configured
        dw = self._compute_sparse_outer(pre_spikes=pre_spikes, post_spikes=post_spikes)
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


@LearningStrategyRegistry.register(
    "stdp",
    config_class=STDPConfig,
    aliases=["spike_timing", "spike_timing_dependent_plasticity"],
    description="Spike-timing dependent plasticity with LTP/LTD windows",
    version="1.0",
)
class STDPStrategy(LearningStrategy):
    """Spike-Timing Dependent Plasticity.

    Trace-based STDP:
        - LTP: pre_trace × post_spike (pre before post)
        - LTD: post_trace × pre_spike (post before pre)
    """

    def __init__(self, config: Optional[STDPConfig] = None):
        super().__init__(config or STDPConfig())
        self.stdp_config: STDPConfig = self.config

        # Timestep (set by update_temporal_parameters)
        self._dt_ms: Optional[float] = None

        # Trace manager (initialized lazily when we know dimensions)
        self._trace_manager: Optional[EligibilityTraceManager] = None

        # Firing rate tracking for activity-dependent LTD gating
        # Uses EMA with tau=100ms (100 timesteps averaging window)
        self._firing_rate_tau_ms: float = 100.0
        self._firing_rate_decay: Optional[float] = None
        self.firing_rates: Optional[torch.Tensor] = None  # Per-neuron running average

        # Retrograde signaling (endocannabinoid-like)
        # Tracks strong postsynaptic depolarization to gate plasticity
        self._retrograde_decay: Optional[float] = None
        self.retrograde_signal: Optional[torch.Tensor] = None  # Per-neuron retrograde signal [0-1]

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        self._dt_ms = dt_ms
        # Compute firing rate decay factor: alpha = 1 - dt/tau
        self._firing_rate_decay = 1.0 - dt_ms / self._firing_rate_tau_ms
        # Compute retrograde signal decay factor
        self._retrograde_decay = 1.0 - dt_ms / self.stdp_config.retrograde_tau_ms
        # NOTE: Trace manager computes decay factors on-the-fly in update_traces()

    def _ensure_trace_manager(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
    ) -> None:
        """Initialize trace manager if needed or reinitialize if dimensions changed.

        Args:
            pre_spikes: Presynaptic spikes [n_pre] (1D)
            post_spikes: Postsynaptic spikes [n_post] (1D)
        """
        n_pre = pre_spikes.shape[0]
        n_post = post_spikes.shape[0]
        device = pre_spikes.device

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
                config=EligibilitySTDPConfig(
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
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply STDP learning with optional neuromodulator and BCM modulation.

        LTP: A+ × pre_trace × post_spike
        LTD: A- × post_trace × pre_spike

        Args:
            weights: Current weights [n_post, n_pre]
            pre_spikes: Presynaptic spikes [n_pre] (1D)
            post_spikes: Postsynaptic spikes [n_post] (1D)
            **kwargs: Optional modulations:
                - dopamine (float): Dopamine level for DA-modulated STDP
                - acetylcholine (float): ACh level (favors LTP/encoding)
                - norepinephrine (float): NE level (inverted-U modulation)
                - bcm_modulation (float|Tensor): BCM metaplasticity factor
                - oscillation_phase (float): Phase for phase-locked STDP
                - learning_strategy (SpikingLearningRule): Rule type for conditional modulation
        """
        # Ensure 1D inputs
        if pre_spikes.dim() != 1:
            pre_spikes = pre_spikes.squeeze()
        if post_spikes.dim() != 1:
            post_spikes = post_spikes.squeeze()

        assert pre_spikes.dim() == 1 and post_spikes.dim() == 1, "STDPStrategy expects 1D inputs"

        # Initialize trace manager if needed
        self._ensure_trace_manager(pre_spikes, post_spikes)
        assert self._trace_manager is not None, "Trace manager must be initialized"

        # Initialize firing rates if needed
        n_post = post_spikes.shape[0]
        if self.firing_rates is None or self.firing_rates.shape[0] != n_post:
            self.firing_rates = torch.zeros(n_post, device=post_spikes.device, dtype=torch.float32)

        # Initialize retrograde signal if needed
        if self.stdp_config.retrograde_enabled:
            if self.retrograde_signal is None or self.retrograde_signal.shape[0] != n_post:
                self.retrograde_signal = torch.zeros(n_post, device=post_spikes.device, dtype=torch.float32)

        # Ensure dt_ms is set
        if self._dt_ms is None:
            # Auto-initialize with default dt_ms if not set (for testing/standalone use)
            self.update_temporal_parameters(dt_ms=1.0)  # Default 1ms timestep

        # Update traces and compute LTP/LTD (dt_ms guaranteed non-None after auto-init)
        assert self._dt_ms is not None
        self._trace_manager.update_traces(pre_spikes, post_spikes, self._dt_ms)
        ltp, ltd = self._trace_manager.compute_ltp_ltd_separate(pre_spikes, post_spikes)

        # Update firing rate tracking (EMA over spikes)
        assert self._firing_rate_decay is not None, "Firing rate decay not initialized"
        assert self.firing_rates is not None, "Firing rates not initialized"
        self.firing_rates = self._firing_rate_decay * self.firing_rates + (1 - self._firing_rate_decay) * post_spikes.float()

        # Update retrograde signal (endocannabinoid-like)
        # Released when postsynaptic neuron strongly depolarizes
        # Acts as a "this was important" signal that gates plasticity
        # Strong spiking activity triggers retrograde messenger release
        if self.stdp_config.retrograde_enabled and self.retrograde_signal is not None:
            assert self._retrograde_decay is not None, "Retrograde decay not initialized"
            # Retrograde release is triggered by spiking activity
            # The signal accumulates and decays slowly (1s timescale)
            spike_contribution = post_spikes.float()  # 1.0 when neuron fires

            # Scale by recent firing rate: frequent firing = stronger retrograde release
            # This captures "how important is this neuron's activity?"
            rate_scaling = (self.firing_rates / self.stdp_config.retrograde_threshold).clamp(0, 2.0)
            weighted_contribution = spike_contribution * (0.5 + 0.5 * rate_scaling)

            # Update retrograde signal with slow decay
            self.retrograde_signal = (
                self._retrograde_decay * self.retrograde_signal +
                (1 - self._retrograde_decay) * weighted_contribution
            )

        # Extract modulation kwargs
        dopamine = kwargs.get("dopamine", 0.0)
        acetylcholine = kwargs.get("acetylcholine", 0.0)
        norepinephrine = kwargs.get("norepinephrine", 0.0)
        bcm_modulation = kwargs.get("bcm_modulation", 1.0)
        oscillation_phase = kwargs.get("oscillation_phase", 0.0)
        learning_strategy = kwargs.get("learning_strategy", None)

        # Apply retrograde signaling (endocannabinoid-like gating)
        # LTP is gated: only strong postsynaptic responses drive potentiation
        # LTD is enhanced: weak responses lead to depotentiation (extinction-like)
        if self.stdp_config.retrograde_enabled and self.retrograde_signal is not None:
            # Retrograde signal represents "postsynaptic neuron cares about this"
            # Range [0, 1]: 0 = no recent strong activity, 1 = strong recent activity
            retro_gate = self.retrograde_signal.clamp(0, 1)  # [n_post]

            # LTP gating: require minimum retrograde signal
            # This prevents spurious correlations from driving potentiation
            # Only synapses that contribute to strong postsynaptic firing get strengthened
            ltp_gate = ((retro_gate - self.stdp_config.retrograde_ltp_gate) /
                       (1.0 - self.stdp_config.retrograde_ltp_gate)).clamp(0, 1)
            if isinstance(ltp, torch.Tensor):
                ltp = ltp * ltp_gate.unsqueeze(1)  # [n_post, n_pre]

            # LTD enhancement: low retrograde signal enhances depression
            # This implements extinction learning - connections that don't contribute
            # to strong responses get depressed
            ltd_enhance = 1.0 + (1.0 - retro_gate) * (self.stdp_config.retrograde_ltd_enhance - 1.0)
            if isinstance(ltd, torch.Tensor):
                ltd = ltd * ltd_enhance.unsqueeze(1)  # [n_post, n_pre]

        # Apply neuromodulator modulations to LTP
        if isinstance(ltp, torch.Tensor):
            # Dopamine modulation (for dopamine-STDP)
            if learning_strategy is not None and hasattr(learning_strategy, "name"):
                # Check if this is dopamine-STDP rule
                if "DOPAMINE" in learning_strategy.name:
                    ltp = ltp * (1.0 + dopamine)

            # Acetylcholine modulation (high ACh = favor LTP/encoding)
            ach_modulation = 0.5 + 0.5 * acetylcholine  # Range: [0.5, 1.5]
            ltp = ltp * ach_modulation

            # Norepinephrine modulation (inverted-U: moderate NE optimal)
            ne_modulation = 1.0 - 0.5 * abs(norepinephrine - 0.5)  # Peak at 0.5
            ltp = ltp * ne_modulation

            # Phase modulation (for phase-STDP)
            if learning_strategy is not None and hasattr(learning_strategy, "name"):
                if "PHASE" in learning_strategy.name:
                    phase_mod = 0.5 + 0.5 * np.cos(oscillation_phase)
                    ltp = ltp * phase_mod

            # BCM modulation
            if isinstance(bcm_modulation, torch.Tensor):
                ltp = ltp * bcm_modulation.unsqueeze(1)
            else:
                ltp = ltp * bcm_modulation

        # Apply neuromodulator modulations to LTD
        if isinstance(ltd, torch.Tensor):
            # CRITICAL FIX: DISABLE STDP LTD when BCM is active
            # With sparse firing (2% cortex, 8% thalamus), STDP LTD dominates because:
            # 1. Pre-post pairs rarely occur within STDP window (~20ms)
            # 2. LTP occurs infrequently (Total LTP ~ 0.0005 per 100 timesteps)
            # 3. LTD occurs frequently (Total LTD ~ 0.028 per 100 timesteps)
            # 4. LTP/LTD ratio ~ 0.02 → guaranteed collapse despite activity
            #
            # BCM provides sufficient homeostatic depression via theta adaptation.
            # STDP should ONLY provide Hebbian potentiation (causal timing).
            # This prevents premature weight collapse during bootstrap.
            #
            # Future: Re-enable STDP LTD once network is stable and firing >5%

            # Check if BCM is co-active via kwargs (indicates Composite Strategy)
            bcm_active = kwargs.get("bcm_modulation") is not None

            if bcm_active:
                # BCM handles depression → STDP LTD disabled
                ltd = ltd * 0.0  # Zero out LTD
            else:
                # No BCM → Keep STDP LTD but with strict conditional gating
                # CONDITIONAL LTD: Only apply depression when postsynaptic firing rate exceeds threshold
                # Uses tracked firing rate (EMA over 1 second) not instantaneous spikes
                # This prevents weight collapse during bootstrap when neurons fire sparsely
                # firing_rates is [n_post] 1D tensor, ltd is [n_post, n_pre] 2D tensor

                # CRITICAL FIX: Raise activity threshold for LTD from 0.001 to 0.01
                # At 0.001, 76% of neurons pass threshold → excessive LTD dominates sparse firing
                # At 0.01, only ~40% pass → LTD only applied to reliably active neurons
                # This prevents premature weight collapse before STDP timing windows align
                ltd_threshold = max(self.stdp_config.activity_threshold, 0.01)  # Enforce minimum 0.01
                activity_mask = self.firing_rates >= ltd_threshold  # [n_post]

                # ADDITIONAL PROTECTION: Reduce LTD magnitude when LTP is actively occurring
                # This prevents the situation where sparse firing causes LTD to dominate even though
                # some synapses ARE successfully potentiating (but infrequently due to sparse timing).
                # Compute LTP/LTD ratio per synapse to protect actively learning connections
                if isinstance(ltp, torch.Tensor):
                    # Protect synapses with recent LTP by scaling down LTD proportionally
                    # If ltp > ltd, reduce ltd by the ratio (let potentiation win)
                    # This implements a "recent success" memory that prevents premature depression
                    ltp_protection = (ltp / (ltd + 1e-8)).clamp(0, 1)  # [n_post, n_pre], range [0, 1]
                    ltd = ltd * (1.0 - 0.5 * ltp_protection)  # Reduce LTD by up to 50% where LTP active

                # Expand mask to match ltd shape [n_post, n_pre]
                # Only allow LTD for neurons above threshold
                ltd = ltd * activity_mask.unsqueeze(1)  # [n_post, n_pre]
            if learning_strategy is not None and hasattr(learning_strategy, "name"):
                if "DOPAMINE" in learning_strategy.name:
                    ltd = ltd * (1.0 - 0.5 * max(0.0, dopamine))

            # Acetylcholine modulation (high ACh = reduce LTD/favor encoding)
            ach_ltd_suppression = 1.0 - 0.3 * acetylcholine  # Range: [0.7, 1.0]
            ltd = ltd * ach_ltd_suppression

            # Norepinephrine modulation (inverted-U)
            ne_modulation = 1.0 - 0.5 * abs(norepinephrine - 0.5)
            ltd = ltd * ne_modulation

            # Phase modulation (for phase-STDP)
            if learning_strategy is not None and hasattr(learning_strategy, "name"):
                if "PHASE" in learning_strategy.name:
                    phase_mod = 0.5 - 0.5 * np.cos(oscillation_phase)
                    ltd = ltd * phase_mod

        # Compute weight change
        dw = ltp - ltd if isinstance(ltp, torch.Tensor) or isinstance(ltd, torch.Tensor) else 0

        # Apply bounds
        old_weights = weights.clone()
        new_weights = self._apply_bounds(weights, dw) if isinstance(dw, torch.Tensor) else weights

        metrics = self._compute_metrics(
            old_weights,
            new_weights,
            dw if isinstance(dw, torch.Tensor) else torch.zeros_like(weights),
        )
        if self._trace_manager is not None:
            metrics["pre_trace_mean"] = self._trace_manager.input_trace.mean().item()
            metrics["post_trace_mean"] = self._trace_manager.output_trace.mean().item()

        return new_weights, metrics


@LearningStrategyRegistry.register(
    "bcm",
    config_class=BCMConfig,
    aliases=["bienenstock_cooper_munro"],
    description="Bienenstock-Cooper-Munro with sliding threshold",
    version="1.0",
)
class BCMStrategy(LearningStrategy):
    """Bienenstock-Cooper-Munro learning with sliding threshold.

    BCM function: φ(c, θ) = c(c - θ)
    Weight update: Δw = lr × pre × φ(post, θ)

    The threshold θ adapts to track E[post²], providing automatic
    metaplasticity that stabilizes learning.
    """

    def __init__(self, config: Optional[BCMConfig] = None):
        super().__init__(config or BCMConfig())
        self.bcm_config: BCMConfig = self.config

        # Decay factor (computed in update_temporal_parameters)
        self._dt_ms: Optional[float] = None
        self.register_buffer(
            "decay_theta",
            torch.tensor(0.0, dtype=torch.float32),
        )

        # Sliding threshold (per-neuron)
        self.theta: Optional[torch.Tensor] = None

        # Firing rate tracking for activity-dependent LTD gating
        # Uses EMA with tau=100ms (100 timesteps averaging window)
        self._firing_rate_tau_ms: float = 100.0
        self.register_buffer(
            "decay_firing_rate",
            torch.tensor(0.0, dtype=torch.float32),
        )
        self.firing_rates: Optional[torch.Tensor] = None  # Per-neuron running average

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update decay factors when brain timestep changes.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        self._dt_ms = dt_ms
        tau = self.bcm_config.tau_theta
        self.decay_theta: torch.Tensor = torch.tensor(
            1.0 - dt_ms / tau, dtype=torch.float32, device=self.decay_theta.device
        )
        # Compute firing rate decay factor
        self.decay_firing_rate: torch.Tensor = torch.tensor(
            1.0 - dt_ms / self._firing_rate_tau_ms, dtype=torch.float32, device=self.decay_firing_rate.device
        )

    def _init_theta(self, n_post: int) -> None:
        """Initialize threshold if needed.

        Args:
            n_post: Number of postsynaptic neurons
            device: Ignored - uses module's device for reliability
        """
        # Use module's device from existing buffer (reliable source of truth)
        # decay_theta is always present and moves with .to(device)
        module_device = self.decay_theta.device

        needs_init = (
            self.theta is None
            or self.theta.shape[0] != n_post
            or self.theta.device != module_device  # Device changed!
        )

        if needs_init:
            # Create new tensor on module's device
            new_theta = torch.full(
                (n_post,),
                self.bcm_config.theta_init,
                device=module_device,
            )
            # Check if already registered as buffer
            if "theta" not in self._buffers:
                # First time: delete attribute and register as buffer so it moves with .to(device)
                if hasattr(self, "theta"):
                    delattr(self, "theta")
                self.register_buffer("theta", new_theta, persistent=False)
            else:
                # Already registered - replace by re-registering (handles both device and shape change)
                del self._buffers["theta"]
                self.register_buffer("theta", new_theta, persistent=False)

    def compute_phi(self, post_spikes: torch.Tensor, firing_rates: torch.Tensor) -> torch.Tensor:
        """Compute BCM modulation function with firing-rate-based LTD gating.

        φ(c, θ) = c × (c - θ) / θ

        BCM depression occurs when c < θ (below sliding threshold).
        We block depression when firing rate is too low (cold-start protection).
        Uses tracked firing rate (EMA over 1 second), not instantaneous spikes.

        This allows:
        - LTP always when c > theta (neuron above its history)
        - LTD only when: (c < theta) AND (firing_rate > activity_threshold)

        Prevents cold-start collapse while allowing BCM selectivity once active.

        Args:
            post_spikes: Postsynaptic spikes [n_post] (1D)
            firing_rates: Tracked firing rates [n_post] (1D)

        Returns:
            BCM modulation [n_post] (1D)
        """
        if self.theta is None:
            return torch.ones_like(post_spikes)

        c = post_spikes.float()
        phi = c * (c - self.theta) / (self.theta + 1e-8)

        # FIRING-RATE-BASED LTD GATING: Block depression when firing rate is too low
        # Uses tracked firing rate (not instantaneous spikes) for stable gating
        # This prevents weight collapse during bootstrap when neurons fire sparsely
        depression_mask = phi < 0  # Where BCM would depress (c < theta)
        barely_active = firing_rates < self.bcm_config.activity_threshold  # Low firing rate
        invalid_depression = depression_mask & barely_active
        phi = torch.where(invalid_depression, torch.zeros_like(phi), phi)

        return phi

    def _update_theta(self, post_spikes: torch.Tensor) -> None:
        """Update sliding threshold.

        Args:
            post_spikes: Postsynaptic activity [n_post] (1D)
        """
        cfg = self.bcm_config

        c = post_spikes.float()
        c_p = c.pow(cfg.p)

        # EMA update
        self.theta = self.decay_theta * self.theta + (1 - self.decay_theta) * c_p

        # Clamp
        self.theta = self.theta.clamp(cfg.theta_min, cfg.theta_max)

    def update_threshold(self, post_spikes: torch.Tensor) -> None:
        """Update sliding threshold (public API).

        This is the standard public interface for updating BCM thresholds.

        Args:
            post_spikes: Postsynaptic activity [n_post] (1D)
        """
        # Initialize theta if needed
        if self.theta is None:
            self._init_theta(post_spikes.shape[0])
        self._update_theta(post_spikes)

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply BCM learning.

        Δw[j,i] = lr × pre[i] × φ(post[j], θ[j])

        Args:
            weights: Current weights [n_post, n_pre]
            pre_spikes: Presynaptic activity [n_pre] (1D)
            post_spikes: Postsynaptic activity [n_post] (1D)
        """
        # Auto-initialize temporal parameters if not set
        if self._dt_ms is None:
            self.update_temporal_parameters(dt_ms=1.0)

        cfg = self.bcm_config

        # Ensure 1D inputs
        if pre_spikes.dim() != 1:
            pre_spikes = pre_spikes.squeeze()
        if post_spikes.dim() != 1:
            post_spikes = post_spikes.squeeze()

        assert pre_spikes.dim() == 1 and post_spikes.dim() == 1, "BCMStrategy expects 1D inputs"

        n_post = post_spikes.shape[0]
        self._init_theta(n_post)

        # Initialize firing rates if needed
        module_device = self.decay_theta.device
        if self.firing_rates is None or self.firing_rates.shape[0] != n_post or self.firing_rates.device != module_device:
            new_firing_rates = torch.zeros(n_post, device=module_device, dtype=torch.float32)
            if "firing_rates" not in self._buffers:
                if hasattr(self, "firing_rates"):
                    delattr(self, "firing_rates")
                self.register_buffer("firing_rates", new_firing_rates, persistent=False)
            else:
                del self._buffers["firing_rates"]
                self.register_buffer("firing_rates", new_firing_rates, persistent=False)

        # Update firing rate tracking (EMA over spikes)
        assert self.firing_rates is not None
        self.firing_rates = self.decay_firing_rate * self.firing_rates + (1 - self.decay_firing_rate) * post_spikes.float().to(module_device)

        # Compute BCM modulation using tracked firing rates
        phi = self.compute_phi(post_spikes, self.firing_rates)  # [n_post]

        # Weight update: dw[j,i] = lr × pre[i] × φ[j]
        dw = cfg.learning_rate * torch.outer(phi, pre_spikes.float())

        # Activity-dependent weight decay (biologically inspired)
        # - Full decay when post-synaptic neurons are active (use-dependent pruning)
        # - Minimal decay when silent (prevent collapse during bootstrap/silence)
        # Biology: Active synapses undergo turnover, silent synapses are preserved
        if cfg.weight_decay > 0:
            # Compute decay factor per neuron based on activity level
            # [n_post] -> decay_factor for each post-synaptic neuron
            activity_ratio = (self.firing_rates / cfg.min_activity_for_decay).clamp(0.0, 1.0)
            # Interpolate: silent -> silent_decay_factor, active -> 1.0
            effective_decay = cfg.silent_decay_factor + (1.0 - cfg.silent_decay_factor) * activity_ratio
            # Apply decay per neuron: dw[j, :] -= decay[j] * weights[j, :]
            dw -= cfg.weight_decay * effective_decay.unsqueeze(1) * weights

        # Update threshold
        self._update_theta(post_spikes)

        # Apply bounds
        old_weights = weights.clone()
        new_weights = self._apply_bounds(weights, dw)

        metrics = self._compute_metrics(old_weights, new_weights, dw)
        metrics["theta_mean"] = self.theta.mean().item() if self.theta is not None else 0.0
        metrics["phi_mean"] = phi.mean().item()

        return new_weights, metrics


@LearningStrategyRegistry.register(
    "three_factor",
    config_class=ThreeFactorConfig,
    aliases=["rl", "dopamine", "threefactor"],
    description="Three-factor learning: eligibility × neuromodulator",
    version="1.0",
)
class ThreeFactorStrategy(LearningStrategy):
    """Three-factor reinforcement learning rule.

    Δw = lr × eligibility × modulator

    - Eligibility: accumulated spike-timing correlations (Hebbian)
    - Modulator: dopamine, reward signal, or TD error

    Key insight: Without modulator, NO learning occurs (not reduced, NONE).
    """

    def __init__(self, config: Optional[ThreeFactorConfig] = None):
        super().__init__(config or ThreeFactorConfig())
        self.tf_config: ThreeFactorConfig = self.config

        # Decay factors (computed in update_temporal_parameters)
        self._dt_ms: Optional[float] = None
        self.register_buffer(
            "decay_elig",
            torch.tensor(0.0, dtype=torch.float32),
        )

        # Eligibility trace (Hebbian correlation)
        self.eligibility: Optional[torch.Tensor] = None

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update decay factors when brain timestep changes.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        self._dt_ms = dt_ms
        self.decay_elig: torch.Tensor = torch.tensor(
            1.0 - dt_ms / self.tf_config.eligibility_tau,
            dtype=torch.float32,
            device=self.decay_elig.device,
        )

    def update_eligibility(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """Update eligibility trace with current activity.

        Eligibility accumulates Hebbian correlations until modulator arrives.

        Args:
            pre_spikes: Presynaptic spikes [n_pre] (1D)
            post_spikes: Postsynaptic spikes [n_post] (1D)
        """
        # Auto-initialize dt_ms if not set (for testing/standalone use)
        if self._dt_ms is None:
            self.update_temporal_parameters(dt_ms=1.0)

        # Ensure 1D inputs
        if pre_spikes.dim() != 1:
            pre_spikes = pre_spikes.squeeze()
        if post_spikes.dim() != 1:
            post_spikes = post_spikes.squeeze()

        n_post = post_spikes.shape[0]
        n_pre = pre_spikes.shape[0]

        # Use module's device from existing buffer (reliable source of truth)
        # decay_elig is always present and moves with .to(device)
        module_device = self.decay_elig.device

        # Initialize or update eligibility buffer
        needs_init = (
            self.eligibility is None
            or self.eligibility.shape != (n_post, n_pre)
            or self.eligibility.device != module_device  # Device changed!
        )

        if needs_init:
            # Create new tensor on module's device
            new_elig = torch.zeros(n_post, n_pre, device=module_device)

            # Check if already registered as buffer
            if "eligibility" not in self._buffers:
                # First time: delete attribute and register as buffer so it moves with .to(device)
                if hasattr(self, "eligibility"):
                    delattr(self, "eligibility")
                self.register_buffer("eligibility", new_elig, persistent=False)
            else:
                # Already registered - replace by re-registering (handles both device and shape change)
                del self._buffers["eligibility"]
                self.register_buffer("eligibility", new_elig, persistent=False)

        # Decay existing eligibility
        self.eligibility = self.decay_elig * self.eligibility

        # Add new Hebbian correlation: outer product [n_post, n_pre]
        # Ensure hebbian is on the same device as eligibility
        hebbian = torch.outer(post_spikes.float().to(module_device), pre_spikes.float().to(module_device))
        self.eligibility = self.eligibility + hebbian

        return self.eligibility

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        modulator: float = 0.0,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply three-factor learning.

        Args:
            weights: Current weights
            pre_spikes: Presynaptic activity
            post_spikes: Postsynaptic activity
            modulator: Neuromodulatory signal (dopamine, reward, etc.)

        Returns:
            Updated weights and metrics
        """
        cfg = self.tf_config

        # Update eligibility
        self.update_eligibility(pre_spikes, post_spikes)

        # No learning without modulator
        if abs(modulator) < 0.01:
            return weights, {
                "modulator": modulator,
                "eligibility_mean": (
                    self.eligibility.mean().item() if self.eligibility is not None else 0.0
                ),
                "ltp": 0.0,
                "ltd": 0.0,
                "net_change": 0.0,
            }

        # Three-factor update
        assert self.eligibility is not None, "Eligibility must be initialized"
        dw = cfg.learning_rate * self.eligibility * modulator

        # Apply bounds
        old_weights = weights.clone()
        new_weights = self._apply_bounds(weights, dw)

        metrics = self._compute_metrics(old_weights, new_weights, dw)
        metrics["modulator"] = modulator
        metrics["eligibility_mean"] = (
            self.eligibility.mean().item() if self.eligibility is not None else 0.0
        )

        return new_weights, metrics


@LearningStrategyRegistry.register(
    "error_corrective",
    config_class=ErrorCorrectiveConfig,
    aliases=["delta", "supervised", "error"],
    description="Supervised error-corrective learning (delta rule)",
    version="1.0",
)
class ErrorCorrectiveStrategy(LearningStrategy):
    """Supervised error-corrective learning (delta rule).

    Δw = lr × pre × (target - actual)

    Used in cerebellum-like circuits for supervised motor learning.
    """

    def __init__(self, config: Optional[ErrorCorrectiveConfig] = None):
        super().__init__(config or ErrorCorrectiveConfig())
        self.ec_config: ErrorCorrectiveConfig = self.config

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply error-corrective learning.

        Args:
            weights: Current weights [n_post, n_pre]
            pre_spikes: Presynaptic activity (input) [n_pre] (1D)
            post_spikes: Postsynaptic activity (actual output) [n_post] (1D)
            target: Target output [n_post] (1D)

        Returns:
            Updated weights and metrics
        """
        cfg = self.ec_config

        if target is None:
            return weights, {"error": 0.0, "ltp": 0.0, "ltd": 0.0, "net_change": 0.0}

        # Ensure 1D inputs
        if pre_spikes.dim() != 1:
            pre_spikes = pre_spikes.squeeze()
        if post_spikes.dim() != 1:
            post_spikes = post_spikes.squeeze()
        if target.dim() != 1:
            target = target.squeeze()

        assert (
            pre_spikes.dim() == 1 and post_spikes.dim() == 1 and target.dim() == 1
        ), "ErrorCorrectiveStrategy expects 1D inputs"

        # Compute error
        error = target.float() - post_spikes.float()

        # Check threshold
        if error.abs().max() < cfg.error_threshold:
            return weights, {"error": 0.0, "ltp": 0.0, "ltd": 0.0, "net_change": 0.0}

        # Delta rule: dw[j,i] = lr × error[j] × pre[i]
        dw = cfg.learning_rate * torch.outer(error, pre_spikes.float())

        # Apply bounds
        old_weights = weights.clone()
        new_weights = self._apply_bounds(weights, dw)

        metrics = self._compute_metrics(old_weights, new_weights, dw)
        metrics["error"] = error.abs().mean().item()

        return new_weights, metrics


@LearningStrategyRegistry.register(
    "composite",
    description="Compose multiple learning strategies",
    version="1.0",
)
class CompositeStrategy(LearningStrategy):
    """Compose multiple learning strategies.

    Strategies are applied sequentially, with each one potentially
    modulating the output of the previous.
    """

    def __init__(
        self,
        strategies: List[LearningStrategy],
        config: Optional[LearningConfig] = None,
    ):
        super().__init__(config or LearningConfig())
        self.strategies = nn.ModuleList(strategies)

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply all strategies sequentially.

        Each strategy receives the weights output by the previous one.
        Metrics are merged with prefixes.
        """
        current_weights = weights
        all_metrics: Dict[str, float] = {}

        for i, strategy in enumerate(self.strategies):
            if isinstance(strategy, LearningStrategy):
                current_weights, metrics = strategy.compute_update(
                    weights=current_weights,
                    pre_spikes=pre_spikes,
                    post_spikes=post_spikes,
                    **kwargs
                )
                # Prefix metrics with strategy index
                for key, value in metrics.items():
                    all_metrics[f"s{i}_{key}"] = value

        return current_weights, all_metrics


# =============================================================================
# Convenience: Strategy Factory
# =============================================================================


def create_strategy(strategy_name: str, **config_kwargs: Any) -> LearningStrategy:
    """Factory function to create learning strategies by name.

    Args:
        strategy_name: One of 'hebbian', 'stdp', 'bcm', 'three_factor', 'error_corrective'
        **config_kwargs: Configuration parameters for the strategy

    Returns:
        Configured learning strategy
    """
    name = strategy_name.lower().replace("-", "_").replace(" ", "_")

    if name == "hebbian":
        return HebbianStrategy(HebbianConfig(**config_kwargs))
    elif name == "stdp":
        return STDPStrategy(STDPConfig(**config_kwargs))
    elif name == "bcm":
        return BCMStrategy(BCMConfig(**config_kwargs))
    elif name in ("three_factor", "threefactor", "rl"):
        return ThreeFactorStrategy(ThreeFactorConfig(**config_kwargs))
    elif name in ("error_corrective", "delta", "supervised"):
        return ErrorCorrectiveStrategy(ErrorCorrectiveConfig(**config_kwargs))
    else:
        raise ValueError(f"Unknown learning strategy: {strategy_name}")
