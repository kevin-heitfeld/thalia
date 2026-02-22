"""Learning Rule Strategies: Pluggable Learning Algorithms for Brain Components."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn

from thalia.constants import DEFAULT_DT_MS
from thalia.utils import validate_spike_tensor

from .eligibility_trace_manager import EligibilityTraceManager


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
    a_minus: float = 0.012  # LTD amplitude (matches biological STDP ratio ~1.2%)
    tau_plus: float = 20.0  # LTP time constant (ms)
    tau_minus: float = 20.0  # LTD time constant (ms)
    activity_threshold: float = 0.01  # Minimum postsynaptic activity (1%) to enable LTD

    # Eligibility traces (for three-factor learning)
    eligibility_tau_ms: float = 1000.0  # Eligibility trace decay time (~1 second)
    heterosynaptic_ratio: float = 0.3  # Fraction of LTD applied to non-active synapses

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


@dataclass
class D1STDPConfig(LearningConfig):
    """Configuration for D1 MSN three-factor STDP.

    Multi-timescale eligibility traces for striatal corticostriatal plasticity.
    Biology (Yagishita 2014): Fast traces (~500 ms) capture immediate coincidences;
    slow traces (~60 s) consolidate tags for delayed reward credit assignment.
    D1 receptors (Gs-coupled): DA+ → LTP, DA- → LTD.
    """

    fast_eligibility_tau_ms: float = 500.0    # Fast trace decay (~500 ms)
    slow_eligibility_tau_ms: float = 60000.0  # Slow trace decay (~60 s)
    eligibility_consolidation_rate: float = 0.01  # Fast→slow transfer rate per step
    slow_trace_weight: float = 0.3            # α in combined = fast + α*slow


@dataclass
class D2STDPConfig(D1STDPConfig):
    """Configuration for D2 MSN three-factor STDP.

    Identical fields to :class:`D1STDPConfig`; the dopamine signal is inverted
    inside :class:`D2STDPStrategy` to reflect Gi-coupled D2 receptor physiology
    (DA+ → LTD, DA- → LTP).
    """


@dataclass
class PredictiveCodingConfig(LearningConfig):
    """Configuration for anti-Hebbian temporal predictive-coding learning.

    Biology: L5/L6 → L4 feedback weights learn to *predict and suppress* L4
    activity.  Co-activation of (delayed) pre and post cells strengthens the
    inhibitory weight, so correct predictions silence L4 (no update) while
    wrong predictions strengthen feedback inhibition (error-driven learning).

    Args:
        prediction_delay_steps: Number of timesteps by which pre-synaptic
            spikes are delayed before computing the update.  Default ``1``
            corresponds to one-step look-back (previous pre → current post).
    """

    prediction_delay_steps: int = 1


# =============================================================================
# Base Learning Strategy
# =============================================================================


class LearningStrategy(nn.Module, ABC):
    """Base class for learning strategies with common functionality."""

    def __init__(self, config: LearningConfig):
        super().__init__()
        self.config = config
        self._dt_ms: float = DEFAULT_DT_MS

    def setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Called once by BrainBuilder.build() after synapse dimensions are known.

        Subclasses must override this to eagerly initialize all registered buffers
        and submodules.  The default is a no-op for strategies that carry no
        per-synapse state (e.g. a pure neuromodulator gate with no trace tensors).

        After setup() returns, all state must be registered via register_buffer()
        or as nn.Module children so that .to(device), .state_dict(), and
        .parameters() all work correctly.

        Args:
            n_pre:  Number of presynaptic neurons for this synapse.
            n_post: Number of postsynaptic neurons for this synapse.
            device: Device to allocate state tensors on.
        """
        ...

    def ensure_setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Call setup() lazily if state has not been initialised yet or sizes changed.

        Concrete subclasses should override this to check their own state tensors
        and re-call setup() when the synapse dimensions change between calls.
        The default implementation delegates to setup() unconditionally only on
        the very first call (i.e., when the subclass hasn't registered any state).
        """
        ...

    @abstractmethod
    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute weight update using this learning rule.

        Args:
            weights: Current weight matrix [n_post, n_pre]
            pre_spikes: Presynaptic activity [n_pre]
            post_spikes: Postsynaptic activity [n_post]
            **kwargs: Strategy-specific inputs (target, reward, etc.)

        Returns:
            Updated weight matrix (do NOT clamp — apply_learning() owns clamping)
        """

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        self._dt_ms = dt_ms


# =============================================================================
# STDP and BCM Strategies
# =============================================================================


class STDPStrategy(LearningStrategy):
    """Spike-Timing Dependent Plasticity.

    Trace-based STDP:
        - LTP: pre_trace × post_spike (pre before post)
        - LTD: post_trace × pre_spike (post before pre)

    Call setup(n_pre, n_post, device) once before compute_update().
    """

    def __init__(self, config: STDPConfig):
        super().__init__(config)

        # Scalar caches for decay factors (populated in setup / update_temporal_parameters)
        self._firing_rate_tau_ms: float = 100.0
        self._firing_rate_decay: float = math.exp(-DEFAULT_DT_MS / self._firing_rate_tau_ms)
        self._retrograde_decay: float = math.exp(-DEFAULT_DT_MS / config.retrograde_tau_ms)

        # trace_manager is created as a proper nn.Module child in setup()
        # (not during __init__ because n_pre/n_post are not yet known)

    # ------------------------------------------------------------------
    # Eager initialisation
    # ------------------------------------------------------------------

    def setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Eagerly initialise all per-synapse state.

        Must be called once (by BrainBuilder.build()) before compute_update().
        Assigns trace_manager as an nn.Module child so .to(device) and
        state_dict() automatically include all trace tensors.
        """
        # nn.Module child assignment registers it automatically
        self.trace_manager = EligibilityTraceManager(n_pre, n_post, self.config, device)

        # Registered buffers travel with .to(device) and appear in state_dict()
        self.register_buffer("firing_rates",      torch.zeros(n_post, device=device))
        self.register_buffer("retrograde_signal", torch.zeros(n_post, device=device))

        cfg = self.config
        self._firing_rate_decay = math.exp(-self._dt_ms / self._firing_rate_tau_ms)
        self._retrograde_decay  = math.exp(-self._dt_ms / cfg.retrograde_tau_ms)

    def ensure_setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Call setup() if not yet initialised or if synapse dimensions changed."""
        if (
            not hasattr(self, "trace_manager")
            or self.trace_manager.n_input  != n_pre
            or self.trace_manager.n_output != n_post
        ):
            self.setup(n_pre, n_post, device)

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply STDP learning with optional neuromodulator modulation.

        LTP: A+ × pre_trace × post_spike
        LTD: A- × post_trace × pre_spike

        setup() must be called before the first compute_update() call.

        Args:
            weights: Current weights [n_post, n_pre]
            pre_spikes: Presynaptic spikes [n_pre] (1D)
            post_spikes: Postsynaptic spikes [n_post] (1D)
            **kwargs: Optional modulations:
                - dopamine (float): Dopamine level for DA-modulated STDP
                - acetylcholine (float): ACh level (favors LTP/encoding)
                - norepinephrine (float): NE level (inverted-U modulation)
        """
        self.ensure_setup(pre_spikes.shape[0], post_spikes.shape[0], pre_spikes.device)
        validate_spike_tensor(pre_spikes, tensor_name="pre_spikes")
        validate_spike_tensor(post_spikes, tensor_name="post_spikes")

        # Extract modulation kwargs
        dopamine       = kwargs.get("dopamine", 0.0)
        acetylcholine  = kwargs.get("acetylcholine", 0.0)
        norepinephrine = kwargs.get("norepinephrine", 0.0)

        cfg = self.config

        # Update traces and compute LTP/LTD (EligibilityTraceManager is a registered submodule)
        self.trace_manager.update_traces(pre_spikes, post_spikes, self._dt_ms)
        ltp, ltd = self.trace_manager.compute_ltp_ltd_separate(pre_spikes, post_spikes)

        # Update firing rate tracking (EMA over spikes)
        self.firing_rates = self._firing_rate_decay * self.firing_rates + (1 - self._firing_rate_decay) * post_spikes.float()

        # Update retrograde signal (endocannabinoid-like)
        if cfg.retrograde_enabled:
            spike_contribution = post_spikes.float()
            rate_scaling = (self.firing_rates / cfg.retrograde_threshold).clamp(0, 2.0)
            weighted_contribution = spike_contribution * (0.5 + 0.5 * rate_scaling)
            self.retrograde_signal = (
                self._retrograde_decay * self.retrograde_signal +
                (1 - self._retrograde_decay) * weighted_contribution
            )

        # Apply retrograde signaling (endocannabinoid-like gating)
        # LTP is gated: only strong postsynaptic responses drive potentiation
        # LTD is enhanced: weak responses lead to depotentiation (extinction-like)
        if cfg.retrograde_enabled:
            # Retrograde signal represents "postsynaptic neuron cares about this"
            # Range [0, 1]: 0 = no recent strong activity, 1 = strong recent activity
            retro_gate = self.retrograde_signal.clamp(0, 1)  # [n_post]

            # LTP gating: require minimum retrograde signal
            # This prevents spurious correlations from driving potentiation
            # Only synapses that contribute to strong postsynaptic firing get strengthened
            ltp_gate = ((retro_gate - cfg.retrograde_ltp_gate) /
                       (1.0 - cfg.retrograde_ltp_gate)).clamp(0, 1)
            if isinstance(ltp, torch.Tensor):
                ltp = ltp * ltp_gate.unsqueeze(1)  # [n_post, n_pre]

            # LTD enhancement: low retrograde signal enhances depression
            # This implements extinction learning - connections that don't contribute
            # to strong responses get depressed
            ltd_enhance = 1.0 + (1.0 - retro_gate) * (cfg.retrograde_ltd_enhance - 1.0)
            if isinstance(ltd, torch.Tensor):
                ltd = ltd * ltd_enhance.unsqueeze(1)  # [n_post, n_pre]

        # Apply neuromodulator modulations to LTP
        if isinstance(ltp, torch.Tensor):
            # Apply neuromodulator modulations to LTP
            # Dopamine modulation: high DA enhances LTP (favors learning)
            ltp = ltp * (1.0 + dopamine)

            # Acetylcholine modulation (high ACh = favor LTP/encoding)
            ach_modulation = 0.5 + 0.5 * acetylcholine  # Range: [0.5, 1.5]
            ltp = ltp * ach_modulation

            # Norepinephrine modulation (inverted-U: moderate NE optimal)
            ne_modulation = 1.0 - 0.5 * abs(norepinephrine - 0.5)  # Peak at 0.5
            ltp = ltp * ne_modulation

        # Apply neuromodulator modulations to LTD
        if isinstance(ltd, torch.Tensor):
            # CONDITIONAL LTD: Only apply depression when postsynaptic firing rate exceeds threshold
            # Uses tracked firing rate (EMA over 1 second) not instantaneous spikes
            # firing_rates is [n_post] 1D tensor, ltd is [n_post, n_pre] 2D tensor

            ltd_threshold = cfg.activity_threshold
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

            # Apply neuromodulator modulations to LTD
            # Dopamine modulation: high DA reduces LTD (favors learning)
            ltd = ltd * (1.0 - 0.5 * max(0.0, dopamine))

            # Acetylcholine modulation (high ACh = reduce LTD/favor encoding)
            ach_ltd_suppression = 1.0 - 0.3 * acetylcholine  # Range: [0.7, 1.0]
            ltd = ltd * ach_ltd_suppression

            # Norepinephrine modulation (inverted-U)
            ne_modulation = 1.0 - 0.5 * abs(norepinephrine - 0.5)
            ltd = ltd * ne_modulation

        # Compute weight change
        dw = ltp - ltd if isinstance(ltp, torch.Tensor) or isinstance(ltd, torch.Tensor) else 0

        if isinstance(dw, torch.Tensor):
            # Return raw updated weights — clamping is owned by apply_learning() on NeuralRegion
            new_weights = weights + dw
        else:
            new_weights = weights

        return new_weights

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes."""
        super().update_temporal_parameters(dt_ms)
        # Use true exponential decay (not linear 1-dt/tau which goes negative for dt>tau)
        self._firing_rate_decay = math.exp(-dt_ms / self._firing_rate_tau_ms)
        self._retrograde_decay  = math.exp(-dt_ms / self.config.retrograde_tau_ms)
        # Propagate to trace_manager if already set up
        if hasattr(self, "trace_manager"):
            self.trace_manager.update_temporal_parameters(dt_ms)


class BCMStrategy(LearningStrategy):
    """Bienenstock-Cooper-Munro learning with sliding threshold.

    BCM function: φ(c, θ) = c(c - θ)
    Weight update: Δw = lr × pre × φ(post, θ)

    The threshold θ adapts to track E[firing_rate²], providing automatic
    metaplasticity that stabilizes learning.

    Call setup(n_pre, n_post, device) once before compute_update().
    """

    def __init__(self, config: BCMConfig):
        super().__init__(config)

        # Scalar decay factors (dimension-independent; computed in update_temporal_parameters)
        # Stored as plain Python floats — the actual per-neuron state is in setup() buffers
        self._firing_rate_tau_ms: float = 100.0

        # Pre-compute decay scalars at default dt (refreshed in update_temporal_parameters)
        self._decay_theta_val:        float = math.exp(-DEFAULT_DT_MS / config.tau_theta)
        self._decay_firing_rate_val:  float = math.exp(-DEFAULT_DT_MS / self._firing_rate_tau_ms)

    # ------------------------------------------------------------------
    # Eager initialisation
    # ------------------------------------------------------------------

    def setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Eagerly register per-neuron buffers.

        Must be called once (by BrainBuilder.build()) before compute_update().
        """
        self.register_buffer("theta",        torch.full((n_post,), self.config.theta_init, device=device))
        self.register_buffer("firing_rates", torch.zeros(n_post, device=device))

    def ensure_setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Call setup() if not yet initialised or if post-population size changed."""
        if not hasattr(self, "theta") or self.theta.shape[0] != n_post:
            self.setup(n_pre, n_post, device)

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
        c = post_spikes.float()
        phi = c * (c - self.theta) / (self.theta + 1e-8)

        # FIRING-RATE-BASED LTD GATING: Block depression when firing rate is too low
        # Uses tracked firing rate (not instantaneous spikes) for stable gating
        depression_mask = phi < 0  # Where BCM would depress (c < theta)
        barely_active = firing_rates < self.config.activity_threshold  # Low firing rate
        invalid_depression = depression_mask & barely_active
        phi = torch.where(invalid_depression, torch.zeros_like(phi), phi)

        return phi

    def _update_theta(self) -> None:
        """Update sliding threshold using tracked firing rates.

        Uses the per-neuron firing rate EMA rather than raw instantaneous spikes,
        so the threshold tracks average activity, not single-timestep volatility.
        """
        assert hasattr(self, "theta"), "BCMStrategy.setup() must be called before _update_theta()"
        cfg = self.config
        r_p = self.firing_rates.pow(cfg.p)
        # In-place EMA to avoid triggering a new tensor allocation
        self.theta.mul_(self._decay_theta_val).add_(r_p, alpha=1.0 - self._decay_theta_val)
        self.theta.clamp_(cfg.theta_min, cfg.theta_max)

    def update_threshold(self, post_spikes: torch.Tensor) -> None:
        """Update sliding threshold (public API).

        This is the standard public interface for updating BCM thresholds.
        Caller should update firing_rates before calling this.

        Args:
            post_spikes: Postsynaptic activity [n_post] (1D).  Kept for
                         API compatibility; actual theta update uses self.firing_rates.
        """
        self._update_theta()

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply BCM learning.

        Δw[j,i] = lr × pre[i] × φ(post[j], θ[j])

        Args:
            weights: Current weights [n_post, n_pre]
            pre_spikes: Presynaptic activity [n_pre] (1D)
            post_spikes: Postsynaptic activity [n_post] (1D)
        """
        self.ensure_setup(pre_spikes.shape[0], post_spikes.shape[0], pre_spikes.device)
        validate_spike_tensor(pre_spikes, tensor_name="pre_spikes")
        validate_spike_tensor(post_spikes, tensor_name="post_spikes")

        cfg = self.config

        # Update firing rate EMA (in-place)
        self.firing_rates.mul_(self._decay_firing_rate_val).add_(
            post_spikes.float(), alpha=1.0 - self._decay_firing_rate_val
        )

        # Compute BCM modulation using tracked firing rates
        phi = self.compute_phi(post_spikes, self.firing_rates)  # [n_post]

        # Weight update: dw[j,i] = lr × pre[i] × φ[j]
        dw = cfg.learning_rate * torch.outer(phi, pre_spikes.float())

        # Activity-dependent weight decay (biologically inspired)
        # - Full decay when post-synaptic neurons are active (use-dependent pruning)
        # Biology: Active synapses undergo turnover, silent synapses are preserved
        if cfg.weight_decay > 0:
            # Compute decay factor per neuron based on activity level
            # [n_post] -> decay_factor for each post-synaptic neuron
            activity_ratio = (self.firing_rates / cfg.min_activity_for_decay).clamp(0.0, 1.0)
            # Interpolate: silent -> silent_decay_factor, active -> 1.0
            effective_decay = cfg.silent_decay_factor + (1.0 - cfg.silent_decay_factor) * activity_ratio
            # Apply decay per neuron: dw[j, :] -= decay[j] * weights[j, :]
            dw -= cfg.weight_decay * effective_decay.unsqueeze(1) * weights

        # Update threshold using firing rate EMA
        self._update_theta()

        # Return weight change without internal clamping (apply_learning() owns clamping)
        return weights + dw

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update decay factors when brain timestep changes.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        super().update_temporal_parameters(dt_ms)
        self._decay_theta_val       = math.exp(-dt_ms / self.config.tau_theta)
        self._decay_firing_rate_val = math.exp(-dt_ms / self._firing_rate_tau_ms)


class ThreeFactorStrategy(LearningStrategy):
    """Three-factor reinforcement learning rule.

    Δw = lr × eligibility × modulator

    - Eligibility: accumulated spike-timing correlations (Hebbian)
    - Modulator: dopamine, reward signal, or TD error

    Key insight: Without modulator, NO learning occurs (not reduced, NONE).
    """

    def __init__(self, config: ThreeFactorConfig):
        super().__init__(config)

        # Scalar decay factor — dimension-independent, pre-computed at default dt
        self._decay_elig_val: float = math.exp(-DEFAULT_DT_MS / config.eligibility_tau)

    # ------------------------------------------------------------------
    # Eager initialisation
    # ------------------------------------------------------------------

    def setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Eagerly register eligibility buffer.

        Must be called once (by BrainBuilder.build()) before compute_update().
        """
        self.register_buffer("eligibility", torch.zeros(n_post, n_pre, device=device))

    def ensure_setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Call setup() if not yet initialised or if synapse dimensions changed."""
        if (
            not hasattr(self, "eligibility")
            or self.eligibility.shape != (n_post, n_pre)
        ):
            self.setup(n_pre, n_post, device)

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
        self.ensure_setup(pre_spikes.shape[0], post_spikes.shape[0], pre_spikes.device)
        # Ensure 1D inputs
        if pre_spikes.dim() != 1:
            pre_spikes = pre_spikes.squeeze()
        if post_spikes.dim() != 1:
            post_spikes = post_spikes.squeeze()

        # Decay in-place
        self.eligibility.mul_(self._decay_elig_val)

        # Add Hebbian correlation: outer product [n_post, n_pre], normalized
        pre_rate = pre_spikes.float().mean() + 1e-6
        post_rate = post_spikes.float().mean() + 1e-6
        normalization = (pre_rate * post_rate).sqrt()

        hebbian = torch.outer(post_spikes.float(), pre_spikes.float())
        self.eligibility.add_(hebbian / normalization)

        # Clamp to prevent saturation (biological bound on synaptic tags)
        self.eligibility.clamp_(0.0, 1.0)

        return self.eligibility

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply three-factor learning.

        Args:
            weights: Current weights
            pre_spikes: Presynaptic activity
            post_spikes: Postsynaptic activity
            **kwargs: Must include 'modulator' (float) representing dopamine/reward signal

        Returns:
            Updated weights
        """
        self.ensure_setup(pre_spikes.shape[0], post_spikes.shape[0], pre_spikes.device)
        validate_spike_tensor(pre_spikes, tensor_name="pre_spikes")
        validate_spike_tensor(post_spikes, tensor_name="post_spikes")

        cfg = self.config

        modulator: float = kwargs.get("modulator", 0.0)

        # Update eligibility
        self.update_eligibility(pre_spikes, post_spikes)

        # Three-factor update — no internal clamping (apply_learning() owns it)
        dw = cfg.learning_rate * self.eligibility * modulator
        return weights + dw

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update decay factor when brain timestep changes."""
        super().update_temporal_parameters(dt_ms)
        self._decay_elig_val = math.exp(-dt_ms / self.config.eligibility_tau)


# =============================================================================
# D1/D2 MSN Three-Factor STDP
# =============================================================================


class D1STDPStrategy(LearningStrategy):
    """Three-factor STDP for D1 MSNs.

    Update rule::

        Δw = (fast_trace + α * slow_trace) × DA × lr

    Traces update every timestep (eligibility tags); weight change scales with the
    current per-neuron dopamine concentration ``DA``.  When ``DA = 0`` the traces
    still accumulate — only the weight update is gated, matching the biological
    three-factor rule.

    Args:
        config: :class:`D1STDPConfig` (or subclass) instance.
    """

    def __init__(self, config: D1STDPConfig) -> None:
        super().__init__(config)
        self._fast_decay: float = math.exp(-DEFAULT_DT_MS / config.fast_eligibility_tau_ms)
        self._slow_decay: float = math.exp(-DEFAULT_DT_MS / config.slow_eligibility_tau_ms)

    # ------------------------------------------------------------------
    # Eager / lazy setup
    # ------------------------------------------------------------------

    def setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Register fast and slow eligibility trace buffers."""
        self.register_buffer("fast_trace", torch.zeros(n_post, n_pre, device=device))
        self.register_buffer("slow_trace", torch.zeros(n_post, n_pre, device=device))

    def ensure_setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Call :meth:`setup` if not yet initialised or if dimensions changed."""
        if (
            not hasattr(self, "fast_trace")
            or self.fast_trace.shape != (n_post, n_pre)
        ):
            self.setup(n_pre, n_post, device)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply three-factor D1 update.

        Args:
            weights: Current weight matrix ``[n_post, n_pre]``.
            pre_spikes: Presynaptic spikes ``[n_pre]``.
            post_spikes: Postsynaptic spikes ``[n_post]``.
            **kwargs:
                dopamine (Tensor | float): Per-neuron ``[n_post]`` or scalar
                    dopamine concentration.  Positive → LTP, negative → LTD.

        Returns:
            Updated weight tensor (same shape as *weights*).
        """
        self.ensure_setup(pre_spikes.shape[0], post_spikes.shape[0], pre_spikes.device)
        validate_spike_tensor(pre_spikes, tensor_name="pre_spikes")
        validate_spike_tensor(post_spikes, tensor_name="post_spikes")

        cfg = self.config
        dopamine: Any = kwargs.get("dopamine", 0.0)

        # Hebbian outer product: [n_post, n_pre]
        eligibility_update = torch.outer(post_spikes.float(), pre_spikes.float())

        # Multi-timescale trace update (in-place for efficiency)
        self.fast_trace.mul_(self._fast_decay).add_(eligibility_update)
        self.slow_trace.mul_(self._slow_decay).add_(
            self.fast_trace * cfg.eligibility_consolidation_rate
        )

        combined = self.fast_trace + cfg.slow_trace_weight * self.slow_trace

        # DA modulation: broadcast [n_post] → [n_post, 1] or keep scalar
        if isinstance(dopamine, torch.Tensor):
            da: Any = dopamine.unsqueeze(-1)
        else:
            da = dopamine

        weight_update = combined * da * cfg.learning_rate
        return weights + weight_update

    # ------------------------------------------------------------------
    # Temporal parameter management
    # ------------------------------------------------------------------

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Recompute decay scalars when the brain timestep changes."""
        super().update_temporal_parameters(dt_ms)
        cfg = self.config
        self._fast_decay = math.exp(-dt_ms / cfg.fast_eligibility_tau_ms)
        self._slow_decay = math.exp(-dt_ms / cfg.slow_eligibility_tau_ms)


class D2STDPStrategy(D1STDPStrategy):
    """Three-factor STDP for D2 MSNs.

    Identical to :class:`D1STDPStrategy` except the dopamine signal is
    **inverted** before the weight update, reflecting Gi-coupled D2 receptor
    physiology (DA+ → LTD, DA- → LTP).

    Args:
        config: :class:`D2STDPConfig` (or :class:`D1STDPConfig`) instance.
    """

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Invert dopamine signal then delegate to :class:`D1STDPStrategy`."""
        dopamine = kwargs.get("dopamine", 0.0)
        if isinstance(dopamine, torch.Tensor):
            inverted: Any = -dopamine
        else:
            inverted = -dopamine
        return super().compute_update(
            weights=weights,
            pre_spikes=pre_spikes,
            post_spikes=post_spikes,
            **{**kwargs, "dopamine": inverted},
        )


# =============================================================================
# Predictive Coding / Anti-Hebbian
# =============================================================================


class PredictiveCodingStrategy(LearningStrategy):
    """Anti-Hebbian temporal predictive coding.

    Update rule::

        Δw = lr × post ⊗ delayed_pre

    The strategy maintains an internal ring-buffer so that *pre_spikes* passed
    to :meth:`compute_update` are the **current** (undelayed) spikes; the
    strategy internally retrieves spikes from ``prediction_delay_steps`` steps
    ago.  Pass a ``learning_rate_override`` kwarg to apply neuromodulatory
    scaling (e.g. dopamine) without changing the config.

    Args:
        config: :class:`PredictiveCodingConfig` instance.
    """

    def __init__(self, config: PredictiveCodingConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Eager / lazy setup
    # ------------------------------------------------------------------

    def setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Register the pre-spike ring buffer."""
        delay = max(1, self.config.prediction_delay_steps)
        self.register_buffer(
            "pre_spike_buffer",
            torch.zeros(delay, n_pre, device=device),
        )

    def ensure_setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Call :meth:`setup` if not yet initialised or if *n_pre* changed."""
        delay = max(1, self.config.prediction_delay_steps)
        if (
            not hasattr(self, "pre_spike_buffer")
            or self.pre_spike_buffer.shape != (delay, n_pre)
        ):
            self.setup(n_pre, n_post, device)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply anti-Hebbian predictive update.

        Args:
            weights: Current weight matrix ``[n_post, n_pre]``.
            pre_spikes: **Current-timestep** presynaptic spikes ``[n_pre]``.
                The strategy retrieves the internally buffered delayed version.
            post_spikes: Postsynaptic spikes ``[n_post]``.
            **kwargs:
                learning_rate_override (float): Optional effective learning rate
                    that replaces ``config.learning_rate`` for this call.  Use
                    when neuromodulation (e.g. dopamine) scales the base rate.

        Returns:
            Updated weight tensor (same shape as *weights*).
        """
        n_pre = pre_spikes.shape[0]
        self.ensure_setup(n_pre, post_spikes.shape[0], pre_spikes.device)
        validate_spike_tensor(pre_spikes, tensor_name="pre_spikes")
        validate_spike_tensor(post_spikes, tensor_name="post_spikes")

        lr: float = kwargs.get("learning_rate_override", self.config.learning_rate)

        # Oldest slot in the ring buffer = most delayed pre-spikes
        delayed_pre = self.pre_spike_buffer[0]  # [n_pre]

        # Anti-Hebbian outer product: strengthen inhibition on co-activation
        dw = lr * torch.outer(post_spikes.float(), delayed_pre.float())

        # Advance ring buffer: shift existing entries back, insert current spikes
        self.pre_spike_buffer = torch.roll(self.pre_spike_buffer, shifts=1, dims=0)
        self.pre_spike_buffer[0] = pre_spikes.float()

        return weights + dw


# =============================================================================
# Composite Strategy (for chaining multiple strategies together)
# =============================================================================


class CompositeStrategy(LearningStrategy):
    """Composite learning strategy that applies multiple strategies sequentially.

    This is a proper ``nn.Module`` so sub-strategies' buffers/parameters move
    with ``.to(device)`` and appear in ``state_dict()``.

    Args:
        strategies: Ordered list of :class:`LearningStrategy` instances.  They
            are applied left-to-right; each receives the weights returned by
            the previous strategy.
    """

    def __init__(self, strategies: List[LearningStrategy]) -> None:
        if not strategies:
            raise ValueError("CompositeStrategy requires at least one sub-strategy")
        super().__init__(config=strategies[0].config)
        self.sub_strategies: nn.ModuleList = nn.ModuleList(strategies)

    def setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Propagate setup to all sub-strategies."""
        for s in self.sub_strategies:
            if isinstance(s, LearningStrategy):
                s.setup(n_pre, n_post, device)

    def ensure_setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Propagate ensure_setup to all sub-strategies."""
        for s in self.sub_strategies:
            if isinstance(s, LearningStrategy):
                s.ensure_setup(n_pre, n_post, device)

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply all sub-strategies sequentially."""
        for s in self.sub_strategies:
            if isinstance(s, LearningStrategy):
                weights = s.compute_update(
                    weights=weights,
                    pre_spikes=pre_spikes,
                    post_spikes=post_spikes,
                    **kwargs,
                )
        return weights

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Propagate timestep update to all sub-strategies."""
        super().update_temporal_parameters(dt_ms)
        for s in self.sub_strategies:
            if isinstance(s, LearningStrategy):
                s.update_temporal_parameters(dt_ms)
