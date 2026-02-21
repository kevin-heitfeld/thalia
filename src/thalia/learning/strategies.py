"""Learning Rule Strategies: Pluggable Learning Algorithms for Brain Components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn

from thalia.constants import DEFAULT_DT_MS
from thalia.utils import clamp_weights, validate_spike_tensor

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


# =============================================================================
# Base Learning Strategy
# =============================================================================


class LearningStrategy(nn.Module, ABC):
    """Base class for learning strategies with common functionality."""

    def __init__(self, config: LearningConfig):
        super().__init__()
        self.config = config
        self._dt_ms: float = DEFAULT_DT_MS

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
            Updated weight matrix
        """

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        self._dt_ms = dt_ms

# =============================================================================
# Strategy Implementations
# =============================================================================


class STDPStrategy(LearningStrategy):
    """Spike-Timing Dependent Plasticity.

    Trace-based STDP:
        - LTP: pre_trace × post_spike (pre before post)
        - LTD: post_trace × pre_spike (post before pre)
    """

    def __init__(self, config: STDPConfig):
        super().__init__(config)

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
            self._trace_manager = EligibilityTraceManager(
                n_input=n_pre,
                n_output=n_post,
                config=self.config,
                device=device,
            )

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
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
        """
        validate_spike_tensor(pre_spikes, tensor_name="pre_spikes")
        validate_spike_tensor(post_spikes, tensor_name="post_spikes")

        # Extract modulation kwargs
        # These are passed by regions that track neuromodulator concentrations (e.g., cortex)
        dopamine = kwargs.get("dopamine", 0.0)
        acetylcholine = kwargs.get("acetylcholine", 0.0)
        norepinephrine = kwargs.get("norepinephrine", 0.0)

        # Initialize trace manager if needed
        self._ensure_trace_manager(pre_spikes, post_spikes)
        assert self._trace_manager is not None, "Trace manager must be initialized"

        cfg = self.config

        # Initialize firing rates if needed
        n_post = post_spikes.shape[0]
        if self.firing_rates is None or self.firing_rates.shape[0] != n_post:
            self.firing_rates = torch.zeros(n_post, device=post_spikes.device, dtype=torch.float32)

        # Initialize retrograde signal if needed
        if cfg.retrograde_enabled:
            if self.retrograde_signal is None or self.retrograde_signal.shape[0] != n_post:
                self.retrograde_signal = torch.zeros(n_post, device=post_spikes.device, dtype=torch.float32)

        # Update traces and compute LTP/LTD
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
        if cfg.retrograde_enabled and self.retrograde_signal is not None:
            assert self._retrograde_decay is not None, "Retrograde decay not initialized"
            # Retrograde release is triggered by spiking activity
            # The signal accumulates and decays slowly (1s timescale)
            spike_contribution = post_spikes.float()  # 1.0 when neuron fires

            # Scale by recent firing rate: frequent firing = stronger retrograde release
            # This captures "how important is this neuron's activity?"
            rate_scaling = (self.firing_rates / cfg.retrograde_threshold).clamp(0, 2.0)
            weighted_contribution = spike_contribution * (0.5 + 0.5 * rate_scaling)

            # Update retrograde signal with slow decay
            self.retrograde_signal = (
                self._retrograde_decay * self.retrograde_signal +
                (1 - self._retrograde_decay) * weighted_contribution
            )

        # Apply retrograde signaling (endocannabinoid-like gating)
        # LTP is gated: only strong postsynaptic responses drive potentiation
        # LTD is enhanced: weak responses lead to depotentiation (extinction-like)
        if cfg.retrograde_enabled and self.retrograde_signal is not None:
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
            # Apply weight change with bounds
            new_weights = clamp_weights(weights + dw, cfg.w_min, cfg.w_max, inplace=False)
        else:
            new_weights = weights

        return new_weights

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        super().update_temporal_parameters(dt_ms)
        # Compute firing rate decay factor: alpha = 1 - dt/tau
        self._firing_rate_decay = 1.0 - dt_ms / self._firing_rate_tau_ms
        # Compute retrograde signal decay factor
        self._retrograde_decay = 1.0 - dt_ms / self.config.retrograde_tau_ms
        # NOTE: Trace manager computes decay factors on-the-fly in update_traces()


class BCMStrategy(LearningStrategy):
    """Bienenstock-Cooper-Munro learning with sliding threshold.

    BCM function: φ(c, θ) = c(c - θ)
    Weight update: Δw = lr × pre × φ(post, θ)

    The threshold θ adapts to track E[post²], providing automatic
    metaplasticity that stabilizes learning.
    """

    def __init__(self, config: BCMConfig):
        super().__init__(config)

        # Decay factor (computed in update_temporal_parameters)
        self.register_buffer("decay_theta", torch.tensor(0.0, dtype=torch.float32))

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
                self.config.theta_init,
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
        depression_mask = phi < 0  # Where BCM would depress (c < theta)
        barely_active = firing_rates < self.config.activity_threshold  # Low firing rate
        invalid_depression = depression_mask & barely_active
        phi = torch.where(invalid_depression, torch.zeros_like(phi), phi)

        return phi

    def _update_theta(self, post_spikes: torch.Tensor) -> None:
        """Update sliding threshold.

        Args:
            post_spikes: Postsynaptic activity [n_post] (1D)
        """
        cfg = self.config

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
    ) -> torch.Tensor:
        """Apply BCM learning.

        Δw[j,i] = lr × pre[i] × φ(post[j], θ[j])

        Args:
            weights: Current weights [n_post, n_pre]
            pre_spikes: Presynaptic activity [n_pre] (1D)
            post_spikes: Postsynaptic activity [n_post] (1D)
        """
        validate_spike_tensor(pre_spikes, tensor_name="pre_spikes")
        validate_spike_tensor(post_spikes, tensor_name="post_spikes")

        cfg = self.config

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

        # Apply weight change with bounds
        new_weights = clamp_weights(weights + dw, cfg.w_min, cfg.w_max, inplace=False)

        return new_weights

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update decay factors when brain timestep changes.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        super().update_temporal_parameters(dt_ms)
        tau = self.config.tau_theta
        self.decay_theta: torch.Tensor = torch.tensor(
            1.0 - dt_ms / tau, dtype=torch.float32, device=self.decay_theta.device
        )
        # Compute firing rate decay factor
        self.decay_firing_rate: torch.Tensor = torch.tensor(
            1.0 - dt_ms / self._firing_rate_tau_ms, dtype=torch.float32, device=self.decay_firing_rate.device
        )


class ThreeFactorStrategy(LearningStrategy):
    """Three-factor reinforcement learning rule.

    Δw = lr × eligibility × modulator

    - Eligibility: accumulated spike-timing correlations (Hebbian)
    - Modulator: dopamine, reward signal, or TD error

    Key insight: Without modulator, NO learning occurs (not reduced, NONE).
    """

    def __init__(self, config: ThreeFactorConfig):
        super().__init__(config)

        # Decay factors (computed in update_temporal_parameters)
        self.register_buffer("decay_elig", torch.tensor(0.0, dtype=torch.float32))

        # Eligibility trace (Hebbian correlation)
        self.eligibility: Optional[torch.Tensor] = None

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
        # Normalize by spike rates to prevent saturation with high activity
        # Ensure hebbian is on the same device as eligibility
        pre_rate = pre_spikes.float().mean() + 1e-6  # Avoid div by zero
        post_rate = post_spikes.float().mean() + 1e-6
        normalization = torch.sqrt(pre_rate * post_rate)  # Geometric mean of rates

        hebbian = torch.outer(post_spikes.float().to(module_device), pre_spikes.float().to(module_device))
        self.eligibility = self.eligibility + hebbian / normalization

        # Clamp to prevent saturation (biological bound on synaptic tags)
        # Typical range: [0, 1] normalized to max correlation strength
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
        validate_spike_tensor(pre_spikes, tensor_name="pre_spikes")
        validate_spike_tensor(post_spikes, tensor_name="post_spikes")

        cfg = self.config

        modulator: float = kwargs.get("modulator", 0.0)

        # Update eligibility
        self.update_eligibility(pre_spikes, post_spikes)

        # Three-factor update
        assert self.eligibility is not None, "Eligibility must be initialized"
        dw = cfg.learning_rate * self.eligibility * modulator

        # Apply weight change with bounds
        new_weights = clamp_weights(weights + dw, cfg.w_min, cfg.w_max, inplace=False)

        return new_weights

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update decay factors when brain timestep changes.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        super().update_temporal_parameters(dt_ms)
        self.decay_elig: torch.Tensor = torch.tensor(
            1.0 - dt_ms / self.config.eligibility_tau,
            dtype=torch.float32,
            device=self.decay_elig.device,
        )


class CompositeStrategy:
    """Composite learning strategy that applies multiple strategies sequentially."""

    @staticmethod
    def compute_update(
        strategies: List[LearningStrategy],
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply all strategies sequentially."""
        current_weights = weights

        for strategy in strategies:
            if isinstance(strategy, LearningStrategy):
                current_weights = strategy.compute_update(
                    weights=current_weights,
                    pre_spikes=pre_spikes,
                    post_spikes=post_spikes,
                    **kwargs
                )

        return current_weights

    @staticmethod
    def update_temporal_parameters(strategies: List[LearningStrategy], dt_ms: float) -> None:
        """Update temporal parameters for all sub-strategies."""
        for strategy in strategies:
            strategy.update_temporal_parameters(dt_ms)
