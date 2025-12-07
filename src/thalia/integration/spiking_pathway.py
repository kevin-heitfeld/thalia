"""
SpikingPathway: Fully spiking inter-region connections with temporal dynamics.

This module implements biologically realistic spiking pathways between
brain regions, supporting both rate and temporal coding.

Why Spiking Pathways Matter:
============================
1. STABILITY: More spikes = more samples = lower variance in estimates
2. TEMPORAL CODING: Spike timing carries information beyond rate
3. SYNCHRONIZATION: Phase relationships between regions coordinate processing
4. BIOLOGICAL REALISM: All brain connections are spiking

Temporal Coding Mechanisms:
===========================
1. SPIKE TIMING: First-spike latency encodes stimulus intensity
2. PHASE CODING: Spikes locked to theta/gamma oscillations
3. SYNCHRONY: Coincident spikes across neurons encode binding
4. BURST CODING: Burst vs single spikes encode different information

Learning Rules:
===============
1. STDP: Spike-timing dependent plasticity (causal = LTP, anti-causal = LTD)
2. PHASE-STDP: STDP modulated by oscillation phase
3. BURST-STDP: Bursts trigger different plasticity than single spikes
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import numpy as np

from thalia.core.pathway_protocol import BaseNeuralPathway
from thalia.core.stp import ShortTermPlasticity, STPConfig, STPType
from thalia.core.utils import clamp_weights
from thalia.core.weight_init import WeightInitializer
from thalia.learning.bcm import BCMRule, BCMConfig


class TemporalCoding(Enum):
    """Temporal coding schemes for spike information."""
    RATE = auto()           # Information in firing rate
    LATENCY = auto()        # Information in first-spike latency
    PHASE = auto()          # Information in spike phase relative to oscillation
    SYNCHRONY = auto()      # Information in coincident spikes
    BURST = auto()          # Information in burst patterns


class SpikingLearningRule(Enum):
    """Learning rules for spiking pathways."""
    STDP = auto()           # Classic spike-timing dependent plasticity
    PHASE_STDP = auto()     # STDP modulated by oscillation phase
    TRIPLET_STDP = auto()   # Triplet rule (more biologically accurate)
    DOPAMINE_STDP = auto()  # Dopamine-modulated STDP (reward learning)
    REPLAY_STDP = auto()    # Only active during replay/sleep


@dataclass
class SpikingPathwayConfig:
    """Configuration for a spiking pathway.

    Attributes:
        source_size: Number of neurons in source region
        target_size: Number of neurons in target region

        # Neuron model parameters
        tau_mem_ms: Membrane time constant
        tau_syn_ms: Synaptic time constant
        v_thresh: Spike threshold
        v_reset: Reset voltage after spike
        v_rest: Resting membrane potential
        refractory_ms: Refractory period

        # STDP parameters
        learning_rule: Which plasticity rule to use
        stdp_lr: Learning rate
        tau_plus_ms: LTP time constant
        tau_minus_ms: LTD time constant
        a_plus: LTP amplitude
        a_minus: LTD amplitude

        # Weight bounds
        w_min: Minimum weight
        w_max: Maximum weight
        soft_bounds: Use soft bounds

        # Temporal coding
        temporal_coding: Which temporal coding scheme
        oscillation_freq_hz: Frequency for phase coding
        phase_precision: How tightly spikes lock to phase

        # Connectivity
        sparsity: Fraction of connections to keep
        topographic: Use topographic connectivity
        axonal_delay_ms: Axonal conduction delay
        delay_variability: Variability in delays

        # Homeostasis
        synaptic_scaling: Enable homeostatic scaling
        target_rate: Target firing rate for scaling

        device: Compute device
    """
    source_size: int
    target_size: int

    # Neuron model
    tau_mem_ms: float = 20.0
    tau_syn_ms: float = 5.0
    v_thresh: float = 1.0
    v_reset: float = 0.0
    v_rest: float = 0.0
    refractory_ms: float = 2.0

    # STDP
    learning_rule: SpikingLearningRule = SpikingLearningRule.STDP
    stdp_lr: float = 0.01
    tau_plus_ms: float = 20.0
    tau_minus_ms: float = 20.0
    a_plus: float = 1.0
    a_minus: float = 1.0

    # Weights
    w_min: float = 0.0
    w_max: float = 1.0
    soft_bounds: bool = True
    init_mean: float = 0.3
    init_std: float = 0.1

    # Temporal coding
    temporal_coding: TemporalCoding = TemporalCoding.RATE
    oscillation_freq_hz: float = 8.0  # Theta rhythm
    phase_precision: float = 0.5  # How tightly spikes lock to phase

    # Connectivity
    sparsity: float = 1.0
    topographic: bool = False
    axonal_delay_ms: float = 1.0
    delay_variability: float = 0.2

    # Homeostasis
    synaptic_scaling: bool = True
    target_rate: float = 0.1
    scaling_tau_ms: float = 1000.0

    # Short-Term Plasticity (STP)
    stp_enabled: bool = False
    stp_type: STPType = STPType.DEPRESSING  # Preset synapse type
    stp_config: Optional[STPConfig] = None  # Custom STP parameters (overrides stp_type)

    # BCM sliding threshold (metaplasticity)
    bcm_enabled: bool = False
    bcm_config: Optional[BCMConfig] = None  # Custom BCM parameters

    device: str = "cpu"


class SpikingPathway(BaseNeuralPathway):
    """
    Fully spiking inter-region pathway with temporal dynamics.
    
    Inherits from BaseNeuralPathway, implementing the NeuralPathway protocol
    interface with a standardized API for inter-region connections.

    Unlike rate-based pathways, this implements actual spiking neurons with:
    - Leaky integrate-and-fire dynamics
    - Synaptic currents with temporal filtering
    - Axonal conduction delays
    - **STDP learning (automatic during every forward pass)**
    - Support for temporal coding schemes

    The pathway acts as a "mini-region" that transforms spikes
    from source to target while maintaining temporal structure.
    
    Protocol Compliance:
    - forward(): Process spikes through pathway (**learning happens automatically**)
    - reset_state(): Clear membrane potentials, traces, delays
    - get_diagnostics(): Report activity and learning metrics
    
    Note on Learning:
    Like brain regions (Prefrontal, Hippocampus, etc.), this pathway
    ALWAYS learns during forward passes via STDP. There is no separate
    learn() method - plasticity happens continuously and automatically.

    Example:
        pathway = SpikingPathway(SpikingPathwayConfig(
            source_size=64,
            target_size=128,
            temporal_coding=TemporalCoding.PHASE,
            oscillation_freq_hz=8.0,  # Theta
        ))

        for t in range(n_timesteps):
            target_spikes = pathway(source_spikes[t], dt=1.0, time_ms=t)
            # STDP learning happens automatically during forward pass!

        # Check learning metrics
        metrics = pathway.get_learning_metrics()
    """

    def __init__(self, config: SpikingPathwayConfig):
        super().__init__()
        self.config = config

        # =====================================================================
        # WEIGHTS with axonal delays
        # =====================================================================
        self.weights = nn.Parameter(
            self._initialize_weights(),
            requires_grad=False,
        )

        # Axonal delays (in timesteps, will be converted from ms)
        delays = WeightInitializer.gaussian(
            n_output=config.target_size,
            n_input=config.source_size,
            mean=config.axonal_delay_ms,
            std=config.delay_variability * config.axonal_delay_ms,
            device=config.device
        ).clamp(min=0.1)
        self.register_buffer("axonal_delays", delays)

        # Connectivity mask
        if config.sparsity < 1.0:
            mask = WeightInitializer.sparse_random(
                n_output=config.target_size,
                n_input=config.source_size,
                sparsity=config.sparsity,
                device=config.device
            )
            self.register_buffer("connectivity_mask", mask)
        else:
            self.connectivity_mask = None

        # =====================================================================
        # NEURON STATE (for target neurons in this pathway)
        # =====================================================================
        # Membrane potential
        self.register_buffer(
            "membrane",
            torch.zeros(config.target_size, device=config.device) + config.v_rest,
        )

        # Synaptic current (filtered input)
        self.register_buffer(
            "synaptic_current",
            torch.zeros(config.target_size, device=config.device),
        )

        # Refractory counter
        self.register_buffer(
            "refractory",
            torch.zeros(config.target_size, device=config.device),
        )

        # =====================================================================
        # SPIKE TRACES for STDP
        # =====================================================================
        # Pre-synaptic traces (one per source neuron)
        self.register_buffer(
            "pre_trace",
            torch.zeros(config.source_size, device=config.device),
        )

        # Post-synaptic traces (one per target neuron)
        self.register_buffer(
            "post_trace",
            torch.zeros(config.target_size, device=config.device),
        )

        # =====================================================================
        # DELAY BUFFER for axonal delays
        # =====================================================================
        max_delay_steps = int(config.axonal_delay_ms * 2 / 1.0) + 1  # Assume dt=1ms
        self.register_buffer(
            "delay_buffer",
            torch.zeros(max_delay_steps, config.source_size, device=config.device),
        )
        self.delay_buffer_idx = 0

        # =====================================================================
        # OSCILLATION STATE for phase coding
        # =====================================================================
        self.oscillation_phase = 0.0  # Current phase (0 to 2π)

        # =====================================================================
        # LEARNING STATE
        # =====================================================================
        self.dopamine_level = 0.0
        self.replay_active = False
        self.total_ltp = 0.0
        self.total_ltd = 0.0

        # Firing rate tracking for homeostasis
        self.register_buffer(
            "firing_rate_estimate",
            torch.zeros(config.target_size, device=config.device) + config.target_rate,
        )

        # =====================================================================
        # SHORT-TERM PLASTICITY (STP)
        # =====================================================================
        if config.stp_enabled:
            stp_cfg = config.stp_config
            if stp_cfg is None:
                # Use preset based on stp_type
                stp_cfg = STPConfig.from_type(config.stp_type, dt=1.0)
            self.stp = ShortTermPlasticity(
                n_pre=config.source_size,
                n_post=config.target_size,
                config=stp_cfg,
                per_synapse=True,  # Per-synapse STP for pathways
            )
            self.stp.to(config.device)
        else:
            self.stp = None

        # =====================================================================
        # BCM SLIDING THRESHOLD (Metaplasticity)
        # =====================================================================
        if config.bcm_enabled:
            bcm_cfg = config.bcm_config or BCMConfig()
            self.bcm = BCMRule(
                n_post=config.target_size,
                config=bcm_cfg,
            )
            self.bcm.to(config.device)
        else:
            self.bcm = None

    def _initialize_weights(self) -> torch.Tensor:
        """Initialize weights with optional topographic structure."""
        cfg = self.config

        if cfg.topographic:
            # Use topographic initialization from registry
            weights = WeightInitializer.topographic(
                n_output=cfg.target_size,
                n_input=cfg.source_size,
                base_weight=cfg.init_mean,
                sigma_factor=4.0,
                boost_strength=0.3,
                device=cfg.device
            )
        else:
            # Use Gaussian initialization from registry
            weights = WeightInitializer.gaussian(
                n_output=cfg.target_size,
                n_input=cfg.source_size,
                mean=cfg.init_mean,
                std=cfg.init_std,
                device=cfg.device
            )

        return weights.clamp(cfg.w_min, cfg.w_max)

    def forward(
        self,
        source_spikes: torch.Tensor,
        dt: float = 1.0,
        time_ms: float = 0.0,
    ) -> torch.Tensor:
        """Process source spikes and generate target spikes.

        Args:
            source_spikes: Binary spike tensor from source (source_size,)
            dt: Timestep in ms
            time_ms: Current time in ms (for phase coding)

        Returns:
            Binary spike tensor for target (target_size,)
        """
        cfg = self.config

        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        # Ensure 1D
        source_spikes = source_spikes.squeeze()
        assert source_spikes.shape[-1] == cfg.source_size, (
            f"SpikingPathway.forward: source_spikes has shape {source_spikes.shape} "
            f"but source_size={cfg.source_size}. Check pathway input dimensions."
        )

        # =====================================================================
        # 1. UPDATE OSCILLATION PHASE (for phase coding)
        # =====================================================================
        if cfg.temporal_coding == TemporalCoding.PHASE:
            self.oscillation_phase = (
                2 * np.pi * cfg.oscillation_freq_hz * time_ms / 1000.0
            ) % (2 * np.pi)

        # =====================================================================
        # 2. HANDLE AXONAL DELAYS
        # =====================================================================
        # Store current spikes in delay buffer
        self.delay_buffer[self.delay_buffer_idx] = source_spikes

        # Get delayed spikes for each synapse
        # For simplicity, use average delay (per-synapse delays would need more complex indexing)
        avg_delay_steps = int(cfg.axonal_delay_ms / dt)
        delayed_idx = (self.delay_buffer_idx - avg_delay_steps) % self.delay_buffer.shape[0]
        delayed_spikes = self.delay_buffer[delayed_idx]

        # Advance buffer index
        self.delay_buffer_idx = (self.delay_buffer_idx + 1) % self.delay_buffer.shape[0]

        # =====================================================================
        # 3. COMPUTE SYNAPTIC INPUT (with optional STP)
        # =====================================================================
        effective_weights = self.weights
        if self.connectivity_mask is not None:
            effective_weights = self.weights * self.connectivity_mask

        # Apply Short-Term Plasticity if enabled
        if self.stp is not None:
            # Get STP efficacy based on presynaptic activity
            # efficacy shape: (n_pre, n_post) for per-synapse STP
            stp_efficacy = self.stp(delayed_spikes.unsqueeze(0)).squeeze(0)
            # Modulate weights by STP efficacy
            effective_weights = effective_weights * stp_efficacy.T  # Transpose for (n_post, n_pre)

        # Weighted sum of delayed spikes
        synaptic_input = torch.matmul(effective_weights, delayed_spikes)

        # =====================================================================
        # 4. UPDATE SYNAPTIC CURRENT (temporal filtering)
        # =====================================================================
        syn_decay = np.exp(-dt / cfg.tau_syn_ms)
        self.synaptic_current = self.synaptic_current * syn_decay + synaptic_input

        # =====================================================================
        # 5. UPDATE MEMBRANE POTENTIAL
        # =====================================================================
        # Decay membrane toward rest
        mem_decay = np.exp(-dt / cfg.tau_mem_ms)
        self.membrane = (
            cfg.v_rest +
            (self.membrane - cfg.v_rest) * mem_decay +
            self.synaptic_current * (1 - mem_decay)
        )

        # =====================================================================
        # 6. GENERATE SPIKES
        # =====================================================================
        # Decrease refractory counters
        self.refractory = (self.refractory - dt).clamp(min=0)

        # Check for threshold crossing (only if not refractory)
        can_spike = self.refractory <= 0

        # Apply phase modulation for phase coding
        if cfg.temporal_coding == TemporalCoding.PHASE:
            # Threshold is lower at preferred phase (phase=0)
            phase_modulation = 1.0 - cfg.phase_precision * np.cos(self.oscillation_phase)
            effective_thresh = cfg.v_thresh * phase_modulation
        else:
            effective_thresh = cfg.v_thresh

        # Generate spikes
        target_spikes = ((self.membrane >= effective_thresh) & can_spike).float()

        # Reset spiking neurons
        spike_mask = target_spikes > 0
        self.membrane[spike_mask] = cfg.v_reset
        self.refractory[spike_mask] = cfg.refractory_ms

        # =====================================================================
        # 7. UPDATE STDP TRACES
        # =====================================================================
        # Pre-synaptic trace
        pre_decay = np.exp(-dt / cfg.tau_plus_ms)
        self.pre_trace = self.pre_trace * pre_decay + source_spikes

        # Post-synaptic trace
        post_decay = np.exp(-dt / cfg.tau_minus_ms)
        self.post_trace = self.post_trace * post_decay + target_spikes

        # =====================================================================
        # 8. APPLY STDP LEARNING
        # =====================================================================
        if cfg.learning_rule != SpikingLearningRule.REPLAY_STDP or self.replay_active:
            self._apply_stdp(source_spikes, target_spikes, dt)

        # =====================================================================
        # 9. UPDATE FIRING RATE ESTIMATE (for homeostasis)
        # =====================================================================
        rate_decay = np.exp(-dt / cfg.scaling_tau_ms)
        self.firing_rate_estimate = (
            self.firing_rate_estimate * rate_decay +
            target_spikes * (1 - rate_decay)
        )

        # Apply synaptic scaling
        if cfg.synaptic_scaling:
            self._apply_synaptic_scaling(dt)

        return target_spikes

    def _apply_stdp(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        dt: float,
    ) -> None:
        """Apply STDP weight updates with optional BCM modulation."""
        cfg = self.config

        # =====================================================================
        # BCM MODULATION (if enabled)
        # =====================================================================
        # BCM scales the learning rate based on postsynaptic activity relative
        # to a sliding threshold. This prevents runaway potentiation/depression.
        bcm_modulation = 1.0
        if self.bcm is not None:
            # Compute BCM modulation factor based on postsynaptic activity
            # φ(c, θ) = c(c - θ) / θ  →  positive above threshold, negative below
            post_activity = post_spikes.mean()  # Average firing rate
            bcm_modulation = self.bcm.compute_phi(post_activity)
            # Update the sliding threshold based on recent activity
            self.bcm.update_threshold(post_activity)

        # LTP: post spike with pre trace → strengthen
        # For each post-synaptic spike, potentiate synapses from neurons with recent pre-spikes
        if post_spikes.sum() > 0:
            ltp = cfg.a_plus * torch.outer(post_spikes, self.pre_trace)

            # Modulate by dopamine for dopamine-STDP
            if cfg.learning_rule == SpikingLearningRule.DOPAMINE_STDP:
                ltp = ltp * (1 + self.dopamine_level)

            # Modulate by phase for phase-STDP
            if cfg.learning_rule == SpikingLearningRule.PHASE_STDP:
                # LTP is stronger at preferred phase
                phase_mod = 0.5 + 0.5 * np.cos(self.oscillation_phase)
                ltp = ltp * phase_mod

            # Apply BCM modulation
            if isinstance(bcm_modulation, torch.Tensor):
                ltp = ltp * bcm_modulation.unsqueeze(1)
            else:
                ltp = ltp * bcm_modulation

            self.total_ltp += ltp.sum().item()
        else:
            ltp = 0

        # LTD: pre spike with post trace → weaken
        if pre_spikes.sum() > 0:
            ltd = cfg.a_minus * torch.outer(self.post_trace, pre_spikes)

            if cfg.learning_rule == SpikingLearningRule.DOPAMINE_STDP:
                # LTD is reduced by dopamine (protects good synapses)
                ltd = ltd * (1 - 0.5 * max(0, self.dopamine_level))

            if cfg.learning_rule == SpikingLearningRule.PHASE_STDP:
                # LTD is stronger at anti-phase
                phase_mod = 0.5 - 0.5 * np.cos(self.oscillation_phase)
                ltd = ltd * phase_mod

            self.total_ltd += ltd.sum().item()
        else:
            ltd = 0

        # Compute weight change
        dw = cfg.stdp_lr * (ltp - ltd)

        # Apply soft bounds
        if cfg.soft_bounds and isinstance(dw, torch.Tensor):
            w_norm = (self.weights.data - cfg.w_min) / (cfg.w_max - cfg.w_min)
            ltp_factor = 1.0 - w_norm
            ltd_factor = w_norm
            dw_pos = dw.clamp(min=0) * ltp_factor
            dw_neg = dw.clamp(max=0) * ltd_factor
            dw = dw_pos + dw_neg

        # Apply connectivity mask
        if self.connectivity_mask is not None and isinstance(dw, torch.Tensor):
            dw = dw * self.connectivity_mask

        # Update weights
        if isinstance(dw, torch.Tensor):
            self.weights.data += dw
            clamp_weights(self.weights.data, cfg.w_min, cfg.w_max)

    def _apply_synaptic_scaling(self, dt: float) -> None:
        """Apply homeostatic synaptic scaling."""
        cfg = self.config

        # Compute scaling factor per neuron
        rate_ratio = cfg.target_rate / (self.firing_rate_estimate + 1e-6)

        # Gentle scaling (1% per timestep toward target)
        scale_factor = 1.0 + 0.01 * (rate_ratio - 1.0)
        scale_factor = scale_factor.clamp(0.99, 1.01)  # Limit change rate

        # Apply to outgoing weights of each target neuron
        self.weights.data *= scale_factor.unsqueeze(1)
        clamp_weights(self.weights.data, cfg.w_min, cfg.w_max)

    def reset_state(self) -> None:
        """Reset all state (call between trials)."""
        cfg = self.config

        self.membrane.fill_(cfg.v_rest)
        self.synaptic_current.zero_()
        self.refractory.zero_()
        self.pre_trace.zero_()
        self.post_trace.zero_()
        self.delay_buffer.zero_()
        self.delay_buffer_idx = 0
        self.oscillation_phase = 0.0

    def reset_traces(self) -> None:
        """Reset eligibility traces (alias for reset_state for compatibility)."""
        self.reset_state()

    def learn(
        self,
        source_activity: Optional[torch.Tensor] = None,
        target_activity: Optional[torch.Tensor] = None,
        dopamine: float = 0.0,
        **kwargs,
    ) -> Dict[str, float]:
        """
        External learning interface for compatibility with non-spiking pathways.

        In spiking pathways, learning happens continuously via STDP during
        forward passes. This method allows triggering additional learning
        updates with dopamine modulation.

        Args:
            source_activity: Pre-synaptic activity [source_size] (uses traces if None)
            target_activity: Post-synaptic activity [target_size] (uses traces if None)
            dopamine: Dopamine signal for modulated learning
            **kwargs: Additional arguments (ignored)

        Returns:
            Dict with learning metrics
        """
        # Set dopamine level for modulated STDP
        self.set_dopamine(dopamine)

        # If activities provided, run a learning step
        if source_activity is not None and target_activity is not None:
            cfg = self.config

            # Ensure correct shapes
            pre_trace = torch.clamp(source_activity.flatten(), 0, 1)
            post_trace = torch.clamp(target_activity.flatten(), 0, 1)

            # Resize to match weight dimensions if needed
            if pre_trace.shape[0] != cfg.source_size:
                pre_resized = torch.zeros(cfg.source_size, device=pre_trace.device)
                pre_resized[:min(pre_trace.shape[0], cfg.source_size)] = pre_trace[:cfg.source_size]
                pre_trace = pre_resized
            if post_trace.shape[0] != cfg.target_size:
                post_resized = torch.zeros(cfg.target_size, device=post_trace.device)
                post_resized[:min(post_trace.shape[0], cfg.target_size)] = post_trace[:cfg.target_size]
                post_trace = post_resized

            # STDP update: weights are [target_size, source_size]
            # LTP: post-before-pre → strengthen (post outer pre)
            # LTD: pre-before-post → weaken (pre outer post)^T = (post outer pre) negated
            # For simplicity, use Hebbian: strengthen when both active
            ltp = torch.outer(post_trace, pre_trace)  # [target, source]

            lr = cfg.stdp_lr

            # Dopamine modulation
            if cfg.learning_rule == SpikingLearningRule.DOPAMINE_STDP:
                lr *= (1.0 + dopamine)

            # Update weights - simple Hebbian with dopamine gating
            dw = lr * cfg.a_plus * ltp * (1.0 + dopamine)

            # Soft bounds
            if cfg.soft_bounds:
                dw = dw * (cfg.w_max - self.weights.data) * (self.weights.data - cfg.w_min)

            self.weights.data += dw
            clamp_weights(self.weights.data, cfg.w_min, cfg.w_max)

            self.total_ltp += ltp.sum().item()

        return self.get_learning_metrics()

    def set_dopamine(self, level: float) -> None:
        """Set dopamine level for modulated learning."""
        self.dopamine_level = level

    def set_replay_mode(self, active: bool) -> None:
        """Enable/disable replay mode for replay-gated STDP."""
        self.replay_active = active

    def get_learning_metrics(self) -> Dict[str, float]:
        """Get learning metrics since last reset."""
        return {
            "total_ltp": self.total_ltp,
            "total_ltd": self.total_ltd,
            "ltp_ltd_ratio": self.total_ltp / max(self.total_ltd, 1e-6),
            "mean_weight": self.weights.data.mean().item(),
            "weight_std": self.weights.data.std().item(),
            "mean_firing_rate": self.firing_rate_estimate.mean().item(),
        }

    def reset_learning_metrics(self) -> None:
        """Reset accumulated learning metrics."""
        self.total_ltp = 0.0
        self.total_ltd = 0.0

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics."""
        return {
            "weight_mean": self.weights.data.mean().item(),
            "weight_std": self.weights.data.std().item(),
            "weight_min": self.weights.data.min().item(),
            "weight_max": self.weights.data.max().item(),
            "membrane_mean": self.membrane.mean().item(),
            "membrane_std": self.membrane.std().item(),
            "firing_rate_mean": self.firing_rate_estimate.mean().item(),
            "pre_trace_norm": self.pre_trace.norm().item(),
            "post_trace_norm": self.post_trace.norm().item(),
            "oscillation_phase": self.oscillation_phase,
            "dopamine_level": self.dopamine_level,
            "replay_active": self.replay_active,
            "total_ltp": self.total_ltp,
            "total_ltd": self.total_ltd,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get pathway state for checkpointing.

        Returns state dictionary with:
        - weights: Learnable synaptic weights
        - neuron_state: Membrane potentials, synaptic currents, refractory timers
        - stdp_traces: Pre/post-synaptic eligibility traces
        - delays: Axonal delay buffer
        - homeostasis: Firing rate estimates
        - stp_state: Short-term plasticity (if enabled)
        - bcm_state: BCM sliding thresholds (if enabled)
        - learning_state: Dopamine level, LTP/LTD accumulators
        """
        state = {
            "weights": self.weights.data.clone(),
            "neuron_state": {
                "membrane": self.membrane.clone(),
                "synaptic_current": self.synaptic_current.clone(),
                "refractory": self.refractory.clone(),
            },
            "stdp_traces": {
                "pre_trace": self.pre_trace.clone(),
                "post_trace": self.post_trace.clone(),
            },
            "delays": {
                "buffer": self.delay_buffer.clone(),
                "buffer_idx": self.delay_buffer_idx,
            },
            "homeostasis": {
                "firing_rate_estimate": self.firing_rate_estimate.clone(),
            },
            "learning_state": {
                "dopamine_level": self.dopamine_level,
                "replay_active": self.replay_active,
                "total_ltp": self.total_ltp,
                "total_ltd": self.total_ltd,
                "oscillation_phase": self.oscillation_phase,
            },
        }

        # Add STP state if enabled
        if self.stp is not None:
            state["stp_state"] = self.stp.get_state()

        # Add BCM state if enabled
        if self.bcm is not None:
            state["bcm_state"] = {
                "theta": self.bcm.theta.clone() if self.bcm.theta is not None else None,
            }

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load pathway state from checkpoint.

        Args:
            state: State dictionary from get_state()

        Note:
            Restores weights, neuron state, traces, and learning components.
            Device is inferred from current weights.
        """
        device = self.weights.device

        # Restore weights
        self.weights.data.copy_(state["weights"].to(device))

        # Restore neuron state
        neuron_state = state["neuron_state"]
        self.membrane.copy_(neuron_state["membrane"].to(device))
        self.synaptic_current.copy_(neuron_state["synaptic_current"].to(device))
        self.refractory.copy_(neuron_state["refractory"].to(device))

        # Restore STDP traces
        stdp_traces = state["stdp_traces"]
        self.pre_trace.copy_(stdp_traces["pre_trace"].to(device))
        self.post_trace.copy_(stdp_traces["post_trace"].to(device))

        # Restore delay buffer
        delays = state["delays"]
        self.delay_buffer.copy_(delays["buffer"].to(device))
        self.delay_buffer_idx = delays["buffer_idx"]

        # Restore homeostasis
        homeostasis = state["homeostasis"]
        self.firing_rate_estimate.copy_(homeostasis["firing_rate_estimate"].to(device))

        # Restore learning state
        learning_state = state["learning_state"]
        self.dopamine_level = learning_state["dopamine_level"]
        self.replay_active = learning_state["replay_active"]
        self.total_ltp = learning_state["total_ltp"]
        self.total_ltd = learning_state["total_ltd"]
        self.oscillation_phase = learning_state["oscillation_phase"]

        # Restore STP state if present
        if "stp_state" in state and self.stp is not None:
            self.stp.load_state(state["stp_state"])

        # Restore BCM state if present
        if "bcm_state" in state and self.bcm is not None:
            bcm_state = state["bcm_state"]
            if bcm_state["theta"] is not None:
                self.bcm.theta = bcm_state["theta"].to(device)
