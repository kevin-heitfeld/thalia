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

from thalia.config.base import NeuralComponentConfig
from thalia.core.utils import clamp_weights
from thalia.core.neuron_constants import (
    TAU_MEM_STANDARD,
    TAU_SYN_EXCITATORY,
    V_THRESHOLD_STANDARD,
    V_RESET_STANDARD,
    V_REST_STANDARD,
    TAU_REF_STANDARD,
)
from thalia.core.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.regions.base import NeuralComponent, LearningRule
from thalia.core.stp import ShortTermPlasticity, STPConfig, STPType
from thalia.core.weight_init import WeightInitializer
from thalia.core.eligibility_utils import EligibilityTraceManager, STDPConfig
from thalia.learning.bcm import BCMRule, BCMConfig
from thalia.core.component_registry import register_pathway


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
class SpikingPathwayConfig(NeuralComponentConfig):
    """Configuration for a spiking pathway.

    Inherits n_neurons, dt_ms, device, dtype, seed from NeuralComponentConfig.
    The dt_ms should be set from GlobalConfig.dt_ms by Brain.

    Note: For pathways, you can set either n_neurons OR target_size (they're synonyms).
    If both are provided, target_size takes precedence. If neither is set, defaults to 100.
    The n_neurons field represents the pathway's intermediate neuron population size.

    Attributes:
        source_size: Number of neurons in source region
        target_size: Synonym for n_neurons (pathway's intermediate population size)

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

    Note: Set source_size and either n_neurons or target_size when creating config.
    They will be automatically synchronized.
    """
    # Required field - set this when creating config!
    source_size: int = 0  # Must be set explicitly
    target_size: int = 100  # Synonym for n_neurons (pathway neuron population)

    def __post_init__(self):
        """Synchronize n_neurons and target_size (they're synonyms for pathways)."""
        # If target_size was explicitly set (not default), use it for n_neurons
        if self.target_size != 100:
            self.n_neurons = self.target_size
        # Otherwise, use n_neurons value (from parent or explicitly set)
        else:
            self.target_size = self.n_neurons

        # Synchronize learning_rate with stdp_lr for compatibility with NeuralComponent
        # If stdp_lr was explicitly set, it takes precedence
        if not hasattr(self, 'learning_rate'):
            self.learning_rate = self.stdp_lr

    # Neuron model
    tau_mem_ms: float = TAU_MEM_STANDARD
    tau_syn_ms: float = TAU_SYN_EXCITATORY
    v_thresh: float = V_THRESHOLD_STANDARD
    v_reset: float = V_RESET_STANDARD
    v_rest: float = V_REST_STANDARD
    refractory_ms: float = TAU_REF_STANDARD

    # STDP
    learning_rule: SpikingLearningRule = SpikingLearningRule.STDP
    stdp_lr: float = 0.01
    tau_plus_ms: float = 20.0
    tau_minus_ms: float = 20.0
    a_plus: float = 1.0
    a_minus: float = 1.0
    max_trace: float = 10.0  # Maximum trace value to prevent runaway accumulation

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


@register_pathway(
    "spiking",
    description="Fully spiking inter-region pathway with LIF neurons, STDP learning, and temporal coding",
    version="2.0",
    author="Thalia Project"
)
class SpikingPathway(NeuralComponent):
    """
    Fully spiking inter-region pathway with temporal dynamics.

    Inherits from NeuralComponent, implementing the NeuralPathway protocol
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
        super().__init__(config)
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
        # NEURON STATE (handled by ConductanceLIF)
        # =====================================================================
        self.neurons = self._create_neurons()

        # Synaptic current (filtered input - still needed for temporal filtering)
        self.register_buffer(
            "synaptic_current",
            torch.zeros(config.target_size, device=config.device),
        )

        # =====================================================================
        # ELIGIBILITY TRACE MANAGER for STDP
        # =====================================================================
        # Consolidated trace management using EligibilityTraceManager
        stdp_config = STDPConfig(
            stdp_tau_ms=config.tau_plus_ms,  # Use tau_plus for trace decay
            eligibility_tau_ms=1000.0,        # Long eligibility trace (not used in pathways)
            stdp_lr=config.stdp_lr,
            a_plus=config.a_plus,
            a_minus=config.a_minus,
            w_min=config.w_min,
            w_max=config.w_max,
            heterosynaptic_ratio=getattr(config, 'heterosynaptic_ratio', 0.3),
        )
        self._trace_manager = EligibilityTraceManager(
            n_input=config.source_size,
            n_output=config.target_size,
            config=stdp_config,
            device=config.device,
        )

        # =====================================================================
        # BACKWARD COMPATIBILITY PROPERTIES
        # =====================================================================
        # Provide access to traces for external code

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
        self.norepinephrine_level = 0.0
        self.acetylcholine_level = 0.0
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

    @property
    def pre_trace(self) -> torch.Tensor:
        """Pre-synaptic trace (backward compatibility)."""
        return self._trace_manager.input_trace

    @property
    def post_trace(self) -> torch.Tensor:
        """Post-synaptic trace (backward compatibility)."""
        return self._trace_manager.output_trace

    def _create_neurons(self) -> ConductanceLIF:
        """Create neuron model for pathway.

        Returns ConductanceLIF with parameters matching pathway config.
        Pathways use the same neuron models as regions for consistency.
        """
        cfg = self.config
        neuron_config = ConductanceLIFConfig(
            tau_mem=cfg.tau_mem_ms,
            v_rest=cfg.v_rest,
            v_reset=cfg.v_reset,
            v_threshold=cfg.v_thresh,
            tau_ref=cfg.refractory_ms,
            dt_ms=cfg.dt_ms,
            device=cfg.device,
        )
        neurons = ConductanceLIF(n_neurons=cfg.target_size, config=neuron_config)
        return neurons

    def _get_learning_rule(self) -> LearningRule:
        """Return the primary learning rule for this pathway.

        SpikingPathway uses STDP (spike-timing dependent plasticity)
        as its primary learning mechanism.
        """
        return LearningRule.STDP

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

        return clamp_weights(weights, cfg.w_min, cfg.w_max, inplace=False)

    def forward(
        self,
        input_spikes: torch.Tensor,
        time_ms: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """Process source spikes and generate target spikes.

        Args:
            input_spikes: Binary spike tensor from source (source_size,)
            time_ms: Current time in ms (for phase coding)
            **kwargs: Additional arguments (ignored)

        Returns:
            Binary spike tensor for target (target_size,)

        Note:
            Timestep (dt_ms) is obtained from self.config.dt_ms (set from GlobalConfig)
        """
        # Get timestep from config
        dt = self.config.dt_ms
        cfg = self.config

        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        # Ensure 1D
        source_spikes = input_spikes.squeeze()
        assert source_spikes.shape[-1] == cfg.source_size, (
            f"SpikingPathway.forward: source_spikes has shape {source_spikes.shape} "
            f"but source_size={cfg.source_size}. Check pathway input dimensions."
        )

        # =====================================================================
        # 1. UPDATE OSCILLATION PHASE (for phase coding)
        # =====================================================================
        if cfg.temporal_coding == TemporalCoding.PHASE:
            # Use brain-wide gamma oscillator (no fallback - must be provided by Brain)
            if hasattr(self, '_oscillator_phases') and 'gamma' in self._oscillator_phases:
                self.oscillation_phase = self._oscillator_phases['gamma']
            else:
                # No fallback - pathways must receive brain oscillators
                self.oscillation_phase = 0.0

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
            # efficacy shape: [n_pre] for per-neuron or [n_pre, n_post] for per-synapse STP
            stp_efficacy = self.stp(delayed_spikes)
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
        # 5. GENERATE SPIKES using ConductanceLIF
        # =====================================================================
        # Apply phase modulation for phase coding by temporarily adjusting threshold
        if cfg.temporal_coding == TemporalCoding.PHASE:
            # Threshold is lower at preferred phase (phase=0)
            phase_modulation = 1.0 - cfg.phase_precision * np.cos(self.oscillation_phase)
            original_thresh = self.neurons.config.v_threshold
            self.neurons.config.v_threshold = original_thresh * phase_modulation
            target_spikes, _ = self.neurons(self.synaptic_current)
            self.neurons.config.v_threshold = original_thresh  # Restore
        else:
            target_spikes, _ = self.neurons(self.synaptic_current)

        # =====================================================================
        # 7. UPDATE STDP TRACES
        # =====================================================================
        # Use trace manager for consolidated trace updates
        self._trace_manager.update_traces(
            input_spikes=source_spikes,
            output_spikes=target_spikes,
            dt_ms=dt,
        )

        # =====================================================================
        # 8. APPLY STDP LEARNING
        # =====================================================================
        if cfg.learning_rule != SpikingLearningRule.REPLAY_STDP or self.replay_active:
            self._apply_stdp(source_spikes, target_spikes)

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
            self._apply_synaptic_scaling()

        return target_spikes

    def _apply_stdp(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
    ) -> None:
        """Apply STDP weight updates with optional BCM modulation.

        Note:
            Timestep (dt_ms) obtained from self.config
        """
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

        # Use trace manager to compute raw LTP/LTD (without combining or soft bounds)
        ltp, ltd = self._trace_manager.compute_ltp_ltd_separate(
            input_spikes=pre_spikes,
            output_spikes=post_spikes,
        )

        # Apply neuromodulator and phase modulations to LTP
        # Modulate by dopamine for dopamine-STDP
        if cfg.learning_rule == SpikingLearningRule.DOPAMINE_STDP:
            ltp = ltp * (1 + self.dopamine_level)

        # Modulate by acetylcholine (high ACh = favor LTP/encoding)
        # Biologically: ACh from nucleus basalis enhances encoding
        ach_modulation = 0.5 + 0.5 * self.acetylcholine_level  # Range: [0.5, 1.5]
        ltp = ltp * ach_modulation

        # Modulate by norepinephrine (inverted-U: moderate NE optimal)
        # Biologically: NE from locus coeruleus affects learning rate
        # Too low = weak learning, optimal (~0.5) = strong, too high = noisy
        ne_modulation = 1.0 - 0.5 * abs(self.norepinephrine_level - 0.5)  # Peak at 0.5
        ltp = ltp * ne_modulation

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

        # Apply neuromodulator and phase modulations to LTD
        if cfg.learning_rule == SpikingLearningRule.DOPAMINE_STDP:
            # LTD is reduced by dopamine (protects good synapses)
            ltd = ltd * (1 - 0.5 * max(0, self.dopamine_level))

        # Modulate by acetylcholine (high ACh = reduce LTD/favor encoding)
        # Biologically: ACh shifts LTP/LTD balance toward LTP
        ach_ltd_suppression = 1.0 - 0.3 * self.acetylcholine_level  # Range: [0.7, 1.0]
        ltd = ltd * ach_ltd_suppression

        # Norepinephrine also affects LTD with inverted-U
        ne_modulation = 1.0 - 0.5 * abs(self.norepinephrine_level - 0.5)
        ltd = ltd * ne_modulation

        if cfg.learning_rule == SpikingLearningRule.PHASE_STDP:
            # LTD is stronger at anti-phase
            phase_mod = 0.5 - 0.5 * np.cos(self.oscillation_phase)
            ltd = ltd * phase_mod

        # Track LTP/LTD for diagnostics (only if tensors, not scalar 0)
        if isinstance(ltp, torch.Tensor):
            self.total_ltp += ltp.sum().item()
        if isinstance(ltd, torch.Tensor):
            self.total_ltd += ltd.sum().item()

        # Compute weight change
        dw = cfg.stdp_lr * (ltp - ltd)        # Apply connectivity mask
        if self.connectivity_mask is not None and isinstance(dw, torch.Tensor):
            dw = dw * self.connectivity_mask

        # Update weights
        if isinstance(dw, torch.Tensor):
            self.weights.data += dw
            clamp_weights(self.weights.data, cfg.w_min, cfg.w_max)

    def _apply_synaptic_scaling(self) -> None:
        """Apply homeostatic synaptic scaling.

        Note:
            Timestep (dt_ms) obtained from self.config
        """
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
        self.neurons.reset_state()
        self.synaptic_current.zero_()
        self._trace_manager.reset_traces()
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

    def set_neuromodulators(self, dopamine: float, norepinephrine: float, acetylcholine: float) -> None:
        """Set all neuromodulator levels for modulated learning.

        This consolidated method sets all three neuromodulators at once,
        which is more efficient than three separate calls.

        Args:
            dopamine: DA level - modulates STDP learning rate, eligibility traces, reward-gating
            norepinephrine: NE level - modulates neural gain, learning rate (inverted-U)
            acetylcholine: ACh level - modulates LTP/LTD balance, encoding vs retrieval

        Biological effects:
        - Dopamine: Higher DA = stronger plasticity, reward-gated learning
        - Norepinephrine: Inverted-U (moderate NE optimal), affects gain and SNR
        - Acetylcholine: High ACh = favor LTP/encoding, low ACh = favor retrieval
        """
        self.dopamine_level = dopamine
        self.norepinephrine_level = norepinephrine
        self.acetylcholine_level = acetylcholine

    def set_replay_mode(self, active: bool) -> None:
        """Enable/disable replay mode for replay-gated STDP."""
        self.replay_active = active

    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Optional[Dict[str, float]] = None,
        theta_slot: int = 0,
        coupled_amplitudes: Optional[Dict[str, float]] = None,
    ) -> None:
        """Receive oscillator phases from brain broadcast.

        Pathways use brain-wide oscillators for synchronized processing:
        - **Phase coding**: Uses gamma oscillator for temporal coding
        - **Attention modulation**: Beta/alpha coupling affects gain
        - **Replay gating**: Ripple oscillator triggers memory consolidation
        - **State-dependent plasticity**: Different learning in different brain states

        All pathways are synchronized to the same brain-wide rhythms for
        coordinated inter-region communication and state-dependent processing.

        Args:
            phases: Oscillator phases in radians {'theta': ..., 'gamma': ..., 'ripple': ..., etc}
            signals: Oscillator signal values {'theta': ..., 'gamma': ..., etc}
            theta_slot: Current theta slot [0, n_slots-1] for sequence encoding
            coupled_amplitudes: Effective amplitudes per oscillator (pre-computed by coupling)

        Example:
            # Pathways automatically use brain oscillators:
            gamma_phase = self._oscillator_phases['gamma']  # For phase coding
            beta_amp = self._coupled_amplitudes['beta']     # For attention gain
            ripple_phase = self._oscillator_phases['ripple']  # For replay

        Note:
            Called automatically by Brain before each forward() call.
            Pathways MUST receive oscillators from Brain - no local fallbacks.
        """
        # Store for use by all pathways (required for synchronized processing)
        self._oscillator_phases = phases
        self._oscillator_signals = signals or {}
        self._oscillator_theta_slot = theta_slot
        self._coupled_amplitudes = coupled_amplitudes or {}

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
            "membrane_mean": self.neurons.membrane.mean().item(),
            "membrane_std": self.neurons.membrane.std().item(),
            "firing_rate_mean": self.firing_rate_estimate.mean().item(),
            "pre_trace_norm": self.pre_trace.norm().item(),
            "post_trace_norm": self.post_trace.norm().item(),
            "oscillation_phase": self.oscillation_phase,
            "dopamine_level": self.dopamine_level,
            "replay_active": self.replay_active,
            "total_ltp": self.total_ltp,
            "total_ltd": self.total_ltd,
        }

    def add_neurons(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Add neurons to pathway (expands target dimension).

        Pathways expand the target side (output) when adding neurons.
        This maintains connectivity with source while increasing capacity.

        Args:
            n_new: Number of target neurons to add
            initialization: Weight init strategy ('sparse_random', 'xavier', 'uniform')
            sparsity: Connection sparsity for new neurons (if sparse_random)

        Implementation:
            1. Expand weight matrix: [target, source] → [target+n_new, source]
            2. Expand all target-side state tensors (membrane, traces, etc.)
            3. Expand axonal delays and connectivity mask
            4. Update config
        """
        cfg = self.config
        device = self.weights.device
        old_target_size = cfg.target_size
        new_target_size = old_target_size + n_new

        # 1. Expand weight matrix [target, source] → [target+n_new, source]
        old_weights = self.weights.data.clone()

        # Initialize new weights for added neurons
        if initialization == 'xavier':
            new_weights = WeightInitializer.xavier(
                n_output=n_new,
                n_input=cfg.source_size,
                device=device
            )
        elif initialization == 'uniform':
            new_weights = WeightInitializer.uniform(
                n_output=n_new,
                n_input=cfg.source_size,
                low=cfg.w_min,
                high=cfg.w_max,
                device=device
            )
        else:  # sparse_random (default)
            new_weights = WeightInitializer.sparse_random(
                n_output=n_new,
                n_input=cfg.source_size,
                sparsity=sparsity,
                mean=cfg.init_mean,
                std=cfg.init_std,
                device=device
            )

        # Clamp to weight bounds
        new_weights = clamp_weights(new_weights, cfg.w_min, cfg.w_max, inplace=False)

        # Concatenate old and new weights
        expanded_weights = torch.cat([old_weights, new_weights], dim=0)
        self.weights = nn.Parameter(expanded_weights, requires_grad=False)

        # 2. Expand axonal delays [target, source] → [target+n_new, source]
        new_delays = WeightInitializer.gaussian(
            n_output=n_new,
            n_input=cfg.source_size,
            mean=cfg.axonal_delay_ms,
            std=cfg.delay_variability * cfg.axonal_delay_ms,
            device=device
        ).clamp(min=0.1)
        expanded_delays = torch.cat([self.axonal_delays, new_delays], dim=0)
        self.register_buffer("axonal_delays", expanded_delays)

        # 3. Expand connectivity mask if present
        if self.connectivity_mask is not None:
            new_mask = WeightInitializer.sparse_random(
                n_output=n_new,
                n_input=cfg.source_size,
                sparsity=cfg.sparsity,
                device=device
            )
            expanded_mask = torch.cat([self.connectivity_mask, new_mask], dim=0)
            self.register_buffer("connectivity_mask", expanded_mask)

        # 4. Expand neuron object (preserve old state)
        old_neuron_state = self.neurons.get_state()

        neuron_config = ConductanceLIFConfig(
            tau_mem=cfg.tau_mem_ms,
            v_rest=cfg.v_rest,
            v_reset=cfg.v_reset,
            v_threshold=cfg.v_thresh,
            tau_ref=cfg.refractory_ms,
            dt_ms=cfg.dt_ms,
            device=device,
        )
        self.neurons = ConductanceLIF(n_neurons=new_target_size, config=neuron_config)

        # Restore old state + initialize new neurons to rest
        self.neurons.reset_state()  # Initialize all to rest
        if old_neuron_state.get("membrane") is not None:
            self.neurons.membrane[:old_target_size] = old_neuron_state["membrane"].to(device)
        if old_neuron_state.get("refractory") is not None:
            self.neurons.refractory[:old_target_size] = old_neuron_state["refractory"].to(device)
        if old_neuron_state.get("g_E") is not None:
            self.neurons.g_E[:old_target_size] = old_neuron_state["g_E"].to(device)
        if old_neuron_state.get("g_I") is not None:
            self.neurons.g_I[:old_target_size] = old_neuron_state["g_I"].to(device)
        if old_neuron_state.get("g_adapt") is not None:
            self.neurons.g_adapt[:old_target_size] = old_neuron_state["g_adapt"].to(device)

        # Synaptic current
        new_current = torch.zeros(n_new, device=device)
        expanded_current = torch.cat([self.synaptic_current, new_current], dim=0)
        self.register_buffer("synaptic_current", expanded_current)

        # Grow trace manager to match new target size
        self._trace_manager = self._trace_manager.add_neurons(n_new)

        # Firing rate estimate
        new_firing_rate = torch.zeros(n_new, device=device) + cfg.target_rate
        expanded_firing_rate = torch.cat([self.firing_rate_estimate, new_firing_rate], dim=0)
        self.register_buffer("firing_rate_estimate", expanded_firing_rate)

        # 5. Update STP if enabled
        if self.stp is not None:
            # STP tracks post-synaptic resources
            self.stp = ShortTermPlasticity(
                n_pre=cfg.source_size,
                n_post=new_target_size,
                config=self.stp.config,
                per_synapse=True,
            )
            self.stp.to(device)

        # 6. Update BCM if enabled
        if self.bcm is not None:
            # BCM tracks sliding threshold per target neuron
            old_bcm_config = self.bcm.config
            self.bcm = BCMRule(
                n_post=new_target_size,
                config=old_bcm_config,
            )
            self.bcm.to(device)

        # 7. Update config
        self.config.target_size = new_target_size
        self.config.n_neurons = new_target_size  # Keep n_neurons in sync

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
                "neurons": self.neurons.get_state(),
                "synaptic_current": self.synaptic_current.clone(),
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
        self.neurons.load_state(neuron_state["neurons"])
        self.synaptic_current.copy_(neuron_state["synaptic_current"].to(device))

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

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete pathway state for checkpointing (BrainComponent protocol).

        Returns:
            Dictionary with keys:
            - 'weights': Dict[str, torch.Tensor] - All learnable parameters
            - 'pathway_state': Dict[str, Any] - Dynamic state
            - 'config': SpikingPathwayConfig - Configuration
            - 'class_name': str - Class name for reconstruction
        """
        return {
            'weights': {
                'weights': self.weights.detach().clone(),
            },
            'pathway_state': self.get_state(),  # Use existing get_state()
            'config': self.config,
            'class_name': self.__class__.__name__,
        }

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Restore complete pathway state from checkpoint (BrainComponent protocol).

        Args:
            state: Dictionary from get_full_state()

        Raises:
            ValueError: If state is incompatible
        """
        # Verify class matches
        if state.get('class_name') != self.__class__.__name__:
            raise ValueError(
                f"State class mismatch: expected {self.__class__.__name__}, "
                f"got {state.get('class_name')}"
            )

        # Use existing load_state() for pathway state
        self.load_state(state['pathway_state'])
