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

FILE ORGANIZATION (941 lines)
==============================
Lines 1-115:   Module docstring, imports, config classes
Lines 116-250: SpikingPathway class __init__, neuron initialization
Lines 251-400: Forward pass (spike propagation with delays)
Lines 401-530: Learning (STDP, trace management)
Lines 531-650: Growth (grow_output/grow_input for pathway expansion)
Lines 651-800: Diagnostics and health monitoring
Lines 801-941: Utility methods (reset_state, checkpoint support)

NAVIGATION TIP: Use VSCode's "Go to Symbol" (Ctrl+Shift+O) to jump between methods.
"""

from enum import Enum, auto
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np

from thalia.core.base.component_config import PathwayConfig
from thalia.core.errors import CheckpointError
from thalia.utils.core_utils import clamp_weights
from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.components.synapses.stp import ShortTermPlasticity, STPConfig
from thalia.components.synapses.weight_init import WeightInitializer
from thalia.learning.eligibility.trace_manager import EligibilityTraceManager, STDPConfig
from thalia.managers.component_registry import register_pathway
from thalia.regions.base import NeuralComponent
from thalia.learning.rules.bcm import BCMRule, BCMConfig
from thalia.learning.rules.strategies import STDPConfig as StrategySTDPConfig
from thalia.learning.strategy_registry import create_hippocampus_strategy
from thalia.learning.homeostasis.synaptic_homeostasis import UnifiedHomeostasis, UnifiedHomeostasisConfig
from thalia.core.diagnostics_keys import DiagnosticKeys as DK


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


@register_pathway(
    "spiking",
    description="Fully spiking inter-region pathway with LIF neurons, STDP learning, and temporal coding",
    version="2.0",
    author="Thalia Project",
    config_class=PathwayConfig,
)
class SpikingPathway(NeuralComponent):
    """Fully spiking inter-region pathway with temporal dynamics and plasticity.

    Inherits from NeuralComponent, implementing the BrainComponent protocol
    with standardized API for inter-region connections.

    **Key Features**:
    - Leaky integrate-and-fire (LIF) neuron dynamics
    - Synaptic currents with temporal filtering
    - Axonal conduction delays (realistic transmission)
    - **Continuous STDP learning** (automatic during every forward pass)
    - Support for temporal coding schemes (rate, latency, phase, synchrony)

    **Biological Reality**:
    Pathways are NOT passive wires - they are active neural populations that:
    - Filter and transform information
    - Learn continuously via spike-timing dependent plasticity
    - Introduce realistic delays (2-10ms for long-range projections)
    - Maintain temporal structure critical for synchronization

    **Protocol Compliance** (BrainComponent):
    - forward(): Process spikes AND apply learning (standard PyTorch, ADR-007)
    - reset_state(): Clear membrane potentials, traces, delay buffers
    - get_diagnostics(): Report activity, health, and learning metrics
    - grow_output(): Grow pathway when target region grows
    - grow_input(): Grow pathway when source region grows
    - check_health(): Detect pathologies (silence, saturation, etc.)

    **Learning Design**:
    Like brain regions, this pathway ALWAYS learns during forward passes.
    There is no separate learn() method - plasticity is continuous and
    automatic, modulated by neuromodulators (dopamine, etc.).

    **Usage Example**:

    .. code-block:: python

        # Create pathway between cortex and hippocampus
        pathway = SpikingPathway(PathwayConfig(
            n_input=256,   # Cortex output size
            n_output=128,  # Hippocampus input size
            axonal_delay_ms=5.0,  # Long-range delay
            stdp_lr=0.001,
        ))

        # Process spikes (learning happens automatically)
        for t in range(n_timesteps):
            hippo_input = pathway(cortex_output[t], dt=1.0)
            # STDP applied automatically based on spike timing!

        # Monitor pathway health
        health = pathway.check_health()
        if not health.is_healthy:
            print(f\"Pathway issues: {health.issues}\")

        # Get learning metrics
        diag = pathway.get_diagnostics()
        print(f\"Weight change: {diag['learning']['weight_change']}\")
    """

    def __init__(self, config: PathwayConfig):
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
            n_output=config.n_output,
            n_input=config.n_input,
            mean=config.axonal_delay_ms,
            std=config.delay_variability * config.axonal_delay_ms,
            device=config.device
        ).clamp(min=0.1)
        self.register_buffer("axonal_delays", delays)

        # Connectivity mask
        if config.sparsity < 1.0:
            mask = WeightInitializer.sparse_random(
                n_output=config.n_output,
                n_input=config.n_input,
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
            torch.zeros(config.n_output, device=config.device),
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
            n_input=config.n_input,
            n_output=config.n_output,
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
            torch.zeros(max_delay_steps, config.n_input, device=config.device),
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
            torch.zeros(config.n_output, device=config.device) + config.activity_target,
        )

        # =====================================================================
        # SHORT-TERM PLASTICITY (STP)
        # =====================================================================
        if config.stp_enabled:
            stp_cfg = config.stp_config
            if stp_cfg is None:
                # Use preset based on stp_type (convert string to STPType if needed)
                from thalia.components.synapses.stp import STPType
                stp_type = config.stp_type
                if isinstance(stp_type, str):
                    stp_type = STPType[stp_type]
                stp_cfg = STPConfig.from_type(stp_type, dt=1.0)
            self.stp = ShortTermPlasticity(
                n_pre=config.n_input,
                n_post=config.n_output,
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
                n_post=config.n_output,
                config=bcm_cfg,
            )
            self.bcm.to(config.device)
        else:
            self.bcm = None

        # =====================================================================
        # LEARNING STRATEGY (Strategy Pattern)
        # =====================================================================
        # Use hippocampus-style STDP strategy via factory helper
        # (Pathways use similar spike-timing rules as hippocampal connections)
        stdp_cfg = StrategySTDPConfig(
            learning_rate=config.stdp_lr,
            a_plus=config.a_plus,
            a_minus=config.a_minus,
            tau_plus=config.tau_plus_ms,
            tau_minus=config.tau_minus_ms,
            dt_ms=config.dt_ms,
            w_min=config.w_min,
            w_max=config.w_max,
        )
        self.learning_strategy = create_hippocampus_strategy(stdp_config=stdp_cfg)

        # =====================================================================
        # HOMEOSTASIS (UnifiedHomeostasis)
        # =====================================================================
        homeostasis_config = UnifiedHomeostasisConfig(
            weight_budget=config.weight_budget * config.n_input,  # Total budget per neuron
            w_min=config.w_min,
            w_max=config.w_max,
            soft_normalization=config.soft_normalization,
            normalization_rate=config.normalization_rate,
            device=config.device,
        )
        self.homeostasis = UnifiedHomeostasis(homeostasis_config)

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
        neurons = ConductanceLIF(n_neurons=cfg.n_output, config=neuron_config)
        return neurons

    def _initialize_weights(self) -> torch.Tensor:
        """Initialize weights with optional topographic structure."""
        cfg = self.config

        if cfg.topographic:
            # Use topographic initialization from registry
            weights = WeightInitializer.topographic(
                n_output=cfg.n_output,
                n_input=cfg.n_input,
                base_weight=cfg.init_mean,
                sigma_factor=4.0,
                boost_strength=0.3,
                device=cfg.device
            )
        else:
            # Use Gaussian initialization from registry
            weights = WeightInitializer.gaussian(
                n_output=cfg.n_output,
                n_input=cfg.n_input,
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
            input_spikes: Binary spike tensor from source (n_input,)
            time_ms: Current time in ms (for phase coding)
            **kwargs: Additional arguments (ignored)

        Returns:
            Binary spike tensor for target (n_output,)

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
        assert source_spikes.shape[-1] == cfg.n_input, (
            f"SpikingPathway.forward: source_spikes has shape {source_spikes.shape} "
            f"but n_input={cfg.n_input}. Check pathway input dimensions."
        )

        # =====================================================================
        # 1. UPDATE OSCILLATION PHASE (for phase coding)
        # =====================================================================
        # Convert string temporal_coding to enum if needed
        temporal_coding = cfg.temporal_coding
        if isinstance(temporal_coding, str):
            temporal_coding = TemporalCoding[temporal_coding]

        if temporal_coding == TemporalCoding.PHASE:
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
        if temporal_coding == TemporalCoding.PHASE:
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
        # 8. APPLY STDP LEARNING (via Strategy Pattern)
        # =====================================================================
        # Convert string learning_rule to enum if needed
        learning_rule = cfg.learning_rule
        if isinstance(learning_rule, str):
            learning_rule = SpikingLearningRule[learning_rule]

        if learning_rule != SpikingLearningRule.REPLAY_STDP or self.replay_active:
            # Use strategy pattern for learning (inherited from LearningStrategyMixin)
            # Apply neuromodulator and BCM modulation via kwargs

            # Compute BCM modulation if enabled (per-neuron)
            bcm_modulation = 1.0
            if self.bcm is not None:
                # Per-neuron activity (n_post,) for per-neuron BCM thresholds
                post_activity = target_spikes.float()
                # BCM returns per-neuron modulation factors (n_post,)
                bcm_modulation = self.bcm.compute_phi(post_activity)
                self.bcm.update_threshold(post_activity)

            # Apply learning with modulations
            _ = self.apply_strategy_learning(
                pre_activity=source_spikes,
                post_activity=target_spikes,
                weights=self.weights,
                dopamine=self.dopamine_level,
                acetylcholine=self.acetylcholine_level,
                norepinephrine=self.norepinephrine_level,
                bcm_modulation=bcm_modulation,
                oscillation_phase=self.oscillation_phase,
                learning_rule=cfg.learning_rule,
            )

        # =====================================================================
        # 9. UPDATE FIRING RATE ESTIMATE (for homeostasis)
        # =====================================================================
        rate_decay = np.exp(-dt / cfg.activity_tau_ms)
        self.firing_rate_estimate = (
            self.firing_rate_estimate * rate_decay +
            target_spikes * (1 - rate_decay)
        )

        # Apply synaptic scaling
        if cfg.homeostasis_enabled:
            self._apply_synaptic_scaling()

        return target_spikes

    def _apply_synaptic_scaling(self) -> None:
        """Apply homeostatic synaptic scaling.

        Delegates to UnifiedHomeostasis for constraint-based weight normalization.
        """
        if self.config.homeostasis_enabled:
            self.weights.data = self.homeostasis.normalize_weights(
                self.weights.data,
                dim=1,  # Normalize rows (per-neuron)
            )

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
            DK.WEIGHT_MEAN: self.weights.data.mean().item(),
            DK.WEIGHT_STD: self.weights.data.std().item(),
            DK.FIRING_RATE: self.firing_rate_estimate.mean().item(),
        }

    def reset_learning_metrics(self) -> None:
        """Reset accumulated learning metrics."""
        self.total_ltp = 0.0
        self.total_ltd = 0.0

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics using DiagnosticsMixin helpers.

        Returns pathway state including weight statistics, membrane dynamics,
        eligibility traces, oscillation phase, and learning metrics.
        """
        diagnostics = {}

        # Weight statistics using mixin helper
        diagnostics.update(self.weight_diagnostics(self.weights.data, prefix=""))

        # Membrane dynamics - safe access
        if self.neurons.membrane is not None:
            diagnostics["membrane_mean"] = self.neurons.membrane.mean().item()
            diagnostics["membrane_std"] = self.neurons.membrane.std().item()
        else:
            diagnostics["membrane_mean"] = 0.0
            diagnostics["membrane_std"] = 0.0

        if self.firing_rate_estimate is not None:
            diagnostics["firing_rate_mean"] = self.firing_rate_estimate.mean().item()
        else:
            diagnostics["firing_rate_mean"] = 0.0

        # Eligibility traces using mixin helper
        diagnostics.update(self.trace_diagnostics(self.pre_trace, prefix="pre"))
        diagnostics.update(self.trace_diagnostics(self.post_trace, prefix="post"))

        # Oscillation and neuromodulation state
        diagnostics["oscillation_phase"] = self.oscillation_phase
        diagnostics["dopamine_level"] = self.dopamine_level
        diagnostics["replay_active"] = self.replay_active

        # Learning metrics
        diagnostics["total_ltp"] = self.total_ltp
        diagnostics["total_ltd"] = self.total_ltd

        return diagnostics

    def check_health(self) -> Any:  # Returns HealthReport
        """Check for pathological states in the pathway.

        Detects:
        - Silence: Firing rate too low
        - Runaway activity: Firing rate too high
        - Weight saturation
        - NaN/Inf values

        Returns:
            HealthReport with is_healthy flag and list of issues
        """
        from thalia.diagnostics.health_monitor import HealthReport

        issues = []

        # Check for NaN/Inf in weights
        if torch.isnan(self.weights).any() or torch.isinf(self.weights).any():
            issues.append("NaN or Inf values in weights")

        # Check firing rate if available
        if self.firing_rate_estimate is not None:
            mean_rate = self.firing_rate_estimate.mean().item()
            if mean_rate < 0.01:  # Less than 1% firing
                issues.append(f"Very low firing rate: {mean_rate:.4f}")
            elif mean_rate > 0.9:  # More than 90% firing
                issues.append(f"Runaway activity: {mean_rate:.4f}")

        # Check weight saturation
        if hasattr(self.config, 'weight_clipping') and self.config.weight_clipping is not None:
            min_clip, max_clip = self.config.weight_clipping
            saturated = (self.weights <= min_clip + 0.01).float().mean()
            if saturated > 0.5:
                issues.append(f"Weight saturation: {saturated:.1%} at minimum")

        return HealthReport(
            is_healthy=(len(issues) == 0),
            issues=issues,
            warnings=[],  # Could add warnings for borderline cases
        )

    def get_capacity_metrics(self) -> Dict[str, float]:
        """Get capacity and resource utilization metrics.

        Returns:
            Dictionary with:
            - n_neurons: Number of neurons in pathway
            - n_connections: Total number of synaptic connections
            - effective_connections: Non-zero connections (after sparsification)
            - sparsity: Fraction of zero-weight connections
            - utilization: Fraction of active neurons (recent firing)
        """
        n_neurons = self.config.n_output
        n_connections = self.weights.numel()
        
        # Count effective (non-zero) connections
        effective = (self.weights.abs() > 1e-6).sum().item()
        sparsity = 1.0 - (effective / n_connections)

        # Estimate utilization from firing rate
        utilization = 0.0
        if self.firing_rate_estimate is not None:
            utilization = (self.firing_rate_estimate > 0.01).float().mean().item()

        return {
            "n_neurons": float(n_neurons),
            "n_connections": float(n_connections),
            "effective_connections": float(effective),
            "sparsity": sparsity,
            "utilization": utilization,
        }

    # =========================================================================
    # UNIFIED GROWTH API
    # =========================================================================

    def grow_input(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Grow pathway input dimension (pre-synaptic side).

        When the source region grows, pathways receiving spikes FROM that region
        must also grow their input dimension to maintain connectivity.

        Args:
            n_new: Number of input neurons to add
            initialization: Weight init strategy ('sparse_random', 'xavier', 'uniform')
            sparsity: Connection sparsity for new neurons (if sparse_random)

        Example:
            >>> # Source region grows
            >>> cortex.grow_output(20)  # Add 20 neurons
            >>> # Pathway must adapt to new source size
            >>> cortex_to_hippocampus.grow_input(20)

        Implementation:
            1. Expand weight matrix: [target, source] → [target, source+n_new]
            2. Expand all source-side state tensors (pre_traces, delays)
            3. Expand connectivity mask
            4. Update config

        Note:
            Unlike grow_output(), this does NOT expand neurons (they belong to target).
            Only synaptic weights and source-side traces are expanded.
        """
        cfg = self.config
        device = self.weights.device
        old_source_size = cfg.n_input
        new_source_size = old_source_size + n_new

        # 1. Expand weight matrix [target, source] → [target, source+n_new]
        old_weights = self.weights.data.clone()

        # Initialize new weights for added source neurons
        if initialization == 'xavier':
            new_weights = WeightInitializer.xavier(
                n_output=cfg.n_output,
                n_input=n_new,
                device=device
            )
        elif initialization == 'uniform':
            new_weights = WeightInitializer.uniform(
                n_output=cfg.n_output,
                n_input=n_new,
                low=cfg.w_min,
                high=cfg.w_max,
                device=device
            )
        else:  # sparse_random (default)
            new_weights = WeightInitializer.sparse_random(
                n_output=cfg.n_output,
                n_input=n_new,
                sparsity=sparsity,
                mean=cfg.init_mean,
                std=cfg.init_std,
                device=device
            )

        # Clamp to weight bounds
        new_weights = clamp_weights(new_weights, cfg.w_min, cfg.w_max, inplace=False)

        # Concatenate old and new weights along input dimension
        expanded_weights = torch.cat([old_weights, new_weights], dim=1)
        self.weights = nn.Parameter(expanded_weights, requires_grad=False)

        # 2. Expand axonal delays [target, source] → [target, source+n_new]
        new_delays = WeightInitializer.gaussian(
            n_output=cfg.n_output,
            n_input=n_new,
            mean=cfg.axonal_delay_ms,
            std=cfg.delay_variability * cfg.axonal_delay_ms,
            device=device
        ).clamp(min=0.1)
        expanded_delays = torch.cat([self.axonal_delays, new_delays], dim=1)
        self.register_buffer("axonal_delays", expanded_delays)

        # 3. Expand connectivity mask if present
        if self.connectivity_mask is not None:
            new_mask = WeightInitializer.sparse_random(
                n_output=cfg.n_output,
                n_input=n_new,
                sparsity=cfg.sparsity,
                device=device
            )
            expanded_mask = torch.cat([self.connectivity_mask, new_mask], dim=1)
            self.register_buffer("connectivity_mask", expanded_mask)

        # 4. Expand pre-synaptic traces (source-side state)
        # Pre-traces track source neuron activity for STDP
        # Note: pre_trace is accessed via _trace_manager, so we need to grow the manager
        # The trace manager doesn't have a public grow method, so we recreate it

        # Create new trace manager with expanded source dimension
        trace_config = STDPConfig(
            stdp_tau_ms=cfg.tau_plus_ms,  # Use config tau values
            eligibility_tau_ms=cfg.eligibility_tau_ms,
            stdp_lr=cfg.stdp_lr,
            a_plus=cfg.a_plus,
            a_minus=cfg.a_minus,
            w_min=cfg.w_min,
            w_max=cfg.w_max,
            heterosynaptic_ratio=cfg.heterosynaptic_ratio,
        )
        old_input_trace = self._trace_manager.input_trace.clone()
        old_output_trace = self._trace_manager.output_trace.clone()

        self._trace_manager = EligibilityTraceManager(
            n_input=new_source_size,
            n_output=cfg.n_output,
            config=trace_config,
            device=device,
        )

        # Restore old traces + initialize new source neurons
        self._trace_manager.input_trace[:old_source_size] = old_input_trace
        self._trace_manager.output_trace[:] = old_output_trace

        # Update learning strategy's trace manager if using strategy-based learning
        if hasattr(self, 'learning_strategy') and self.learning_strategy is not None:
            if hasattr(self.learning_strategy, '_trace_manager'):
                self.learning_strategy._trace_manager = self._trace_manager

        # 5. Expand delay buffer: [max_delay_steps, old_source] → [max_delay_steps, new_source]
        max_delay_steps = self.delay_buffer.shape[0]
        old_delay_buffer = self.delay_buffer.clone()
        new_delay_buffer = torch.zeros(max_delay_steps, new_source_size, device=device)
        new_delay_buffer[:, :old_source_size] = old_delay_buffer
        self._buffers["delay_buffer"] = new_delay_buffer

        # 6. Update STP if enabled
        if self.stp is not None:
            # STP must track new pre-synaptic neurons
            self.stp = ShortTermPlasticity(
                n_pre=new_source_size,
                n_post=cfg.n_output,
                config=self.stp.config,
                per_synapse=True,
            )
            self.stp.to(device)

        # 7. Update config
        self.config.n_input = new_source_size

    def grow_output(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Grow pathway output dimension (post-synaptic side).

        When the target region grows, pathways sending spikes TO that region
        must also grow their output dimension to maintain connectivity.

        Args:
            n_new: Number of output neurons to add
            initialization: Weight init strategy ('sparse_random', 'xavier', 'uniform')
            sparsity: Connection sparsity for new neurons (if sparse_random)

        Example:
            >>> # Target region grows
            >>> hippocampus.grow_output(15)  # Add 15 neurons
            >>> # Pathway must adapt to new target size
            >>> cortex_to_hippocampus.grow_output(15)

        Implementation:
            1. Expand weight matrix: [target, source] → [target+n_new, source]
            2. Expand all target-side state tensors (membrane, traces, etc.)
            3. Expand axonal delays and connectivity mask
            4. Update config
        """
        cfg = self.config
        device = self.weights.device
        old_target_size = cfg.n_output
        new_target_size = old_target_size + n_new

        # 1. Expand weight matrix [target, source] → [target+n_new, source]
        old_weights = self.weights.data.clone()

        # Initialize new weights for added neurons
        if initialization == 'xavier':
            new_weights = WeightInitializer.xavier(
                n_output=n_new,
                n_input=cfg.n_input,
                device=device
            )
        elif initialization == 'uniform':
            new_weights = WeightInitializer.uniform(
                n_output=n_new,
                n_input=cfg.n_input,
                low=cfg.w_min,
                high=cfg.w_max,
                device=device
            )
        else:  # sparse_random (default)
            new_weights = WeightInitializer.sparse_random(
                n_output=n_new,
                n_input=cfg.n_input,
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
            n_input=cfg.n_input,
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
                n_input=cfg.n_input,
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
        # grow_dimension returns new manager, must reassign
        # pre_trace and post_trace properties will automatically use new manager
        self._trace_manager = self._trace_manager.grow_dimension(n_new)

        # Update learning strategy's trace manager if using strategy-based learning
        if hasattr(self, 'learning_strategy') and self.learning_strategy is not None:
            if hasattr(self.learning_strategy, '_trace_manager'):
                self.learning_strategy._trace_manager = self._trace_manager

        # Firing rate estimate
        new_firing_rate = torch.zeros(n_new, device=device) + cfg.activity_target
        expanded_firing_rate = torch.cat([self.firing_rate_estimate, new_firing_rate], dim=0)
        self.register_buffer("firing_rate_estimate", expanded_firing_rate)

        # 5. Update STP if enabled
        if self.stp is not None:
            # STP tracks post-synaptic resources
            self.stp = ShortTermPlasticity(
                n_pre=cfg.n_input,
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
        self.config.n_output = new_target_size
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
            - 'config': PathwayConfig - Configuration
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
            raise CheckpointError(
                f"State class mismatch: expected {self.__class__.__name__}, "
                f"got {state.get('class_name')}"
            )

        # Use existing load_state() for pathway state
        self.load_state(state['pathway_state'])
