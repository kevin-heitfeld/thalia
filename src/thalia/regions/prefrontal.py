"""
Prefrontal Cortex - Gated Working Memory and Executive Control

The prefrontal cortex (PFC) is specialized for:
- Working memory maintenance
- Rule learning and flexible switching
- Executive control and decision making
- Goal-directed behavior

Key Features:
=============
1. GATED WORKING MEMORY:
   - Information can be actively maintained against decay
   - Dopamine gates what enters/updates working memory
   - Persistent activity through recurrent connections

2. DOPAMINE GATING:
   - DA burst → "update gate open" → allow new info into WM
   - DA dip → "maintain" → protect current WM contents
   - Similar to LSTM/GRU gating mechanisms in deep learning

3. CONTEXT-DEPENDENT LEARNING:
   - Rule neurons learn to represent abstract rules
   - Same sensory input → different outputs based on context
   - Supports flexible behavior switching

4. SLOW INTEGRATION:
   - Longer time constants than sensory areas
   - Allows integration over longer timescales
   - Supports temporal abstraction

Biological Basis:
=================
- Layer 2/3 recurrent circuits for WM maintenance
- D1/D2 receptors modulate gain and gating
- Strong connections with striatum (for action selection)
- Connections with hippocampus (for episodic retrieval)

When to Use:
============
- Working memory tasks (maintain info over delays)
- Rule learning (learn context-dependent responses)
- Sequence generation (use rules to generate behavior)
- Any task requiring flexible, goal-directed control
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from thalia.core.utils import ensure_batch_dim, ensure_1d, clamp_weights, cosine_similarity_safe
from thalia.core.stp import ShortTermPlasticity, STPConfig, STPType
from thalia.regions.base import (
    BrainRegion,
    RegionConfig,
    RegionState,
    LearningRule,
)
from thalia.core.neuron import ConductanceLIF, ConductanceLIFConfig


@dataclass
class PrefrontalConfig(RegionConfig):
    """Configuration specific to prefrontal cortex.

    PFC implements DOPAMINE-GATED STDP:
    - STDP creates eligibility traces from spike timing
    - Dopamine gates what enters working memory and what gets learned
    - High DA → update WM and learn new associations
    - Low DA → maintain WM and protect existing patterns
    """

    # Working memory parameters
    wm_decay_tau_ms: float = 500.0  # How fast WM decays (slow!)
    wm_noise_std: float = 0.01  # Noise in WM maintenance

    # Gating parameters
    gate_threshold: float = 0.5  # DA level to open update gate
    gate_strength: float = 2.0  # How strongly gating affects updates

    # Dopamine parameters
    dopamine_tau_ms: float = 100.0  # DA decay time constant
    dopamine_baseline: float = 0.2  # Tonic DA level

    # Learning rates
    wm_lr: float = 0.1  # Learning rate for WM update weights
    rule_lr: float = 0.01  # Learning rate for rule weights
    stdp_lr: float = 0.02  # STDP learning rate for spike-based version

    # STDP parameters
    stdp_tau_ms: float = 20.0  # STDP trace decay
    heterosynaptic_ratio: float = 0.3  # LTD for non-active synapses

    # Recurrent connections for WM maintenance
    recurrent_strength: float = 0.8  # Self-excitation for persistence
    recurrent_inhibition: float = 0.2  # Lateral inhibition

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION (SFA)
    # =========================================================================
    # PFC pyramidal neurons show adaptation. This helps prevent runaway
    # activity during sustained working memory maintenance.
    pfc_adapt_increment: float = 0.2  # Adaptation per spike
    pfc_adapt_tau: float = 150.0      # Adaptation time constant (ms)

    # =========================================================================
    # SHORT-TERM PLASTICITY (STP) for recurrent connections
    # =========================================================================
    # PFC recurrent connections show SHORT-TERM DEPRESSION, preventing
    # frozen attractors. This allows working memory to be updated.
    stp_recurrent_enabled: bool = True

    # Weight constraints
    soft_bounds: bool = True

    # Synaptic scaling for homeostasis
    synaptic_scaling_enabled: bool = True
    synaptic_scaling_target: float = 0.4
    synaptic_scaling_rate: float = 0.001


@dataclass
class PrefrontalState(RegionState):
    """State for prefrontal cortex region."""
    # Working memory contents
    working_memory: Optional[torch.Tensor] = None

    # Gate state
    update_gate: Optional[torch.Tensor] = None

    # Rule representation
    active_rule: Optional[torch.Tensor] = None

    # Dopamine level (override base)
    dopamine: float = 0.2  # Start at baseline


class DopamineGatingSystem:
    """Dopamine-based gating for working memory updates.

    Unlike striatal dopamine (which determines LTP vs LTD direction),
    prefrontal dopamine gates what information enters working memory:
    - High DA → gate open → update WM with new input
    - Low DA → gate closed → maintain current WM
    """

    def __init__(
        self,
        n_neurons: int,
        tau_ms: float = 100.0,
        baseline: float = 0.2,
        threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.n_neurons = n_neurons
        self.tau_ms = tau_ms
        self.baseline = baseline
        self.threshold = threshold
        self.device = torch.device(device)

        self.level = baseline  # Current DA level

    def reset(self):
        """Reset to baseline."""
        self.level = self.baseline

    def update(self, signal: float, dt: float = 1.0) -> float:
        """Update dopamine level with new signal.

        Args:
            signal: External dopamine signal (-1 to 1)
            dt: Timestep in ms

        Returns:
            Current dopamine level
        """
        # Decay toward baseline
        decay = torch.exp(torch.tensor(-dt / self.tau_ms)).item()
        self.level = self.baseline + (self.level - self.baseline) * decay

        # Add signal
        self.level += signal

        # Clamp to valid range
        self.level = max(0.0, min(1.0, self.level))

        return self.level

    def get_gate(self) -> float:
        """Get current gating value (0-1).

        Returns smooth gate value based on dopamine level.
        """
        # Sigmoid around threshold
        return 1.0 / (1.0 + torch.exp(torch.tensor(
            -10 * (self.level - self.threshold)
        )).item())


class Prefrontal(BrainRegion):
    """
    Prefrontal Cortex implementation with gated working memory.

    Key features:
    - Working memory maintained through recurrent activity
    - Dopamine gates what enters working memory
    - Slow time constants for temporal integration
    - Context-dependent rule learning
    """

    def __init__(self, config: PrefrontalConfig):
        """
        Initialize prefrontal cortex.

        Args:
            config: PFC configuration
        """
        # Store config before calling super().__init__ which calls abstract methods
        self.pfc_config = config
        super().__init__(config)

        # Recurrent weights for WM maintenance
        self.rec_weights = nn.Parameter(
            torch.randn(config.n_output, config.n_output, device=self.device) * 0.1
        )
        # Initialize with some self-excitation
        with torch.no_grad():
            self.rec_weights.data += torch.eye(
                config.n_output, device=self.device
            ) * config.recurrent_strength

        # Lateral inhibition weights
        self.inhib_weights = nn.Parameter(
            torch.ones(config.n_output, config.n_output, device=self.device)
            * config.recurrent_inhibition
        )
        with torch.no_grad():
            self.inhib_weights.data.fill_diagonal_(0.0)

        # Dopamine gating system
        self.dopamine_system = DopamineGatingSystem(
            n_neurons=config.n_output,
            tau_ms=config.dopamine_tau_ms,
            baseline=config.dopamine_baseline,
            threshold=config.gate_threshold,
            device=config.device,
        )

        # STDP traces for spike-based learning
        self.input_trace = torch.zeros(config.n_input, device=self.device)
        self.output_trace = torch.zeros(config.n_output, device=self.device)

        # Eligibility trace for dopamine-gated learning
        self.stdp_eligibility = torch.zeros(
            config.n_output, config.n_input, device=self.device
        )

        # Initialize working memory state
        self.state = PrefrontalState(
            working_memory=torch.zeros(1, config.n_output, device=self.device),
            update_gate=torch.zeros(1, config.n_output, device=self.device),
            dopamine=config.dopamine_baseline,
        )

    def _get_learning_rule(self) -> LearningRule:
        """PFC uses dopamine-gated STDP learning."""
        return LearningRule.HEBBIAN

    def _initialize_weights(self) -> torch.Tensor:
        """Initialize feedforward weights."""
        # Xavier initialization
        std = (2.0 / (self.pfc_config.n_input + self.pfc_config.n_output)) ** 0.5
        return nn.Parameter(
            torch.randn(
                self.pfc_config.n_output,
                self.pfc_config.n_input,
                device=torch.device(self.pfc_config.device),
            ) * std
        )

    def _create_neurons(self) -> ConductanceLIF:
        """Create conductance-based LIF neurons with slow dynamics and SFA."""
        cfg = self.pfc_config
        # Slower dynamics for temporal integration + spike-frequency adaptation
        neuron_config = ConductanceLIFConfig(
            g_L=0.02,  # Slower leak (τ_m ≈ 50ms with C_m=1.0)
            tau_E=10.0,  # Slower excitatory (for integration)
            tau_I=15.0,  # Slower inhibitory
            adapt_increment=cfg.pfc_adapt_increment,  # SFA enabled!
            tau_adapt=cfg.pfc_adapt_tau,
        )
        neurons = ConductanceLIF(cfg.n_output, neuron_config)

        # =====================================================================
        # SHORT-TERM PLASTICITY for recurrent connections
        # =====================================================================
        # PFC recurrent connections show SHORT-TERM DEPRESSION, preventing
        # frozen attractors. This allows working memory to be updated.
        if cfg.stp_recurrent_enabled:
            device = torch.device(cfg.device)
            self.stp_recurrent = ShortTermPlasticity(
                n_pre=cfg.n_output,
                n_post=cfg.n_output,
                config=STPConfig.from_type(STPType.DEPRESSING, dt=cfg.dt_ms),
                per_synapse=True,
            )
            self.stp_recurrent.to(device)
        else:
            self.stp_recurrent = None

        return neurons

    def reset(self) -> None:
        """Reset state for new episode."""
        super().reset()
        self.neurons.reset_state(1)
        self.dopamine_system.reset()

        # Reset STP state
        if hasattr(self, 'stp_recurrent') and self.stp_recurrent is not None:
            self.stp_recurrent.reset_state(1)

        # Reset STDP traces
        self.input_trace.zero_()
        self.output_trace.zero_()
        self.stdp_eligibility.zero_()

        self.state = PrefrontalState(
            membrane=torch.zeros(1, self.config.n_output, device=self.device),
            spikes=torch.zeros(1, self.config.n_output, device=self.device),
            working_memory=torch.zeros(1, self.config.n_output, device=self.device),
            update_gate=torch.zeros(1, self.config.n_output, device=self.device),
            dopamine=self.pfc_config.dopamine_baseline,
        )

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset state with specific batch size."""
        from thalia.core.utils import assert_single_instance
        assert_single_instance(batch_size, "PrefrontalCortex")
        
        self.neurons.reset_state(batch_size)
        self.dopamine_system.reset()

        # Reset STDP traces
        self.input_trace.zero_()
        self.output_trace.zero_()
        self.stdp_eligibility.zero_()

        self.state = PrefrontalState(
            membrane=torch.zeros(batch_size, self.config.n_output, device=self.device),
            spikes=torch.zeros(batch_size, self.config.n_output, device=self.device),
            working_memory=torch.zeros(batch_size, self.config.n_output, device=self.device),
            update_gate=torch.ones(batch_size, self.config.n_output, device=self.device),
            dopamine=self.pfc_config.dopamine_baseline,
        )

    def forward(
        self,
        input_spikes: torch.Tensor,
        dt: float = 1.0,
        encoding_mod: float = 1.0,
        retrieval_mod: float = 1.0,
        dopamine_signal: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Process input through prefrontal cortex.

        Args:
            input_spikes: Input spike pattern [batch, n_input]
            dt: Time step in ms
            encoding_mod: Theta modulation for encoding (opens gate for new info)
            retrieval_mod: Theta modulation for retrieval (maintains WM)
            dopamine_signal: External DA signal for gating (-1 to 1)
            **kwargs: Additional inputs

        Returns:
            Output spikes [batch, n_output]
        """
        batch_size = input_spikes.shape[0]

        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        assert input_spikes.shape[-1] == self.pfc_config.n_input, (
            f"PrefrontalCortex.forward: input_spikes has shape {input_spikes.shape} "
            f"but n_input={self.pfc_config.n_input}. Check that input matches PFC config."
        )

        # Ensure state is initialized
        if self.state.working_memory is None:
            self.reset_state(batch_size)

        # Ensure batch size matches
        if self.state.working_memory.shape[0] != batch_size:
            self.reset_state(batch_size)

        # Update dopamine and get gate value
        da_level = self.dopamine_system.update(dopamine_signal, dt)
        gate = self.dopamine_system.get_gate()
        self.state.dopamine = da_level

        # =====================================================================
        # THETA MODULATION
        # =====================================================================
        # Encoding phase (theta trough): gate new info into WM
        # Retrieval phase (theta peak): maintain WM and boost recurrence
        ff_gain = 0.5 + 0.5 * encoding_mod  # 0.5-1.0: boost input during encoding
        rec_gain = 0.5 + 0.5 * retrieval_mod  # 0.5-1.0: boost recurrence during retrieval

        # Feedforward input - modulated by encoding phase
        ff_input = torch.matmul(input_spikes.float(), self.weights.t()) * ff_gain

        # =====================================================================
        # RECURRENT INPUT WITH STP (prevents frozen WM attractors)
        # =====================================================================
        # Without STP, the same WM pattern is reinforced forever.
        # With DEPRESSING STP, frequently-used synapses get temporarily weaker,
        # allowing WM to be updated with new information.
        if (hasattr(self, 'stp_recurrent') and self.stp_recurrent is not None
            and self.state.working_memory is not None):
            # Apply STP to recurrent connections
            # stp_efficacy has shape (batch, n_output, n_output) - per-synapse modulation
            stp_efficacy = self.stp_recurrent(
                self.state.working_memory.float()
            )
            # Effective weights: broadcast rec_weights with per-batch STP efficacy
            # rec_weights is (n_output, n_output), stp_efficacy is (batch, n_output, n_output)
            # For recurrent: WM @ (weights * efficacy).T = WM @ weights.T * efficacy (with proper dims)
            # Use einsum for clean batched matmul with per-synapse modulation
            # wm[b, i] * rec_weights[j, i] * stp_efficacy[b, i, j] → output[b, j]
            effective_rec_weights = self.rec_weights.unsqueeze(0) * stp_efficacy.transpose(-2, -1)
            rec_input = torch.einsum('bi,bji->bj', self.state.working_memory.float(), effective_rec_weights) * rec_gain
        else:
            # Recurrent input from working memory - modulated by retrieval phase
            rec_input = torch.matmul(
                self.state.working_memory.float() if self.state.working_memory is not None else torch.zeros(1, self.pfc_config.n_output, device=input_spikes.device),
                self.rec_weights.t()
            ) * rec_gain

        # Lateral inhibition
        inhib = torch.matmul(
            self.state.working_memory.float() if self.state.working_memory is not None else torch.zeros(1, self.pfc_config.n_output, device=input_spikes.device),
            self.inhib_weights.t()
        )

        # Total excitation and inhibition
        g_exc = (ff_input + rec_input).clamp(min=0)
        g_inh = inhib.clamp(min=0)

        # Run through neurons
        output_spikes, _ = self.neurons(g_exc, g_inh)

        # Update working memory with gating
        # High gate (high DA) → update with new activity
        # Low gate (low DA) → maintain current WM
        gate_tensor = torch.full_like(self.state.working_memory, gate)
        self.state.update_gate = gate_tensor

        # WM decay
        decay = torch.exp(torch.tensor(-dt / self.pfc_config.wm_decay_tau_ms))

        # Gated update: WM = gate * new_input + (1-gate) * decayed_old
        new_wm = (
            gate_tensor * output_spikes.float() +
            (1 - gate_tensor) * self.state.working_memory * decay
        )

        # Add noise for stochasticity
        noise = torch.randn_like(new_wm) * self.pfc_config.wm_noise_std
        self.state.working_memory = (new_wm + noise).clamp(min=0, max=1)

        # Store state
        self.state.spikes = output_spikes

        # Output shape check
        assert output_spikes.shape == (batch_size, self.pfc_config.n_output), (
            f"PrefrontalCortex.forward: output_spikes has shape {output_spikes.shape} "
            f"but expected ({batch_size}, {self.pfc_config.n_output}). "
            f"Check PFC neuron or weight configuration."
        )

        # Apply continuous plasticity (learning happens as part of forward dynamics)
        self._apply_plasticity(input_spikes, output_spikes, dt)

        return output_spikes

    def _apply_plasticity(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        dt: float = 1.0,
    ) -> None:
        """Apply dopamine-gated STDP learning continuously.

        This is called automatically at each forward() timestep.
        Uses spike-timing dependent plasticity gated by dopamine:
        - STDP creates eligibility traces from spike timing
        - Dopamine gates whether eligibility becomes lasting plasticity
        - High DA → learn associations (eligibility → weight change)
        - Low DA → protect existing patterns (eligibility decays unused)
        """
        if not self.plasticity_enabled:
            return

        input_spikes = ensure_batch_dim(input_spikes)
        output_spikes = ensure_batch_dim(output_spikes)

        cfg = self.pfc_config

        # Get effective learning rate (dopamine modulated)
        effective_lr = self.get_effective_learning_rate(cfg.stdp_lr)
        if effective_lr < 1e-8:
            return

        # Get 1D versions for torch.outer
        input_1d = ensure_1d(input_spikes)
        output_1d = ensure_1d(output_spikes)

        # ======================================================================
        # Update STDP traces
        # ======================================================================
        trace_decay = 1.0 - dt / cfg.stdp_tau_ms
        self.input_trace = self.input_trace * trace_decay + input_1d
        self.output_trace = self.output_trace * trace_decay + output_1d

        # STDP rule: post spike with pre trace → LTP
        ltp = torch.outer(output_1d, self.input_trace)
        ltd = torch.outer(self.output_trace, input_1d)

        # Apply soft bounds
        if cfg.soft_bounds:
            w_normalized = (self.weights - self.config.w_min) / (self.config.w_max - self.config.w_min + 1e-6)
            ltp_factor = 1.0 - w_normalized
            ltd_factor = w_normalized
            ltp = ltp * ltp_factor
            ltd = ltd * ltd_factor

        # Compute STDP weight change with dopamine-modulated learning rate
        stdp_dw = effective_lr * (ltp - ltd)

        # ======================================================================
        # HETEROSYNAPTIC PLASTICITY
        # ======================================================================
        # When a postsynaptic neuron fires but a presynaptic neuron was INACTIVE,
        # weaken that synapse. This prevents dominant synapses from monopolizing
        # the neuron and allows flexible working memory updates.
        # inactive_pre[i,j] = 1 if pre j was inactive when post i fired
        inactive_pre = output_1d.unsqueeze(1) * (1.0 - self.input_trace.unsqueeze(0))
        hetero_ltd = cfg.heterosynaptic_ratio * effective_lr * inactive_pre
        stdp_dw = stdp_dw - hetero_ltd

        # Accumulate into eligibility trace
        eligibility_decay = 1.0 - dt / 100.0  # 100ms eligibility window
        self.stdp_eligibility = self.stdp_eligibility * eligibility_decay + stdp_dw

        # ======================================================================
        # Dopamine-gated learning
        # ======================================================================
        gate = self.dopamine_system.get_gate()

        # Only apply eligibility if gate is open (high DA)
        if gate > 0.1:  # Minimum gate to trigger learning
            dW = self.stdp_eligibility * gate

            with torch.no_grad():
                self.weights.data += dW
                clamp_weights(self.weights.data, self.config.w_min, self.config.w_max)

                # Synaptic scaling for homeostasis
                if cfg.synaptic_scaling_enabled:
                    mean_weight = self.weights.data.mean()
                    scaling = 1.0 + cfg.synaptic_scaling_rate * (cfg.synaptic_scaling_target - mean_weight)
                    self.weights.data *= scaling
                    clamp_weights(self.weights.data, self.config.w_min, self.config.w_max)

        # ======================================================================
        # Update recurrent weights to strengthen WM patterns
        # ======================================================================
        if self.state.working_memory is not None and gate > 0.1:
            wm_mean = self.state.working_memory.mean(dim=0)
            dW_rec = cfg.rule_lr * gate * torch.outer(wm_mean, wm_mean)
            with torch.no_grad():
                self.rec_weights.data += dW_rec
                self.rec_weights.data.fill_diagonal_(
                    cfg.recurrent_strength
                )  # Maintain self-excitation
                self.rec_weights.data.clamp_(0.0, 1.0)

    def set_context(self, context: torch.Tensor) -> None:
        """
        Set the current context/rule in working memory.

        This allows explicit control of PFC state for rule-based tasks.

        Args:
            context: Context pattern [batch, n_output] or [n_output]
        """
        context = ensure_batch_dim(context)

        self.state.working_memory = context.to(self.device).float()
        self.state.active_rule = context.to(self.device).float()

    def get_working_memory(self) -> torch.Tensor:
        """Get current working memory contents."""
        if self.state.working_memory is None:
            return torch.zeros(1, self.config.n_output, device=self.device)
        return self.state.working_memory

    def maintain(self, n_steps: int = 10, dt: float = 1.0) -> Dict[str, Any]:
        """
        Run maintenance steps without external input.

        Useful for testing WM persistence.

        Args:
            n_steps: Number of maintenance steps
            dt: Time step in ms

        Returns:
            Metrics about maintenance
        """
        if self.state.working_memory is None:
            return {"error": "No working memory to maintain"}

        initial_wm = self.state.working_memory.clone()
        batch_size = self.state.working_memory.shape[0]

        for _ in range(n_steps):
            # No external input, low DA (maintenance mode)
            null_input = torch.zeros(batch_size, self.config.n_input, device=self.device)
            self.forward(null_input, dopamine_signal=-0.3, dt=dt)

        final_wm = self.state.working_memory

        # Compute retention using safe cosine similarity
        retention = cosine_similarity_safe(
            initial_wm.flatten(), final_wm.flatten()
        ).item()

        return {
            "n_steps": n_steps,
            "retention": retention,
            "initial_activity": initial_wm.mean().item(),
            "final_activity": final_wm.mean().item(),
        }

    def get_state(self) -> PrefrontalState:
        """Get current state."""
        return self.state
