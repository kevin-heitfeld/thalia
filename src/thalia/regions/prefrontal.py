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

from thalia.regions.base import (
    BrainRegion,
    RegionConfig,
    RegionState,
    LearningRule,
)
from thalia.core.neuron import ConductanceLIF, ConductanceLIFConfig


@dataclass
class PrefrontalConfig(RegionConfig):
    """Configuration specific to prefrontal cortex."""

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

    # Recurrent connections for WM maintenance
    recurrent_strength: float = 0.8  # Self-excitation for persistence
    recurrent_inhibition: float = 0.2  # Lateral inhibition

    # Weight constraints
    soft_bounds: bool = True


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

        # Initialize working memory state
        self.state = PrefrontalState(
            working_memory=torch.zeros(1, config.n_output, device=self.device),
            update_gate=torch.zeros(1, config.n_output, device=self.device),
            dopamine=config.dopamine_baseline,
        )

    def _get_learning_rule(self) -> LearningRule:
        """PFC uses Hebbian learning with gating."""
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
        """Create conductance-based LIF neurons with slow dynamics."""
        # Slower dynamics for temporal integration
        neuron_config = ConductanceLIFConfig(
            g_L=0.02,  # Slower leak (τ_m ≈ 50ms with C_m=1.0)
            tau_E=10.0,  # Slower excitatory (for integration)
            tau_I=15.0,  # Slower inhibitory
        )
        return ConductanceLIF(self.pfc_config.n_output, neuron_config)

    def reset(self) -> None:
        """Reset state for new episode."""
        super().reset()
        self.neurons.reset_state(1)
        self.dopamine_system.reset()

        self.state = PrefrontalState(
            membrane=torch.zeros(1, self.config.n_output, device=self.device),
            spikes=torch.zeros(1, self.config.n_output, device=self.device),
            working_memory=torch.zeros(1, self.config.n_output, device=self.device),
            update_gate=torch.zeros(1, self.config.n_output, device=self.device),
            dopamine=self.pfc_config.dopamine_baseline,
        )

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset state with specific batch size."""
        self.neurons.reset_state(batch_size)
        self.dopamine_system.reset()

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
        dopamine_signal: float = 0.0,
        dt: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Process input through prefrontal cortex.

        Args:
            input_spikes: Input spike pattern [batch, n_input]
            dopamine_signal: External DA signal for gating (-1 to 1)
            dt: Time step in ms
            **kwargs: Additional inputs

        Returns:
            Output spikes [batch, n_output]
        """
        batch_size = input_spikes.shape[0]

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

        # Feedforward input
        ff_input = torch.matmul(input_spikes.float(), self.weights.t())

        # Recurrent input from working memory
        rec_input = torch.matmul(
            self.state.working_memory.float(),
            self.rec_weights.t()
        )

        # Lateral inhibition
        inhib = torch.matmul(
            self.state.working_memory.float(),
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

        return output_spikes

    def learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        reward: float = 0.0,
        dt: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Apply gated Hebbian learning.

        Learning is modulated by dopamine:
        - High DA (gate open) → learn associations with new inputs
        - Low DA (gate closed) → consolidate existing WM representations

        Args:
            input_spikes: Input spikes [batch, n_input]
            output_spikes: Output spikes [batch, n_output]
            reward: Reward signal (triggers DA burst)
            dt: Time step in ms
            **kwargs: Additional learning signals

        Returns:
            Dictionary with learning metrics
        """
        # Dopamine modulates learning rate
        gate = self.dopamine_system.get_gate()
        effective_lr = self.pfc_config.wm_lr * (0.2 + 0.8 * gate)

        # Hebbian update: dW = lr * post * pre^T
        pre_mean = input_spikes.float().mean(dim=0)
        post_mean = output_spikes.float().mean(dim=0)

        dW = effective_lr * torch.outer(post_mean, pre_mean)

        # Apply soft bounds
        if self.pfc_config.soft_bounds:
            # Scale by distance to bounds
            upper_room = self.config.w_max - self.weights.data
            lower_room = self.weights.data - self.config.w_min
            dW = torch.where(dW > 0, dW * upper_room, dW * lower_room)

        with torch.no_grad():
            self.weights.data += dW
            self.weights.data.clamp_(self.config.w_min, self.config.w_max)

        # Also update recurrent weights to strengthen WM patterns
        if self.state.working_memory is not None:
            wm_mean = self.state.working_memory.mean(dim=0)
            dW_rec = self.pfc_config.rule_lr * gate * torch.outer(wm_mean, wm_mean)
            with torch.no_grad():
                self.rec_weights.data += dW_rec
                self.rec_weights.data.fill_diagonal_(
                    self.pfc_config.recurrent_strength
                )  # Maintain self-excitation
                self.rec_weights.data.clamp_(0.0, 1.0)

        return {
            "effective_lr": effective_lr,
            "gate_value": gate,
            "dopamine": self.state.dopamine,
            "ltp": (dW > 0).sum().item(),
            "ltd": (dW < 0).sum().item(),
            "wm_activity": self.state.working_memory.mean().item()
            if self.state.working_memory is not None else 0.0,
        }

    def set_context(self, context: torch.Tensor) -> None:
        """
        Set the current context/rule in working memory.

        This allows explicit control of PFC state for rule-based tasks.

        Args:
            context: Context pattern [batch, n_output] or [n_output]
        """
        if context.dim() == 1:
            context = context.unsqueeze(0)

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

        # Compute retention
        retention = torch.nn.functional.cosine_similarity(
            initial_wm.flatten(), final_wm.flatten(), dim=0
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
