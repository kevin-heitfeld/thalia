"""
Leaky Integrate-and-Fire (LIF) neuron model.

The LIF neuron is the fundamental building block of THALIA networks.
It accumulates input over time, leaks toward a resting potential,
and fires a spike when threshold is reached.

Membrane dynamics:
    τ_m * dV/dt = -(V - V_rest) + R * I

When V >= V_threshold:
    - Emit spike
    - V = V_reset
    - Enter refractory period
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from thalia.config.base import NeuralComponentConfig


# =============================================================================
# CURRENT-BASED LIF NEURON
# =============================================================================


@dataclass
class LIFConfig(NeuralComponentConfig):
    """Configuration for LIF neuron parameters.

    Inherits n_neurons, dt, device, dtype, seed from NeuralComponentConfig.

    Attributes:
        tau_mem: Membrane time constant in ms (default: 20.0)
            Controls how quickly the membrane potential decays toward rest.
            Larger values = slower decay = longer memory of inputs.
        v_rest: Resting membrane potential (default: 0.0)
        v_reset: Reset potential after spike (default: 0.0)
        v_threshold: Spike threshold (default: 1.0)
        tau_ref: Refractory period in ms (default: 2.0)
            Absolute refractory period during which neuron cannot fire.

        # Adaptation parameters (spike-frequency adaptation)
        tau_adapt: Adaptation time constant in ms (default: 100.0)
            Controls how quickly adaptation current decays.
        adapt_increment: Adaptation current increment per spike (default: 0.0)
            Set > 0 to enable spike-frequency adaptation.
            After each spike, adaptation current increases by this amount,
            making the neuron temporarily less excitable.

        # Noise parameters
        noise_std: Standard deviation of membrane noise (default: 0.0)
            Gaussian noise added to membrane potential each timestep.
            Enables stochastic firing and spontaneous activity.

        # Membrane bounds
        v_min: Minimum membrane potential (default: None = no bound)
            If set, membrane is clamped to this value. Biologically justified
            by ion channel reversal potentials (e.g., GABA_A reversal ~ -70mV).
            Prevents runaway hyperpolarization from inhibitory currents.
    """
    tau_mem: float = 20.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_threshold: float = 1.0
    tau_ref: float = 2.0
    v_min: Optional[float] = None  # Minimum membrane potential (reversal limit)

    # Adaptation (spike-frequency adaptation)
    tau_adapt: float = 100.0
    adapt_increment: float = 0.0  # 0 = no adaptation

    # Noise
    noise_std: float = 0.0  # 0 = deterministic

    @property
    def decay(self) -> float:
        """Membrane decay factor per timestep."""
        return torch.exp(torch.tensor(-self.dt / self.tau_mem)).item()

    @property
    def adapt_decay(self) -> float:
        """Adaptation current decay factor per timestep."""
        return torch.exp(torch.tensor(-self.dt / self.tau_adapt)).item()

    @property
    def ref_steps(self) -> int:
        """Refractory period in timesteps."""
        return int(self.tau_ref / self.dt)


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron layer.

    Implements a population of LIF neurons with shared parameters.
    Supports batch processing and GPU acceleration.

    Features:
        - Membrane potential with exponential leak
        - Absolute refractory period
        - Spike-frequency adaptation (optional)
        - Stochastic noise injection (optional)

    Args:
        n_neurons: Number of neurons in the layer
        config: LIF configuration parameters

    Example:
        >>> # Basic usage
        >>> lif = LIFNeuron(n_neurons=100)
        >>> lif.reset_state()
        >>> for t in range(100):
        ...     spikes, voltage = lif(input_current[t])

        >>> # With adaptation and noise
        >>> config = LIFConfig(
        ...     tau_mem=20.0,
        ...     adapt_increment=0.1,  # Enable adaptation
        ...     noise_std=0.05        # Add noise
        ... )
        >>> lif = LIFNeuron(n_neurons=100, config=config)
    """

    def __init__(
        self,
        n_neurons: int,
        config: Optional[LIFConfig] = None
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config or LIFConfig()

        # Register constants as buffers (moves with device)
        self.register_buffer(
            "decay",
            torch.tensor(self.config.decay, dtype=torch.float32)
        )
        self.register_buffer(
            "adapt_decay",
            torch.tensor(self.config.adapt_decay, dtype=torch.float32)
        )
        self.register_buffer(
            "v_threshold",
            torch.tensor(self.config.v_threshold, dtype=torch.float32)
        )
        self.register_buffer(
            "v_rest",
            torch.tensor(self.config.v_rest, dtype=torch.float32)
        )
        self.register_buffer(
            "v_reset",
            torch.tensor(self.config.v_reset, dtype=torch.float32)
        )

        # Optional membrane minimum (for reversal potential bounds)
        if self.config.v_min is not None:
            self.register_buffer(
                "v_min",
                torch.tensor(self.config.v_min, dtype=torch.float32)
            )
        else:
            self.v_min = None

        # State variables (initialized on first forward or reset)
        self.membrane: Optional[torch.Tensor] = None
        self.refractory: Optional[torch.Tensor] = None
        self.adaptation: Optional[torch.Tensor] = None  # Adaptation current

    def reset_state(self) -> None:
        """Reset neuron state to resting potential (ADR-005: 1D tensors).

        Creates 1D state tensors for single-brain architecture.
        """
        device = self.decay.device

        self.membrane = torch.full(
            (self.n_neurons,),
            self.config.v_rest,
            device=device,
            dtype=torch.float32
        )
        self.refractory = torch.zeros(
            (self.n_neurons,),
            device=device,
            dtype=torch.int32
        )
        self.adaptation = torch.zeros(
            (self.n_neurons,),
            device=device,
            dtype=torch.float32
        )

    def forward(
        self,
        input_current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one timestep of input (ADR-005: 1D tensors).

        Args:
            input_current: Input current to neurons [n_neurons] (1D)

        Returns:
            spikes: Binary bool spike tensor [n_neurons] (1D)
            membrane: Membrane potentials after update [n_neurons] (1D)
        """
        # Initialize state if needed
        if self.membrane is None:
            self.reset_state()

        # ADR-005: Expect 1D tensors
        assert input_current.dim() == 1, (
            f"LIFNeuron.forward: Expected 1D input (ADR-005), got shape {input_current.shape}"
        )
        assert input_current.shape[0] == self.n_neurons, (
            f"LIFNeuron.forward: input has {input_current.shape[0]} neurons, expected {self.n_neurons}"
        )

        # Decrement refractory counter
        self.refractory = torch.clamp(self.refractory - 1, min=0)

        # Check which neurons are not in refractory period
        not_refractory = (self.refractory == 0).float()

        # Decay adaptation current
        self.adaptation = self.adaptation * self.adapt_decay

        # Compute effective input (subtract adaptation current)
        effective_input = input_current - self.adaptation

        # Add noise if configured
        if self.config.noise_std > 0:
            noise = torch.randn_like(self.membrane) * self.config.noise_std
            effective_input = effective_input + noise

        # Leaky integration (only for non-refractory neurons)
        # V(t+1) = decay * (V(t) - V_rest) + V_rest + I_eff(t) * not_refractory
        self.membrane = (
            self.decay * (self.membrane - self.v_rest)
            + self.v_rest
            + effective_input * not_refractory
        )

        # Clamp membrane to minimum (reversal potential bound)
        # Biologically justified: ion channels have reversal potentials
        # that physically prevent membrane from going too negative
        if self.v_min is not None:
            self.membrane = torch.clamp(self.membrane, min=self.v_min.item())

        # Spike generation (bool for biological accuracy and memory efficiency)
        spikes = self.membrane >= self.v_threshold

        # Reset spiking neurons
        self.membrane = torch.where(
            spikes,
            self.v_reset.expand_as(self.membrane),
            self.membrane
        )

        # Set refractory period for spiking neurons
        self.refractory = torch.where(
            spikes,
            torch.full_like(self.refractory, self.config.ref_steps),
            self.refractory
        )

        # Increment adaptation for spiking neurons (need float for arithmetic)
        if self.config.adapt_increment > 0:
            self.adaptation = self.adaptation + spikes.float() * self.config.adapt_increment

        return spikes, self.membrane.clone()

    def get_state(self) -> dict[str, torch.Tensor]:
        """Get current neuron state for analysis/saving.

        Returns:
            Dictionary containing membrane, refractory, and adaptation state
        """
        return {
            "membrane": self.membrane.clone() if self.membrane is not None else None,
            "refractory": self.refractory.clone() if self.refractory is not None else None,
            "adaptation": self.adaptation.clone() if self.adaptation is not None else None,
        }

    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore neuron state from checkpoint.

        Args:
            state: Dictionary from get_state()
        """
        # Infer device from existing tensors or incoming state
        device = (self.membrane.device if self.membrane is not None
                 else state["membrane"].device if state["membrane"] is not None
                 else torch.device("cpu"))

        if state["membrane"] is not None:
            self.membrane = state["membrane"].to(device)
        if state["refractory"] is not None:
            self.refractory = state["refractory"].to(device)
        if state["adaptation"] is not None:
            self.adaptation = state["adaptation"].to(device)

    def _apply(self, fn, recurse: bool = True):
        """Apply a function to all tensors, including state variables.

        This is called by .to(), .cuda(), .cpu() etc. to move tensors to devices.
        Override to also move non-parameter state tensors.

        Args:
            fn: Function to apply to each tensor
            recurse: Whether to recurse into child modules

        Returns:
            Self (for chaining)
        """
        # Call parent to handle parameters and buffers
        super()._apply(fn, recurse)

        # Apply to state variables if they exist
        if self.membrane is not None:
            self.membrane = fn(self.membrane)
        if self.refractory is not None:
            self.refractory = fn(self.refractory)
        if self.adaptation is not None:
            self.adaptation = fn(self.adaptation)

        return self

    def __repr__(self) -> str:
        adapt_str = f", adapt={self.config.adapt_increment}" if self.config.adapt_increment > 0 else ""
        noise_str = f", noise={self.config.noise_std}" if self.config.noise_std > 0 else ""
        return (
            f"LIFNeuron(n={self.n_neurons}, "
            f"τ_m={self.config.tau_mem}ms, "
            f"θ={self.config.v_threshold}"
            f"{adapt_str}{noise_str})"
        )


# =============================================================================
# CONDUCTANCE-BASED LIF NEURON
# =============================================================================


@dataclass
class ConductanceLIFConfig(NeuralComponentConfig):
    """Configuration for conductance-based LIF neuron.

    Inherits n_neurons, dt, device, dtype, seed from NeuralComponentConfig.

    This implements biologically realistic membrane dynamics where currents
    depend on the difference between membrane potential and reversal potentials.

    Membrane equation:
        C_m * dV/dt = g_L(E_L - V) + g_E(E_E - V) + g_I(E_I - V)

    Key advantages over current-based LIF:
    - Natural saturation (can't exceed reversal potentials)
    - Proper shunting inhibition (divisive, not subtractive)
    - Realistic voltage-dependent current flow
    - No need for artificial v_min clamping

    Attributes:
        # Membrane properties
        C_m: Membrane capacitance in arbitrary units (default: 1.0)
            Scales the effective time constant. Higher = slower dynamics.
        g_L: Leak conductance (default: 0.05)
            Controls passive membrane time constant: τ_m = C_m / g_L
            With C_m=1.0, g_L=0.05 gives τ_m ≈ 20ms

        # Reversal potentials (in normalized units, threshold = 1.0)
        E_L: Leak/resting reversal potential (default: 0.0)
            Where membrane settles with no input. Typically -65mV in biology.
        E_E: Excitatory reversal potential (default: 3.0)
            AMPA/NMDA reversal. Typically 0mV in biology.
            Set > threshold so excitation can drive spiking.
        E_I: Inhibitory reversal potential (default: -0.5)
            GABA_A reversal. Typically -70mV in biology.
            Set < E_L for hyperpolarizing inhibition.

        # Synaptic conductance time constants
        tau_E: Excitatory conductance decay (ms) (default: 5.0)
            AMPA is fast (1-2ms), NMDA is slow (50-100ms).
            This is a simplified single time constant.
        tau_I: Inhibitory conductance decay (ms) (default: 10.0)
            GABA_A is ~5-10ms.

        # Spike parameters
        v_threshold: Spike threshold (default: 1.0)
        v_reset: Reset potential after spike (default: 0.0)
        tau_ref: Absolute refractory period in ms (default: 2.0)

        # Adaptation
        tau_adapt: Adaptation time constant in ms (default: 100.0)
        adapt_increment: Adaptation conductance increment per spike (default: 0.0)
            When > 0, adds adaptation conductance (slow K+ current)
        E_adapt: Adaptation reversal potential (default: -0.5)
            Typically hyperpolarizing (like slow K+), set below E_L

        # Noise
        noise_std: Membrane noise standard deviation (default: 0.0)
    """
    # Membrane properties
    C_m: float = 1.0
    g_L: float = 0.05  # τ_m = C_m/g_L = 20ms

    # Reversal potentials (normalized units)
    E_L: float = 0.0    # Leak/rest (≈ -65mV scaled to 0)
    E_E: float = 3.0    # Excitatory (≈ 0mV, well above threshold)
    E_I: float = -0.5   # Inhibitory (≈ -70mV, below rest)

    # Synaptic time constants
    tau_E: float = 5.0   # Excitatory (AMPA-like)
    tau_I: float = 10.0  # Inhibitory (GABA_A-like)

    # Spike parameters
    v_threshold: float = 1.0
    v_reset: float = 0.0
    tau_ref: float = 2.0

    # Adaptation (conductance-based)
    tau_adapt: float = 100.0
    adapt_increment: float = 0.0
    E_adapt: float = -0.5  # Adaptation reversal (hyperpolarizing, like slow K+)

    # Noise
    noise_std: float = 0.0

    @property
    def tau_m(self) -> float:
        """Effective membrane time constant."""
        return self.C_m / self.g_L

    @property
    def g_E_decay(self) -> float:
        """Excitatory conductance decay factor per timestep."""
        return torch.exp(torch.tensor(-self.dt / self.tau_E)).item()

    @property
    def g_I_decay(self) -> float:
        """Inhibitory conductance decay factor per timestep."""
        return torch.exp(torch.tensor(-self.dt / self.tau_I)).item()

    @property
    def adapt_decay(self) -> float:
        """Adaptation conductance decay factor per timestep."""
        return torch.exp(torch.tensor(-self.dt / self.tau_adapt)).item()

    @property
    def ref_steps(self) -> int:
        """Refractory period in timesteps."""
        return int(self.tau_ref / self.dt)


class ConductanceLIF(nn.Module):
    """Conductance-based Leaky Integrate-and-Fire neuron.

    Implements biologically realistic membrane dynamics with reversal potentials.

    Key features:
    - Voltage-dependent currents: I = g(E - V)
    - Natural saturation at reversal potentials
    - Proper shunting inhibition (divisive effect on membrane)
    - Separate excitatory and inhibitory conductances
    - Spike-frequency adaptation (optional)

    Membrane equation:
        C_m * dV/dt = g_L(E_L - V) + g_E(E_E - V) + g_I(E_I - V) + g_adapt(E_L - V)

    The conductances g_E and g_I are driven by synaptic inputs and decay
    exponentially. This creates a natural low-pass filter on inputs.

    Args:
        n_neurons: Number of neurons in the layer
        config: ConductanceLIFConfig with parameters

    Example:
        >>> config = ConductanceLIFConfig(
        ...     E_E=3.0,    # Excitatory reversal (above threshold)
        ...     E_I=-0.5,   # Inhibitory reversal (below rest)
        ...     tau_E=5.0,  # Fast excitation
        ...     tau_I=10.0, # Slower inhibition
        ... )
        >>> neuron = ConductanceLIF(n_neurons=100, config=config)
        >>> neuron.reset_state()
        >>>
        >>> # Input conductances, not currents!
        >>> g_exc = torch.rand(1, 100) * 0.1  # Excitatory input
        >>> g_inh = torch.rand(1, 100) * 0.05  # Inhibitory input
        >>> spikes, voltage = neuron(g_exc, g_inh)
    """

    def __init__(
        self,
        n_neurons: int,
        config: Optional[ConductanceLIFConfig] = None
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config or ConductanceLIFConfig()

        # Register constants as buffers
        self.register_buffer("C_m", torch.tensor(self.config.C_m))
        self.register_buffer("g_L", torch.tensor(self.config.g_L))
        self.register_buffer("E_L", torch.tensor(self.config.E_L))
        self.register_buffer("E_E", torch.tensor(self.config.E_E))
        self.register_buffer("E_I", torch.tensor(self.config.E_I))
        self.register_buffer("E_adapt", torch.tensor(self.config.E_adapt))
        # Per-neuron threshold for intrinsic plasticity support
        self.register_buffer(
            "v_threshold",
            torch.full((n_neurons,), self.config.v_threshold, dtype=torch.float32)
        )
        self.register_buffer("v_reset", torch.tensor(self.config.v_reset))
        self.register_buffer("g_E_decay", torch.tensor(self.config.g_E_decay))
        self.register_buffer("g_I_decay", torch.tensor(self.config.g_I_decay))
        self.register_buffer("adapt_decay", torch.tensor(self.config.adapt_decay))

        # State variables
        self.membrane: Optional[torch.Tensor] = None
        self.g_E: Optional[torch.Tensor] = None  # Excitatory conductance
        self.g_I: Optional[torch.Tensor] = None  # Inhibitory conductance
        self.g_adapt: Optional[torch.Tensor] = None  # Adaptation conductance
        self.refractory: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        """Reset neuron state to resting potential.

        Initializes all state tensors as 1D [n_neurons] per ADR-005.
        """
        device = self.C_m.device

        # Membrane starts at leak reversal (resting potential)
        self.membrane = torch.full(
            (self.n_neurons,),
            self.config.E_L,
            device=device,
            dtype=torch.float32
        )

        # All conductances start at zero
        self.g_E = torch.zeros(
            self.n_neurons,
            device=device,
            dtype=torch.float32
        )
        self.g_I = torch.zeros(
            self.n_neurons,
            device=device,
            dtype=torch.float32
        )
        self.g_adapt = torch.zeros(
            self.n_neurons,
            device=device,
            dtype=torch.float32
        )

        self.refractory = torch.zeros(
            self.n_neurons,
            device=device,
            dtype=torch.int32
        )

    def adjust_thresholds(
        self,
        delta: torch.Tensor,
        min_threshold: float = 0.5,
        max_threshold: float = 2.0
    ) -> None:
        """Adjust per-neuron thresholds for intrinsic plasticity.

        Args:
            delta: Threshold adjustment per neuron, shape (n_neurons,) or (batch, n_neurons)
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
        """
        # Handle batch dimension - average across batch if present
        if delta.dim() == 2:
            delta = delta.mean(dim=0)

        # Apply adjustment and clamp to valid range
        self.v_threshold = (self.v_threshold + delta).clamp(min_threshold, max_threshold)

    def forward(
        self,
        g_exc_input: torch.Tensor,
        g_inh_input: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one timestep of conductance input.

        Args:
            g_exc_input: Excitatory conductance input, shape [n_neurons] (1D, ADR-005)
                This is ADDED to the excitatory conductance state.
            g_inh_input: Inhibitory conductance input, shape [n_neurons] (1D, ADR-005)
                Optional. If None, no inhibitory input is applied.

        Returns:
            spikes: Binary spike tensor, shape [n_neurons] (1D bool, ADR-004/005)
            membrane: Membrane potentials after update, shape [n_neurons] (1D)
        """
        # Initialize state if needed
        if self.membrane is None:
            self.reset_state()

        # Assert 1D input (ADR-005: No Batch Dimension)
        assert g_exc_input.dim() == 1, (
            f"ConductanceLIF.forward: g_exc_input must be 1D [n_neurons], "
            f"got shape {g_exc_input.shape}. See ADR-005: No Batch Dimension."
        )
        assert g_exc_input.shape[0] == self.n_neurons, (
            f"ConductanceLIF.forward: g_exc_input has {g_exc_input.shape[0]} neurons "
            f"but expected {self.n_neurons}."
        )

        # Decrement refractory counter (in-place)
        self.refractory = (self.refractory - 1).clamp_(min=0)
        not_refractory = self.refractory == 0

        # Update synaptic conductances (in-place decay + add input)
        self.g_E.mul_(self.g_E_decay).add_(g_exc_input)
        if g_inh_input is not None:
            self.g_I.mul_(self.g_I_decay).add_(g_inh_input)
        else:
            self.g_I.mul_(self.g_I_decay)

        # Decay adaptation conductance (in-place)
        self.g_adapt.mul_(self.adapt_decay)

        # Compute total conductance (for effective time constant)
        # Pre-add g_L to avoid extra addition
        g_total = self.g_L + self.g_E + self.g_I + self.g_adapt

        # Compute equilibrium potential (weighted average of reversals)
        # Fused: V_inf = (g_L*E_L + g_E*E_E + g_I*E_I + g_adapt*E_adapt) / g_total
        # Pre-compute g_L*E_L once (it's a constant)
        V_inf = (
            self.g_L * self.E_L +
            self.g_E * self.E_E +
            self.g_I * self.E_I +
            self.g_adapt * self.E_adapt
        ) / g_total

        # Effective time constant: τ_eff = C_m / g_total
        # V(t+dt) = V_inf + (V(t) - V_inf) * exp(-dt * g_total / C_m)
        decay_factor = torch.exp((-self.config.dt / self.C_m) * g_total)

        # Update membrane for non-refractory neurons
        # Fused: new_V = V_inf + (V - V_inf) * decay = V_inf * (1 - decay) + V * decay
        V_diff = self.membrane - V_inf
        new_membrane = V_inf + V_diff * decay_factor

        # Add noise only if configured (branch elimination)
        if self.config.noise_std > 0:
            new_membrane = new_membrane + torch.randn_like(self.membrane) * self.config.noise_std

        # Apply only to non-refractory neurons
        self.membrane = torch.where(not_refractory, new_membrane, self.membrane)

        # Spike generation (bool for biological accuracy and memory efficiency)
        spikes = self.membrane >= self.v_threshold

        # Combined spike handling: reset membrane AND set refractory in one pass
        # This avoids creating intermediate tensors
        if spikes.any():
            self.membrane = torch.where(spikes, self.v_reset, self.membrane)
            self.refractory = torch.where(
                spikes,
                self.config.ref_steps,  # scalar broadcasts
                self.refractory
            )
            # Increment adaptation for spiking neurons (need float for arithmetic)
            if self.config.adapt_increment > 0:
                self.g_adapt = self.g_adapt + spikes.float() * self.config.adapt_increment

        return spikes, self.membrane

    def forward_current(
        self,
        input_current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience method: convert current to conductance and process.

        This allows drop-in replacement of current-based LIF.
        Positive current → excitatory conductance
        Negative current → inhibitory conductance

        Args:
            input_current: Input current, shape (batch, n_neurons)
                Positive values become excitatory conductance.
                Negative values become inhibitory conductance.

        Returns:
            spikes, membrane: Same as forward()
        """
        # Split into excitatory and inhibitory components
        # Scale by (E_E - E_L) to match current-based behavior at rest
        g_exc = torch.clamp(input_current, min=0) / (self.E_E - self.E_L)
        g_inh = torch.clamp(-input_current, min=0) / (self.E_L - self.E_I)

        return self.forward(g_exc, g_inh)

    def get_state(self) -> dict[str, Optional[torch.Tensor]]:
        """Get current neuron state for analysis/saving."""
        return {
            "membrane": self.membrane.clone() if self.membrane is not None else None,
            "g_E": self.g_E.clone() if self.g_E is not None else None,
            "g_I": self.g_I.clone() if self.g_I is not None else None,
            "g_adapt": self.g_adapt.clone() if self.g_adapt is not None else None,
            "refractory": self.refractory.clone() if self.refractory is not None else None,
        }

    def load_state(self, state: dict[str, Optional[torch.Tensor]]) -> None:
        """Restore neuron state from checkpoint.

        Args:
            state: Dictionary from get_state()
        """
        # Infer device from existing tensors or incoming state
        device = (self.membrane.device if self.membrane is not None
                 else state["membrane"].device if state["membrane"] is not None
                 else torch.device("cpu"))

        if state["membrane"] is not None:
            self.membrane = state["membrane"].to(device)
        if state["g_E"] is not None:
            self.g_E = state["g_E"].to(device)
        if state["g_I"] is not None:
            self.g_I = state["g_I"].to(device)
        if state["g_adapt"] is not None:
            self.g_adapt = state["g_adapt"].to(device)
        if state["refractory"] is not None:
            self.refractory = state["refractory"].to(device)

    def _apply(self, fn, recurse: bool = True):
        """Apply a function to all tensors, including state variables.

        This is called by .to(), .cuda(), .cpu() etc. to move tensors to devices.
        Override to also move non-parameter state tensors.

        Args:
            fn: Function to apply to each tensor
            recurse: Whether to recurse into child modules

        Returns:
            Self (for chaining)
        """
        # Call parent to handle parameters and buffers
        super()._apply(fn, recurse)

        # Apply to state variables if they exist
        if self.membrane is not None:
            self.membrane = fn(self.membrane)
        if self.g_E is not None:
            self.g_E = fn(self.g_E)
        if self.g_I is not None:
            self.g_I = fn(self.g_I)
        if self.g_adapt is not None:
            self.g_adapt = fn(self.g_adapt)
        if self.refractory is not None:
            self.refractory = fn(self.refractory)

        return self

    def __repr__(self) -> str:
        return (
            f"ConductanceLIF(n={self.n_neurons}, "
            f"τ_m={self.config.tau_m:.1f}ms, "
            f"E_L={self.config.E_L}, E_E={self.config.E_E}, E_I={self.config.E_I}, "
            f"θ={self.config.v_threshold})"
        )
