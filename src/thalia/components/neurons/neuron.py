"""Conductance-Based Leaky Integrate-and-Fire (LIF) Neuron Model.

This module implements biologically realistic spiking neurons using
conductance-based membrane dynamics where synaptic currents depend on
the driving force (difference between membrane potential and reversal potential).

**Membrane Dynamics**:
=====================
.. math::

    C_m \\\\frac{dV}{dt} = g_L(E_L - V) + g_E(E_E - V) + g_I(E_I - V)

Where:
- V: membrane potential
- g_L, g_E, g_I: leak, excitatory, inhibitory conductances
- E_L, E_E, E_I: reversal potentials for each conductance

**Key Advantages** over current-based LIF:
==========================================
1. **Natural Saturation**: Currents vanish as V approaches reversal potential
2. **Shunting Inhibition**: Inhibition is divisive (not just subtractive)
3. **Voltage-Dependent**: Current magnitude depends on driving force
4. **Biological Realism**: Matches real neuron biophysics more closely

**Spike Generation**:
When V ≥ V_threshold:
- Emit spike (binary: 0 or 1, per ADR-004)
- Reset: V → V_reset
- Enter refractory period (τ_ref ms, no spiking)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from thalia.config.neuron_config import BaseNeuronConfig
from thalia.constants.neuron import (
    E_EXCITATORY,
    E_INHIBITORY,
    E_LEAK,
    G_LEAK_STANDARD,
    MEMBRANE_CAPACITANCE_STANDARD,
    TAU_SYN_EXCITATORY,
    TAU_SYN_INHIBITORY,
)

# =============================================================================
# CONDUCTANCE-BASED LIF NEURON
# =============================================================================


@dataclass
class ConductanceLIFConfig(BaseNeuronConfig):
    """Configuration for conductance-based LIF neuron.

    Inherits device, dtype, seed, dt_ms from BaseConfig via BaseNeuronConfig.
    Inherits tau_mem, v_rest, v_reset, v_threshold, tau_ref from BaseNeuronConfig.

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
    C_m: float = MEMBRANE_CAPACITANCE_STANDARD
    g_L: float = G_LEAK_STANDARD  # τ_m = C_m/g_L = 20ms

    # Reversal potentials (normalized units)
    E_L: float = E_LEAK  # Leak/rest (≈ -65mV scaled to 0)
    E_E: float = E_EXCITATORY  # Excitatory (≈ 0mV, well above threshold)
    E_I: float = E_INHIBITORY  # Inhibitory (≈ -70mV, below rest)

    # Synaptic time constants
    tau_E: float = TAU_SYN_EXCITATORY  # Excitatory (AMPA-like)
    tau_I: float = TAU_SYN_INHIBITORY  # Inhibitory (GABA_A-like)

    # Adaptation (conductance-based)
    tau_adapt: float = 100.0
    adapt_increment: float = 0.0
    E_adapt: float = -0.5  # Adaptation reversal (hyperpolarizing, like slow K+)

    # Noise (enable by default for biological realism)
    noise_std: float = 0.005  # With OU noise, this gives low spontaneous rate but sufficient exploration
    noise_tau_ms: float = 5.0  # Ornstein-Uhlenbeck correlation time (5-10ms biologically)
    use_ou_noise: bool = True  # Use autocorrelated (OU) noise instead of white noise

    @property
    def tau_m(self) -> float:
        """Effective membrane time constant."""
        return self.C_m / self.g_L


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
        >>> neuron = ConductanceLIF(n_neurons=100, config=config, device='cpu')
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
        config: Optional[ConductanceLIFConfig] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config or ConductanceLIFConfig()

        # Determine device
        if device is None:
            device = torch.device("cpu")

        # Register constants as buffers
        self.register_buffer("C_m", torch.tensor(self.config.C_m, device=device))
        self.register_buffer("g_L", torch.tensor(self.config.g_L, device=device))
        self.register_buffer("E_L", torch.tensor(self.config.E_L, device=device))
        self.register_buffer("E_E", torch.tensor(self.config.E_E, device=device))
        self.register_buffer("E_I", torch.tensor(self.config.E_I, device=device))
        self.register_buffer("E_adapt", torch.tensor(self.config.E_adapt, device=device))
        # Per-neuron threshold for intrinsic plasticity support
        self.register_buffer(
            "v_threshold",
            torch.full((n_neurons,), self.config.v_threshold, dtype=torch.float32, device=device),
        )
        self.register_buffer("v_reset", torch.tensor(self.config.v_reset, device=device))

        # Cached decay factors (computed via update_temporal_parameters)
        self.register_buffer("_g_E_decay", torch.tensor(1.0, device=device))
        self.register_buffer("_g_I_decay", torch.tensor(1.0, device=device))
        self.register_buffer("_adapt_decay", torch.tensor(1.0, device=device))
        self._dt_ms: Optional[float] = None  # Tracks current dt

        # State variables
        self.membrane: Optional[torch.Tensor] = None
        self.g_E: Optional[torch.Tensor] = None  # Excitatory conductance
        self.g_I: Optional[torch.Tensor] = None  # Inhibitory conductance
        self.g_adapt: Optional[torch.Tensor] = None  # Adaptation conductance
        self.refractory: Optional[torch.Tensor] = None
        self.ou_noise: Optional[torch.Tensor] = None  # Ornstein-Uhlenbeck noise state

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update decay factors for new timestep.

        Called by brain when dt changes. Recomputes cached decay factors
        for conductances based on new dt and time constants.

        Args:
            dt_ms: New timestep in milliseconds

        Note:
            Uses register_buffer to ensure decay factors are moved with model
            and included in state_dict for checkpointing.
        """
        self._dt_ms = dt_ms

        device = self._g_E_decay.device

        # Recompute excitatory conductance decay: exp(-dt / tau_E)
        self._g_E_decay: torch.Tensor = torch.tensor(
            float(torch.exp(torch.tensor(-dt_ms / self.config.tau_E)).item()),
            device=device,
        )

        # Recompute inhibitory conductance decay: exp(-dt / tau_I)
        self._g_I_decay = torch.tensor(
            float(torch.exp(torch.tensor(-dt_ms / self.config.tau_I)).item()),
            device=device,
        )

        # Recompute adaptation conductance decay: exp(-dt / tau_adapt)
        self._adapt_decay = torch.tensor(
            float(torch.exp(torch.tensor(-dt_ms / self.config.tau_adapt)).item()),
            device=device,
        )

    def reset_state(self) -> None:
        """Reset neuron state to resting potential.

        Initializes all state tensors as 1D [n_neurons] per ADR-005.
        """
        device = self.C_m.device

        # Membrane starts at leak reversal (resting potential)
        dev = torch.device(str(device)) if device else torch.device("cpu")
        membrane = torch.full((self.n_neurons,), self.config.E_L, device=dev, dtype=torch.float32)
        # Remove old attribute/buffer if exists, then register as buffer
        if hasattr(self, "membrane"):
            delattr(self, "membrane")
        self.register_buffer("membrane", membrane, persistent=False)

        # All conductances start at zero
        g_E = torch.zeros(self.n_neurons, device=dev, dtype=torch.float32)
        if hasattr(self, "g_E"):
            delattr(self, "g_E")
        self.register_buffer("g_E", g_E, persistent=False)

        g_I = torch.zeros(self.n_neurons, device=dev, dtype=torch.float32)
        if hasattr(self, "g_I"):
            delattr(self, "g_I")
        self.register_buffer("g_I", g_I, persistent=False)

        g_adapt = torch.zeros(self.n_neurons, device=dev, dtype=torch.float32)
        if hasattr(self, "g_adapt"):
            delattr(self, "g_adapt")
        self.register_buffer("g_adapt", g_adapt, persistent=False)

        refractory = torch.zeros(self.n_neurons, device=dev, dtype=torch.int32)
        if hasattr(self, "refractory"):
            delattr(self, "refractory")
        self.register_buffer("refractory", refractory, persistent=False)

        # Initialize pre-spike membrane (for WTA selection)
        # This is updated before spike check in forward()
        membrane_pre_spike = torch.full((self.n_neurons,), self.config.E_L, device=dev, dtype=torch.float32)
        if hasattr(self, "membrane_pre_spike"):
            delattr(self, "membrane_pre_spike")
        self.register_buffer("membrane_pre_spike", membrane_pre_spike, persistent=False)

        # Initialize OU noise state (autocorrelated noise)
        if self.config.use_ou_noise:
            ou_noise = torch.zeros(self.n_neurons, device=dev, dtype=torch.float32)
            if hasattr(self, "ou_noise"):
                delattr(self, "ou_noise")
            self.register_buffer("ou_noise", ou_noise, persistent=False)
        else:
            self.ou_noise = None

    def adjust_thresholds(
        self, delta: torch.Tensor, min_threshold: float = 0.5, max_threshold: float = 2.0
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
        self.v_threshold: torch.Tensor  # Type annotation for mypy
        self.v_threshold = (self.v_threshold + delta).clamp(min_threshold, max_threshold)  # type: ignore[has-type]

    def forward(
        self, g_exc_input: torch.Tensor, g_inh_input: Optional[torch.Tensor] = None
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
        assert self.refractory is not None, "refractory must be initialized"
        self.refractory = (self.refractory - 1).clamp_(min=0)  # type: ignore[assignment]
        not_refractory = self.refractory == 0

        # Check that temporal parameters have been initialized
        if self._dt_ms is None:
            # Auto-initialize with default dt_ms if not set (for testing/standalone use)
            # In production, DynamicBrain calls update_temporal_parameters() during init
            self.update_temporal_parameters(dt_ms=1.0)  # Default 1ms timestep

        # Update synaptic conductances (in-place decay + add input)
        assert self.g_E is not None, "g_E must be initialized"
        assert self.g_I is not None, "g_I must be initialized"
        self.g_E.mul_(self._g_E_decay).add_(g_exc_input)  # type: ignore[union-attr, arg-type]
        if g_inh_input is not None:
            self.g_I.mul_(self._g_I_decay).add_(g_inh_input)  # type: ignore[union-attr, arg-type]
        else:
            self.g_I.mul_(self._g_I_decay)  # type: ignore[union-attr, arg-type]

        # Decay adaptation conductance (in-place) - handle None case
        if self.g_adapt is not None:
            self.g_adapt.mul_(self._adapt_decay)  # type: ignore[union-attr, arg-type]
        else:
            # Initialize if missing (can happen after loading old checkpoints)
            assert self.membrane is not None, "membrane must be initialized"
            self.g_adapt = torch.zeros(self.n_neurons, device=self.membrane.device)  # type: ignore[union-attr]

        # Compute total conductance (for effective time constant)
        # Pre-add g_L to avoid extra addition
        g_total = self.g_L + self.g_E + self.g_I + self.g_adapt  # type: ignore[operator]

        # Compute equilibrium potential (weighted average of reversals)
        # Fused: V_inf = (g_L*E_L + g_E*E_E + g_I*E_I + g_adapt*E_adapt) / g_total
        # Pre-compute g_L*E_L once (it's a constant)
        V_inf = (
            self.g_L * self.E_L  # type: ignore[operator]
            + self.g_E * self.E_E  # type: ignore[operator]
            + self.g_I * self.E_I  # type: ignore[operator]
            + self.g_adapt * self.E_adapt  # type: ignore[operator]
        ) / g_total  # type: ignore[operator]

        # Effective time constant: τ_eff = C_m / g_total
        # V(t+dt) = V_inf + (V(t) - V_inf) * exp(-dt * g_total / C_m)
        assert self._dt_ms is not None, "dt_ms must be set via update_temporal_parameters"
        decay_factor = torch.exp((-self._dt_ms / self.C_m) * g_total)  # type: ignore[operator]

        # Update membrane for non-refractory neurons
        # Fused: new_V = V_inf + (V - V_inf) * decay = V_inf * (1 - decay) + V * decay
        assert self.membrane is not None, "membrane must be initialized"
        V_diff = self.membrane - V_inf  # type: ignore[operator]
        new_membrane = V_inf + V_diff * decay_factor

        # Add noise only if configured
        if self.config.noise_std > 0:
            if self.config.use_ou_noise:
                # Ornstein-Uhlenbeck (colored) noise: dx = -x/τ*dt + σ*sqrt(2/τ)*dW
                # Discrete: x(t+dt) = x(t)*exp(-dt/τ) + σ*sqrt(1-exp(-2*dt/τ))*randn()
                if self.ou_noise is None:
                    # Initialize OU noise if not present (e.g., after loading old checkpoint)
                    self.ou_noise = torch.zeros_like(self.membrane)

                ou_decay = torch.exp(torch.tensor(-self._dt_ms / self.config.noise_tau_ms))  # type: ignore[arg-type]
                ou_std = self.config.noise_std * torch.sqrt(1 - ou_decay**2)  # Stationary variance
                self.ou_noise = self.ou_noise * ou_decay + torch.randn_like(self.membrane) * ou_std  # type: ignore[assignment, arg-type]
                new_membrane = new_membrane + self.ou_noise  # type: ignore[arg-type]
            else:
                # White noise (legacy, uncorrelated)
                new_membrane = new_membrane + torch.randn_like(self.membrane) * self.config.noise_std  # type: ignore[arg-type]

        # Apply only to non-refractory neurons
        self.membrane = torch.where(not_refractory, new_membrane, self.membrane)  # type: ignore[assignment, arg-type]

        # Store pre-spike membrane for WTA selection (before reset)
        # This is needed because post-spike membrane is reset to v_reset,
        # making it useless for winner-take-all selection
        self.membrane_pre_spike = self.membrane.clone()  # type: ignore[union-attr]

        # Spike generation (bool for biological accuracy and memory efficiency)
        self.v_threshold = torch.tensor(self.v_threshold)  # Ensure it's a Tensor for mypy
        spikes = self.membrane >= self.v_threshold  # type: ignore[operator, arg-type]

        # Combined spike handling: reset membrane AND set refractory in one pass
        # This avoids creating intermediate tensors
        if spikes.any():
            self.membrane = torch.where(spikes, self.v_reset, self.membrane)  # type: ignore[assignment, arg-type]
            # Compute refractory steps from dt_ms
            assert self._dt_ms is not None, "dt_ms must be set"
            ref_steps = int(self.config.tau_ref / self._dt_ms)
            self.refractory = torch.where(
                spikes,
                ref_steps,
                self.refractory,  # scalar broadcasts  # type: ignore[arg-type]
            )
            # Increment adaptation for spiking neurons (need float for arithmetic)
            if self.config.adapt_increment > 0:
                self.g_adapt = self.g_adapt + spikes.float() * self.config.adapt_increment

        return spikes, self.membrane

    def forward_current(self, input_current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        e_e_val = torch.tensor(self.E_E) if not isinstance(self.E_E, torch.Tensor) else self.E_E  # type: ignore[arg-type]
        e_l_val = torch.tensor(self.E_L) if not isinstance(self.E_L, torch.Tensor) else self.E_L  # type: ignore[arg-type]
        e_i_val = torch.tensor(self.E_I) if not isinstance(self.E_I, torch.Tensor) else self.E_I  # type: ignore[arg-type]
        g_exc = torch.clamp(input_current, min=0) / (e_e_val - e_l_val)  # type: ignore[operator]
        g_inh = torch.clamp(-input_current, min=0) / (e_l_val - e_i_val)  # type: ignore[operator]

        return self.forward(g_exc, g_inh)

    def get_state(self) -> dict[str, Optional[torch.Tensor]]:
        """Get current neuron state for analysis/saving."""
        return {
            "membrane": self.membrane.clone() if self.membrane is not None else None,
            "g_E": self.g_E.clone() if self.g_E is not None else None,
            "g_I": self.g_I.clone() if self.g_I is not None else None,
            "g_adapt": self.g_adapt.clone() if self.g_adapt is not None else None,
            "refractory": self.refractory.clone() if self.refractory is not None else None,
            "ou_noise": self.ou_noise.clone() if self.ou_noise is not None else None,
        }

    def load_state(self, state: dict[str, Optional[torch.Tensor]]) -> None:
        """Restore neuron state from checkpoint.

        Args:
            state: Dictionary from get_state()
        """
        # Infer device from module's parameters/buffers (most reliable for device transfer)
        # Use C_m buffer which is always present and on the correct device after .to()
        if hasattr(self, "C_m") and self.C_m is not None:
            device = self.C_m.device
        elif self.membrane is not None:
            device = self.membrane.device
        elif state["membrane"] is not None:
            device = state["membrane"].device
        else:
            device = torch.device("cpu")

        dev_cast = torch.device(str(device))  # Cast for type compatibility
        if state["membrane"] is not None:
            self.membrane = state["membrane"].to(dev_cast)  # type: ignore[assignment, arg-type]
        if state["g_E"] is not None:
            self.g_E = state["g_E"].to(dev_cast)  # type: ignore[assignment, arg-type]
        if state["g_I"] is not None:
            self.g_I = state["g_I"].to(dev_cast)  # type: ignore[assignment, arg-type]
        if state["g_adapt"] is not None:
            self.g_adapt = state["g_adapt"].to(dev_cast)  # type: ignore[assignment, arg-type]
        if state["refractory"] is not None:
            self.refractory = state["refractory"].to(dev_cast)  # type: ignore[assignment, arg-type]
        # Restore OU noise if present in state
        if state.get("ou_noise") is not None:
            self.ou_noise = state["ou_noise"].to(dev_cast)  # type: ignore[assignment, arg-type]

    def grow_neurons(self, n_new: int) -> None:
        """Grow neuron population by adding new neurons.

        Preserves existing neuron state and expands all state tensors.
        New neurons start at resting potential with zero conductances.
        This is more efficient than recreating the entire neuron module.

        Args:
            n_new: Number of neurons to add

        Example:
            >>> neurons = ConductanceLIF(n_neurons=100, config=config, device='cpu')
            >>> neurons.reset_state()
            >>> # ... training ...
            >>> neurons.grow_neurons(20)  # Now 120 neurons, old state preserved
            >>> assert neurons.n_neurons == 120
        """
        if n_new <= 0:
            return

        old_n = self.n_neurons
        new_n = old_n + n_new
        device = self.C_m.device

        # Update neuron count
        self.n_neurons = new_n

        # Get device from existing tensors
        assert self.v_threshold is not None, "v_threshold must be initialized"
        device = self.v_threshold.device  # type: ignore[union-attr]
        dev_cast = torch.device(str(device))  # Cast for type compatibility

        # Expand per-neuron threshold buffer (registered buffer)
        new_thresholds = torch.full(
            (n_new,), self.config.v_threshold, dtype=torch.float32, device=dev_cast  # type: ignore[arg-type]
        )
        self.v_threshold = torch.cat([self.v_threshold, new_thresholds])  # type: ignore[assignment, list-item]

        # Expand state tensors (only if already initialized)
        if self.membrane is not None:
            # Preserve old state, initialize new neurons at resting potential
            new_membrane = torch.full((n_new,), self.config.E_L, device=dev_cast, dtype=torch.float32)  # type: ignore[arg-type]
            self.membrane = torch.cat([self.membrane, new_membrane])  # type: ignore[assignment, list-item]

            # Zero conductances for new neurons
            new_zeros = torch.zeros(n_new, device=dev_cast, dtype=torch.float32)  # type: ignore[arg-type]
            assert self.g_E is not None and self.g_I is not None and self.g_adapt is not None
            self.g_E = torch.cat([self.g_E, new_zeros])  # type: ignore[assignment, list-item]
            self.g_I = torch.cat([self.g_I, new_zeros])  # type: ignore[assignment, list-item]
            self.g_adapt = torch.cat([self.g_adapt, new_zeros])  # type: ignore[assignment, list-item]

            # Zero refractory for new neurons
            new_ref = torch.zeros(n_new, device=dev_cast, dtype=torch.int32)  # type: ignore[arg-type]
            assert self.refractory is not None
            self.refractory = torch.cat([self.refractory, new_ref])  # type: ignore[assignment, list-item]

            # Extend OU noise for new neurons (if using OU noise)
            if hasattr(self, "ou_noise") and self.ou_noise is not None:
                new_ou = torch.zeros(n_new, device=dev_cast, dtype=torch.float32)  # type: ignore[arg-type]
                self.ou_noise = torch.cat([self.ou_noise, new_ou])  # type: ignore[assignment, list-item]

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
