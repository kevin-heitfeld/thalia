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

**Type Safety**:
================
This module uses type aliases from thalia.units for dimensional analysis:
- ConductanceTensor: Synaptic/membrane conductances (≥ 0)
- VoltageTensor: Membrane potentials
- CurrentTensor: Membrane currents (derived from g × (E - V))

Type checkers (mypy/pyright) will catch unit mismatches at development time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from thalia.units import ConductanceTensor, VoltageTensor

# =============================================================================
# CONDUCTANCE-BASED LIF NEURON
# =============================================================================


@dataclass
class ConductanceLIFConfig:
    """Configuration for conductance-based LIF neuron.

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

        tau_mem: Membrane time constant in ms (default: 20.0)
            Controls how quickly the membrane potential decays toward rest.
            Larger values = slower decay = longer memory of inputs.
            Standard pyramidal neurons: 15-30ms
            Fast-spiking interneurons: 5-15ms

        v_rest: Resting membrane potential (default: 0.0)
            Membrane potential with no input.
            In normalized units where threshold = 1.0
            Biological equivalent: ~-65mV

        v_reset: Reset potential after spike (default: 0.0)
            Where membrane is set after spike emission.
            Typically equals v_rest for simplicity.
            Biological equivalent: ~-70mV

        v_threshold: Spike threshold (default: 1.0)
            Membrane potential at which spike is emitted.
            In normalized units (threshold = 1.0 by convention)
            Biological equivalent: ~-55mV

        tau_ref: Absolute refractory period in ms (default: 2.0)
            Duration during which neuron cannot fire after a spike.
            Biological range: 1-5ms depending on neuron type

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

    device: str = "cpu"

    tau_mem: float = 20.0  # Membrane time constant (ms)
    v_rest: float = 0.0  # Resting potential
    v_reset: float = 0.0  # Reset after spike
    v_threshold: float = 1.0  # Spike threshold
    tau_ref: float = 5.0  # Refractory period (ms)

    # Membrane properties
    C_m: float = 1.0
    g_L: float = 0.05  # τ_m = C_m/g_L = 20ms

    # Reversal potentials (normalized units)
    E_L: float = 0.0  # Leak/rest (≈ -65mV scaled to 0)
    E_E: float = 3.0  # Excitatory (≈ 0mV, well above threshold)
    E_I: float = -0.5  # Inhibitory (≈ -70mV, below rest)

    # Synaptic time constants
    tau_E: float = 5.0  # Excitatory (AMPA-like)
    tau_I: float = 10.0  # Inhibitory (GABA_A-like)

    # Adaptation (conductance-based)
    tau_adapt: float = 100.0
    adapt_increment: float = 0.0
    E_adapt: float = -0.5  # Adaptation reversal (hyperpolarizing, like slow K+)

    # Noise (enable by default for biological realism)
    noise_std: float = 0.02  # Increased to break pathological synchronization (2% of threshold)
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
        self.device = device

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

        # Per-neuron refractory period (heterogeneous to prevent synchronization)
        # Biological neurons have variable refractory periods (3-7ms typical range)
        # This naturally decorrelates population activity
        tau_ref_mean = self.config.tau_ref
        tau_ref_std = tau_ref_mean * 0.4  # 40% coefficient of variation (increased from 20% for stronger decorrelation)
        self.register_buffer(
            "tau_ref_per_neuron",
            torch.normal(
                mean=tau_ref_mean,
                std=tau_ref_std,
                size=(n_neurons,),
                device=device
            ).clamp(min=3.0, max=tau_ref_mean * 1.6)  # Clamp to minimum 3ms (biological floor) and 160% of mean
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
        self.v_threshold = (self.v_threshold + delta).clamp(min_threshold, max_threshold)

    def _get_additional_conductances(self) -> list[tuple[torch.Tensor, float]]:
        """Get additional intrinsic conductances beyond g_E, g_I, g_adapt.

        Override this in specialized neurons to add intrinsic conductances
        (e.g., I_h pacemaker, SK channels, gap junctions).

        Returns:
            List of (conductance_tensor, reversal_potential) tuples.
            Each conductance is shape [n_neurons].

        Example:
            return [
                (self.g_ih, 0.8),  # I_h pacemaker
                (self.g_sk * self.sk_activation, -0.5),  # SK channels
            ]
        """
        return []

    def forward(
        self,
        g_exc_input: ConductanceTensor,
        g_inh_input: Optional[ConductanceTensor] = None,
    ) -> tuple[torch.Tensor, VoltageTensor]:
        """Process one timestep of conductance input.

        **CRITICAL**: Inputs MUST be conductances, NOT currents!
        If you have currents, you cannot convert them to conductances without
        knowing the driving force. Use conductance-based synaptic weights instead.

        Args:
            g_exc_input: Excitatory conductance input, shape [n_neurons] (1D, ADR-005)
                Units: Conductance (normalized by g_L)
                This is ADDED to the excitatory conductance state.
            g_inh_input: Inhibitory conductance input, shape [n_neurons] (1D, ADR-005)
                Units: Conductance (normalized by g_L)
                Optional. If None, no inhibitory input is applied.

        Returns:
            spikes: Binary spike tensor, shape [n_neurons] (1D bool, ADR-004/005)
            membrane: Membrane potentials after update, shape [n_neurons] (1D)
        """
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
        if self.refractory is None:
            # Initialize with HETEROGENEOUS refractory states to prevent synchronization
            # Random initial refractory values prevent all neurons exiting refractory simultaneously
            # This is biologically realistic - neurons are never perfectly synchronized initially
            # Use per-neuron tau_ref for heterogeneous refractory periods
            max_ref_steps = int(self.tau_ref_per_neuron.max().item() / 1.0)  # Assume dt=1ms for init
            self.refractory = torch.randint(
                0, max(1, max_ref_steps),
                (self.n_neurons,),
                dtype=torch.int32,
                device=g_exc_input.device
            )
        self.refractory = (self.refractory - 1).clamp_(min=0)
        not_refractory = self.refractory == 0

        # Check that temporal parameters have been initialized
        if self._dt_ms is None:
            # Auto-initialize with default dt_ms if not set (for testing/standalone use)
            self.update_temporal_parameters(dt_ms=1.0)  # Default 1ms timestep

        # Update synaptic conductances (in-place decay + add input)
        if self.g_E is None:
            # Initialize if missing (first forward pass or after loading old checkpoints)
            self.g_E = torch.zeros(self.n_neurons, device=g_exc_input.device)
        if self.g_I is None:
            self.g_I = torch.zeros(self.n_neurons, device=g_exc_input.device)

        self.g_E.mul_(self._g_E_decay).add_(g_exc_input)
        # CRITICAL FIX (2025-01): Clamp conductances to prevent numerical issues
        # Conductances are physical quantities and cannot be negative
        self.g_E.clamp_(min=0.0)

        if g_inh_input is not None:
            self.g_I.mul_(self._g_I_decay).add_(g_inh_input)
        else:
            self.g_I.mul_(self._g_I_decay)

        self.g_I.clamp_(min=0.0)

        # Decay adaptation conductance (in-place) - handle None case
        if self.g_adapt is not None:
            self.g_adapt.mul_(self._adapt_decay)
        else:
            # Initialize if missing
            self.g_adapt = torch.zeros(self.n_neurons, device=g_exc_input.device)

        # Compute total conductance (for effective time constant)
        # Start with standard conductances
        g_total = self.g_L + self.g_E + self.g_I + self.g_adapt

        # Compute equilibrium potential (weighted average of reversals)
        # Start with standard conductances
        V_inf_numerator = (
            self.g_L * self.E_L
            + self.g_E * self.E_E
            + self.g_I * self.E_I
            + self.g_adapt * self.E_adapt
        )

        # Add intrinsic conductances from specialized neurons (e.g., I_h, SK, gap junctions)
        additional_conductances = self._get_additional_conductances()
        for g, E_rev in additional_conductances:
            g_total = g_total + g
            V_inf_numerator = V_inf_numerator + g * E_rev

        V_inf = V_inf_numerator / g_total

        # Effective time constant: τ_eff = C_m / g_total
        # V(t+dt) = V_inf + (V(t) - V_inf) * exp(-dt * g_total / C_m)
        assert self._dt_ms is not None, "dt_ms must be set via update_temporal_parameters"
        decay_factor = torch.exp((-self._dt_ms / self.C_m) * g_total)

        # Update membrane for non-refractory neurons
        # Fused: new_V = V_inf + (V - V_inf) * decay = V_inf * (1 - decay) + V * decay
        if self.membrane is None:
            # Initialize membrane with HETEROGENEOUS values to prevent synchronization
            # Uniform distribution between E_L and partway to threshold prevents pathological synchrony
            # This is critical for biological realism - neurons never start identically
            # Handle both normalized units (threshold > E_L) and absolute mV (threshold < E_L)
            e_l_val = self.E_L.item()
            thresh_val = self.v_threshold[0].item()

            # Initialize between E_L and 50% toward threshold
            if thresh_val > e_l_val:
                # Normalized units (threshold above rest)
                v_min = e_l_val
                v_max = e_l_val + 0.5 * (thresh_val - e_l_val)
            else:
                # Absolute mV scale (threshold below rest, e.g., -55mV threshold, 0mV rest)
                # In this case, we want to start slightly depolarized from E_L
                v_min = e_l_val + 0.5 * (thresh_val - e_l_val)  # Halfway to threshold
                v_max = e_l_val  # Up to rest

            v_init = torch.empty((self.n_neurons,), device=g_exc_input.device)
            v_init.uniform_(v_min, v_max)
            self.membrane = v_init

        V_diff = self.membrane - V_inf
        new_membrane = V_inf + V_diff * decay_factor

        # Add noise only if configured
        if self.config.noise_std > 0:
            if self.config.use_ou_noise:
                # Ornstein-Uhlenbeck (colored) noise: dx = -x/τ*dt + σ*sqrt(2/τ)*dW
                # Discrete: x(t+dt) = x(t)*exp(-dt/τ) + σ*sqrt(1-exp(-2*dt/τ))*randn()
                if self.ou_noise is None:
                    # Initialize OU noise if not present (e.g., after loading old checkpoint)
                    self.ou_noise = torch.zeros_like(self.membrane)

                ou_decay = torch.exp(torch.tensor(-self._dt_ms / self.config.noise_tau_ms))
                ou_std = self.config.noise_std * torch.sqrt(1 - ou_decay**2)  # Stationary variance
                self.ou_noise = self.ou_noise * ou_decay + torch.randn_like(self.membrane) * ou_std
                new_membrane = new_membrane + self.ou_noise
            else:
                # White noise (legacy, uncorrelated)
                new_membrane = new_membrane + torch.randn_like(self.membrane) * self.config.noise_std

        # Update membrane for ALL neurons (including refractory)
        # Neurons continue integrating during refractory, they just can't spike
        # This allows charge buildup during refractory period for fast re-spiking
        self.membrane = new_membrane

        # Store pre-spike membrane for WTA selection (before reset)
        # This is needed because post-spike membrane is reset to v_reset,
        # making it useless for winner-take-all selection
        self.membrane_pre_spike = self.membrane.clone()

        # Spike generation (bool for biological accuracy and memory efficiency)
        # Only allow spikes from non-refractory neurons
        above_threshold = self.membrane >= self.v_threshold
        spikes = above_threshold & not_refractory  # Can only spike if not refractory

        # Combined spike handling: reset membrane AND set refractory in one pass
        # This avoids creating intermediate tensors
        if spikes.any():
            self.membrane = torch.where(spikes, self.v_reset, self.membrane)
            # Compute per-neuron refractory steps from heterogeneous tau_ref
            assert self._dt_ms is not None, "dt_ms must be set"
            ref_steps_per_neuron = (self.tau_ref_per_neuron / self._dt_ms).int()
            self.refractory = torch.where(
                spikes,
                ref_steps_per_neuron,  # Now per-neuron, not scalar!
                self.refractory,
            )
            # Increment adaptation for spiking neurons (need float for arithmetic)
            if self.config.adapt_increment > 0:
                self.g_adapt = self.g_adapt + spikes.float() * self.config.adapt_increment

        return spikes, self.membrane

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
