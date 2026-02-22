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

Type checkers (mypy/pyright) will catch unit mismatches at development time.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from typing import Optional, Union
import math

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
            Can be a scalar (same for all neurons) or tensor (per-neuron).
            Controls how quickly the membrane potential decays toward rest.
            Larger values = slower decay = longer memory of inputs.
            Standard pyramidal neurons: 15-30ms
            Fast-spiking interneurons: 3-8ms
            PFC delay neurons: 100-500ms

        v_rest: Resting membrane potential (default: 0.0)
            Membrane potential with no input.
            In normalized units where threshold = 1.0
            Biological equivalent: ~-65mV

        v_reset: Reset potential after spike (default: 0.0)
            Where membrane is set after spike emission.
            Typically equals v_rest for simplicity.
            Biological equivalent: ~-70mV

        v_threshold: Spike threshold (default: 1.0)
            Can be a scalar (same for all neurons) or tensor (per-neuron).
            Membrane potential at which spike is emitted.
            In normalized units (threshold = 1.0 by convention)
            Biological equivalent: ~-55mV
            Per-neuron heterogeneity (5-15% CV) prevents synchronization.

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

    # RNG configuration (counter-based per-neuron for true independence)
    region_name: str  # Brain region identifier (e.g., "cortex", "striatum")
    population_name: str  # Population identifier (e.g., "L4", "D1_MSN")
    rng_seed: Optional[int] = None  # Master seed for deriving per-neuron seeds (None = hash from identity)

    device: str = "cpu"

    tau_mem: Union[float, torch.Tensor] = 20.0  # Membrane time constant (ms) - scalar or per-neuron
    v_rest: float = 0.0  # Resting potential
    v_reset: float = 0.0  # Reset after spike
    v_threshold: Union[float, torch.Tensor] = 1.0  # Spike threshold - scalar or per-neuron
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

    # NMDA conductance (slow excitation for temporal integration)
    tau_nmda: float = 100.0  # NMDA decay time constant (80-150ms biologically)
    E_nmda: float = 3.0  # NMDA reversal potential (same as AMPA)
    nmda_ratio: float = 0.0  # NMDA/(AMPA+NMDA) ratio - REDUCED from 0.3 to prevent accumulation
    # TODO: Consider making nmda_ratio a per-source parameter for more biological realism (different synapses have different NMDA contributions)

    # Adaptation (conductance-based)
    tau_adapt: float = 100.0
    adapt_increment: float = 0.0
    E_adapt: float = -0.5  # Adaptation reversal (hyperpolarizing, like slow K+)

    # Noise (enable by default for biological realism)
    noise_std: float = 0.08  # 8% of threshold - balanced for stability without over-synchronization
    noise_tau_ms: float = 3.0  # 3ms temporal correlation
    use_ou_noise: bool = True  # Use autocorrelated (OU) noise instead of white noise

    # T-type Ca²⁺ channels (for thalamic pacemaker neurons)
    # Enables rebound bursting: hyperpolarization → de-inactivate → depolarizing rebound → burst
    # Creates intrinsic 7-14 Hz oscillations (spindles/alpha)
    enable_t_channels: bool = False  # Enable for thalamic relay and TRN neurons
    g_T: float = 0.15  # T-channel conductance (when enabled)
    E_Ca: float = 4.0  # Calcium reversal potential (highly depolarizing)
    tau_h_T_ms: float = 50.0  # T-channel de-inactivation time constant (40-80ms)
    V_half_h_T: float = -0.3  # Half-activation voltage for h_T (hyperpolarized)
    k_h_T: float = 0.15  # Slope of h_T activation curve

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

    @property
    def device(self) -> torch.device:
        """Device where tensors are located."""
        return torch.device(self.config.device)

    def __init__(self, n_neurons: int, config: ConductanceLIFConfig):
        """Initialize conductance-based LIF neuron."""
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config

        # Register constants as buffers
        self.register_buffer("C_m", torch.tensor(self.config.C_m, device=self.device))
        self.register_buffer("g_L", torch.tensor(self.config.g_L, device=self.device))
        self.register_buffer("E_L", torch.tensor(self.config.E_L, device=self.device))
        self.register_buffer("E_E", torch.tensor(self.config.E_E, device=self.device))
        self.register_buffer("E_I", torch.tensor(self.config.E_I, device=self.device))
        self.register_buffer("E_nmda", torch.tensor(self.config.E_nmda, device=self.device))
        self.register_buffer("E_adapt", torch.tensor(self.config.E_adapt, device=self.device))

        # Per-neuron tau_mem for heterogeneous time constants (frequency diversity)
        # This enables different neuron types to resonate at different frequencies:
        # - Fast-spiking (3-8ms): Gamma oscillations (40-80 Hz)
        # - Standard pyramidal (15-30ms): Alpha/beta (8-15 Hz)
        # - PFC delay neurons (100-500ms): Persistent activity (<2 Hz)
        if isinstance(config.tau_mem, (int, float)):
            # Scalar: same tau_mem for all neurons
            self.register_buffer(
                "tau_mem_per_neuron",
                torch.full((n_neurons,), float(config.tau_mem), dtype=torch.float32, device=self.device),
            )
        else:
            # Tensor: per-neuron tau_mem (already specified)
            assert config.tau_mem.shape[0] == n_neurons, (
                f"tau_mem tensor has {config.tau_mem.shape[0]} elements but expected {n_neurons}"
            )
            self.register_buffer(
                "tau_mem_per_neuron",
                config.tau_mem.to(device=self.device, dtype=torch.float32),
            )

        # Per-neuron threshold for intrinsic plasticity support
        # Supports both scalar (uniform) and tensor (per-neuron heterogeneity)
        if isinstance(config.v_threshold, (int, float)):
            # Scalar: same threshold for all neurons
            self.register_buffer(
                "v_threshold",
                torch.full((n_neurons,), float(config.v_threshold), dtype=torch.float32, device=self.device),
            )
        else:
            # Tensor: per-neuron threshold (already specified)
            assert config.v_threshold.shape[0] == n_neurons, (
                f"v_threshold tensor has {config.v_threshold.shape[0]} elements but expected {n_neurons}"
            )
            self.register_buffer(
                "v_threshold",
                config.v_threshold.to(device=self.device, dtype=torch.float32),
            )

        # HOMEOSTATIC INTRINSIC EXCITABILITY: Per-neuron leak conductance scaling
        # Biology: Neurons modulate ion channel densities to maintain target firing rate
        # Lower g_L → higher input resistance → more excitable (Turrigiano & Nelson 2004)
        # This is the CORRECT way to implement homeostatic gain (not multiplying synaptic conductances!)
        # Start at 1.0 (normal leak), will adapt based on activity
        self.register_buffer(
            "g_L_scale",
            torch.ones(n_neurons, dtype=torch.float32, device=self.device),
        )

        # =============================================================================
        # RNG INDEPENDENCE: Per-Neuron Counter-Based Seeds
        # =============================================================================
        # CRITICAL: Each neuron gets a unique seed to eliminate spurious correlations
        # Noise generation is f(seed_i, timestep) using Philox algorithm

        # Derive base seed from identity (refactor-proof, order-independent)
        if config.rng_seed is not None:
            base_seed = config.rng_seed
        else:
            # Use md5 for stable hashing across Python sessions (hash() is salted)
            key_string = f"{config.region_name}_{config.population_name}"
            key_hash = int(md5(key_string.encode()).hexdigest()[:8], 16)
            base_seed = key_hash % (2**31)  # Fit in int32

        # Create per-neuron seeds (each neuron gets unique seed)
        # Shape: (n_neurons,) - one seed per neuron
        neuron_seeds = []
        for neuron_id in range(self.n_neurons):
            # Hierarchical key: hash(base_seed, neuron_id)
            neuron_key = f"{base_seed}_{neuron_id}"
            neuron_hash = int(md5(neuron_key.encode()).hexdigest()[:8], 16)
            neuron_seeds.append(neuron_hash % (2**31))

        # Store as tensor for vectorized Philox operations (directly on device)
        self.register_buffer(
            "neuron_seeds",
            torch.tensor(neuron_seeds, dtype=torch.int64, device=self.device)
        )

        # Timestep counter for temporal determinism (counter-based RNG paradigm)
        self.rng_timestep = 0  # NOT a buffer - changes every forward pass

        # Per-neuron refractory period (heterogeneous to prevent synchronization)
        # Biological neurons have variable refractory periods (3-7ms typical range)
        # This naturally decorrelates population activity
        # CRITICAL: Higher CV (60%) for fast neurons prevents pathological synchrony
        # observed at 330 Hz with uniform parameters
        tau_ref_mean = self.config.tau_ref
        # Adaptive CV: fast neurons (tau_ref < 4ms) get 60% CV, others get 40% CV
        # This specifically targets fast-spiking interneurons that were causing 330 Hz lockstep
        tau_ref_cv = 0.60 if tau_ref_mean < 4.0 else 0.40
        tau_ref_std = tau_ref_mean * tau_ref_cv

        # Set biologically-appropriate bounds: fast neurons (< 4ms) clamped tighter
        if tau_ref_mean < 4.0:
            # Fast neurons: 2.5-5ms (prevents >150 Hz firing)
            min_ref, max_ref = 2.5, 5.0
        else:
            # Standard neurons: 2.5-8ms (allows proper refractory variability)
            min_ref, max_ref = 2.5, tau_ref_mean * 1.6

        self.register_buffer(
            "tau_ref_per_neuron",
            torch.normal(
                mean=tau_ref_mean,
                std=tau_ref_std,
                size=(n_neurons,),
                device=self.device
            ).clamp(min=min_ref, max=max_ref)
        )
        self.register_buffer("v_reset", torch.tensor(self.config.v_reset, device=self.device))

        # Cached decay factors (computed via update_temporal_parameters)
        self.register_buffer("_g_E_decay", torch.tensor(1.0, device=self.device))
        self.register_buffer("_g_I_decay", torch.tensor(1.0, device=self.device))
        self.register_buffer("_g_nmda_decay", torch.tensor(1.0, device=self.device))
        self.register_buffer("_adapt_decay", torch.tensor(1.0, device=self.device))
        self._dt_ms: Optional[float] = None  # Tracks current dt

        # State variables
        self.membrane: Optional[torch.Tensor] = None
        self.g_E: Optional[torch.Tensor] = None  # Excitatory conductance (AMPA)
        self.g_I: Optional[torch.Tensor] = None  # Inhibitory conductance (GABA_A)
        self.g_nmda: Optional[torch.Tensor] = None  # NMDA conductance (slow excitation)
        self.g_adapt: Optional[torch.Tensor] = None  # Adaptation conductance
        self.refractory: Optional[torch.Tensor] = None
        self.ou_noise: Optional[torch.Tensor] = None  # Ornstein-Uhlenbeck noise state

        # T-type Ca²⁺ channel state (for thalamic pacemaker neurons)
        self.h_T: Optional[torch.Tensor] = None  # T-channel de-inactivation variable (0-1)
        self._h_T_decay: Optional[torch.Tensor] = None  # Cached decay factor

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update decay factors for new timestep.

        Called by brain when dt changes. Recomputes cached decay factors
        for conductances based on new dt and time constants.

        Args:
            dt_ms: New timestep in milliseconds
        """
        self._dt_ms = dt_ms

        device = self.device

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

        # Recompute NMDA conductance decay: exp(-dt / tau_nmda)
        self._g_nmda_decay = torch.tensor(
            float(torch.exp(torch.tensor(-dt_ms / self.config.tau_nmda)).item()),
            device=device,
        )

        # Recompute adaptation conductance decay: exp(-dt / tau_adapt)
        self._adapt_decay = torch.tensor(
            float(torch.exp(torch.tensor(-dt_ms / self.config.tau_adapt)).item()),
            device=device,
        )

        # Recompute T-channel de-inactivation decay (if enabled)
        if self.config.enable_t_channels:
            self._h_T_decay = torch.tensor(
                float(torch.exp(torch.tensor(-dt_ms / self.config.tau_h_T_ms)).item()),
                device=device,
            )

    def adjust_thresholds(
        self,
        delta: torch.Tensor,
        min_threshold: float,
        max_threshold: float,
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
        self.v_threshold = (self.v_threshold + delta).clamp(min=min_threshold, max=max_threshold)

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

    # =============================================================================
    # RNG INDEPENDENCE: Vectorized Philox Implementation
    # =============================================================================
    # Counter-based RNG using Philox4x32 algorithm for GPU-parallelized per-neuron noise

    def _philox_uniform(self, counters: torch.Tensor) -> torch.Tensor:
        """Philox 4x32 -> uniform [0,1) using full int64 math."""
        W0, W1 = 0x9E3779B9, 0xBB67AE85
        rounds = 10

        x = counters.clone().to(torch.int64)

        for r in range(rounds):
            lo = x & 0xFFFFFFFF
            hi = (x >> 32) & 0xFFFFFFFF

            lo = (lo * W0) & 0xFFFFFFFF
            hi = (hi * W1) & 0xFFFFFFFF

            # mix hi/lo without truncating the original x
            x = ((hi << 32) | lo) ^ (W0 * (r + 1))

        # normalize
        u = ((x & 0xFFFFFFFF) + 1).float() / (2**32 + 2)  # avoid exact 0 or 1
        return u

    def _generate_vectorized_noise_philox(self) -> torch.Tensor:
        """Generate fully independent per-neuron Gaussian noise (std=1) using Philox.

        Returns:
            (n_neurons,) tensor of Gaussian noise.
        """
        # Create unique counters per neuron and timestep
        counters = self.neuron_seeds.to(torch.int64) * 2654435761 + self.rng_timestep

        # Generate two independent uniforms per neuron
        u1 = self._philox_uniform(counters)
        u2 = self._philox_uniform(counters + 1)  # offset for independence

        # Box-Muller transform: uniform -> standard Gaussian
        z = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2 * math.pi * u2)
        return z

    def __call__(self, *args, **kwds):
        assert False, f"{self.__class__.__name__} instances should not be called directly. Use forward() instead."
        return super().__call__(*args, **kwds)

    @torch.no_grad()
    def forward(
        self,
        g_ampa_input: Optional[ConductanceTensor],
        g_gaba_a_input: Optional[ConductanceTensor],
        g_nmda_input: Optional[ConductanceTensor],
        g_gap_input: Optional[ConductanceTensor] = None,
        E_gap_reversal: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, VoltageTensor]:
        """Process one timestep of conductance input.

        **CRITICAL**: Inputs MUST be conductances, NOT currents!
        If you have currents, you cannot convert them to conductances without
        knowing the driving force. Use conductance-based synaptic weights instead.

        **RNG Independence** (Critical for Biological Accuracy):
        Each neuron gets an independent RNG stream seeded from hash(region, population, neuron_id).
        Noise is generated as f(seed_i, timestep) using counter-based approach (no reseeding).
        This eliminates spurious cross-region and intra-population correlations from shared noise.

        Biological correlations must be added explicitly (e.g., shared inputs, gap junctions).

        Args:
            g_ampa_input: Fast excitatory conductance (AMPA), shape [n_neurons] (1D, ADR-005)
                Units: Conductance (normalized by g_L)
                Tau ~5ms for rapid synaptic transmission
            g_gaba_a_input: Fast inhibitory conductance (GABA_A), shape [n_neurons] (1D, ADR-005)
                Units: Conductance (normalized by g_L)
                Tau ~10ms for rapid inhibition
            g_nmda_input: Slow excitatory conductance (NMDA), shape [n_neurons] (1D, ADR-005)
                Units: Conductance (normalized by g_L)
                Tau ~100ms for temporal integration
            g_gap_input: Gap junction conductance, shape [n_neurons] (1D, ADR-005, optional)
                Units: Conductance (normalized by g_L)
                Direct electrical coupling between neurons (instantaneous)
                Biology: Ultra-fast synchronization (<0.1ms)
            E_gap_reversal: Gap junction reversal potential, shape [n_neurons] (1D, optional)
                Dynamic reversal = weighted average of neighbor voltages
                Must be provided if g_gap_input is provided
                Physics: I_gap = g_gap × (E_gap - V), where E_gap = avg(V_neighbors)

        Returns:
            spikes: Binary spike tensor, shape [n_neurons] (1D bool, ADR-004/005)
            membrane: Membrane potentials after update, shape [n_neurons] (1D)
        """
        # Advance timestep counter (used for counter-based noise generation)
        # CRITICAL: This MUST be at the start before any noise generation
        self.rng_timestep += 1

        if g_ampa_input is None:
            g_ampa_input = ConductanceTensor(torch.zeros(self.n_neurons, device=self.device))
        if g_gaba_a_input is None:
            g_gaba_a_input = ConductanceTensor(torch.zeros(self.n_neurons, device=self.device))
        if g_nmda_input is None:
            g_nmda_input = ConductanceTensor(torch.zeros(self.n_neurons, device=self.device))

        assert g_ampa_input.dim() == 1, (
            f"ConductanceLIF.forward: g_ampa_input must be 1D [n_neurons], "
            f"got shape {g_ampa_input.shape}. See ADR-005: No Batch Dimension."
        )
        assert g_ampa_input.shape[0] == self.n_neurons, (
            f"ConductanceLIF.forward: g_ampa_input has {g_ampa_input.shape[0]} neurons "
            f"but expected {self.n_neurons}."
        )
        assert g_gaba_a_input.dim() == 1, (
            f"ConductanceLIF.forward: g_gaba_a_input must be 1D [n_neurons], "
            f"got shape {g_gaba_a_input.shape}. See ADR-005: No Batch Dimension."
        )
        assert g_gaba_a_input.shape[0] == self.n_neurons, (
            f"ConductanceLIF.forward: g_gaba_a_input has {g_gaba_a_input.shape[0]} neurons "
            f"but expected {self.n_neurons}."
        )
        assert g_nmda_input.dim() == 1, (
            f"ConductanceLIF.forward: g_nmda_input must be 1D [n_neurons], "
            f"got shape {g_nmda_input.shape}. See ADR-005: No Batch Dimension."
        )
        assert g_nmda_input.shape[0] == self.n_neurons, (
            f"ConductanceLIF.forward: g_nmda_input has {g_nmda_input.shape[0]} neurons "
            f"but expected {self.n_neurons}."
        )

        # Validate gap junction inputs (both must be provided together)
        if g_gap_input is not None or E_gap_reversal is not None:
            assert g_gap_input is not None and E_gap_reversal is not None, (
                "ConductanceLIF.forward: g_gap_input and E_gap_reversal must both be provided or both be None"
            )
            assert g_gap_input.dim() == 1, (
                f"ConductanceLIF.forward: g_gap_input must be 1D [n_neurons], "
                f"got shape {g_gap_input.shape}. See ADR-005: No Batch Dimension."
            )
            assert g_gap_input.shape[0] == self.n_neurons, (
                f"ConductanceLIF.forward: g_gap_input has {g_gap_input.shape[0]} neurons "
                f"but expected {self.n_neurons}."
            )
            assert E_gap_reversal.dim() == 1, (
                f"ConductanceLIF.forward: E_gap_reversal must be 1D [n_neurons], "
                f"got shape {E_gap_reversal.shape}."
            )
            assert E_gap_reversal.shape[0] == self.n_neurons, (
                f"ConductanceLIF.forward: E_gap_reversal has {E_gap_reversal.shape[0]} neurons "
                f"but expected {self.n_neurons}."
            )

        # Check that temporal parameters have been initialized
        if self._dt_ms is None:
            # Auto-initialize with default dt_ms if not set (for testing/standalone use)
            self.update_temporal_parameters(dt_ms=1.0)  # Default 1ms timestep
        assert self._dt_ms is not None, "dt_ms must be set via update_temporal_parameters"

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
                device=self.device
            )
        self.refractory = (self.refractory - 1).clamp_(min=0)
        not_refractory = self.refractory == 0

        # Update synaptic conductances (in-place decay + add input)
        if self.g_E is None:
            # Initialize if missing (first forward pass or after loading old checkpoints)
            self.g_E = torch.zeros(self.n_neurons, device=self.device)
        if self.g_I is None:
            self.g_I = torch.zeros(self.n_neurons, device=self.device)

        self.g_E.mul_(self._g_E_decay).add_(g_ampa_input)
        # CRITICAL FIX (2025-01): Clamp conductances to prevent numerical issues
        # Conductances are physical quantities and cannot be negative
        self.g_E.clamp_(min=0.0)

        self.g_I.mul_(self._g_I_decay).add_(g_gaba_a_input)
        self.g_I.clamp_(min=0.0)

        # Update NMDA conductance (slow excitation, tau~100ms)
        if self.g_nmda is None:
            self.g_nmda = torch.zeros(self.n_neurons, device=self.device)

        self.g_nmda.mul_(self._g_nmda_decay).add_(g_nmda_input)
        self.g_nmda.clamp_(min=0.0)

        # Decay adaptation conductance (in-place) - handle None case
        if self.g_adapt is not None:
            self.g_adapt.mul_(self._adapt_decay)
        else:
            # Initialize if missing
            self.g_adapt = torch.zeros(self.n_neurons, device=self.device)

        # HOMEOSTATIC INTRINSIC EXCITABILITY: Apply per-neuron leak conductance scaling
        # Biology: Neurons modulate leak channel density (g_L) to control excitability
        # Lower g_L → higher input resistance (R = 1/g_L) → same input produces larger voltage deflection
        # This is the biologically correct way to implement homeostatic gain control
        g_L_effective = self.g_L * self.g_L_scale  # [n_neurons]

        # Compute total conductance (for effective time constant)
        # Include NMDA for slow temporal integration
        # Use effective leak conductance (modulated by homeostasis)
        g_total = g_L_effective + self.g_E + self.g_I + self.g_nmda + self.g_adapt

        # Compute equilibrium potential (weighted average of reversals)
        # Include NMDA contribution
        # Use effective leak conductance
        V_inf_numerator = (
            g_L_effective * self.E_L
            + self.g_E * self.E_E
            + self.g_I * self.E_I
            + self.g_nmda * self.E_nmda
            + self.g_adapt * self.E_adapt
        )

        # Add gap junction conductances (if provided)
        # Gap junctions have DYNAMIC reversal potential = weighted average of neighbor voltages
        # This is fundamentally different from chemical synapses (fixed reversals)
        if g_gap_input is not None and E_gap_reversal is not None:
            g_total = g_total + g_gap_input
            V_inf_numerator = V_inf_numerator + g_gap_input * E_gap_reversal

        # Update T-channel de-inactivation state (if enabled)
        # T-channels de-inactivate during hyperpolarization, creating rebound bursts
        # Must be computed BEFORE additional_conductances to be included in V_inf
        if self.config.enable_t_channels and self.membrane is not None:
            # Steady-state de-inactivation: h_T_inf = 1 / (1 + exp((V - V_half) / k))
            # High h_T when hyperpolarized (V < V_half) → ready to generate rebound burst
            # Low h_T when depolarized (V > V_half) → channels inactivated
            h_T_inf = 1.0 / (1.0 + torch.exp((self.membrane - self.config.V_half_h_T) / self.config.k_h_T))

            # Exponential relaxation toward steady-state
            # h_T(t+dt) = h_T_inf + (h_T - h_T_inf) * exp(-dt/tau_h_T)
            if self.h_T is not None:
                self.h_T = h_T_inf + (self.h_T - h_T_inf) * self._h_T_decay
            else:
                # Initialize if somehow missed earlier
                self.h_T = h_T_inf

            # Compute T-current for membrane update
            # Activation is instantaneous (voltage-dependent), de-inactivation is slow (h_T)
            # m_T_inf = 1 / (1 + exp((V_half_m_T - V) / k_m_T))
            # Using V_half_m_T = -0.5 (activates near rest), k_m_T = 0.1 (steep activation)
            V_half_m_T = -0.5  # Activation threshold (normalized)
            k_m_T = 0.1  # Activation slope
            m_T_inf = 1.0 / (1.0 + torch.exp((V_half_m_T - self.membrane) / k_m_T))

            # T-current: I_T = g_T * m_T * h_T * (E_Ca - V)
            # This creates depolarizing current when h_T is high (after hyperpolarization)
            # Convert to conductance form: g_T_effective = g_T * m_T * h_T
            g_T_eff = self.config.g_T * m_T_inf * self.h_T

            # Add T-conductance to membrane dynamics
            g_total = g_total + g_T_eff
            V_inf_numerator = V_inf_numerator + g_T_eff * self.config.E_Ca

        # Add intrinsic conductances from specialized neurons (e.g., I_h, SK)
        # NOTE: Gap junctions should now be passed via g_gap_input parameter, not via this hook
        additional_conductances = self._get_additional_conductances()
        for g, E_rev in additional_conductances:
            g_total = g_total + g
            V_inf_numerator = V_inf_numerator + g * E_rev

        V_inf = V_inf_numerator / g_total

        # Effective time constant: τ_eff = C_m / g_total
        # But we also need to account for per-neuron tau_mem diversity
        # The true membrane dynamics depend on both conductance-based τ_eff AND intrinsic τ_mem
        # We model this as: τ_combined = (τ_mem * τ_eff) / (τ_mem + τ_eff)
        # For simplicity and biological accuracy, we use tau_mem to scale the decay:
        # decay = exp(-dt / tau_mem * g_total / g_L_effective)
        # Where g_total/g_L_effective gives the conductance-based speed-up factor
        # Per-neuron decay factor incorporating both tau_mem and conductance state
        # Fast neurons (small tau_mem) decay quickly, slow neurons (large tau_mem) integrate longer
        # Use effective g_L (homeostatic modulation affects time constant)
        decay_factor = torch.exp((-self._dt_ms / self.tau_mem_per_neuron) * (g_total / g_L_effective))

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

            v_init = torch.empty((self.n_neurons,), device=self.device)
            v_init.uniform_(v_min, v_max)
            self.membrane = v_init

            # Initialize T-channel de-inactivation variable (if enabled)
            if self.config.enable_t_channels:
                # Start at steady-state for resting potential
                # h_T is high when hyperpolarized, low when depolarized
                # h_T_inf = 1 / (1 + exp((V - V_half) / k))
                h_T_init = 1.0 / (1.0 + torch.exp((v_init - self.config.V_half_h_T) / self.config.k_h_T))
                self.h_T = h_T_init

        V_diff = self.membrane - V_inf
        new_membrane = V_inf + V_diff * decay_factor

        # Add noise only if configured (PER-NEURON independent noise)
        if self.config.noise_std > 0:
            # Generate per-neuron noise using vectorized Philox (fully GPU-parallelized)
            # CRITICAL: This replaces torch.randn_like() to eliminate shared RNG correlations
            noise = self._generate_vectorized_noise_philox()

            if self.config.use_ou_noise:
                # Ornstein-Uhlenbeck (colored) noise: dx = -x/τ*dt + σ*sqrt(2/τ)*dW
                # Discrete: x(t+dt) = x(t)*exp(-dt/τ) + σ*sqrt(1-exp(-2*dt/τ))*randn()
                if self.ou_noise is None:
                    # Initialize OU noise if not present (e.g., after loading old checkpoint)
                    # Initialize directly on device (avoid CPU→GPU transfer)
                    self.ou_noise = torch.zeros(self.n_neurons, device=self.device)

                ou_decay = torch.exp(torch.tensor(-self._dt_ms / self.config.noise_tau_ms))
                ou_std = self.config.noise_std * torch.sqrt(1 - ou_decay**2)  # Stationary variance
                self.ou_noise = self.ou_noise * ou_decay + noise * ou_std
                new_membrane = new_membrane + self.ou_noise
            else:
                # White noise (legacy, uncorrelated temporally but now per-neuron independent)
                new_membrane = new_membrane + noise * self.config.noise_std

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
