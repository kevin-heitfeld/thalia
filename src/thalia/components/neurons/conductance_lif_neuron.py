"""Conductance-Based Leaky Integrate-and-Fire (LIF) Neuron Model.

This module implements biologically realistic spiking neurons using
conductance-based membrane dynamics where synaptic currents depend on
the driving force (difference between membrane potential and reversal potential).

**Membrane Dynamics**:
=====================

    dV/dt = g_L(E_L - V) + g_E(E_E - V) + g_I(E_I - V)

Where:
- V: membrane potential
- g_L, g_E, g_I: leak, excitatory, inhibitory conductances
- E_L, E_E, E_I: reversal potentials for each conductance

(C_m normalised to 1; time unit = ms; conductances in units of g_L)

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
This module uses type aliases from thalia.typing for dimensional analysis:
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

from thalia import GlobalConfig
import thalia.utils.rng as rng
from thalia.typing import ConductanceTensor, GapJunctionReversal, VoltageTensor


# =============================================================================
# Helper functions for conductance decay and stable string hashing
# =============================================================================


KNUTH_MULTIPLICATIVE_HASH = 2654435761


def string_hash_md5(s: str) -> int:
    """Stable hash function for strings using MD5. Returns 31-bit int safe for int64 arithmetic."""
    return int(md5(s.encode()).hexdigest()[:8], 16) % (2**31)


def _decay(dt_ms: float, tau: Union[float, torch.Tensor], device: torch.device) -> torch.Tensor:
    if isinstance(tau, torch.Tensor):
        return torch.exp(-dt_ms / tau)
    else:
        return torch.tensor(math.exp(-dt_ms / tau), device=device)


def _ensure(t: Optional[torch.Tensor], n_neurons: int, device: torch.device) -> torch.Tensor:
    return t if t is not None else torch.zeros(n_neurons, device=device)


# =============================================================================
# CONDUCTANCE-BASED LIF NEURON
# =============================================================================


@dataclass
class ConductanceLIFConfig:
    """Configuration for conductance-based LIF neuron.

    This implements biologically realistic membrane dynamics where currents
    depend on the difference between membrane potential and reversal potentials.

    Membrane equation (C_m normalised to 1):
        dV/dt = g_L(E_L - V) + g_E(E_E - V) + g_I(E_I - V)

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

        v_reset: Reset potential after spike (default: 0.0)
            Where membrane is set after spike emission.
            Typically equals ``E_L`` (rest = reset). Set below ``E_L`` for
            after-hyperpolarization (e.g. -0.10 for serotonin neurons).
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
        g_L: Leak conductance (default: 0.05)
            Controls passive membrane time constant: τ_m = 1 / g_L (with C_m normalised to 1).
            With g_L=0.05 gives τ_m = 20ms; g_L=0.125 gives τ_m = 8ms (fast-spiking).

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

    # =========================================================================
    # RNG configuration (counter-based per-neuron for true independence)
    # =========================================================================
    region_name: str  # Brain region identifier
    population_name: str  # Population identifier

    # device: str = "cpu"  # Device for tensors (e.g., "cpu", "cuda:0")

    # =========================================================================
    # Membrane properties
    # =========================================================================
    tau_mem: Union[float, torch.Tensor] = 20.0  # Membrane time constant (ms) - scalar or per-neuron
    v_reset: float = 0.0  # Reset after spike; set below E_L for after-hyperpolarization
    v_threshold: Union[float, torch.Tensor] = 1.0  # Spike threshold - scalar or per-neuron
    tau_ref: float = 5.0  # Refractory period (ms)

    g_L: float = 0.05  # Leak conductance; τ_m = 1/g_L = 20ms (C_m normalised to 1)

    # =========================================================================
    # Reversal potentials (normalized units)
    # =========================================================================
    E_L: float = 0.0  # Leak/rest (≈ -65mV scaled to 0)
    E_E: float = 3.0  # Excitatory (≈ 0mV, well above threshold)
    E_I: float = -0.5  # Inhibitory (≈ -70mV, below rest)

    # =========================================================================
    # Synaptic time constants
    # =========================================================================
    tau_E: float = 5.0    # Excitatory (AMPA-like)
    tau_I: float = 10.0   # Inhibitory (GABA_A-like, fast Cl⁻ ionotropic)

    # GABA_B slow inhibitory channel (metabotropic K⁺)
    # Biology: tau_decay ~250-800 ms, deeper hyperpolarisation (E_GABA_B ~ -90 mV)
    tau_GABA_B: float = 400.0  # GABA_B conductance decay (ms); 250-800 ms biologically
    E_GABA_B: float = -0.8     # GABA_B reversal (normalised; more negative than E_I = -0.5)

    # NMDA conductance (slow excitation for temporal integration)
    tau_nmda: float = 100.0  # NMDA decay time constant (80-150ms biologically)
    E_nmda: float = 3.0      # NMDA reversal potential (same as AMPA)

    # =========================================================================
    # Noise
    # =========================================================================
    noise_std: float = 0.08    # 8% of threshold - balanced for stability without over-synchronization
    noise_tau_ms: float = 3.0  # 3ms temporal correlation

    # =========================================================================
    # NMDA Mg²⁺ voltage-dependent block parameters
    # =========================================================================
    nmda_mg_k: float = 5.0       # Sigmoid slope (steeper = sharper voltage-gate)
    nmda_mg_v_half: float = 0.5  # Half-unblock voltage in normalized units

    # =========================================================================
    # Gap junction parameters (for interneuron coupling)
    # =========================================================================
    # FIX — dendritic voltage estimate for single-compartment Mg²⁺ block.
    # Biology: NMDA Mg²⁺ unblocking happens at the synapse site (local dendritic voltage),
    # not the soma. Local AMPA drive depolarises the dendrite beyond somatic voltage,
    # enabling coincidence detection even when the soma is near rest.
    # V_dend_est = V_soma + g_E * (E_E - V_soma) * dendrite_coupling_scale
    # At rest with moderate AMPA: V_dend_est > V_soma → partial NMDA unblock (correct)
    # At rest with no AMPA: V_dend_est ≈ V_soma → minimal NMDA unblock (correct)
    # Set to 0.0 to revert to legacy somatic Mg²⁺ block.
    dendrite_coupling_scale: float = 0.2  # Fraction of AMPA excitatory drive added to dendritic V estimate

    # =========================================================================
    # Adaptation
    # =========================================================================
    tau_adapt: float = 100.0
    adapt_increment: float = 0.0
    E_adapt: float = -0.5  # Adaptation reversal (hyperpolarizing, like slow K+)

    # =========================================================================
    # I_h (HCN) pacemaker current parameters
    # =========================================================================
    # I_h (HCN / "funny") current — voltage-dependent pacemaker
    # Activates on HYPERPOLARIZATION (opposite to most channels).
    # Creates a depolarising "sag" that drives the membrane back toward rest and underlies
    # rhythmic pacemaker activity in STN, thalamic relay, and VTA/SNc neurons.
    #   E_h ≈ -45 mV → normalised ≈ -0.3 (between E_L=0 and E_I=-0.5)
    #   tau_h ~100 ms (slow activation, creates long depolarising ramp)
    enable_ih: bool = False   # Set True in STN, thalamus relay, VTA/SNc configs
    g_h_max: float = 0.05     # Maximum HCN conductance (scales pacemaker strength)
    E_h: float = -0.3         # HCN reversal potential (normalised units, between rest and E_I)
    V_half_h: float = -0.3    # Half-activation voltage (negative = activated by hyperpolarisation)
    k_h: float = 0.10         # Slope factor (small → steep voltage-dependence)
    tau_h_ms: float = 100.0   # Activation time constant (ms); slow ramp → pacemaker

    # =========================================================================
    # Type I (T-type) Ca²⁺ channels for rebound bursting
    # =========================================================================
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
        """Passive membrane time constant (ms): τ_m = 1 / g_L (C_m normalised to 1)."""
        return 1.0 / self.g_L


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
        dV/dt = g_L(E_L - V) + g_E(E_E - V) + g_I(E_I - V) + g_adapt(E_L - V)

    (C_m normalised to 1.)

    The conductances g_E and g_I are driven by synaptic inputs and decay
    exponentially. This creates a natural low-pass filter on inputs.

    Args:
        n_neurons: Number of neurons in the layer
        config: ConductanceLIFConfig with parameters
    """

    def __init__(self, n_neurons: int, config: ConductanceLIFConfig, device: str = "cpu"):
        """Initialize conductance-based LIF neuron."""
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config

        # =============================================================================
        # Register constants as buffers
        # =============================================================================
        self.g_L: torch.Tensor
        self.E_L: torch.Tensor
        self.E_E: torch.Tensor
        self.E_I: torch.Tensor
        self.E_nmda: torch.Tensor
        self.E_GABA_B: torch.Tensor
        self.E_adapt: torch.Tensor
        self.v_reset: torch.Tensor
        self.register_buffer("g_L", torch.tensor(config.g_L, device=device))
        self.register_buffer("E_L", torch.tensor(config.E_L, device=device))
        self.register_buffer("E_E", torch.tensor(config.E_E, device=device))
        self.register_buffer("E_I", torch.tensor(config.E_I, device=device))
        self.register_buffer("E_nmda", torch.tensor(config.E_nmda, device=device))
        self.register_buffer("E_GABA_B", torch.tensor(config.E_GABA_B, device=device))
        self.register_buffer("E_adapt", torch.tensor(config.E_adapt, device=device))
        self.register_buffer("v_reset", torch.tensor(config.v_reset, device=device))

        # =============================================================================
        # Per-neuron tau_mem for heterogeneous time constants (frequency diversity)
        # =============================================================================
        # This enables different neuron types to resonate at different frequencies:
        # - Fast-spiking (3-8ms): Gamma oscillations (40-80 Hz)
        # - Standard pyramidal (15-30ms): Alpha/beta (8-15 Hz)
        # - PFC delay neurons (100-500ms): Persistent activity (<2 Hz)
        self.tau_mem_per_neuron: torch.Tensor
        if isinstance(config.tau_mem, (int, float)):
            self.register_buffer(
                "tau_mem_per_neuron",
                torch.full((n_neurons,), float(config.tau_mem), dtype=torch.float32, device=device),
            )
        else:
            assert config.tau_mem.shape[0] == n_neurons
            self.register_buffer(
                "tau_mem_per_neuron",
                config.tau_mem.to(device=device, dtype=torch.float32),
            )

        # =============================================================================
        # Per-neuron threshold for intrinsic plasticity support
        # =============================================================================
        self.v_threshold: torch.Tensor
        if isinstance(config.v_threshold, (int, float)):
            self.register_buffer(
                "v_threshold",
                torch.full((n_neurons,), float(config.v_threshold), dtype=torch.float32, device=device),
            )
        else:
            assert config.v_threshold.shape[0] == n_neurons
            self.register_buffer(
                "v_threshold",
                config.v_threshold.to(device=device, dtype=torch.float32),
            )

        # =============================================================================
        # Cached decay factors (computed via update_temporal_parameters)
        # =============================================================================
        self._dt_ms: Optional[float] = None  # Tracks current dt
        self._g_E_decay: torch.Tensor
        self._g_I_decay: torch.Tensor
        self._g_nmda_decay: torch.Tensor
        self._g_GABA_B_decay: torch.Tensor
        self._V_soma_decay: torch.Tensor
        self._adapt_decay: torch.Tensor
        self.register_buffer("_g_E_decay",      torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_g_I_decay",      torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_g_nmda_decay",   torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_g_GABA_B_decay", torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_V_soma_decay",   torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_adapt_decay",    torch.tensor(1.0, device=device), persistent=False)

        # =============================================================================
        # HOMEOSTATIC INTRINSIC EXCITABILITY: Per-neuron leak conductance scaling
        # =============================================================================
        # Biology: Neurons modulate ion channel densities to maintain target firing rate
        # Lower g_L → higher input resistance → more excitable (Turrigiano & Nelson 2004)
        # This is the CORRECT way to implement homeostatic gain (not multiplying synaptic conductances!)
        # Start at 1.0 (normal leak), will adapt based on activity
        self.g_L_scale: torch.Tensor
        self.register_buffer("g_L_scale", torch.ones(n_neurons, dtype=torch.float32, device=device))

        # =============================================================================
        # RNG INDEPENDENCE: Per-Neuron Counter-Based Seeds
        # =============================================================================
        # Each neuron gets a unique seed to eliminate spurious correlations
        # Noise generation is f(seed_i, timestep) using Philox algorithm

        # Create per-neuron seeds (each neuron gets unique seed)
        # Derive seed from identity (refactor-proof, order-independent)
        # Hierarchical key: hash(region, population, neuron_id)
        # Use md5 for stable hashing across Python sessions (hash() is salted)
        neuron_seeds = [
            string_hash_md5(f"{config.region_name}_{config.population_name}_{neuron_id}")
            for neuron_id in range(self.n_neurons)
        ]
        self._neuron_seeds: torch.Tensor
        self.register_buffer("_neuron_seeds", torch.tensor(neuron_seeds, dtype=torch.int64, device=device))

        # Pre-scaled seeds: multiply by the Knuth golden-ratio constant (floor(2^32/φ)) once
        # at init so the per-timestep noise path only needs an addition, not a multiply.
        # The constant spreads adjacent neuron indices maximally across the 64-bit counter
        # space, preventing counter collisions between neurons at nearby timesteps.
        self._neuron_seeds_scaled: torch.Tensor
        self.register_buffer("_neuron_seeds_scaled", self._neuron_seeds * KNUTH_MULTIPLICATIVE_HASH)

        # Timestep counter for runtime noise (NOT a buffer - changes every forward pass)
        # Init-time calls use (1<<31) offset: unreachable by runtime (simulation steps << 2^31)
        self._rng_timestep = 0
        _ctr = self._neuron_seeds_scaled
        _u_v       = rng.philox_uniform(_ctr + (1 << 31))       # voltage init
        _u_r       = rng.philox_uniform(_ctr + (1 << 31) + 1)   # refractory init
        _gauss_ref = rng.philox_gaussian(_ctr + (1 << 31) + 2)  # tau_ref distribution (uses +2 and +3)

        # =============================================================================
        # Ornstein-Uhlenbeck noise state
        # =============================================================================
        self.ou_noise: torch.Tensor
        self._ou_decay: torch.Tensor
        self._ou_std: torch.Tensor
        self.register_buffer("ou_noise", torch.zeros(self.n_neurons, device=device), persistent=True)
        self.register_buffer("_ou_decay", torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_ou_std",   torch.tensor(1.0, device=device), persistent=False)

        # =============================================================================
        # Heterogeneous initial membrane potentials for desynchronization
        # =============================================================================
        # Uniform distribution between E_L and partway to threshold prevents pathological synchrony
        # If threshold is above rest, start between E_L and halfway to threshold (more excitable)
        # If threshold is below rest, start between halfway to threshold and E_L (less excitable)
        threshold_above_rest_mask = self.v_threshold > self.E_L
        v0 = self.E_L
        v1 = self.E_L + 0.5 * (self.v_threshold - self.E_L)
        v_min = torch.where(threshold_above_rest_mask, v0, v1)
        v_max = torch.where(threshold_above_rest_mask, v1, v0)
        v_init = v_min + _u_v * (v_max - v_min)

        # =============================================================================
        # Initialize conductances and state variables
        # =============================================================================
        self.V_soma: VoltageTensor = v_init                                     # Membrane potential (V_soma)
        self.g_E: torch.Tensor = torch.zeros(self.n_neurons, device=device)       # AMPA conductance
        self.g_I: torch.Tensor = torch.zeros(self.n_neurons, device=device)       # GABA_A conductance (fast)
        self.g_GABA_B: torch.Tensor = torch.zeros(self.n_neurons, device=device)  # GABA_B conductance (slow, K⁺)
        self.g_nmda: torch.Tensor = torch.zeros(self.n_neurons, device=device)    # NMDA conductance (slow excitation)
        self.g_adapt: torch.Tensor = torch.zeros(self.n_neurons, device=device)   # Adaptation conductance

        # =============================================================================
        # Per-neuron refractory period for desynchronization
        # =============================================================================
        # Biology: variable tau_ref (3–7 ms) naturally decorrelates population activity.
        tau_ref_mean = config.tau_ref
        tau_ref_cv = 0.60 if tau_ref_mean < 4.0 else 0.40
        tau_ref_std = tau_ref_mean * tau_ref_cv
        if tau_ref_mean < 4.0:
            min_ref, max_ref = 2.5, 5.0
        else:
            min_ref, max_ref = 2.5, tau_ref_mean * 1.6

        self.tau_ref_per_neuron: torch.Tensor
        self.register_buffer(
            "tau_ref_per_neuron",
            (tau_ref_mean + _gauss_ref * tau_ref_std).clamp(min_ref, max_ref),
        )

        # Refractory timer (counts down to 0, prevents spiking when >0)
        # Initialize refractory states from the per-neuron Philox stream.
        self._u_refractory_init: torch.Tensor
        self.register_buffer("_u_refractory_init", _u_r, persistent=False)  # Uniform [0,1) for refractory timer initialization
        self.refractory: Optional[torch.Tensor] = None  # Initialized on first forward pass based on tau_ref_per_neuron

        # =============================================================================
        # I_h (HCN) pacemaker channel state for STN, thalamic relay, and VTA/SNc neurons
        # =============================================================================
        # h_gate is the ACTIVATION variable: high when hyperpolarised, low when depolarised
        # (opposite convention to most gates — HCN is anomalous rectifier)
        self.h_gate: Optional[torch.Tensor] = None   # HCN gate open probability (0-1)
        self._h_decay: Optional[torch.Tensor]
        if config.enable_ih:
            self.register_buffer("_h_decay", torch.tensor(1.0, device=device), persistent=False)
            # Initialize I_h (HCN) gate at steady-state for starting voltage
            self.h_gate = 1.0 / (1.0 + torch.exp((v_init - config.V_half_h) / config.k_h))
        else:
            self._h_decay = None

        # =============================================================================
        # Type I (T-type) Ca²⁺ channel state for thalamic pacemaker neurons
        # =============================================================================
        # T-type Ca²⁺ channel state (for thalamic pacemaker neurons)
        self.h_T: Optional[torch.Tensor] = None  # T-channel de-inactivation variable (0-1)
        self._h_T_decay: Optional[torch.Tensor]
        if config.enable_t_channels:
            self.register_buffer("_h_T_decay", torch.tensor(1.0, device=device), persistent=False)
            # Start at steady-state for resting potential
            # h_T is high when hyperpolarized, low when depolarized
            # h_T_inf = 1 / (1 + exp((V - V_half) / k))
            self.h_T = 1.0 / (1.0 + torch.exp((v_init - config.V_half_h_T) / config.k_h_T))
        else:
            self._h_T_decay = None

    def _next_uniform(self) -> torch.Tensor:
        """Generate uniform random number in [0,1) for each neuron using Philox."""
        u = rng.philox_uniform(self._neuron_seeds_scaled + self._rng_timestep)
        self._rng_timestep += 1
        return u

    def _next_gaussian(self) -> torch.Tensor:
        """Generate Gaussian random number (mean=0, std=1) for each neuron using Philox."""
        g = rng.philox_gaussian(self._neuron_seeds_scaled + self._rng_timestep)
        self._rng_timestep += 2  # philox_gaussian internally uses counters and counters+1
        return g

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update decay factors for new timestep.

        Called by brain when dt changes. Recomputes cached decay factors
        for conductances based on new dt and time constants.

        Args:
            dt_ms: New timestep in milliseconds
        """
        self._dt_ms = dt_ms
        cfg = self.config
        device = self.V_soma.device

        self._g_E_decay      = _decay(dt_ms, cfg.tau_E, device)
        self._g_I_decay      = _decay(dt_ms, cfg.tau_I, device)
        self._g_nmda_decay   = _decay(dt_ms, cfg.tau_nmda, device)
        self._g_GABA_B_decay = _decay(dt_ms, cfg.tau_GABA_B, device)
        self._V_soma_decay   = _decay(dt_ms, self.tau_mem_per_neuron, device)
        self._adapt_decay    = _decay(dt_ms, cfg.tau_adapt, device)
        self._ou_decay       = _decay(dt_ms, cfg.noise_tau_ms, device)
        self._ou_std = cfg.noise_std * torch.sqrt(1.0 - self._ou_decay**2)

        if cfg.enable_t_channels:
            self._h_T_decay = _decay(dt_ms, cfg.tau_h_T_ms, device)

        if cfg.enable_ih:
            self._h_decay = _decay(dt_ms, cfg.tau_h_ms, device)

    def adjust_thresholds(
        self,
        delta: torch.Tensor,
        min_threshold: float,
        max_threshold: float,
    ) -> None:
        """Adjust per-neuron somatic spike thresholds for intrinsic plasticity (homeostasis interface).

        Args:
            delta: Threshold adjustment per neuron [n_neurons]
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
        """
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

    @torch.no_grad()
    def forward(
        self,
        g_ampa_input: Optional[ConductanceTensor],
        g_nmda_input: Optional[ConductanceTensor],
        g_gaba_a_input: Optional[ConductanceTensor],
        g_gaba_b_input: Optional[ConductanceTensor],
        g_gap_input: Optional[ConductanceTensor] = None,
        E_gap_reversal: Optional[GapJunctionReversal] = None,
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
            g_nmda_input: Slow excitatory conductance (NMDA), shape [n_neurons] (1D, ADR-005)
                Units: Conductance (normalized by g_L)
                Tau ~100ms for temporal integration
            g_gaba_a_input: Fast inhibitory conductance (GABA_A), shape [n_neurons] (1D, ADR-005)
                Units: Conductance (normalized by g_L)
                Tau ~10ms for rapid inhibition
            g_gaba_b_input: Slow inhibitory conductance (GABA_B), shape [n_neurons] (1D, ADR-005)
                Units: Conductance (normalized by g_L)
                Tau ~400ms, E_GABA_B ≈ -0.8 (deeper than GABA_A ≈ -0.5)
                Metabotropic K⁺ channel; provides sustained post-synaptic hyperpolarisation
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
        if self._dt_ms is None:
            self.update_temporal_parameters(dt_ms=GlobalConfig.DEFAULT_DT_MS)
        assert self._dt_ms is not None
        dt_ms = self._dt_ms

        if self.refractory is None:
            self.refractory = (self._u_refractory_init * self.tau_ref_per_neuron / self._dt_ms).to(torch.int32)
        assert self.refractory is not None

        device = self.V_soma.device
        cfg = self.config
        n_neurons = self.n_neurons

        g_ampa_basal = _ensure(g_ampa_input, n_neurons, device)
        g_nmda_basal = _ensure(g_nmda_input, n_neurons, device)
        g_gaba_a_basal = _ensure(g_gaba_a_input, n_neurons, device)
        g_gaba_b_basal = _ensure(g_gaba_b_input, n_neurons, device)

        # Validate gap junction inputs (both must be provided together)
        if g_gap_input is not None or E_gap_reversal is not None:
            assert g_gap_input is not None and E_gap_reversal is not None, (
                "ConductanceLIF.forward: g_gap_input and E_gap_reversal must both be provided or both be None"
            )

        # Basal conductances
        self.g_E.mul_(self._g_E_decay).add_(g_ampa_basal).clamp_(min=0.0)
        self.g_nmda.mul_(self._g_nmda_decay).add_(g_nmda_basal).clamp_(min=0.0)
        self.g_I.mul_(self._g_I_decay).add_(g_gaba_a_basal).clamp_(min=0.0)
        self.g_GABA_B.mul_(self._g_GABA_B_decay).add_(g_gaba_b_basal).clamp_(min=0.0)

        # Adaptation decay
        self.g_adapt.mul_(self._adapt_decay)

        # HOMEOSTATIC INTRINSIC EXCITABILITY: Apply per-neuron leak conductance scaling
        # Biology: Neurons modulate leak channel density (g_L) to control excitability
        # Lower g_L → higher input resistance (R = 1/g_L) → same input produces larger voltage deflection
        # This is the biologically correct way to implement homeostatic gain control
        g_L_effective = self.g_L * self.g_L_scale  # [n_neurons]

        # NMDA Mg²⁺ voltage-dependent unblock
        # f_nmda(V) = sigmoid(k * (V_norm - V_half)) — smooth approximation
        # Restores coincidence detection: NMDA only amplifies when postsynaptic cell is depolarized.
        # Without this gate, NMDA accumulates at rest → pathological excitation.
        # With the gate, safe 20% NMDA ratio restores biologically correct temporal integration.
        V_soma = self.V_soma

        # Estimate local dendritic voltage from AMPA drive for Mg²⁺ block.
        # Biology: NMDA unblocking occurs at the synapse site (dendritic voltage), not the soma.
        # Local AMPA depolarization drives the dendrite beyond soma voltage, unblocking nearby
        # NMDA receptors via the coincidence-detection mechanism.
        # V_dend_est = V_soma + g_E * (E_E − V_soma) * dendrite_coupling_scale
        if cfg.dendrite_coupling_scale > 0.0:
            local_ampa_drive = (self.g_E * (self.E_E - V_soma)).clamp(min=0.0)
            V_dend_est = V_soma + local_ampa_drive * cfg.dendrite_coupling_scale
        else:
            V_dend_est = V_soma

        f_nmda = torch.sigmoid(cfg.nmda_mg_k * (V_dend_est - cfg.nmda_mg_v_half))
        g_nmda_effective = self.g_nmda * f_nmda  # Effective (voltage-gated) NMDA conductance

        # Compute total conductance (for effective time constant)
        # Include voltage-gated NMDA for slow temporal integration
        # Use effective leak conductance (modulated by homeostasis)
        g_total = (
            g_L_effective
            + self.g_E
            + g_nmda_effective
            + self.g_I
            + self.g_GABA_B
            + self.g_adapt
        )

        # Compute equilibrium potential (weighted average of reversals)
        # Include voltage-gated NMDA contribution and GABA_B
        # Use effective leak conductance
        V_inf_numerator = (
            g_L_effective * self.E_L
            + self.g_E * self.E_E
            + g_nmda_effective * self.E_nmda
            + self.g_I * self.E_I
            + self.g_GABA_B * self.E_GABA_B
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
        if cfg.enable_t_channels:
            # Steady-state de-inactivation: h_T_inf = 1 / (1 + exp((V - V_half) / k))
            # High h_T when hyperpolarized (V < V_half) → ready to generate rebound burst
            # Low h_T when depolarized (V > V_half) → channels inactivated
            h_T_inf = 1.0 / (1.0 + torch.exp((self.V_soma - cfg.V_half_h_T) / cfg.k_h_T))

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
            m_T_inf = 1.0 / (1.0 + torch.exp((V_half_m_T - self.V_soma) / k_m_T))

            # T-current: I_T = g_T * m_T * h_T * (E_Ca - V)
            # This creates depolarizing current when h_T is high (after hyperpolarization)
            # Convert to conductance form: g_T_effective = g_T * m_T * h_T
            g_T_eff = cfg.g_T * m_T_inf * self.h_T

            # Add T-conductance to membrane dynamics
            g_total = g_total + g_T_eff
            V_inf_numerator = V_inf_numerator + g_T_eff * cfg.E_Ca

        # ---------------------------------------------------------------
        # I_h (HCN pacemaker current)
        # ---------------------------------------------------------------
        # h_inf(V) = 1 / (1 + exp((V - V_half_h) / k_h))
        # Because k_h > 0 and V_half_h is hyperpolarised, h_inf → 1 when V << V_half_h
        # and h_inf → 0 when V >> V_half_h. This is the anomalous-rectifier property.
        # The gate relaxes with time constant tau_h_ms (slow, ~100 ms).
        # I_h is depolarising (E_h > E_I) and provides the sag/pacemaker ramp.
        if cfg.enable_ih:
            h_inf = 1.0 / (1.0 + torch.exp((self.V_soma - cfg.V_half_h) / cfg.k_h))

            if self.h_gate is not None:
                assert self._h_decay is not None, "I_h enabled but _h_decay not initialised; call update_temporal_parameters first"
                self.h_gate = h_inf + (self.h_gate - h_inf) * self._h_decay
            else:
                # Initialise at steady-state for resting voltage
                self.h_gate = h_inf

            g_ih = cfg.g_h_max * self.h_gate  # [n_neurons]
            g_total = g_total + g_ih
            V_inf_numerator = V_inf_numerator + g_ih * cfg.E_h

        # Add intrinsic conductances from specialized neurons (e.g., I_h, SK)
        # NOTE: Gap junctions should now be passed via g_gap_input parameter, not via this hook
        additional_conductances = self._get_additional_conductances()
        for g, E_rev in additional_conductances:
            g_total = g_total + g
            V_inf_numerator = V_inf_numerator + g * E_rev

        V_soma_inf = V_inf_numerator / g_total

        # Effective time constant: τ_eff = 1 / g_total (with C_m = 1)
        # But we also need to account for per-neuron tau_mem diversity
        # The true membrane dynamics depend on both conductance-based τ_eff AND intrinsic τ_mem
        # We model this as: τ_combined = (τ_mem * τ_eff) / (τ_mem + τ_eff)
        # For simplicity and biological accuracy, we use tau_mem to scale the decay:
        # decay = exp(-dt / tau_mem * g_total / g_L_effective)
        # Where g_total/g_L_effective gives the conductance-based speed-up factor
        # Per-neuron decay factor incorporating both tau_mem and conductance state
        # Fast neurons (small tau_mem) decay quickly, slow neurons (large tau_mem) integrate longer
        # Use effective g_L (homeostatic modulation affects time constant)
        # CORRECT: exp(-dt/tau * g_total/g_L) = exp(-dt/tau)^(g_total/g_L) = pow(_V_soma_decay, g_total/g_L)
        V_soma_decay_effective = torch.pow(self._V_soma_decay, g_total / g_L_effective)

        # Update membrane for non-refractory neurons
        new_V_soma = V_soma_inf + (self.V_soma - V_soma_inf) * V_soma_decay_effective

        # Add noise (PER-NEURON independent noise)
        if cfg.noise_std > 0:
            noise = self._next_gaussian()
            # Ornstein-Uhlenbeck (colored) noise: dx = -x/τ*dt + σ*sqrt(2/τ)*dW
            # Discrete: x(t+dt) = x(t)*exp(-dt/τ) + σ*sqrt(1-exp(-2*dt/τ))*randn()
            self.ou_noise = self.ou_noise * self._ou_decay + self._ou_std * noise
            new_V_soma = new_V_soma + self.ou_noise

        # Update membrane for ALL neurons (including refractory)
        # Neurons continue integrating during refractory, they just can't spike
        # This allows charge buildup during refractory period for fast re-spiking
        self.V_soma = new_V_soma

        # Refractory counter
        self.refractory = (self.refractory - 1).clamp_(min=0)
        not_refractory = self.refractory == 0

        # Spike generation (bool for biological accuracy and memory efficiency)
        # Only allow spikes from non-refractory neurons
        above_threshold = self.V_soma >= self.v_threshold
        spikes = above_threshold & not_refractory  # Can only spike if not refractory

        # Combined spike handling: reset membrane AND set refractory in one pass
        # This avoids creating intermediate tensors
        if spikes.any():
            # Reset soma
            self.V_soma = torch.where(spikes, self.v_reset, self.V_soma)
            # Set refractory
            ref_steps = (self.tau_ref_per_neuron / dt_ms).int()
            self.refractory = torch.where(spikes, ref_steps, self.refractory)
            # Adaptation increment
            if cfg.adapt_increment > 0:
                self.g_adapt = self.g_adapt + spikes.float() * cfg.adapt_increment

        return spikes, self.V_soma


# =============================================================================
# TWO-COMPARTMENT CONDUCTANCE LIF NEURON
# =============================================================================


@dataclass
class TwoCompartmentLIFConfig(ConductanceLIFConfig):
    """Configuration for two-compartment (soma + apical dendrite) LIF neuron.

    Extends :class:`ConductanceLIFConfig` with a separate apical dendritic
    compartment coupled to the soma by a conductance ``g_c``.

    **Key biological improvements over single-compartment model**:

    1. **Dendritic NMDA Mg²⁺ block**: Applied at *dendritic* voltage for apical
       NMDA synapses (the biologically correct location), not at somatic voltage.
       This restores proper coincidence detection for top-down feedback signals.

    2. **BAP (Back-Propagating Action Potential)**: When the soma fires, an action
       potential propagates retrograde into the apical dendrite, briefly
       depolarising it.  This enables STDP coincidence detection: apical NMDA
       unblocks when pre-synaptic input arrives just before or during a BAP.

    3. **Dendritic Ca²⁺ spikes**: When apical depolarisation exceeds ``theta_Ca``,
       a regenerative calcium spike occurs (local dendritic depolarisation burst)
       that further drives somatic bursting via the coupling conductance.

    4. **Somadendritic segregation**: Feedforward (basal) and feedback (apical)
       inputs remain spatially separated, implementing the canonical predictive
       coding architecture.

    Extra parameters beyond :class:`ConductanceLIFConfig`:

    Attributes:
        g_c: Somadendritic coupling conductance (default: 0.05).
            Determines how strongly dendrite and soma influence each other.
            Biological range: 0.02–0.10 (weak to moderate coupling).
        C_d: Dendritic membrane capacitance, relative units (default: 0.5).
            Smaller than soma → dendrite charges faster for same current.
        g_L_d: Dendritic leak conductance (default: 0.03).
            Lower than somatic g_L → slower dendritic return to rest → longer
            integration window for coincidence detection.
        bap_amplitude: BAP depolarisation fraction (default: 0.3).
            When soma spikes, dendrite receives: ``ΔV_d = bap_amplitude × (E_Ca − V_dend)``.
            Models partial retrograde spike (attenuated by dendrite length).
        theta_Ca: Ca spike threshold at dendrite (default: 2.0).
            When ``V_dend ≥ theta_Ca``, a Ca spike conductance burst is triggered.
            Set well above E_E (3.0) so only coincident inputs cause Ca spikes.
        g_Ca_spike: Peak calcium conductance during a Ca spike (default: 0.30).
            Adds a fast depolarising burst that propagates to soma via g_c.
        tau_Ca_ms: Ca spike conductance decay time constant in ms (default: 20.0).
    """

    # =============================================================================
    # Dendritic compartment
    # =============================================================================
    g_c: float = 0.05       # Somadendritic coupling conductance
    C_d: float = 0.5        # Dendritic capacitance (relative)
    g_L_d: float = 0.03     # Dendritic leak conductance

    # =============================================================================
    # BAP (back-propagating action potential)
    # =============================================================================
    bap_amplitude: float = 0.3  # Fraction of (E_Ca − V_dend) added to dendrite on soma spike

    # =============================================================================
    # Dendritic Ca²⁺ spike
    # =============================================================================
    theta_Ca: float = 2.0       # Ca spike threshold at dendrite
    g_Ca_spike: float = 0.30    # Peak Ca conductance on Ca spike event
    tau_Ca_ms: float = 20.0     # Ca spike decay time constant (ms)


class TwoCompartmentLIF(nn.Module):
    """Soma + apical-dendrite two-compartment conductance LIF neuron.

    Compartment equations (Euler, per timestep dt; somatic C_m normalised to 1):

    **Soma** (receives basal / proximal inputs):

    .. math::

        \\frac{dV_s}{dt} = g_L(E_L - V_soma) + g_{E,b}(E_E - V_soma) + g_{I,b}(E_I - V_soma)
            + g_{NMDA,b}^{eff}(E_{NMDA} - V_soma) + g_{GABA_B}(E_{GABA_B} - V_soma)
            + g_{adapt}(E_{adapt} - V_soma) + g_c(V_dend - V_soma)

    **Dendrite** (receives apical / distal inputs):

    .. math::

        C_d \\frac{dV_d}{dt} = g_{L,d}(E_L - V_dend) + g_{E,a}(E_E - V_dend) + g_{I,a}(E_I - V_dend)
            + g_{NMDA,a}^{eff}(E_{NMDA} - V_dend) + g_{Ca}(E_{Ca} - V_dend) + g_c(V_soma - V_dend)

    *Critical*: NMDA Mg²⁺ block for **apical** NMDA is gated by ``V_dend`` (not
    ``V_soma``), which is the biologically correct location of the synapse.

    On soma spike:
    - Emit bool spike, reset ``V_soma → v_reset``, enter refractory.
    - BAP: ``V_dend += bap_amplitude × (E_Ca − V_dend)`` (retrograde depolarisation).

    If ``V_dend ≥ theta_Ca``:
    - Trigger Ca spike: ``g_Ca += g_Ca_spike``.  ``g_Ca`` decays with ``tau_Ca_ms``.
    - The Ca current then depolarises the soma further via ``g_c`` coupling.

    Compatible interface with :class:`ConductanceLIF` for homeostatic mechanisms:
    exposes ``n_neurons``, ``g_L_scale``, ``v_threshold``, ``adjust_thresholds()``,
    and ``update_temporal_parameters()``.

    Args:
        n_neurons: Population size.
        config: :class:`TwoCompartmentLIFConfig` with all parameters.
    """

    def __init__(self, n_neurons: int, config: TwoCompartmentLIFConfig, device: str = "cpu"):
        """Initialize two-compartment LIF neuron with separate somatic and dendritic state."""
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config

        # =============================================================================
        # Shared buffers (somatic, identical to ConductanceLIF)
        # =============================================================================
        self.g_L: torch.Tensor
        self.E_L: torch.Tensor
        self.E_E: torch.Tensor
        self.E_I: torch.Tensor
        self.E_nmda: torch.Tensor
        self.E_GABA_B: torch.Tensor
        self.E_Ca: torch.Tensor
        self.E_adapt: torch.Tensor
        self.v_reset: torch.Tensor
        self.register_buffer("g_L",      torch.tensor(config.g_L,      device=device))
        self.register_buffer("E_L",      torch.tensor(config.E_L,      device=device))
        self.register_buffer("E_E",      torch.tensor(config.E_E,      device=device))
        self.register_buffer("E_I",      torch.tensor(config.E_I,      device=device))
        self.register_buffer("E_nmda",   torch.tensor(config.E_nmda,   device=device))
        self.register_buffer("E_GABA_B", torch.tensor(config.E_GABA_B, device=device))
        self.register_buffer("E_Ca",     torch.tensor(config.E_Ca,     device=device))
        self.register_buffer("E_adapt",  torch.tensor(config.E_adapt,  device=device))
        self.register_buffer("v_reset",  torch.tensor(config.v_reset,  device=device))

        # =============================================================================
        # Per-neuron tau_mem for heterogeneous time constants (frequency diversity) (soma)
        # =============================================================================
        self.tau_mem_per_neuron: torch.Tensor
        if isinstance(config.tau_mem, (int, float)):
            self.register_buffer(
                "tau_mem_per_neuron",
                torch.full((n_neurons,), float(config.tau_mem), dtype=torch.float32, device=device),
            )
        else:
            assert config.tau_mem.shape[0] == n_neurons
            self.register_buffer(
                "tau_mem_per_neuron",
                config.tau_mem.to(device=device, dtype=torch.float32),
            )

        # =============================================================================
        # Per-neuron threshold for intrinsic plasticity support
        # =============================================================================
        self.v_threshold: torch.Tensor
        if isinstance(config.v_threshold, (int, float)):
            self.register_buffer(
                "v_threshold",
                torch.full((n_neurons,), float(config.v_threshold), dtype=torch.float32, device=device),
            )
        else:
            assert config.v_threshold.shape[0] == n_neurons
            self.register_buffer(
                "v_threshold",
                config.v_threshold.to(device=device, dtype=torch.float32),
            )

        # =============================================================================
        # Cached decay factors (computed via update_temporal_parameters)
        # =============================================================================
        self._dt_ms: Optional[float] = None  # Tracks current dt
        self._g_E_decay: torch.Tensor
        self._g_I_decay: torch.Tensor
        self._g_nmda_decay: torch.Tensor
        self._g_GABA_B_decay: torch.Tensor
        self._g_Ca_decay: torch.Tensor
        self._V_soma_decay: torch.Tensor
        self._V_dend_decay: torch.Tensor
        self._adapt_decay: torch.Tensor
        self.register_buffer("_g_E_decay",      torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_g_I_decay",      torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_g_nmda_decay",   torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_g_GABA_B_decay", torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_g_Ca_decay",     torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_V_soma_decay",   torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_V_dend_decay",   torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_adapt_decay",    torch.tensor(1.0, device=device), persistent=False)

        # =============================================================================
        # HOMEOSTATIC INTRINSIC EXCITABILITY: Per-neuron leak conductance scaling
        # =============================================================================
        self.g_L_scale: torch.Tensor
        self.register_buffer("g_L_scale", torch.ones(n_neurons, dtype=torch.float32, device=device))

        # =============================================================================
        # RNG INDEPENDENCE: Per-Neuron Counter-Based Seeds
        # =============================================================================
        neuron_seeds = [
            string_hash_md5(f"{config.region_name}_{config.population_name}_{neuron_id}")
            for neuron_id in range(self.n_neurons)
        ]
        self._neuron_seeds: torch.Tensor
        self.register_buffer("_neuron_seeds", torch.tensor(neuron_seeds, dtype=torch.int64, device=device))

        # Pre-scaled seeds: multiply by the Knuth golden-ratio constant (floor(2^32/φ)) once
        # at init so the per-timestep noise path only needs an addition, not a multiply.
        # The constant spreads adjacent neuron indices maximally across the 64-bit counter
        # space, preventing counter collisions between neurons at nearby timesteps.
        self._neuron_seeds_scaled: torch.Tensor
        self.register_buffer("_neuron_seeds_scaled", self._neuron_seeds * KNUTH_MULTIPLICATIVE_HASH)

        # Timestep counter for runtime noise (NOT a buffer - changes every forward pass)
        # Init-time calls use (1<<31) offset: unreachable by runtime (simulation steps << 2^31)
        self._rng_timestep = 0
        _ctr = self._neuron_seeds_scaled
        _u_v       = rng.philox_uniform(_ctr + (1 << 31))       # voltage init
        _u_r       = rng.philox_uniform(_ctr + (1 << 31) + 1)   # refractory init
        _gauss_ref = rng.philox_gaussian(_ctr + (1 << 31) + 2)  # tau_ref distribution (uses +2 and +3)

        # =============================================================================
        # Ornstein-Uhlenbeck noise state
        # =============================================================================
        self.ou_noise: torch.Tensor
        self._ou_decay: torch.Tensor
        self._ou_std: torch.Tensor
        self.register_buffer("ou_noise", torch.zeros(self.n_neurons, device=device), persistent=True)
        self.register_buffer("_ou_decay", torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_ou_std",   torch.tensor(1.0, device=device), persistent=False)

        # =============================================================================
        # Heterogeneous initial membrane potentials for desynchronization
        # =============================================================================
        threshold_above_rest_mask = self.v_threshold > self.E_L
        v0 = self.E_L
        v1 = self.E_L + 0.5 * (self.v_threshold - self.E_L)
        v_min = torch.where(threshold_above_rest_mask, v0, v1)
        v_max = torch.where(threshold_above_rest_mask, v1, v0)
        v_init = v_min + _u_v * (v_max - v_min)

        # =============================================================================
        # Initialize conductances and state variables
        # =============================================================================
        # Somatic state
        self.V_soma: VoltageTensor = v_init                                      # V_soma
        self.g_E_basal: torch.Tensor = torch.zeros(n_neurons, device=device)       # AMPA (basal)
        self.g_I_basal: torch.Tensor = torch.zeros(n_neurons, device=device)       # GABA_A (basal)
        self.g_GABA_B_basal: torch.Tensor = torch.zeros(n_neurons, device=device)  # GABA_B (basal)
        self.g_nmda_basal: torch.Tensor = torch.zeros(n_neurons, device=device)    # NMDA (basal)
        self.g_adapt: torch.Tensor = torch.zeros(n_neurons, device=device)         # SFA conductance

        # Dendritic state
        self.V_dend: VoltageTensor = torch.full((n_neurons,), config.E_L, device=device)  # dendritic voltage
        self.g_E_apical: torch.Tensor = torch.zeros(n_neurons, device=device)             # AMPA (apical)
        self.g_I_apical: torch.Tensor = torch.zeros(n_neurons, device=device)             # GABA_A (apical)
        self.g_nmda_apical: torch.Tensor = torch.zeros(n_neurons, device=device)          # NMDA (apical, blocked at V_dend!)
        self.g_Ca: torch.Tensor = torch.zeros(n_neurons, device=device)                   # Ca spike conductance (transient)

        # =============================================================================
        # Per-neuron refractory period for desynchronization
        # =============================================================================
        # Biology: variable tau_ref (3–7 ms) naturally decorrelates population activity.
        tau_ref_mean = config.tau_ref
        tau_ref_cv = 0.60 if tau_ref_mean < 4.0 else 0.40
        tau_ref_std = tau_ref_mean * tau_ref_cv
        if tau_ref_mean < 4.0:
            min_ref, max_ref = 2.5, 5.0
        else:
            min_ref, max_ref = 2.5, tau_ref_mean * 1.6

        self.tau_ref_per_neuron: torch.Tensor
        self.register_buffer(
            "tau_ref_per_neuron",
            (tau_ref_mean + _gauss_ref * tau_ref_std).clamp(min_ref, max_ref),
        )

        # Refractory timer (counts down to 0, prevents spiking when >0)
        # Initialize refractory states from the per-neuron Philox stream.
        self._u_refractory_init: torch.Tensor
        self.register_buffer("_u_refractory_init", _u_r, persistent=False)  # Uniform [0,1) for refractory timer initialization
        self.refractory: Optional[torch.Tensor] = None  # Initialized on first forward pass based on tau_ref_per_neuron

    def _next_uniform(self) -> torch.Tensor:
        """Generate uniform random number in [0,1) for each neuron using Philox."""
        u = rng.philox_uniform(self._neuron_seeds_scaled + self._rng_timestep)
        self._rng_timestep += 1
        return u

    def _next_gaussian(self) -> torch.Tensor:
        """Generate Gaussian random number (mean=0, std=1) for each neuron using Philox."""
        g = rng.philox_gaussian(self._neuron_seeds_scaled + self._rng_timestep)
        self._rng_timestep += 2  # philox_gaussian internally uses counters and counters+1
        return g

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Recompute all conductance decay factors for a new timestep size."""
        self._dt_ms = dt_ms
        cfg = self.config
        device = self.V_soma.device

        self._g_E_decay      = _decay(dt_ms, cfg.tau_E, device)
        self._g_I_decay      = _decay(dt_ms, cfg.tau_I, device)
        self._g_nmda_decay   = _decay(dt_ms, cfg.tau_nmda, device)
        self._g_GABA_B_decay = _decay(dt_ms, cfg.tau_GABA_B, device)
        self._g_Ca_decay     = _decay(dt_ms, cfg.tau_Ca_ms, device)
        self._V_soma_decay   = _decay(dt_ms, self.tau_mem_per_neuron, device)
        self._V_dend_decay   = _decay(dt_ms, cfg.C_d / cfg.g_L_d, device)
        self._adapt_decay    = _decay(dt_ms, cfg.tau_adapt, device)
        self._ou_decay       = _decay(dt_ms, cfg.noise_tau_ms, device)
        self._ou_std = cfg.noise_std * torch.sqrt(1.0 - self._ou_decay**2)

    def adjust_thresholds(
        self,
        delta: torch.Tensor,
        min_threshold: float,
        max_threshold: float,
    ) -> None:
        """Adjust per-neuron somatic spike thresholds for intrinsic plasticity (homeostasis interface).

        Args:
            delta: Threshold adjustment per neuron [n_neurons]
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
        """
        self.v_threshold = (self.v_threshold + delta).clamp(min=min_threshold, max=max_threshold)

    @torch.no_grad()
    def forward(
        self,
        g_ampa_basal: Optional[ConductanceTensor],
        g_nmda_basal: Optional[ConductanceTensor],
        g_gaba_a_basal: Optional[ConductanceTensor],
        g_gaba_b_basal: Optional[ConductanceTensor],
        g_ampa_apical: Optional[ConductanceTensor] = None,
        g_nmda_apical: Optional[ConductanceTensor] = None,
        g_gaba_a_apical: Optional[ConductanceTensor] = None,
        g_gap_input: Optional[ConductanceTensor] = None,
        E_gap_reversal: Optional[GapJunctionReversal] = None,
    ) -> tuple[torch.Tensor, VoltageTensor, VoltageTensor]:
        """Run one timestep of two-compartment conductance dynamics.

        **CRITICAL**: All conductance inputs must be non-negative.  They represent
        physical conductances (units normalised by ``g_L``), NOT currents.

        Args:
            g_ampa_basal:   AMPA conductance at basal/somatic compartment [n_neurons].
            g_nmda_basal:   NMDA conductance at basal compartment (Mg²⁺ block at V_soma).
            g_gaba_a_basal: GABA_A conductance at basal compartment [n_neurons].
            g_gaba_b_basal: GABA_B conductance at basal compartment [n_neurons].
            g_ampa_apical:  AMPA conductance at apical/dendritic compartment [n_neurons].
            g_nmda_apical:  NMDA conductance at apical compartment.
                            **Mg²⁺ block is computed at V_dend** — the biologically
                            correct dendritic local voltage.
            g_gaba_a_apical: GABA_A conductance at apical compartment [n_neurons].
            g_gap_input:    Gap-junction conductance [n_neurons] (applied to soma).
            E_gap_reversal: Dynamic gap-junction reversal [n_neurons].

        Returns:
            spikes:  bool tensor [n_neurons], True where soma spike was emitted.
            V_soma:  somatic membrane potential [n_neurons].
            V_dend:  dendritic membrane potential [n_neurons].
        """
        if self._dt_ms is None:
            self.update_temporal_parameters(dt_ms=GlobalConfig.DEFAULT_DT_MS)
        assert self._dt_ms is not None
        dt_ms = self._dt_ms

        if self.refractory is None:
            self.refractory = (self._u_refractory_init * self.tau_ref_per_neuron / self._dt_ms).to(torch.int32)
        assert self.refractory is not None

        device = self.V_soma.device
        cfg = self.config
        n_neurons = self.n_neurons

        g_ampa_b  = _ensure(g_ampa_basal, n_neurons, device)
        g_nmda_b  = _ensure(g_nmda_basal, n_neurons, device)
        g_gaba_a_b = _ensure(g_gaba_a_basal, n_neurons, device)
        g_gaba_b_b = _ensure(g_gaba_b_basal, n_neurons, device)
        g_ampa_a  = _ensure(g_ampa_apical, n_neurons, device)
        g_nmda_a  = _ensure(g_nmda_apical, n_neurons, device)
        g_gaba_a_a = _ensure(g_gaba_a_apical, n_neurons, device)

        # Validate gap junction inputs (both must be provided together)
        if g_gap_input is not None or E_gap_reversal is not None:
            assert g_gap_input is not None and E_gap_reversal is not None, (
                "TwoCompartmentLIF.forward: g_gap_input and E_gap_reversal must both be provided or both be None"
            )

        # Basal conductances
        self.g_E_basal.mul_(self._g_E_decay).add_(g_ampa_b).clamp_(min=0.0)
        self.g_nmda_basal.mul_(self._g_nmda_decay).add_(g_nmda_b).clamp_(min=0.0)
        self.g_I_basal.mul_(self._g_I_decay).add_(g_gaba_a_b).clamp_(min=0.0)
        self.g_GABA_B_basal.mul_(self._g_GABA_B_decay).add_(g_gaba_b_b).clamp_(min=0.0)

        # Apical conductances
        self.g_E_apical.mul_(self._g_E_decay).add_(g_ampa_a).clamp_(min=0.0)
        self.g_nmda_apical.mul_(self._g_nmda_decay).add_(g_nmda_a).clamp_(min=0.0)
        self.g_I_apical.mul_(self._g_I_decay).add_(g_gaba_a_a).clamp_(min=0.0)

        # Ca spike conductance (decay only; increment on Ca spike below)
        self.g_Ca.mul_(self._g_Ca_decay).clamp_(min=0.0)

        # Adaptation decay
        self.g_adapt.mul_(self._adapt_decay)

        # Homeostatic g_L scaling
        g_L_soma_effective = self.g_L * self.g_L_scale  # [n_neurons]

        # Mg²⁺ block at SOMA for basal NMDA
        f_nmda_soma = torch.sigmoid(cfg.nmda_mg_k * (self.V_soma - cfg.nmda_mg_v_half))
        g_nmda_b_effective = self.g_nmda_basal * f_nmda_soma

        # Mg²⁺ block at DENDRITE for apical NMDA ← THE KEY BIOLOGICAL FIX
        f_nmda_dend = torch.sigmoid(cfg.nmda_mg_k * (self.V_dend - cfg.nmda_mg_v_half))
        g_nmda_a_effective = self.g_nmda_apical * f_nmda_dend

        # SOMATIC compartment dynamics
        V_soma = self.V_soma
        V_dend = self.V_dend
        g_c = cfg.g_c

        g_soma_total = (
            g_L_soma_effective
            + self.g_E_basal
            + g_nmda_b_effective
            + self.g_I_basal
            + self.g_GABA_B_basal
            + self.g_adapt
            + g_c  # coupling always present
        )
        V_soma_inf_numerator = (
            g_L_soma_effective * self.E_L
            + self.g_E_basal * self.E_E
            + g_nmda_b_effective * self.E_nmda
            + self.g_I_basal * self.E_I
            + self.g_GABA_B_basal * self.E_GABA_B
            + self.g_adapt * self.E_adapt
            + g_c * V_dend  # coupling target = dendritic voltage
        )

        # Gap junctions (soma)
        if g_gap_input is not None and E_gap_reversal is not None:
            g_soma_total = g_soma_total + g_gap_input
            V_soma_inf_numerator = V_soma_inf_numerator + g_gap_input * E_gap_reversal

        V_soma_inf = V_soma_inf_numerator / g_soma_total
        V_soma_decay_effective = torch.pow(self._V_soma_decay, g_soma_total / g_L_soma_effective)
        new_V_soma = V_soma_inf + (V_soma - V_soma_inf) * V_soma_decay_effective

        # DENDRITIC compartment dynamics
        g_L_dend_effective = cfg.g_L_d
        g_dend_total = (
            g_L_dend_effective + self.g_E_apical + self.g_I_apical
            + g_nmda_a_effective + self.g_Ca
            + g_c
        )
        V_dend_inf_numerator = (
            g_L_dend_effective * self.E_L
            + self.g_E_apical * self.E_E
            + self.g_I_apical * self.E_I
            + g_nmda_a_effective * self.E_nmda
            + self.g_Ca * self.E_Ca  # Ca spike drives toward E_Ca
            + g_c * V_soma   # coupling target = somatic voltage
        )

        V_dend_inf = V_dend_inf_numerator / g_dend_total
        V_dend_decay_effective = torch.pow(self._V_dend_decay, g_dend_total / g_L_dend_effective)
        new_V_dend = V_dend_inf + (V_dend - V_dend_inf) * V_dend_decay_effective

        # OU noise on soma
        if cfg.noise_std > 0:
            noise = self._next_gaussian()
            self.ou_noise = self.ou_noise * self._ou_decay + self._ou_std * noise
            new_V_soma = new_V_soma + self.ou_noise

        # Update voltages
        self.V_soma = new_V_soma
        self.V_dend = new_V_dend

        # Ca spike check (BEFORE spike check, so it can influence soma)
        # When dendritic voltage exceeds theta_Ca, trigger a regenerative Ca spike
        ca_spike = (self.V_dend >= cfg.theta_Ca)
        if ca_spike.any():
            self.g_Ca = self.g_Ca + ca_spike.float() * cfg.g_Ca_spike

        # Refractory counter
        self.refractory = (self.refractory - 1).clamp_(min=0)
        not_refractory = self.refractory == 0

        # Spike generation (somatic)
        above_threshold = self.V_soma >= self.v_threshold
        spikes = above_threshold & not_refractory

        if spikes.any():
            # Reset soma
            self.V_soma = torch.where(spikes, self.v_reset, self.V_soma)
            # Set refractory
            ref_steps = (self.tau_ref_per_neuron / dt_ms).int()
            self.refractory = torch.where(spikes, ref_steps, self.refractory)
            # Adaptation increment
            if cfg.adapt_increment > 0:
                self.g_adapt = self.g_adapt + spikes.float() * cfg.adapt_increment
            # BAP: retrograde depolarisation of dendrite
            # V_dend += bap_amplitude * (E_Ca − V_dend)
            bap_dv = cfg.bap_amplitude * (cfg.E_Ca - self.V_dend) * spikes.float()
            self.V_dend = self.V_dend + bap_dv

        return spikes, VoltageTensor(self.V_soma), VoltageTensor(self.V_dend)
