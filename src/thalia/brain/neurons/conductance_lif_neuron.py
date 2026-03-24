"""Conductance-Based Leaky Integrate-and-Fire (LIF) Neuron Model.

This module implements biologically realistic spiking neurons using
conductance-based membrane dynamics where synaptic currents depend on
the driving force (difference between membrane potential and reversal potential).

**Membrane Dynamics**:
=====================

    dV/dt = -g_L·V + g_E(E_E - V) + g_I(E_I - V)

Where:
- V: membrane potential (E_L = 0 by convention)
- g_L, g_E, g_I: leak, excitatory, inhibitory conductances
- E_E, E_I: reversal potentials for excitation and inhibition

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
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from hashlib import md5
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn as nn

from thalia import GlobalConfig
from thalia.errors import ConfigurationError
from thalia.typing import ConductanceTensor, GapJunctionReversal, PopulationName, RegionName, VoltageTensor
from thalia.utils import philox_gaussian, philox_uniform
from thalia.utils.conductance_lif_fused import conductance_lif_step as _clif_step_cpp, is_available as _clif_cpp_available

if TYPE_CHECKING:
    from .conductance_lif_batch import ConductanceLIFBatch


# =============================================================================
# Helper functions
# =============================================================================


def _string_hash_md5_31bit(s: str) -> int:
    """Stable hash function for strings using MD5. Returns 31-bit int safe for int64 arithmetic."""
    return int(md5(s.encode()).hexdigest()[:8], 16) % (2**31)


def _create_neuron_seeds(
    region_name: RegionName,
    population_name: PopulationName,
    n_neurons: int,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Generate per-neuron seeds for RNG independence."""
    KNUTH_MULTIPLICATIVE_HASH = 2654435761
    return torch.tensor([
        _string_hash_md5_31bit(f"{region_name}_{population_name}_{neuron_id}")
        for neuron_id in range(n_neurons)
    ], dtype=torch.int64, device=device) * KNUTH_MULTIPLICATIVE_HASH


def heterogeneous_tau_mem(
    tau_mem_mean: float,
    n_neurons: int,
    device: Union[str, torch.device],
    *,
    cv: float = 0.20,
    clamp_fraction: float = 0.40,
) -> torch.Tensor:
    """Generate heterogeneous membrane time constants for a population of neurons.

    Args:
        tau_mem_mean: Mean membrane time constant (ms).
        n_neurons: Number of neurons.
        device: Torch device.
        cv: Coefficient of variation (std/mean).
            Typical values: 0.10-0.12 for fast-spiking interneurons,
            0.15-0.20 for pyramidal neurons, 0.20-0.25 for neuromodulatory.
            Raised 0.15→0.20: matches other heterogeneity parameter increases.
        clamp_fraction: Symmetric clamp range as fraction of mean (±).
            E.g. 0.40 clamps to [0.6*mean, 1.4*mean].
            Raised 0.30→0.40 for wider distribution tails.
    """
    tau_mem_heterogeneous = torch.normal(
        mean=tau_mem_mean,
        std=tau_mem_mean * cv,
        size=(n_neurons,),
        device=device
    ).clamp(min=tau_mem_mean * (1.0 - clamp_fraction), max=tau_mem_mean * (1.0 + clamp_fraction))
    return tau_mem_heterogeneous


def heterogeneous_v_threshold(
    v_threshold_mean: float,
    n_neurons: int,
    device: Union[str, torch.device],
    *,
    cv: float = 0.30,
    clamp_fraction: float = 0.50,
) -> torch.Tensor:
    """Generate heterogeneous spike thresholds for a population of neurons.

    Args:
        v_threshold_mean: Mean spike threshold (normalised units).
        n_neurons: Number of neurons.
        device: Torch device.
        cv: Coefficient of variation (std/mean).
            Typical values: 0.05-0.08 for fast-spiking interneurons,
            0.15-0.20 for pyramidal neurons, 0.20-0.25 for neuromodulatory.
            Raised 0.20→0.30: FR-CV was 0.09–0.10, expected 1.4–2.0.
            Log-normal rate distribution requires wide threshold diversity.
        clamp_fraction: Symmetric clamp range as fraction of mean (±).
            E.g. 0.50 clamps to [0.50*mean, 1.50*mean].
            Raised 0.30→0.50 to allow wider distribution tails for sparse coding.
    """
    v_threshold_heterogeneous = torch.normal(
        mean=v_threshold_mean,
        std=v_threshold_mean * cv,
        size=(n_neurons,),
        device=device
    ).clamp(min=v_threshold_mean * (1.0 - clamp_fraction), max=v_threshold_mean * (1.0 + clamp_fraction))
    return v_threshold_heterogeneous


def heterogeneous_g_L(
    g_L_mean: float,
    n_neurons: int,
    device: Union[str, torch.device],
    *,
    cv: float = 0.30,
    clamp_fraction: float = 0.50,
) -> torch.Tensor:
    """Generate heterogeneous leak conductances for a population of neurons.

    Different g_L values give each neuron a different input resistance
    (R_in = 1/g_L) and thus a different gain on its f-I curve, producing
    natural firing-rate diversity within the population.

    Args:
        g_L_mean: Mean leak conductance.
        n_neurons: Number of neurons.
        device: Torch device.
        cv: Coefficient of variation (std/mean).
            Typical values: 0.08-0.10 for fast-spiking interneurons,
            0.15-0.20 for pyramidal neurons.
            Raised 0.20→0.30: FR-CV was 0.09–0.10, expected 1.4–2.0.
            Wider g_L diversity gives wider input-resistance distribution.
        clamp_fraction: Symmetric clamp range as fraction of mean (±).
            E.g. 0.50 clamps to [0.50*mean, 1.50*mean].
            Raised 0.35→0.50 to allow wider distribution tails.
    """
    return torch.normal(
        mean=g_L_mean,
        std=g_L_mean * cv,
        size=(n_neurons,),
        device=device,
    ).clamp(min=g_L_mean * (1.0 - clamp_fraction), max=g_L_mean * (1.0 + clamp_fraction))


def heterogeneous_adapt_increment(
    adapt_mean: float,
    n_neurons: int,
    device: Union[str, torch.device],
    *,
    cv: float = 0.30,
    clamp_fraction: float = 0.50,
) -> torch.Tensor:
    """Generate heterogeneous adaptation increments for a population of neurons.

    Different adapt_increment values make some neurons strongly adapting
    (bursty onset, then silence) and others weakly adapting (sustained firing),
    creating natural within-population rate heterogeneity (high FR-CV).

    Args:
        adapt_mean: Mean adaptation increment per spike.
        n_neurons: Number of neurons.
        device: Torch device.
        cv: Coefficient of variation (std/mean).
            Wider than tau_mem_ms/v_threshold because adaptation strength
            varies substantially across neurons of the same type.
            Raised 0.20→0.30: matches v_threshold and g_L CV increase.
        clamp_fraction: Symmetric clamp range as fraction of mean (±).
            E.g. 0.50 clamps to [0.50*mean, 1.50*mean].
            Raised 0.35→0.50 for wider diversity.
    """
    return torch.normal(
        mean=adapt_mean,
        std=adapt_mean * cv,
        size=(n_neurons,),
        device=device,
    ).clamp(min=adapt_mean * (1.0 - clamp_fraction), max=adapt_mean * (1.0 + clamp_fraction))


def heterogeneous_v_reset(
    v_reset_mean: float,
    n_neurons: int,
    device: Union[str, torch.device],
    *,
    std: float = 0.02,
    clamp_range: float = 0.05,
) -> torch.Tensor:
    """Generate heterogeneous reset potentials for a population of neurons.

    Different AHP (after-hyperpolarization) depths arise from variable K⁺
    channel densities (Kv, BK, SK) across neurons.  Deeper reset → longer
    inter-spike interval; shallower → faster recovery.  Well-documented in
    patch-clamp recordings (Storm 1990; Gu et al. 2007).

    Unlike other heterogeneous_* functions, uses absolute std rather than CV
    because v_reset can be zero or negative.

    Args:
        v_reset_mean: Mean reset potential (normalised units).
        n_neurons: Number of neurons.
        device: Torch device.
        std: Absolute standard deviation of reset potential distribution.
            Default 0.02 gives ~2% of threshold variation in AHP depth.
        clamp_range: Symmetric absolute clamp range around mean (±).
            E.g. 0.05 clamps to [mean-0.05, mean+0.05].
    """
    return torch.normal(
        mean=v_reset_mean,
        std=std,
        size=(n_neurons,),
        device=device,
    ).clamp(min=v_reset_mean - clamp_range, max=v_reset_mean + clamp_range)


def heterogeneous_tau_adapt(
    tau_adapt_mean: float,
    n_neurons: int,
    device: Union[str, torch.device],
    *,
    cv: float = 0.25,
    clamp_fraction: float = 0.40,
) -> torch.Tensor:
    """Generate heterogeneous adaptation time constants for a population of neurons.

    Adaptation recovery speed depends on the specific mix of Ca²⁺-activated
    K⁺ channels: BK channels produce fast adaptation (small τ), SK channels
    produce slow adaptation (large τ).  Different SK1/2/3 isoform expression
    ratios across neurons create naturally diverse adaptation kinetics
    (Madison & Nicoll 1984; Bhatt et al. 2005).

    Args:
        tau_adapt_mean: Mean adaptation time constant (ms).
        n_neurons: Number of neurons.
        device: Torch device.
        cv: Coefficient of variation (std/mean).
            Typical values: 0.15-0.20 for interneurons,
            0.20-0.30 for pyramidal neurons.
        clamp_fraction: Symmetric clamp range as fraction of mean (±).
            E.g. 0.40 clamps to [0.60*mean, 1.40*mean].
    """
    return torch.normal(
        mean=tau_adapt_mean,
        std=tau_adapt_mean * cv,
        size=(n_neurons,),
        device=device,
    ).clamp(min=tau_adapt_mean * (1.0 - clamp_fraction), max=tau_adapt_mean * (1.0 + clamp_fraction))


def heterogeneous_noise_std(
    noise_std_mean: float,
    n_neurons: int,
    device: Union[str, torch.device],
    *,
    cv: float = 0.20,
    clamp_fraction: float = 0.40,
) -> torch.Tensor:
    """Generate heterogeneous intrinsic noise levels for a population of neurons.

    Stochastic ion channel gating produces voltage noise that varies with
    channel density and type.  Neurons with fewer copies of a particular ion
    channel show higher noise per channel event.  Input resistance differences
    (partially captured by g_L heterogeneity) also amplify noise differently
    (White et al. 2000; Faisal et al. 2008).

    Args:
        noise_std_mean: Mean noise standard deviation.
        n_neurons: Number of neurons.
        device: Torch device.
        cv: Coefficient of variation (std/mean).
            Typical values: 0.15-0.20 for pyramidal,
            0.10-0.15 for fast-spiking interneurons.
        clamp_fraction: Symmetric clamp range as fraction of mean (±).
            E.g. 0.40 clamps to [0.60*mean, 1.40*mean].
    """
    return torch.normal(
        mean=noise_std_mean,
        std=noise_std_mean * cv,
        size=(n_neurons,),
        device=device,
    ).clamp(min=noise_std_mean * (1.0 - clamp_fraction), max=noise_std_mean * (1.0 + clamp_fraction))


def heterogeneous_dendrite_coupling(
    coupling_mean: float,
    n_neurons: int,
    device: Union[str, torch.device],
    *,
    cv: float = 0.30,
    clamp_fraction: float = 0.40,
) -> torch.Tensor:
    """Generate heterogeneous dendritic coupling scales for a population of neurons.

    Dendritic tree size, branching complexity, and spine density vary
    enormously across neurons — even within the same morphological class.
    Different electrotonic distances from synapse to soma affect the local
    dendritic voltage estimate used for NMDA Mg²⁺ unblocking, making some
    neurons better coincidence detectors than others
    (Spruston 2008; Mainen & Sejnowski 1996).

    Args:
        coupling_mean: Mean dendritic coupling scale.
        n_neurons: Number of neurons.
        device: Torch device.
        cv: Coefficient of variation (std/mean).
            Typical values: 0.20-0.30 for pyramidal neurons,
            0.15-0.20 for interneurons (simpler dendrites).
        clamp_fraction: Symmetric clamp range as fraction of mean (±).
            E.g. 0.40 clamps to [0.60*mean, 1.40*mean].
    """
    return torch.normal(
        mean=coupling_mean,
        std=coupling_mean * cv,
        size=(n_neurons,),
        device=device,
    ).clamp(min=coupling_mean * (1.0 - clamp_fraction), max=coupling_mean * (1.0 + clamp_fraction))


def split_excitatory_conductance(g_exc_total: torch.Tensor, nmda_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Split excitatory conductance into AMPA (fast) and NMDA (slow) components.

    Biology: Excitatory synapses contain both AMPA and NMDA receptors.
    AMPA provides fast transmission (tau~5ms), NMDA provides slow temporal
    integration (tau~100ms) with coincidence-detection via Mg²⁺ voltage-gate.

    Args:
        g_exc_total: Total excitatory conductance to split
        nmda_ratio: Fraction of total conductance that is NMDA

    Returns:
        g_ampa: Fast AMPA conductance (80% of total)
        g_nmda: Slow NMDA conductance (20% of total, voltage-gated downstream)
    """
    ampa_ratio = 1.0 - nmda_ratio
    g_ampa = g_exc_total * ampa_ratio
    g_nmda = g_exc_total * nmda_ratio
    return g_ampa, g_nmda


# =============================================================================
# CONDUCTANCE-BASED LIF NEURON
# =============================================================================


@dataclass
class ConductanceLIFConfig:
    """Configuration for conductance-based LIF neuron.

    This implements biologically realistic membrane dynamics where currents
    depend on the difference between membrane potential and reversal potentials.

    Membrane equation (C_m normalised to 1, E_L = 0):
        dV/dt = -g_L·V + g_E(E_E - V) + g_I(E_I - V)

    Key advantages over current-based LIF:
    - Natural saturation (can't exceed reversal potentials)
    - Proper shunting inhibition (divisive, not subtractive)
    - Realistic voltage-dependent current flow
    - No need for artificial v_min clamping
    """

    # =========================================================================
    # Membrane properties
    # =========================================================================
    tau_mem_ms: Union[float, torch.Tensor]   # Membrane time constant (ms) - scalar or per-neuron
    v_reset: Union[float, torch.Tensor]      # Reset after spike; set below 0 for after-hyperpolarization
                                             # Can be scalar or per-neuron tensor for AHP depth diversity
    v_threshold: Union[float, torch.Tensor]  # Spike threshold - scalar or per-neuron
    tau_ref: float                           # Refractory period (ms)
    g_L: Union[float, torch.Tensor]          # Leak conductance; τ_m = 1/g_L = 20ms (C_m normalised to 1)
                                             # Can be scalar or per-neuron tensor for gain heterogeneity

    # =========================================================================
    # Reversal potentials (normalized units, E_L = 0 by convention)
    # =========================================================================
    E_E: float  # Excitatory (≈ 0mV, well above threshold)
    E_I: float  # Inhibitory (≈ -70mV, below rest)

    # =========================================================================
    # Synaptic time constants
    # =========================================================================
    tau_E: float       # Excitatory (AMPA-like)
    tau_I: float       # Inhibitory (GABA_A-like, fast Cl⁻ ionotropic)

    # NMDA conductance (slow excitation for temporal integration)
    tau_nmda: float    # NMDA decay time constant (80-150ms biologically)
    E_nmda: float      # NMDA reversal potential (same as AMPA)

    # GABA_B slow inhibitory channel (metabotropic K⁺)
    # Biology: tau_decay ~250-800 ms, deeper hyperpolarisation (E_GABA_B ~ -90 mV)
    tau_GABA_B: float  # GABA_B conductance decay (ms); 250-800 ms biologically
    E_GABA_B: float    # GABA_B reversal (normalised; more negative than E_I = -0.5)

    # =========================================================================
    # Noise parameters
    # =========================================================================
    noise_std: Union[float, torch.Tensor]  # Percentage of threshold; can be scalar or per-neuron tensor for noise level diversity
    noise_tau_ms: float                    # Correlation time constant of noise (ms)

    # =========================================================================
    # NMDA Mg²⁺ voltage-dependent block parameters
    # =========================================================================
    # B(V) = 1 / (1 + [Mg²⁺]_o * exp(-0.062 * V_mV) / 3.57)
    # mg_conc: extracellular Mg²⁺ concentration in mM. Physiological ~1 mM.
    # Lower values → less block → more NMDA drive at rest (use for Mg²⁺-free conditions).
    mg_conc: float = 1.0

    # =========================================================================
    # NMDA conductance saturation
    # =========================================================================
    # Soft Michaelis-Menten saturation modelling the finite NMDA receptor pool.
    # g_nmda_eff = g_nmda / (1 + g_nmda / g_nmda_max)
    # Prevents unrealistic NMDA accumulation under sustained high-frequency
    # input (e.g. CA3 autoassociative during SWR, thalamocortical bursts).
    # Set to float('inf') (default) to disable saturation.
    g_nmda_max: float = float('inf')

    # =========================================================================
    # Adaptation
    # =========================================================================
    tau_adapt_ms: Union[float, torch.Tensor] = 100.0  # Can be scalar or per-neuron tensor for adaptation kinetics diversity
    adapt_increment: Union[float, torch.Tensor] = 0.0  # Can be scalar or per-neuron tensor
    E_adapt: float = -0.5  # Adaptation reversal (hyperpolarizing, like slow K+)

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
    dendrite_coupling_scale: Union[float, torch.Tensor] = 0.2  # Fraction of AMPA excitatory drive added to dendritic V estimate
                                                               # Can be scalar or per-neuron tensor for morphological diversity

    # =========================================================================
    # I_h (HCN) pacemaker current parameters
    # =========================================================================
    # I_h (HCN / "funny") current — voltage-dependent pacemaker
    # Activates on HYPERPOLARIZATION (opposite to most channels).
    # Creates a depolarising "sag" that drives the membrane back toward rest and underlies
    # rhythmic pacemaker activity in STN, thalamic relay, and VTA/SNc neurons.
    #   E_h ≈ -45 mV → normalised ≈ -0.3 (between rest=0 and E_I=-0.5)
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
    def tau_m(self) -> Union[float, torch.Tensor]:
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
        dV/dt = -g_L·V + g_E(E_E - V) + g_I(E_I - V) + g_adapt(E_adapt - V)

    (C_m normalised to 1.)

    The conductances g_E and g_I are driven by synaptic inputs and decay
    exponentially. This creates a natural low-pass filter on inputs.

    Args:
        n_neurons: Number of neurons in the layer
        config: ConductanceLIFConfig with parameters
    """

    def __init__(
        self,
        n_neurons: int,
        config: ConductanceLIFConfig,
        region_name: RegionName,
        population_name: PopulationName,
        device: Union[str, torch.device],
    ):
        """Initialize conductance-based LIF neuron."""
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config
        self.device = torch.device(device)

        if region_name == "INVALID" or population_name == "INVALID":
            raise ConfigurationError(
                f"ConductanceLIFConfig for [{region_name}:{population_name}] must have valid region_name and population_name for RNG seeding."
            )

        # =============================================================================
        # Register constants as buffers
        # =============================================================================
        # Precomputed NMDA block constants: B(v) = sigmoid(_nmda_a + _nmda_b * v)
        # Equivalent to 1/(1 + mg_conc * exp(-0.062 * v_mv) / 3.57) but avoids
        # 4 extra per-call tensor ops by folding the v_mv transform into two scalars.
        # With E_L = 0: _nmda_scale = 65/E_E, _nmda_bias_mv = -65
        _nmda_scale = 65.0 / config.E_E
        _nmda_C = config.mg_conc / 3.57 * math.exp(-0.062 * (-65.0))

        self.g_L: torch.Tensor
        self.E_E: torch.Tensor
        self.E_I: torch.Tensor
        self._nmda_a: torch.Tensor
        self._nmda_b: torch.Tensor
        self.E_nmda: torch.Tensor
        self.E_GABA_B: torch.Tensor
        self.E_adapt: torch.Tensor
        self.v_reset: torch.Tensor
        self.adapt_increment: torch.Tensor

        # g_L: per-neuron leak conductance (scalar → broadcast, tensor → per-neuron gain diversity)
        if isinstance(config.g_L, (int, float)):
            warnings.warn(
                f"[{region_name}:{population_name}] ConductanceLIFConfig.g_L provided as scalar ({config.g_L}). "
                f"Using the same g_L for all {n_neurons} neurons."
            )
            self.register_buffer("g_L", torch.full((n_neurons,), float(config.g_L), dtype=torch.float32, device=device))
        else:
            assert config.g_L.shape[0] == n_neurons
            self.register_buffer("g_L", config.g_L.to(device=device, dtype=torch.float32))

        self.register_buffer("E_E",      torch.full((n_neurons,), config.E_E,          dtype=torch.float32, device=device))
        self.register_buffer("E_I",      torch.full((n_neurons,), config.E_I,          dtype=torch.float32, device=device))
        self.register_buffer("_nmda_a",  torch.full((n_neurons,), -math.log(_nmda_C),  dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_nmda_b",  torch.full((n_neurons,), 0.062 * _nmda_scale, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("E_nmda",   torch.full((n_neurons,), config.E_nmda,       dtype=torch.float32, device=device))
        self.register_buffer("E_GABA_B", torch.full((n_neurons,), config.E_GABA_B,     dtype=torch.float32, device=device))
        self.register_buffer("E_adapt",  torch.full((n_neurons,), config.E_adapt,      dtype=torch.float32, device=device))

        # v_reset: per-neuron reset potential (scalar → broadcast, tensor → per-neuron AHP depth diversity)
        if isinstance(config.v_reset, (int, float)):
            if config.v_reset != 0.0:
                warnings.warn(
                    f"[{region_name}:{population_name}] ConductanceLIFConfig.v_reset provided as scalar ({config.v_reset}). "
                    f"Using the same v_reset for all {n_neurons} neurons."
                )
            self.register_buffer("v_reset", torch.full((n_neurons,), float(config.v_reset), dtype=torch.float32, device=device))
        else:
            assert config.v_reset.shape[0] == n_neurons
            self.register_buffer("v_reset", config.v_reset.to(device=device, dtype=torch.float32))

        # adapt_increment: per-neuron adaptation strength (scalar → uniform, tensor → heterogeneous)
        if isinstance(config.adapt_increment, (int, float)):
            # No warning for 0.0: intentionally non-adapting (PV/FSI, SK-based, BG pacemakers)
            if config.adapt_increment != 0.0:
                warnings.warn(
                    f"[{region_name}:{population_name}] ConductanceLIFConfig.adapt_increment provided as scalar ({config.adapt_increment}). "
                    f"Using the same adapt_increment for all {n_neurons} neurons."
                )
            self.register_buffer(
                "adapt_increment",
                torch.full((n_neurons,), float(config.adapt_increment), dtype=torch.float32, device=device),
            )
        else:
            self.register_buffer(
                "adapt_increment",
                config.adapt_increment.to(device=device, dtype=torch.float32),
            )

        # =============================================================================
        # Per-neuron tau_mem_ms for heterogeneous time constants (frequency diversity)
        # =============================================================================
        # This enables different neuron types to resonate at different frequencies:
        # - Fast-spiking (3-8ms): Gamma oscillations (40-80 Hz)
        # - Standard pyramidal (15-30ms): Alpha/beta (8-15 Hz)
        # - PFC delay neurons (100-500ms): Persistent activity (<2 Hz)
        self.tau_mem_ms: torch.Tensor
        if isinstance(config.tau_mem_ms, (int, float)):
            warnings.warn(
                f"[{region_name}:{population_name}] ConductanceLIFConfig.tau_mem_ms provided as scalar ({config.tau_mem_ms} ms). "
                f"Using the same tau_mem_ms for all {n_neurons} neurons."
            )
            self.register_buffer(
                "tau_mem_ms",
                torch.full((n_neurons,), float(config.tau_mem_ms), dtype=torch.float32, device=device),
            )
        else:
            assert config.tau_mem_ms.shape[0] == n_neurons
            self.register_buffer(
                "tau_mem_ms",
                config.tau_mem_ms.to(device=device, dtype=torch.float32),
            )

        # =============================================================================
        # Per-neuron threshold for intrinsic plasticity support
        # =============================================================================
        self.v_threshold: torch.Tensor
        if isinstance(config.v_threshold, (int, float)):
            warnings.warn(
                f"[{region_name}:{population_name}] ConductanceLIFConfig.v_threshold provided as scalar ({config.v_threshold}). "
                f"Using the same v_threshold for all {n_neurons} neurons."
            )
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
        # Per-neuron tau_adapt_ms for heterogeneous adaptation kinetics
        # =============================================================================
        # Different SK/BK channel ratios create diverse adaptation recovery speeds:
        # - Fast adaptation (BK-dominated, 30-80ms): rapid recovery, sustained firing
        # - Slow adaptation (SK-dominated, 100-300ms): strong burst-onset coding
        self.tau_adapt_ms: torch.Tensor
        if isinstance(config.tau_adapt_ms, (int, float)):
            _adapt_active = isinstance(config.adapt_increment, (int, float)) and config.adapt_increment != 0.0 or (
                isinstance(config.adapt_increment, torch.Tensor) and (config.adapt_increment != 0.0).any()
            )
            if config.tau_adapt_ms != 0.0 and _adapt_active:
                warnings.warn(
                    f"[{region_name}:{population_name}] ConductanceLIFConfig.tau_adapt_ms provided as scalar ({config.tau_adapt_ms} ms). "
                    f"Using the same tau_adapt_ms for all {n_neurons} neurons."
                )
            self.register_buffer(
                "tau_adapt_ms",
                torch.full((n_neurons,), float(config.tau_adapt_ms), dtype=torch.float32, device=device),
            )
        else:
            assert config.tau_adapt_ms.shape[0] == n_neurons
            self.register_buffer(
                "tau_adapt_ms",
                config.tau_adapt_ms.to(device=device, dtype=torch.float32),
            )

        # =============================================================================
        # Per-neuron noise_std for heterogeneous intrinsic noise
        # =============================================================================
        # Stochastic ion channel gating creates voltage noise that varies with
        # channel density and type; input resistance differences amplify noise differently.
        self.noise_std: torch.Tensor
        if isinstance(config.noise_std, (int, float)):
            if config.noise_std != 0.0:
                warnings.warn(
                    f"[{region_name}:{population_name}] ConductanceLIFConfig.noise_std provided as scalar ({config.noise_std}). "
                    f"Using the same noise_std for all {n_neurons} neurons."
                )
            self.register_buffer(
                "noise_std",
                torch.full((n_neurons,), float(config.noise_std), dtype=torch.float32, device=device),
            )
        else:
            assert config.noise_std.shape[0] == n_neurons
            self.register_buffer(
                "noise_std",
                config.noise_std.to(device=device, dtype=torch.float32),
            )

        # Feature flags derived from per-neuron parameters
        self._has_noise: bool = bool((self.noise_std > 0).any())
        self._has_dendrite_coupling: bool

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
        self.register_buffer("_g_E_decay",      torch.ones(n_neurons, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_g_I_decay",      torch.ones(n_neurons, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_g_nmda_decay",   torch.ones(n_neurons, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_g_GABA_B_decay", torch.ones(n_neurons, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_V_soma_decay",   torch.ones(n_neurons, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_adapt_decay",    torch.ones(n_neurons, dtype=torch.float32, device=device), persistent=False)

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
        # Pre-scaled seeds: multiply by the Knuth golden-ratio constant (floor(2^32/φ)) once
        # at init so the per-timestep noise path only needs an addition, not a multiply.
        # The constant spreads adjacent neuron indices maximally across the 64-bit counter
        # space, preventing counter collisions between neurons at nearby timesteps.
        self._neuron_seeds: torch.Tensor
        self.register_buffer("_neuron_seeds", _create_neuron_seeds(region_name, population_name, self.n_neurons, device=device))

        # Timestep counter for runtime noise (NOT a buffer - changes every forward pass)
        # Init-time calls use (1<<31) offset: unreachable by runtime (simulation steps << 2^31)
        self._rng_timestep: torch.Tensor
        self.register_buffer("_rng_timestep", torch.tensor(0, dtype=torch.int64, device=device))
        _u_v       = philox_uniform(self._neuron_seeds + (1 << 31))       # voltage init
        _u_r       = philox_uniform(self._neuron_seeds + (1 << 31) + 1)   # refractory init
        _gauss_ref = philox_gaussian(self._neuron_seeds + (1 << 31) + 2)  # tau_ref distribution (uses +2 and +3)

        # =============================================================================
        # Ornstein-Uhlenbeck noise state
        # =============================================================================
        self.ou_noise: torch.Tensor
        self._ou_decay: torch.Tensor
        self._ou_std: torch.Tensor
        self.register_buffer("ou_noise", torch.zeros(self.n_neurons, device=device), persistent=True)
        self.register_buffer("_ou_decay", torch.ones(n_neurons, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_ou_std",   torch.ones(n_neurons, dtype=torch.float32, device=device), persistent=False)

        # =============================================================================
        # Per-neuron dendrite coupling scale
        # =============================================================================
        # Dendritic morphology varies enormously — different tree sizes and branching
        # patterns create different electrotonic distances from synapse to soma,
        # affecting NMDA Mg²⁺ unblocking and coincidence detection sensitivity.
        self._dendrite_coupling_scale: torch.Tensor
        if isinstance(config.dendrite_coupling_scale, (int, float)):
            warnings.warn(
                f"[{region_name}:{population_name}] ConductanceLIFConfig.dendrite_coupling_scale provided as scalar ({config.dendrite_coupling_scale}). "
                f"Using the same dendrite_coupling_scale for all {n_neurons} neurons."
            )
            self.register_buffer(
                "_dendrite_coupling_scale",
                torch.full((n_neurons,), float(config.dendrite_coupling_scale), dtype=torch.float32, device=device),
                persistent=False,
            )
        else:
            assert config.dendrite_coupling_scale.shape[0] == n_neurons
            self.register_buffer(
                "_dendrite_coupling_scale",
                config.dendrite_coupling_scale.to(device=device, dtype=torch.float32),
                persistent=False,
            )
        self._has_dendrite_coupling = bool((self._dendrite_coupling_scale > 0).any())

        # =============================================================================
        # NMDA conductance saturation (Michaelis-Menten receptor pool limit)
        # =============================================================================
        self._has_nmda_saturation: bool = not math.isinf(config.g_nmda_max)
        self._g_nmda_max: torch.Tensor
        self.register_buffer(
            "_g_nmda_max",
            torch.full((n_neurons,), float(config.g_nmda_max), dtype=torch.float32, device=device),
            persistent=False,
        )

        # =============================================================================
        # Heterogeneous initial membrane potentials for desynchronization
        # =============================================================================
        # Uniform distribution between rest (0) and partway to threshold prevents pathological synchrony
        # threshold is always > 0 in normal operation (E_L=0, v_threshold=1.0)
        v_init = 0.5 * self.v_threshold * _u_v  # Uniform in [0, 0.5*threshold]

        # =============================================================================
        # Initialize conductances and state variables
        # =============================================================================
        self.V_soma: VoltageTensor = v_init                                     # Membrane potential (V_soma)
        self.g_E: torch.Tensor = torch.zeros(self.n_neurons, device=device)       # AMPA conductance
        self.g_I: torch.Tensor = torch.zeros(self.n_neurons, device=device)       # GABA_A conductance (fast)
        self.g_GABA_B: torch.Tensor = torch.zeros(self.n_neurons, device=device)  # GABA_B conductance (slow, K⁺)
        self.g_nmda: torch.Tensor = torch.zeros(self.n_neurons, device=device)    # NMDA conductance (slow excitation)
        self.g_adapt: torch.Tensor = torch.zeros(self.n_neurons, device=device)   # Adaptation conductance

        # Pre-allocated zero tensor for None conductance inputs — avoids torch.zeros() allocation in forward
        self._zeros: torch.Tensor
        self.register_buffer("_zeros", torch.zeros(self.n_neurons, device=device), persistent=False)

        # Batch mode: set to non-None by ConductanceLIFBatch._setup_batch_ref()
        self._batch: Optional["ConductanceLIFBatch"] = None

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
        # I_h (HCN) pacemaker channel state and per-neuron parameters
        # =============================================================================
        # h_gate is the ACTIVATION variable: high when hyperpolarised, low when depolarised
        # (opposite convention to most gates — HCN is anomalous rectifier)
        self.h_gate: Optional[torch.Tensor] = None   # HCN gate open probability (0-1)
        self._h_decay: torch.Tensor
        self._g_h_max: torch.Tensor
        self._E_h: torch.Tensor
        self._V_half_h: torch.Tensor
        self._k_h: torch.Tensor
        self.register_buffer("_h_decay",  torch.ones(n_neurons, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_g_h_max",  torch.full((n_neurons,), config.g_h_max if config.enable_ih else 0.0, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_E_h",      torch.full((n_neurons,), config.E_h     if config.enable_ih else 0.0, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_V_half_h", torch.full((n_neurons,), config.V_half_h if config.enable_ih else 0.0, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_k_h",      torch.full((n_neurons,), config.k_h     if config.enable_ih else 0.0, dtype=torch.float32, device=device), persistent=False)
        if config.enable_ih:
            # Initialize I_h (HCN) gate at steady-state for starting voltage
            self.h_gate = 1.0 / (1.0 + torch.exp((v_init - config.V_half_h) / config.k_h))

        # =============================================================================
        # Type I (T-type) Ca²⁺ channel state and per-neuron parameters
        # =============================================================================
        self.h_T: Optional[torch.Tensor] = None  # T-channel de-inactivation variable (0-1)
        self._h_T_decay: torch.Tensor
        self._g_T: torch.Tensor
        self._E_Ca: torch.Tensor
        self._V_half_h_T: torch.Tensor
        self._k_h_T: torch.Tensor
        self.register_buffer("_h_T_decay",  torch.ones(n_neurons, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_g_T",        torch.full((n_neurons,), config.g_T        if config.enable_t_channels else 0.0, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_E_Ca",       torch.full((n_neurons,), config.E_Ca       if config.enable_t_channels else 0.0, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_V_half_h_T", torch.full((n_neurons,), config.V_half_h_T if config.enable_t_channels else 0.0, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_k_h_T",      torch.full((n_neurons,), config.k_h_T      if config.enable_t_channels else 0.0, dtype=torch.float32, device=device), persistent=False)
        if config.enable_t_channels:
            # Start at steady-state for resting potential
            self.h_T = 1.0 / (1.0 + torch.exp((v_init - config.V_half_h_T) / config.k_h_T))

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update decay factors for new timestep.

        Called by brain when dt changes. Recomputes cached decay factors
        for conductances based on new dt and time constants.

        Args:
            dt_ms: New timestep in milliseconds
        """
        self._dt_ms = dt_ms
        config = self.config

        # Compute per-neuron decay factors (fill uniform value into [n_neurons] buffers)
        self._g_E_decay.fill_(math.exp(-dt_ms / config.tau_E))
        self._g_I_decay.fill_(math.exp(-dt_ms / config.tau_I))
        self._g_nmda_decay.fill_(math.exp(-dt_ms / config.tau_nmda))
        self._g_GABA_B_decay.fill_(math.exp(-dt_ms / config.tau_GABA_B))

        # adapt_decay is per-neuron because tau_adapt_ms is per-neuron
        self._adapt_decay.copy_(torch.exp(-dt_ms / self.tau_adapt_ms))

        # V_soma_decay is per-neuron because tau_mem_ms is per-neuron
        self._V_soma_decay.copy_(torch.exp(-dt_ms / self.tau_mem_ms))

        # OU noise decay and std (noise_std is per-neuron)
        ou_decay_val = math.exp(-dt_ms / config.noise_tau_ms)
        self._ou_decay.fill_(ou_decay_val)
        if self._has_noise:
            scale = math.sqrt(1.0 - ou_decay_val**2)
            self._ou_std.copy_(self.noise_std * scale)
        else:
            self._ou_std.zero_()

        # T-channel and I_h decay factors (always registered, even when feature disabled)
        if config.enable_t_channels:
            self._h_T_decay.fill_(math.exp(-dt_ms / config.tau_h_T_ms))

        if config.enable_ih:
            self._h_decay.fill_(math.exp(-dt_ms / config.tau_h_ms))

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

    # =========================================================================
    # BATCH MODE (used by ConductanceLIFBatch)
    # =========================================================================

    def _setup_batch_ref(
        self,
        batch: "ConductanceLIFBatch",
        start: int,
        end: int,
    ) -> None:
        """Register this neuron for batched execution.

        Called by :class:`ConductanceLIFBatch.__init__` after state aliasing.
        Subsequent calls to :meth:`forward` will write inputs into the batch's
        global buffers and return a deferred spike view instead of computing.

        Args:
            batch: The owning ConductanceLIFBatch instance.
            start: Start index into the batch's global tensors (inclusive).
            end: End index into the batch's global tensors (exclusive).
        """
        from .conductance_lif_batch import ConductanceLIFBatch  # avoid circular import at module level
        assert isinstance(batch, ConductanceLIFBatch)
        self._batch: Optional["ConductanceLIFBatch"] = batch
        self._batch_start: int = start
        self._batch_end: int = end
        self._batch_spike_view: torch.Tensor = batch._last_spikes[start:end]

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

    def _compute_nmda_block(self, v: VoltageTensor) -> torch.Tensor:
        """Voltage-dependent Mg²⁺ block of NMDA receptors (Jahr & Stevens 1990).

        B(V) = sigmoid(_nmda_a + _nmda_b * v)
        Equivalent to 1/(1 + mg_conc * exp(-0.062 * v_mv) / 3.57)
        with v_mv the linear V→mV mapping; constants precomputed at init.
        """
        return torch.sigmoid(self._nmda_a + self._nmda_b * v)

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

        # ── BATCH MODE: write inputs to global buffers, return deferred spikes ──
        if self._batch is not None:
            batch = self._batch
            s, e = self._batch_start, self._batch_end
            if g_ampa_input is not None:
                batch.g_ampa_input[s:e] = g_ampa_input
            if g_nmda_input is not None:
                batch.g_nmda_input[s:e] = g_nmda_input
            if g_gaba_a_input is not None:
                batch.g_gaba_a_input[s:e] = g_gaba_a_input
            if g_gaba_b_input is not None:
                batch.g_gaba_b_input[s:e] = g_gaba_b_input
            if g_gap_input is not None:
                batch.g_gap_input[s:e] = g_gap_input
                batch.E_gap_reversal[s:e] = E_gap_reversal
                batch._has_any_gap_junctions = True
            return self._batch_spike_view, self.V_soma

        config = self.config

        # Use pre-allocated zero buffer instead of torch.zeros() per None input
        _z = self._zeros
        g_ampa_basal   = g_ampa_input   if g_ampa_input   is not None else _z
        g_nmda_basal   = g_nmda_input   if g_nmda_input   is not None else _z
        g_gaba_a_basal = g_gaba_a_input if g_gaba_a_input is not None else _z
        g_gaba_b_basal = g_gaba_b_input if g_gaba_b_input is not None else _z

        # Validate gap junction inputs (both must be provided together)
        if g_gap_input is not None or E_gap_reversal is not None:
            assert g_gap_input is not None and E_gap_reversal is not None, (
                "ConductanceLIF.forward: g_gap_input and E_gap_reversal must both be provided or both be None"
            )

        # ── C++ FAST PATH ──
        # Use fused C++ kernel when available and no subclass-specific additional conductances.
        # The C++ kernel fuses ~40 PyTorch tensor ops into one parallelised loop.
        additional_conductances = self._get_additional_conductances()
        if _clif_cpp_available() and len(additional_conductances) == 0:
            return self._forward_cpp(
                g_ampa_basal, g_nmda_basal, g_gaba_a_basal, g_gaba_b_basal,
                g_gap_input, E_gap_reversal, dt_ms, config,
            )

        # ── PYTHON FALLBACK PATH ──
        # Used when C++ kernel is unavailable or subclass has additional conductances.
        return self._forward_python(
            g_ampa_basal, g_nmda_basal, g_gaba_a_basal, g_gaba_b_basal,
            g_gap_input, E_gap_reversal, dt_ms, config, additional_conductances,
        )

    def _forward_cpp(
        self,
        g_ampa_basal: torch.Tensor,
        g_nmda_basal: torch.Tensor,
        g_gaba_a_basal: torch.Tensor,
        g_gaba_b_basal: torch.Tensor,
        g_gap_input: Optional[ConductanceTensor],
        E_gap_reversal: Optional[GapJunctionReversal],
        dt_ms: float,
        config: ConductanceLIFConfig,
    ) -> tuple[torch.Tensor, VoltageTensor]:
        """Fused C++ fast path for ConductanceLIF.forward()."""
        _z = self._zeros
        has_gap = g_gap_input is not None
        has_noise = self._has_noise

        spikes = _clif_step_cpp(
            # State tensors (modified in-place by C++)
            V_soma=self.V_soma,
            g_E=self.g_E,
            g_I=self.g_I,
            g_nmda=self.g_nmda,
            g_GABA_B=self.g_GABA_B,
            g_adapt=self.g_adapt,
            ou_noise=self.ou_noise,
            refractory=self.refractory,
            # Synaptic inputs
            g_ampa_input=g_ampa_basal,
            g_nmda_input=g_nmda_basal,
            g_gaba_a_input=g_gaba_a_basal,
            g_gaba_b_input=g_gaba_b_basal,
            # Per-neuron parameters
            g_E_decay=self._g_E_decay,
            g_I_decay=self._g_I_decay,
            g_nmda_decay=self._g_nmda_decay,
            g_GABA_B_decay=self._g_GABA_B_decay,
            adapt_decay=self._adapt_decay,
            V_soma_decay=self._V_soma_decay,
            g_L=self.g_L,
            g_L_scale=self.g_L_scale,
            v_threshold=self.v_threshold,
            adapt_increment=self.adapt_increment,
            tau_ref_per_neuron=self.tau_ref_per_neuron,
            # Per-neuron constants (tensors)
            v_reset=self.v_reset,
            E_E=self.E_E,
            E_I=self.E_I,
            E_nmda=self.E_nmda,
            E_GABA_B=self.E_GABA_B,
            E_adapt=self.E_adapt,
            dendrite_coupling_scale=self._dendrite_coupling_scale,
            nmda_a=self._nmda_a,
            nmda_b=self._nmda_b,
            g_nmda_max=self._g_nmda_max,
            dt_ms=dt_ms,
            # Noise
            enable_noise=has_noise,
            neuron_seeds=self._neuron_seeds if has_noise else _z.to(torch.int64),
            rng_timestep=int(self._rng_timestep.item()),
            ou_decay=self._ou_decay,
            ou_std=self._ou_std,
            # Gap junctions
            has_gap_junctions=has_gap,
            g_gap_input=g_gap_input if has_gap else _z,
            E_gap_reversal=E_gap_reversal if has_gap else _z,
            # T-channels
            enable_t_channels=config.enable_t_channels,
            h_T=self.h_T if config.enable_t_channels else _z,
            h_T_decay=self._h_T_decay,
            g_T=self._g_T,
            E_Ca=self._E_Ca,
            V_half_h_T=self._V_half_h_T,
            k_h_T=self._k_h_T,
            # I_h (HCN)
            enable_ih=config.enable_ih,
            h_gate=self.h_gate if config.enable_ih else _z,
            h_decay=self._h_decay,
            g_h_max=self._g_h_max,
            E_h=self._E_h,
            V_half_h=self._V_half_h,
            k_h=self._k_h,
        )

        # Advance RNG timestep (C++ consumed seeds + rng_timestep, same as Python path)
        if has_noise:
            self._rng_timestep.add_(2)

        return spikes, self.V_soma

    def _forward_python(
        self,
        g_ampa_basal: torch.Tensor,
        g_nmda_basal: torch.Tensor,
        g_gaba_a_basal: torch.Tensor,
        g_gaba_b_basal: torch.Tensor,
        g_gap_input: Optional[ConductanceTensor],
        E_gap_reversal: Optional[GapJunctionReversal],
        dt_ms: float,
        config: ConductanceLIFConfig,
        additional_conductances: list[tuple[torch.Tensor, float]],
    ) -> tuple[torch.Tensor, VoltageTensor]:
        """Original Python implementation of the ConductanceLIF forward pass."""
        # Cache registered buffers once — avoids nn.Module.__getattr__ on each access
        g_L          = self.g_L
        g_L_scale    = self.g_L_scale
        E_E          = self.E_E
        E_I          = self.E_I
        E_nmda       = self.E_nmda
        E_GABA_B     = self.E_GABA_B
        E_adapt      = self.E_adapt
        v_reset      = self.v_reset
        v_threshold  = self.v_threshold
        tau_ref      = self.tau_ref_per_neuron
        _g_E_decay      = self._g_E_decay
        _g_I_decay      = self._g_I_decay
        _g_nmda_decay   = self._g_nmda_decay
        _g_GABA_B_decay = self._g_GABA_B_decay
        _adapt_decay    = self._adapt_decay
        _V_soma_decay   = self._V_soma_decay
        ou_noise     = self.ou_noise
        _ou_decay    = self._ou_decay
        _ou_std      = self._ou_std

        # Cache state tensors once — avoids nn.Module.__getattr__ on each access.
        g_E      = self.g_E
        g_nmda_s = self.g_nmda
        g_I      = self.g_I
        g_GABA_B = self.g_GABA_B
        g_adapt  = self.g_adapt
        V_soma   = self.V_soma
        refractory = self.refractory

        # Basal conductances
        g_E.mul_(_g_E_decay).add_(g_ampa_basal).clamp_(min=0.0)
        g_nmda_s.mul_(_g_nmda_decay).add_(g_nmda_basal).clamp_(min=0.0)
        # NMDA receptor saturation (finite receptor pool, Michaelis-Menten)
        if self._has_nmda_saturation:
            g_nmda_s.div_(1.0 + g_nmda_s / self._g_nmda_max)
        g_I.mul_(_g_I_decay).add_(g_gaba_a_basal).clamp_(min=0.0)
        g_GABA_B.mul_(_g_GABA_B_decay).add_(g_gaba_b_basal).clamp_(min=0.0)

        # Adaptation decay
        g_adapt.mul_(_adapt_decay)

        # HOMEOSTATIC INTRINSIC EXCITABILITY: Apply per-neuron leak conductance scaling
        # This is the biologically correct way to implement homeostatic gain control
        g_L_effective = g_L * g_L_scale  # [n_neurons]

        # NMDA Mg²⁺ voltage-dependent unblock
        # Restores coincidence detection: NMDA only passes current when the postsynaptic
        # membrane is already depolarised (≥ −50 mV). Without the block, NMDA acts as slow
        # AMPA, breaking CA3 pattern completion and L2/3 cortical attractor dynamics.

        # Estimate local dendritic voltage from AMPA drive for Mg²⁺ block.
        # Biology: NMDA unblocking occurs at the synapse site (dendritic voltage), not the soma.
        # Local AMPA depolarization drives the dendrite beyond soma voltage, unblocking nearby
        # NMDA receptors even when the soma is still near rest — the coincidence-detection
        # mechanism described in Koch et al. (1983) and Jahr & Stevens (1990).
        # V_dend_est = V_soma + g_E * (E_E − V_soma) * dendrite_coupling_scale
        if self._has_dendrite_coupling:
            local_ampa_drive = (g_E * (E_E - V_soma)).clamp(min=0.0)
            V_dend_est = V_soma + local_ampa_drive * self._dendrite_coupling_scale
        else:
            V_dend_est = V_soma

        mg_block = self._compute_nmda_block(V_dend_est)
        g_nmda_effective = g_nmda_s * mg_block  # Effective (voltage-gated) NMDA conductance

        # Compute total conductance (for effective time constant)
        # Include voltage-gated NMDA for slow temporal integration
        # Use effective leak conductance (modulated by homeostasis)
        g_total = (
            g_L_effective
            + g_E
            + g_nmda_effective
            + g_I
            + g_GABA_B
            + g_adapt
        )

        # Compute equilibrium potential (weighted average of reversals)
        # Include voltage-gated NMDA contribution and GABA_B
        # Use effective leak conductance (E_L = 0, so g_L_effective * E_L = 0)
        V_inf_numerator = (
            g_E * E_E
            + g_nmda_effective * E_nmda
            + g_I * E_I
            + g_GABA_B * E_GABA_B
            + g_adapt * E_adapt
        )

        # Add gap junction conductances (if provided)
        # Gap junctions have DYNAMIC reversal potential = weighted average of neighbor voltages
        # This is fundamentally different from chemical synapses (fixed reversals)
        if g_gap_input is not None and E_gap_reversal is not None:
            g_total = g_total + g_gap_input
            V_inf_numerator = V_inf_numerator + g_gap_input * E_gap_reversal

        # Update T-channel de-inactivation state (if enabled)
        # T-channels de-inactivate during hyperpolarization, creating rebound bursts
        if config.enable_t_channels:
            # Steady-state de-inactivation: h_T_inf = 1 / (1 + exp((V - V_half) / k))
            # High h_T when hyperpolarized (V < V_half) → ready to generate rebound burst
            # Low h_T when depolarized (V > V_half) → channels inactivated
            h_T_inf = 1.0 / (1.0 + torch.exp((V_soma - self._V_half_h_T) / self._k_h_T))

            # Exponential relaxation toward steady-state
            # h_T(t+dt) = h_T_inf + (h_T - h_T_inf) * exp(-dt/tau_h_T)
            h_T = self.h_T
            if h_T is not None:
                # sub_(inf).mul_(decay).add_(inf): (h_T - inf)*decay + inf = h_T*decay + inf*(1-decay) ✓
                h_T.sub_(h_T_inf).mul_(self._h_T_decay).add_(h_T_inf)
            else:
                # Initialize if somehow missed earlier
                h_T = h_T_inf
                self.h_T = h_T

            # Compute T-current for membrane update
            # Activation is instantaneous (voltage-dependent), de-inactivation is slow (h_T)
            # m_T_inf = 1 / (1 + exp((V_half_m_T - V) / k_m_T))
            # Using V_half_m_T = -0.5 (activates near rest), k_m_T = 0.1 (steep activation)
            V_half_m_T = -0.5  # Activation threshold (normalized)
            k_m_T = 0.1  # Activation slope
            m_T_inf = 1.0 / (1.0 + torch.exp((V_half_m_T - V_soma) / k_m_T))

            # T-current: I_T = g_T * m_T * h_T * (E_Ca - V)
            # This creates depolarizing current when h_T is high (after hyperpolarization)
            # Convert to conductance form: g_T_effective = g_T * m_T * h_T
            g_T_eff = self._g_T * m_T_inf * h_T

            # Add T-conductance to membrane dynamics
            g_total = g_total + g_T_eff
            V_inf_numerator = V_inf_numerator + g_T_eff * self._E_Ca

        # ---------------------------------------------------------------
        # I_h (HCN pacemaker current)
        # ---------------------------------------------------------------
        # h_inf(V) = 1 / (1 + exp((V - V_half_h) / k_h))
        # Because k_h > 0 and V_half_h is hyperpolarised, h_inf → 1 when V << V_half_h
        # and h_inf → 0 when V >> V_half_h. This is the anomalous-rectifier property.
        # The gate relaxes with time constant tau_h_ms (slow, ~100 ms).
        # I_h is depolarising (E_h > E_I) and provides the sag/pacemaker ramp.
        if config.enable_ih:
            h_inf = 1.0 / (1.0 + torch.exp((V_soma - self._V_half_h) / self._k_h))

            h_gate = self.h_gate
            if h_gate is not None:
                assert self._h_decay is not None, "I_h enabled but _h_decay not initialised; call update_temporal_parameters first"
                h_gate.sub_(h_inf).mul_(self._h_decay).add_(h_inf)
            else:
                # Initialise at steady-state for resting voltage
                h_gate = h_inf
                self.h_gate = h_gate

            g_ih = self._g_h_max * h_gate  # [n_neurons]
            g_total = g_total + g_ih
            V_inf_numerator = V_inf_numerator + g_ih * self._E_h

        # Add intrinsic conductances from specialized neurons (e.g., I_h, SK)
        # NOTE: Gap junctions should now be passed via g_gap_input parameter, not via this hook
        for g, E_rev in additional_conductances:
            g_total = g_total + g
            V_inf_numerator = V_inf_numerator + g * E_rev

        V_soma_inf = V_inf_numerator / g_total

        # Effective time constant: τ_eff = 1 / g_total (with C_m = 1)
        # But we also need to account for per-neuron tau_mem_ms diversity
        # The true membrane dynamics depend on both conductance-based τ_eff AND intrinsic τ_mem
        # We model this as: τ_combined = (τ_mem * τ_eff) / (τ_mem + τ_eff)
        # For simplicity and biological accuracy, we use tau_mem_ms to scale the decay:
        # decay = exp(-dt / tau_mem_ms * g_total / g_L_effective)
        # Where g_total/g_L_effective gives the conductance-based speed-up factor
        # Per-neuron decay factor incorporating both tau_mem_ms and conductance state
        # Fast neurons (small tau_mem_ms) decay quickly, slow neurons (large tau_mem_ms) integrate longer
        # Use effective g_L (homeostatic modulation affects time constant)
        # CORRECT: exp(-dt/tau * g_total/g_L) = exp(-dt/tau)^(g_total/g_L) = pow(_V_soma_decay, g_total/g_L)
        V_soma_decay_effective = torch.pow(_V_soma_decay, g_total / g_L_effective)

        # Update membrane for ALL neurons (including refractory).
        # sub_(inf).mul_(decay).add_(inf): (V - inf)*decay + inf = V*decay + inf*(1-decay) ✓
        V_soma.sub_(V_soma_inf).mul_(V_soma_decay_effective).add_(V_soma_inf)

        # Add noise (PER-NEURON independent noise)
        if self._has_noise:
            noise = philox_gaussian(self._neuron_seeds + self._rng_timestep)
            self._rng_timestep.add_(2)  # philox_gaussian internally uses counters and counters+1
            # Ornstein-Uhlenbeck (colored) noise: dx = -x/τ*dt + σ*sqrt(2/τ)*dW
            # Discrete: x(t+dt) = x(t)*exp(-dt/τ) + σ*sqrt(1-exp(-2*dt/τ))*randn()
            ou_noise.mul_(_ou_decay).add_(noise * _ou_std)
            V_soma.add_(ou_noise)

        # Refractory counter (in-place: no new tensor, no __setattr__)
        refractory.sub_(1).clamp_(min=0)
        not_refractory = refractory == 0

        # Spike generation (bool for biological accuracy and memory efficiency)
        # Only allow spikes from non-refractory neurons
        above_threshold = V_soma >= v_threshold
        spikes = above_threshold & not_refractory  # Can only spike if not refractory

        # Combined spike handling: reset membrane AND set refractory in one pass
        # This avoids creating intermediate tensors
        if spikes.any():
            # Reset soma (copy_ avoids __setattr__ on nn.Module)
            V_soma.copy_(torch.where(spikes, v_reset, V_soma))
            # Set refractory
            ref_steps = (tau_ref / dt_ms).int()
            refractory.copy_(torch.where(spikes, ref_steps, refractory))
            # Adaptation increment
            g_adapt.add_(spikes.float() * self.adapt_increment)

        return spikes, V_soma
