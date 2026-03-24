"""Two-compartment conductance-based leaky integrate-and-fire (LIF) neuron model."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from thalia import GlobalConfig
from thalia.errors import ConfigurationError
from thalia.typing import ConductanceTensor, GapJunctionReversal, PopulationName, RegionName, VoltageTensor
from thalia.utils import decay_tensor, philox_gaussian, philox_uniform
from thalia.utils.two_compartment_lif_fused import (
    is_available as _tclif_cpp_available,
    two_compartment_lif_step as _tclif_step_cpp,
)

from .conductance_lif_neuron import ConductanceLIFConfig, _create_neuron_seeds


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
    """

    # =============================================================================
    # Dendritic compartment
    # =============================================================================
    g_c: float = 0.15       # Somadendritic coupling conductance (history: 0.05→0.15→0.08→0.15)
                             # Reverted 0.08→0.15: lower g_c weakened Ca spike propagation
                             # to soma, worsening coincidence detection gains (1.03-1.08 →
                             # 0.87-1.02).  With bap_amplitude=0.8 and theta_Ca=0.5, the
                             # stronger coupling is needed for effective Ca→soma transfer.
                             # BAP shunting is offset by the raised bap_amplitude.
    C_d: float = 0.5        # Dendritic capacitance (relative)
    g_L_d: float = 0.02     # Dendritic leak conductance (reduced 0.03→0.02: slower
                             # dendritic voltage decay preserves BAP and apical input)

    # =============================================================================
    # BAP (back-propagating action potential)
    # =============================================================================
    bap_amplitude: float = 0.8  # Fraction of (E_Ca − V_dend) added to dendrite on soma spike
                                # (raised 0.3→0.5→0.8: with g_c=0.15, the strong BAP overcomes
                                # coupling-induced shunting to reliably reach theta_Ca)

    # =============================================================================
    # Dendritic Ca²⁺ spike
    # =============================================================================
    theta_Ca: float = 0.5       # Ca spike threshold at dendrite (lowered 2.0→0.8→0.5:
                                # at 0.8 cortical bAP only reached ~0.46 steady-state dendrite
                                # voltage; with BAP boost the peak is ~1.1 which exceeds 0.5
                                # but may not always reach 0.8)
    g_Ca_spike: float = 0.50    # Peak Ca conductance on Ca spike event (raised 0.30→0.50:
                                # stronger regenerative event propagates to soma via g_c)
    tau_Ca_ms: float = 20.0     # Ca spike decay time constant (ms)

    # =============================================================================
    # NMDA plateau potentials (dendritic)
    # =============================================================================
    # Sustained dendritic depolarization triggered by coincident NMDA input and
    # dendritic depolarization (Mg²⁺ unblock).  Provides a 100-300 ms plateau
    # that supports PFC working memory persistent activity without requiring
    # strong recurrent drive alone.  Disabled by default.
    enable_nmda_plateau: bool = False
    nmda_plateau_threshold: float = 0.15   # Min effective NMDA conductance to trigger
    v_dend_plateau_threshold: float = 0.5  # Min dendritic voltage for initiation
    g_nmda_plateau: float = 0.08           # Plateau conductance (drives toward E_nmda)
    tau_plateau_ms: float = 150.0          # Plateau decay time constant (ms)


class TwoCompartmentLIF(nn.Module):
    """Soma + apical-dendrite two-compartment conductance LIF neuron.

    Compartment equations (Euler, per timestep dt; somatic C_m normalised to 1; E_L = 0):

    **Soma** (receives basal / proximal inputs):

    .. math::

        \\frac{dV_s}{dt} = -g_L \\cdot V_soma + g_{E,b}(E_E - V_soma) + g_{I,b}(E_I - V_soma)
            + g_{NMDA,b}^{eff}(E_{NMDA} - V_soma) + g_{GABA_B}(E_{GABA_B} - V_soma)
            + g_{adapt}(E_{adapt} - V_soma) + g_c(V_dend - V_soma)

    **Dendrite** (receives apical / distal inputs):

    .. math::

        C_d \\frac{dV_d}{dt} = -g_{L,d} \\cdot V_dend + g_{E,a}(E_E - V_dend) + g_{I,a}(E_I - V_dend)
            + g_{NMDA,a}^{eff}(E_{NMDA} - V_dend) + g_{Ca}(E_{Ca} - V_dend) + g_c(V_soma - V_dend)

    *Critical*: NMDA Mg²⁺ block for **apical** NMDA is gated by ``V_dend`` (not
    ``V_soma``), which is the biologically correct location of the synapse.

    On soma spike:
    - Emit bool spike, reset ``V_soma → v_reset``, enter refractory.
    - BAP: ``V_dend += bap_amplitude × (E_Ca − V_dend)`` (retrograde depolarisation).

    If ``V_dend ≥ theta_Ca``:
    - Trigger Ca spike: ``g_Ca += g_Ca_spike``.  ``g_Ca`` decays with ``tau_Ca_ms``.
    - The Ca current then depolarises the soma further via ``g_c`` coupling.

    If ``enable_nmda_plateau`` and effective apical NMDA ≥ ``nmda_plateau_threshold``
    and ``V_dend ≥ v_dend_plateau_threshold``:
    - Trigger plateau: ``g_plateau_dend += g_nmda_plateau``.
    - ``g_plateau_dend`` decays with ``tau_plateau_ms`` (100-300 ms).
    - Sustained depolarising conductance drives ``V_dend`` toward ``E_nmda``.

    Compatible interface with :class:`ConductanceLIF` for homeostatic mechanisms:
    exposes ``n_neurons``, ``g_L_scale``, ``v_threshold``, ``adjust_thresholds()``,
    and ``update_temporal_parameters()``.

    Args:
        n_neurons: Population size.
        config: :class:`TwoCompartmentLIFConfig` with all parameters.
    """

    def __init__(
        self,
        n_neurons: int,
        config: TwoCompartmentLIFConfig,
        region_name: RegionName,
        population_name: PopulationName,
        device: Union[str, torch.device],
    ):
        """Initialize two-compartment LIF neuron with separate somatic and dendritic state."""
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config
        self.device = torch.device(device)

        if region_name == "INVALID" or population_name == "INVALID":
            raise ConfigurationError(
                f"ConductanceLIFConfig for [{region_name}:{population_name}] must have valid region_name and population_name for RNG seeding."
            )

        # =============================================================================
        # Shared buffers (somatic, identical to ConductanceLIF)
        # =============================================================================
        # Precomputed NMDA block constants (see ConductanceLIF for derivation)
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
        self.E_Ca: torch.Tensor
        self.E_adapt: torch.Tensor
        self.v_reset: torch.Tensor
        self.adapt_increment: torch.Tensor

        # g_L: per-neuron leak conductance (scalar → broadcast, tensor → per-neuron gain diversity)
        if isinstance(config.g_L, (int, float)):
            warnings.warn(
                f"[{region_name}:{population_name}] TwoCompartmentLIFConfig.g_L provided as scalar ({config.g_L}). "
                f"Using the same g_L for all {n_neurons} neurons."
            )
            self.register_buffer("g_L", torch.full((n_neurons,), float(config.g_L), dtype=torch.float32, device=device))
        else:
            assert config.g_L.shape[0] == n_neurons
            self.register_buffer("g_L", config.g_L.to(device=device, dtype=torch.float32))

        self.register_buffer("E_E",      torch.tensor(config.E_E,          device=device))
        self.register_buffer("E_I",      torch.tensor(config.E_I,          device=device))
        self.register_buffer("_nmda_a",  torch.tensor(-math.log(_nmda_C),  device=device), persistent=False)
        self.register_buffer("_nmda_b",  torch.tensor(0.062 * _nmda_scale, device=device), persistent=False)
        self.register_buffer("E_nmda",   torch.tensor(config.E_nmda,       device=device))
        self.register_buffer("E_GABA_B", torch.tensor(config.E_GABA_B,     device=device))
        self.register_buffer("E_Ca",     torch.tensor(config.E_Ca,         device=device))
        self.register_buffer("E_adapt",  torch.tensor(config.E_adapt,      device=device))
        # v_reset: per-neuron reset potential (scalar → uniform, tensor → heterogeneous AHP depth)
        if isinstance(config.v_reset, (int, float)):
            self.register_buffer("v_reset", torch.full((n_neurons,), float(config.v_reset), dtype=torch.float32, device=device))
        else:
            assert config.v_reset.shape[0] == n_neurons
            self.register_buffer("v_reset", config.v_reset.to(device=device, dtype=torch.float32))

        # tau_adapt_ms: per-neuron adaptation time constant
        self.tau_adapt_ms: torch.Tensor
        if isinstance(config.tau_adapt_ms, (int, float)):
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

        # noise_std: per-neuron intrinsic noise magnitude
        self.noise_std: torch.Tensor
        if isinstance(config.noise_std, (int, float)):
            self._has_noise: bool = config.noise_std > 0.0
            self.register_buffer(
                "noise_std",
                torch.full((n_neurons,), float(config.noise_std), dtype=torch.float32, device=device),
            )
        else:
            self._has_noise = bool(config.noise_std.any())
            self.register_buffer(
                "noise_std",
                config.noise_std.to(device=device, dtype=torch.float32),
            )

        # adapt_increment: per-neuron adaptation strength (scalar → uniform, tensor → heterogeneous)
        if isinstance(config.adapt_increment, (int, float)):
            # No warning for 0.0: intentionally non-adapting (PV/FSI, SK-based, BG pacemakers)
            if config.adapt_increment != 0.0:
                warnings.warn(
                    f"[{region_name}:{population_name}] TwoCompartmentLIFConfig.adapt_increment provided as scalar ({config.adapt_increment}). "
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
        # Per-neuron tau_mem_ms for heterogeneous time constants (frequency diversity) (soma)
        # =============================================================================
        self.tau_mem_ms: torch.Tensor
        if isinstance(config.tau_mem_ms, (int, float)):
            warnings.warn(
                f"[{region_name}:{population_name}] TwoCompartmentLIFConfig.tau_mem_ms provided as scalar ({config.tau_mem_ms} ms). "
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
                f"[{region_name}:{population_name}] TwoCompartmentLIFConfig.v_threshold provided as scalar ({config.v_threshold}). "
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
        self._g_plateau_decay: torch.Tensor
        self.register_buffer("_g_E_decay",      torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_g_I_decay",      torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_g_nmda_decay",   torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_g_GABA_B_decay", torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_g_Ca_decay",     torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_V_soma_decay",   torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_V_dend_decay",   torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_adapt_decay",    torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_g_plateau_decay", torch.tensor(1.0, device=device), persistent=False)

        # =============================================================================
        # HOMEOSTATIC INTRINSIC EXCITABILITY: Per-neuron leak conductance scaling
        # =============================================================================
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
        self.register_buffer("_ou_decay", torch.tensor(1.0, device=device), persistent=False)
        self.register_buffer("_ou_std",   torch.tensor(1.0, device=device), persistent=False)

        # =============================================================================
        # Heterogeneous initial membrane potentials for desynchronization
        # =============================================================================
        # threshold is always > 0 in normal operation (E_L=0, v_threshold=1.0)
        v_init = 0.5 * self.v_threshold * _u_v  # Uniform in [0, 0.5*threshold]

        # =============================================================================
        # Initialize conductances and state variables
        # =============================================================================
        # Somatic state
        self.V_soma: VoltageTensor = v_init                                        # V_soma
        self.g_E_basal: torch.Tensor = torch.zeros(n_neurons, device=device)       # AMPA (basal)
        self.g_I_basal: torch.Tensor = torch.zeros(n_neurons, device=device)       # GABA_A (basal)
        self.g_GABA_B_basal: torch.Tensor = torch.zeros(n_neurons, device=device)  # GABA_B (basal)
        self.g_nmda_basal: torch.Tensor = torch.zeros(n_neurons, device=device)    # NMDA (basal)
        self.g_adapt: torch.Tensor = torch.zeros(n_neurons, device=device)         # SFA conductance

        # Dendritic state
        self.V_dend: VoltageTensor = torch.zeros(n_neurons, device=device)        # dendritic voltage (rest = 0)
        self.g_E_apical: torch.Tensor = torch.zeros(n_neurons, device=device)     # AMPA (apical)
        self.g_I_apical: torch.Tensor = torch.zeros(n_neurons, device=device)     # GABA_A (apical)
        self.g_nmda_apical: torch.Tensor = torch.zeros(n_neurons, device=device)  # NMDA (apical, blocked at V_dend!)
        self.g_Ca: torch.Tensor = torch.zeros(n_neurons, device=device)           # Ca spike conductance (transient)
        self.g_plateau_dend: torch.Tensor = torch.zeros(n_neurons, device=device) # NMDA plateau conductance (sustained)

        # Pre-allocated zero tensor for None conductance inputs — avoids torch.zeros() allocation in forward
        self._zeros: torch.Tensor
        self.register_buffer("_zeros", torch.zeros(self.n_neurons, device=device), persistent=False)

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
        # NMDA conductance saturation (Michaelis-Menten receptor pool limit)
        # =============================================================================
        self._has_nmda_saturation: bool = not math.isinf(config.g_nmda_max)
        self._g_nmda_max: torch.Tensor
        self.register_buffer(
            "_g_nmda_max",
            torch.full((n_neurons,), float(config.g_nmda_max), dtype=torch.float32, device=device),
            persistent=False,
        )

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Recompute all conductance decay factors for a new timestep size."""
        self._dt_ms = dt_ms
        config = self.config
        device = self.V_soma.device

        self._g_E_decay      = decay_tensor(dt_ms, config.tau_E, device)
        self._g_I_decay      = decay_tensor(dt_ms, config.tau_I, device)
        self._g_nmda_decay   = decay_tensor(dt_ms, config.tau_nmda, device)
        self._g_GABA_B_decay = decay_tensor(dt_ms, config.tau_GABA_B, device)
        self._g_Ca_decay     = decay_tensor(dt_ms, config.tau_Ca_ms, device)
        self._V_soma_decay   = decay_tensor(dt_ms, self.tau_mem_ms, device)
        self._V_dend_decay   = decay_tensor(dt_ms, config.C_d / config.g_L_d, device)
        self._adapt_decay    = torch.exp(-dt_ms / self.tau_adapt_ms)
        self._g_plateau_decay = decay_tensor(dt_ms, config.tau_plateau_ms, device)
        self._ou_decay       = decay_tensor(dt_ms, config.noise_tau_ms, device)
        if self._has_noise:
            self._ou_std = self.noise_std * torch.sqrt(1.0 - self._ou_decay**2)
        else:
            self._ou_std = torch.zeros(1, device=device)

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

        config = self.config

        # Use pre-allocated zero buffer instead of torch.zeros() per None input
        _z = self._zeros
        g_ampa_b   = g_ampa_basal    if g_ampa_basal    is not None else _z
        g_nmda_b   = g_nmda_basal    if g_nmda_basal    is not None else _z
        g_gaba_a_b = g_gaba_a_basal  if g_gaba_a_basal  is not None else _z
        g_gaba_b_b = g_gaba_b_basal  if g_gaba_b_basal  is not None else _z
        g_ampa_a   = g_ampa_apical   if g_ampa_apical   is not None else _z
        g_nmda_a   = g_nmda_apical   if g_nmda_apical   is not None else _z
        g_gaba_a_a = g_gaba_a_apical if g_gaba_a_apical is not None else _z

        # Validate gap junction inputs (both must be provided together)
        if g_gap_input is not None or E_gap_reversal is not None:
            assert g_gap_input is not None and E_gap_reversal is not None, (
                "TwoCompartmentLIF.forward: g_gap_input and E_gap_reversal must both be provided or both be None"
            )

        # ── C++ FAST PATH ──
        if _tclif_cpp_available():
            return self._forward_cpp(
                g_ampa_b, g_nmda_b, g_gaba_a_b, g_gaba_b_b,
                g_ampa_a, g_nmda_a, g_gaba_a_a,
                g_gap_input, E_gap_reversal, dt_ms, config,
            )

        # ── PYTHON FALLBACK PATH ──
        return self._forward_python(
            g_ampa_b, g_nmda_b, g_gaba_a_b, g_gaba_b_b,
            g_ampa_a, g_nmda_a, g_gaba_a_a,
            g_gap_input, E_gap_reversal, dt_ms, config,
        )

    def _forward_cpp(
        self,
        g_ampa_b: torch.Tensor,
        g_nmda_b: torch.Tensor,
        g_gaba_a_b: torch.Tensor,
        g_gaba_b_b: torch.Tensor,
        g_ampa_a: torch.Tensor,
        g_nmda_a: torch.Tensor,
        g_gaba_a_a: torch.Tensor,
        g_gap_input: Optional[ConductanceTensor],
        E_gap_reversal: Optional[GapJunctionReversal],
        dt_ms: float,
        config: TwoCompartmentLIFConfig,
    ) -> tuple[torch.Tensor, VoltageTensor, VoltageTensor]:
        """Fused C++ fast path for TwoCompartmentLIF.forward()."""
        _z = self._zeros
        has_gap = g_gap_input is not None
        has_noise = self._has_noise

        spikes = _tclif_step_cpp(
            # Somatic state (modified in-place by C++)
            V_soma=self.V_soma,
            g_E_basal=self.g_E_basal,
            g_I_basal=self.g_I_basal,
            g_GABA_B_basal=self.g_GABA_B_basal,
            g_nmda_basal=self.g_nmda_basal,
            g_adapt=self.g_adapt,
            # Dendritic state (modified in-place by C++)
            V_dend=self.V_dend,
            g_E_apical=self.g_E_apical,
            g_I_apical=self.g_I_apical,
            g_nmda_apical=self.g_nmda_apical,
            g_Ca=self.g_Ca,
            g_plateau=self.g_plateau_dend,
            # Noise state
            ou_noise=self.ou_noise,
            refractory=self.refractory,
            # Basal synaptic inputs
            g_ampa_basal_in=g_ampa_b,
            g_nmda_basal_in=g_nmda_b,
            g_gaba_a_basal_in=g_gaba_a_b,
            g_gaba_b_basal_in=g_gaba_b_b,
            # Apical synaptic inputs
            g_ampa_apical_in=g_ampa_a,
            g_nmda_apical_in=g_nmda_a,
            g_gaba_a_apical_in=g_gaba_a_a,
            # Per-neuron decay factors
            g_E_decay=self._g_E_decay,
            g_I_decay=self._g_I_decay,
            g_nmda_decay=self._g_nmda_decay,
            g_GABA_B_decay=self._g_GABA_B_decay,
            g_Ca_decay=self._g_Ca_decay,
            g_plateau_decay=self._g_plateau_decay,
            adapt_decay=self._adapt_decay,
            V_soma_decay=self._V_soma_decay,
            V_dend_decay=self._V_dend_decay,
            # Per-neuron parameters
            g_L=self.g_L,
            g_L_scale=self.g_L_scale,
            v_threshold=self.v_threshold,
            adapt_increment=self.adapt_increment,
            tau_ref_per_neuron=self.tau_ref_per_neuron,
            # Constants
            v_reset=self.v_reset,
            E_E=self.E_E,
            E_I=self.E_I,
            E_nmda=self.E_nmda,
            E_GABA_B=self.E_GABA_B,
            E_adapt=self.E_adapt,
            E_Ca=self.E_Ca,
            nmda_a=self._nmda_a,
            nmda_b=self._nmda_b,
            g_nmda_max=self._g_nmda_max,
            dt_ms=dt_ms,
            # Two-compartment coupling
            g_c=config.g_c,
            g_L_dend=config.g_L_d,
            bap_amplitude=config.bap_amplitude,
            theta_Ca=config.theta_Ca,
            g_Ca_spike=config.g_Ca_spike,
            # NMDA plateau
            enable_nmda_plateau=config.enable_nmda_plateau,
            nmda_plateau_threshold=config.nmda_plateau_threshold,
            v_dend_plateau_threshold=config.v_dend_plateau_threshold,
            g_nmda_plateau=config.g_nmda_plateau,
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
        )

        # Advance RNG timestep (C++ consumed seeds + rng_timestep, same as Python path)
        if has_noise:
            self._rng_timestep.add_(2)

        return spikes, VoltageTensor(self.V_soma), VoltageTensor(self.V_dend)

    def _forward_python(
        self,
        g_ampa_b: torch.Tensor,
        g_nmda_b: torch.Tensor,
        g_gaba_a_b: torch.Tensor,
        g_gaba_b_b: torch.Tensor,
        g_ampa_a: torch.Tensor,
        g_nmda_a: torch.Tensor,
        g_gaba_a_a: torch.Tensor,
        g_gap_input: Optional[ConductanceTensor],
        E_gap_reversal: Optional[GapJunctionReversal],
        dt_ms: float,
        config: TwoCompartmentLIFConfig,
    ) -> tuple[torch.Tensor, VoltageTensor, VoltageTensor]:
        """Python fallback path for TwoCompartmentLIF.forward()."""
        # Cache state tensors and registered buffers once — avoids nn.Module.__getattr__
        # on each access.  In-place ops modify the same underlying tensor, so caching is safe.
        _g_E_decay      = self._g_E_decay
        _g_nmda_decay   = self._g_nmda_decay
        _g_I_decay      = self._g_I_decay
        _g_GABA_B_decay = self._g_GABA_B_decay
        _g_Ca_decay     = self._g_Ca_decay
        _adapt_decay    = self._adapt_decay
        _V_soma_decay   = self._V_soma_decay
        _V_dend_decay   = self._V_dend_decay
        g_L         = self.g_L
        g_L_scale   = self.g_L_scale
        E_E         = self.E_E
        E_I         = self.E_I
        E_nmda      = self.E_nmda
        E_GABA_B    = self.E_GABA_B
        E_adapt     = self.E_adapt
        E_Ca        = self.E_Ca
        v_reset     = self.v_reset
        v_threshold = self.v_threshold
        tau_ref     = self.tau_ref_per_neuron
        adapt_inc   = self.adapt_increment
        g_E_basal_t   = self.g_E_basal
        g_nmda_basal_t = self.g_nmda_basal
        g_I_basal_t   = self.g_I_basal
        g_GABA_B_basal_t = self.g_GABA_B_basal
        g_E_apical_t  = self.g_E_apical
        g_nmda_apical_t = self.g_nmda_apical
        g_I_apical_t  = self.g_I_apical
        g_Ca_t        = self.g_Ca
        g_adapt_t     = self.g_adapt
        g_plateau_t   = self.g_plateau_dend
        V_soma        = self.V_soma
        V_dend        = self.V_dend
        refractory    = self.refractory

        # Basal conductances
        g_E_basal_t.mul_(_g_E_decay).add_(g_ampa_b).clamp_(min=0.0)
        g_nmda_basal_t.mul_(_g_nmda_decay).add_(g_nmda_b).clamp_(min=0.0)
        g_I_basal_t.mul_(_g_I_decay).add_(g_gaba_a_b).clamp_(min=0.0)
        g_GABA_B_basal_t.mul_(_g_GABA_B_decay).add_(g_gaba_b_b).clamp_(min=0.0)

        # Apical conductances
        g_E_apical_t.mul_(_g_E_decay).add_(g_ampa_a).clamp_(min=0.0)
        g_nmda_apical_t.mul_(_g_nmda_decay).add_(g_nmda_a).clamp_(min=0.0)
        g_I_apical_t.mul_(_g_I_decay).add_(g_gaba_a_a).clamp_(min=0.0)

        # NMDA receptor saturation (finite receptor pool, Michaelis-Menten)
        if self._has_nmda_saturation:
            g_nmda_basal_t.div_(1.0 + g_nmda_basal_t / self._g_nmda_max)
            g_nmda_apical_t.div_(1.0 + g_nmda_apical_t / self._g_nmda_max)

        # Ca spike conductance (decay only; increment on Ca spike below)
        g_Ca_t.mul_(_g_Ca_decay).clamp_(min=0.0)

        # NMDA plateau conductance (decay only; activation after Mg²⁺ block below)
        if config.enable_nmda_plateau:
            g_plateau_t.mul_(self._g_plateau_decay).clamp_(min=0.0)

        # Adaptation decay
        g_adapt_t.mul_(_adapt_decay)

        # Homeostatic g_L scaling
        g_L_soma_effective = g_L * g_L_scale  # [n_neurons]

        # Mg²⁺ block at SOMA for basal NMDA (Jahr & Stevens 1990)
        g_nmda_b_effective = g_nmda_basal_t * self._compute_nmda_block(V_soma)

        # Mg²⁺ block at DENDRITE for apical NMDA (biologically correct: block at synapse site)
        g_nmda_a_effective = g_nmda_apical_t * self._compute_nmda_block(V_dend)

        # NMDA plateau activation: when effective apical NMDA conductance AND dendritic
        # voltage both exceed thresholds, trigger a sustained plateau depolarization.
        # This models the regenerative NMDA-dependent plateau potential observed in
        # PFC pyramidal dendrites (Major et al. 2013; Milojkovic et al. 2004).
        if config.enable_nmda_plateau:
            plateau_trigger = (g_nmda_a_effective >= config.nmda_plateau_threshold) & (V_dend >= config.v_dend_plateau_threshold)
            if plateau_trigger.any():
                g_plateau_t.add_(plateau_trigger.float() * config.g_nmda_plateau)

        # SOMATIC compartment dynamics
        g_c = config.g_c

        g_soma_total = (
            g_L_soma_effective
            + g_E_basal_t
            + g_nmda_b_effective
            + g_I_basal_t
            + g_GABA_B_basal_t
            + g_adapt_t
            + g_c  # coupling always present
        )
        V_soma_inf_numerator = (
            g_E_basal_t * E_E
            + g_nmda_b_effective * E_nmda
            + g_I_basal_t * E_I
            + g_GABA_B_basal_t * E_GABA_B
            + g_adapt_t * E_adapt
            + g_c * V_dend  # coupling target = dendritic voltage
        )

        # Gap junctions (soma)
        if g_gap_input is not None and E_gap_reversal is not None:
            g_soma_total = g_soma_total + g_gap_input
            V_soma_inf_numerator = V_soma_inf_numerator + g_gap_input * E_gap_reversal

        V_soma_inf = V_soma_inf_numerator / g_soma_total
        V_soma_decay_effective = torch.pow(_V_soma_decay, g_soma_total / g_L_soma_effective)
        new_V_soma = V_soma_inf + (V_soma - V_soma_inf) * V_soma_decay_effective

        # DENDRITIC compartment dynamics
        g_L_dend_effective = config.g_L_d
        g_dend_total = (
            g_L_dend_effective + g_E_apical_t + g_I_apical_t
            + g_nmda_a_effective + g_Ca_t
            + g_c
        )
        V_dend_inf_numerator = (
            g_E_apical_t * E_E
            + g_I_apical_t * E_I
            + g_nmda_a_effective * E_nmda
            + g_Ca_t * E_Ca  # Ca spike drives toward E_Ca
            + g_c * V_soma   # coupling target = somatic voltage
        )

        # NMDA plateau: sustained depolarizing conductance driving toward E_nmda
        if config.enable_nmda_plateau:
            g_dend_total = g_dend_total + g_plateau_t
            V_dend_inf_numerator = V_dend_inf_numerator + g_plateau_t * E_nmda

        V_dend_inf = V_dend_inf_numerator / g_dend_total
        V_dend_decay_effective = torch.pow(_V_dend_decay, g_dend_total / g_L_dend_effective)
        new_V_dend = V_dend_inf + (V_dend - V_dend_inf) * V_dend_decay_effective

        # OU noise on soma
        if self._has_noise:
            noise = philox_gaussian(self._neuron_seeds + self._rng_timestep)
            self._rng_timestep.add_(2)  # philox_gaussian internally uses counters and counters+1
            ou_noise = self.ou_noise
            ou_noise.mul_(self._ou_decay).add_(noise * self._ou_std)
            new_V_soma = new_V_soma + ou_noise

        # Update voltages (copy_ avoids __setattr__ on nn.Module)
        V_soma.copy_(new_V_soma)
        V_dend.copy_(new_V_dend)

        # Ca spike check (BEFORE spike check, so it can influence soma)
        # When dendritic voltage exceeds theta_Ca, trigger a regenerative Ca spike
        ca_spike = (V_dend >= config.theta_Ca)
        if ca_spike.any():
            g_Ca_t.add_(ca_spike.float(), alpha=config.g_Ca_spike)

        # Refractory counter (in-place)
        refractory.sub_(1).clamp_(min=0)
        not_refractory = refractory == 0

        # Spike generation (somatic)
        above_threshold = V_soma >= v_threshold
        spikes = above_threshold & not_refractory

        if spikes.any():
            spikes_float = spikes.float()
            # Reset soma (copy_ avoids __setattr__ on nn.Module)
            V_soma.copy_(torch.where(spikes, v_reset, V_soma))
            # Set refractory
            ref_steps = (tau_ref / dt_ms).int()
            refractory.copy_(torch.where(spikes, ref_steps, refractory))
            # Adaptation increment
            g_adapt_t.add_(spikes_float * adapt_inc)
            # BAP: retrograde depolarisation of dendrite
            # V_dend += bap_amplitude * (E_Ca − V_dend)
            bap_dv = config.bap_amplitude * (config.E_Ca - V_dend) * spikes_float
            V_dend.add_(bap_dv)

        return spikes, VoltageTensor(V_soma), VoltageTensor(V_dend)
