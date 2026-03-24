"""Batched ConductanceLIF for the entire brain.

Concatenates all eligible ConductanceLIF populations across all regions into
contiguous global state tensors and runs ONE fused C++ kernel call per timestep,
eliminating ~128 individual Python ConductanceLIF.forward() calls.

**Eligibility**: Only pure ConductanceLIF instances (not subclasses like
SerotoninNeuron, NorepinephrineNeuron, AcetylcholineNeuron) are batched.
TwoCompartmentLIF populations are also excluded (separate neuron model).
Subclass neurons have custom ``_get_additional_conductances()`` that the
fused kernel cannot handle; they remain individually stepped.

**Gap junctions**: Populations with gap junctions (striatum FSI, thalamus TRN,
cerebellum IO, hippocampus inhibitory PV) ARE included. The caller must
compute gap junction conductances before calling ``step()`` and write them
into the global input buffers via ``set_gap_junction_input()``.

**State aliasing**: Individual ConductanceLIF modules' state tensors
(``V_soma``, ``g_E``, etc.) become views into the global arrays, so
diagnostic code that inspects per-population state continues to work.

Population registry maps ``(region_name, population_name)`` to ``(start, end)``
indices into the global arrays for zero-copy spike extraction.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import torch

from thalia.typing import PopulationKey
from thalia.utils.conductance_lif_fused import (
    conductance_lif_step as _clif_step_cpp,
    is_available as _clif_cpp_available,
)
from thalia.utils import decay_tensor

if TYPE_CHECKING:
    from .conductance_lif_neuron import ConductanceLIF

logger = logging.getLogger(__name__)


class ConductanceLIFBatch:
    """Batched ConductanceLIF state for all eligible populations in the brain.

    Holds global contiguous arrays for all neuron state variables.
    Individual :class:`ConductanceLIF` modules' state tensors become views
    into these arrays so state stays in sync.
    """

    def __init__(
        self,
        populations: list[tuple[PopulationKey, ConductanceLIF]],
        device: torch.device,
    ) -> None:
        """Initialize batched state from a list of (key, neuron_module) pairs.

        Args:
            populations: List of ((region_name, pop_name), neuron_module) pairs.
                Must be only pure ConductanceLIF (not subclasses).
            device: Torch device for all tensors.
        """
        # Sort for deterministic ordering
        entries = sorted(populations, key=lambda e: (e[0][0], e[0][1]))

        # Build registry: (region_name, pop_name) -> (start_idx, end_idx)
        total = 0
        self.registry: dict[PopulationKey, tuple[int, int]] = {}
        for key, neuron in entries:
            start = total
            total += neuron.n_neurons
            self.registry[key] = (start, total)

        self.total_neurons = total
        self.device = device
        self._entries = entries

        if total == 0:
            self._empty = True
            logger.info("ConductanceLIFBatch: no eligible populations")
            return
        self._empty = False

        # =====================================================================
        # Global state tensors
        # =====================================================================
        self.V_soma = torch.empty(total, device=device)
        self.g_E = torch.empty(total, device=device)
        self.g_I = torch.empty(total, device=device)
        self.g_nmda = torch.empty(total, device=device)
        self.g_GABA_B = torch.empty(total, device=device)
        self.g_adapt = torch.empty(total, device=device)
        self.ou_noise = torch.empty(total, device=device)
        self.refractory = torch.empty(total, dtype=torch.int32, device=device)

        # Optional channel state (allocated at full size; unused neurons store 0)
        self.h_gate = torch.zeros(total, device=device)
        self.h_T = torch.zeros(total, device=device)

        # =====================================================================
        # Per-neuron parameters (constant across steps, set at init & dt change)
        # =====================================================================
        self.g_E_decay = torch.empty(total, device=device)
        self.g_I_decay = torch.empty(total, device=device)
        self.g_nmda_decay = torch.empty(total, device=device)
        self.g_GABA_B_decay = torch.empty(total, device=device)
        self.adapt_decay = torch.empty(total, device=device)
        self.V_soma_decay = torch.empty(total, device=device)
        self.g_L = torch.empty(total, device=device)
        self.g_L_scale = torch.empty(total, device=device)
        self.v_threshold = torch.empty(total, device=device)
        self.adapt_increment = torch.empty(total, device=device)
        self.tau_ref_per_neuron = torch.empty(total, device=device)
        self._neuron_seeds = torch.empty(total, dtype=torch.int64, device=device)
        self._rng_timestep = torch.zeros(total, dtype=torch.int64, device=device)

        # Per-neuron config scalars that vary across populations
        # (reversal potentials, NMDA params, noise, etc.)
        self._nmda_a = torch.empty(total, device=device)
        self._nmda_b = torch.empty(total, device=device)
        self._ou_decay = torch.empty(total, device=device)
        self._ou_std = torch.empty(total, device=device)

        # Per-neuron boolean feature masks
        self._enable_noise = torch.empty(total, dtype=torch.bool, device=device)
        self._enable_ih = torch.empty(total, dtype=torch.bool, device=device)
        self._enable_t = torch.empty(total, dtype=torch.bool, device=device)

        # Per-neuron I_h parameters (safe non-zero default for k_h to prevent
        # division-by-zero in the C++ kernel when I_h is globally enabled but
        # disabled for specific neurons)
        self._h_decay = torch.zeros(total, device=device)
        self._g_h_max = torch.zeros(total, device=device)
        self._E_h = torch.zeros(total, device=device)
        self._V_half_h = torch.zeros(total, device=device)
        self._k_h = torch.ones(total, device=device)  # non-zero default prevents 0/0 NaN

        # Per-neuron T-channel parameters (safe non-zero defaults for k_h_T
        # to prevent division-by-zero in the C++ kernel when T-channels are
        # globally enabled but disabled for specific neurons)
        self._h_T_decay = torch.zeros(total, device=device)
        self._g_T = torch.zeros(total, device=device)
        self._E_Ca = torch.zeros(total, device=device)
        self._V_half_h_T = torch.zeros(total, device=device)
        self._k_h_T = torch.ones(total, device=device)  # non-zero default prevents 0/0 NaN

        # Per-neuron constants (reversal potentials, NMDA params, dendrite coupling)
        # Now fully per-neuron to match the kernel's tensor API
        self.v_reset = torch.empty(total, device=device)
        self.E_E = torch.empty(total, device=device)
        self.E_I = torch.empty(total, device=device)
        self.E_nmda = torch.empty(total, device=device)
        self.E_GABA_B = torch.empty(total, device=device)
        self.E_adapt = torch.empty(total, device=device)
        self._dendrite_coupling_scale = torch.empty(total, device=device)
        self._g_nmda_max = torch.full((total,), float('inf'), device=device)

        # =====================================================================
        # Input buffers (filled each step by regions)
        # =====================================================================
        self.g_ampa_input = torch.zeros(total, device=device)
        self.g_nmda_input = torch.zeros(total, device=device)
        self.g_gaba_a_input = torch.zeros(total, device=device)
        self.g_gaba_b_input = torch.zeros(total, device=device)
        self.g_gap_input = torch.zeros(total, device=device)
        self.E_gap_reversal = torch.zeros(total, device=device)

        # Track which populations have gap junctions
        self._has_any_gap_junctions = False

        # =====================================================================
        # Copy state & params from individual modules and alias state
        # =====================================================================
        for key, neuron in entries:
            start, end = self.registry[key]
            s = slice(start, end)
            n = neuron.n_neurons
            config = neuron.config

            # Copy current state
            self.V_soma[s] = neuron.V_soma
            self.g_E[s] = neuron.g_E
            self.g_I[s] = neuron.g_I
            self.g_nmda[s] = neuron.g_nmda
            self.g_GABA_B[s] = neuron.g_GABA_B
            self.g_adapt[s] = neuron.g_adapt
            self.ou_noise[s] = neuron.ou_noise
            if neuron.refractory is not None:
                self.refractory[s] = neuron.refractory
            else:
                # Refractory not yet initialized; replicate the lazy init logic
                self.refractory[s] = (neuron._u_refractory_init * neuron.tau_ref_per_neuron / 1.0).to(torch.int32)

            # Copy optional channel state
            if neuron.h_gate is not None:
                self.h_gate[s] = neuron.h_gate
            if neuron.h_T is not None:
                self.h_T[s] = neuron.h_T

            # Copy per-neuron parameters
            self.g_L[s] = neuron.g_L
            self.g_L_scale[s] = neuron.g_L_scale
            self.v_threshold[s] = neuron.v_threshold
            self.adapt_increment[s] = neuron.adapt_increment
            self.tau_ref_per_neuron[s] = neuron.tau_ref_per_neuron
            self._neuron_seeds[s] = neuron._neuron_seeds
            self._rng_timestep[s] = neuron._rng_timestep

            # NMDA block constants (now per-neuron tensors on the neuron)
            self._nmda_a[s] = neuron._nmda_a
            self._nmda_b[s] = neuron._nmda_b

            # Feature masks
            self._enable_noise[s] = neuron._has_noise
            self._enable_ih[s] = config.enable_ih
            self._enable_t[s] = config.enable_t_channels

            # I_h parameters (from per-neuron buffers, zero for disabled)
            self._g_h_max[s] = neuron._g_h_max
            self._E_h[s] = neuron._E_h
            self._V_half_h[s] = neuron._V_half_h
            self._k_h[s] = neuron._k_h

            # T-channel parameters (from per-neuron buffers, zero for disabled)
            self._g_T[s] = neuron._g_T
            self._E_Ca[s] = neuron._E_Ca
            self._V_half_h_T[s] = neuron._V_half_h_T
            self._k_h_T[s] = neuron._k_h_T

            # Per-neuron constants (reversal potentials, v_reset, etc.)
            self.v_reset[s] = neuron.v_reset
            self.E_E[s] = neuron.E_E
            self.E_I[s] = neuron.E_I
            self.E_nmda[s] = neuron.E_nmda
            self.E_GABA_B[s] = neuron.E_GABA_B
            self.E_adapt[s] = neuron.E_adapt
            self._dendrite_coupling_scale[s] = neuron._dendrite_coupling_scale
            self._g_nmda_max[s] = neuron._g_nmda_max

            # ── Alias individual module state with views ──
            neuron.V_soma = self.V_soma[s]
            neuron.g_E = self.g_E[s]
            neuron.g_I = self.g_I[s]
            neuron.g_nmda = self.g_nmda[s]
            neuron.g_GABA_B = self.g_GABA_B[s]
            neuron.g_adapt = self.g_adapt[s]
            neuron.ou_noise = self.ou_noise[s]
            neuron.refractory = self.refractory[s]
            neuron.g_L_scale = self.g_L_scale[s]
            neuron.v_threshold = self.v_threshold[s]
            if config.enable_ih:
                neuron.h_gate = self.h_gate[s]
            if config.enable_t_channels:
                neuron.h_T = self.h_T[s]

        # =====================================================================
        # Pre-allocated spike buffer and per-neuron batch references
        # =====================================================================
        # _last_spikes holds the most recent spikes.  step() fills it **in-place**
        # so that views handed out by _setup_batch_refs remain valid.
        self._last_spikes = torch.zeros(total, dtype=torch.bool, device=device)

        # Tell each ConductanceLIF about its batch slice so that
        # ConductanceLIF.forward() can write inputs here instead of computing.
        for key, neuron in entries:
            start, end = self.registry[key]
            neuron._setup_batch_ref(self, start, end)

        self._use_cpp = _clif_cpp_available()
        self._dt_ms: Optional[float] = None

        logger.info(
            "ConductanceLIFBatch: %d populations, %d total neurons, C++=%s",
            len(entries), total, self._use_cpp,
        )

    # =====================================================================
    # INPUT BUFFER MANAGEMENT
    # =====================================================================

    def clear_inputs(self) -> None:
        """Zero all input buffers. Call at the start of each timestep."""
        if self._empty:
            return
        self.g_ampa_input.zero_()
        self.g_nmda_input.zero_()
        self.g_gaba_a_input.zero_()
        self.g_gaba_b_input.zero_()
        self.g_gap_input.zero_()
        self.E_gap_reversal.zero_()
        self._has_any_gap_junctions = False

    def get_input_views(self, key: PopulationKey) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get writable views into the global input buffers for a population.

        Returns:
            (g_ampa, g_nmda, g_gaba_a, g_gaba_b) views, each shape [n_neurons].
            Write synaptic integration results directly into these.
        """
        start, end = self.registry[key]
        return (
            self.g_ampa_input[start:end],
            self.g_nmda_input[start:end],
            self.g_gaba_a_input[start:end],
            self.g_gaba_b_input[start:end],
        )

    def set_gap_junction_input(
        self,
        key: PopulationKey,
        g_gap: torch.Tensor,
        E_gap: torch.Tensor,
    ) -> None:
        """Set gap junction conductance and reversal for a population."""
        start, end = self.registry[key]
        self.g_gap_input[start:end] = g_gap
        self.E_gap_reversal[start:end] = E_gap
        self._has_any_gap_junctions = True

    def get_population_voltage(self, key: PopulationKey) -> torch.Tensor:
        """Get a read-only view of V_soma for a population (e.g. for gap junctions)."""
        start, end = self.registry[key]
        return self.V_soma[start:end]

    # =====================================================================
    # MAIN STEP
    # =====================================================================

    @torch.no_grad()
    def step(self) -> torch.Tensor:
        """Run one timestep for ALL batched neurons.

        Input buffers must be filled before calling this (via get_input_views,
        set_gap_junction_input, or ConductanceLIF.forward() in batch mode).

        Spikes are written **in-place** into the pre-allocated ``_last_spikes``
        buffer so that per-population spike views (returned by
        ``ConductanceLIF.forward()`` in batch mode) remain valid.

        Returns:
            Global spike tensor [total_neurons] (bool). Use get_spikes()
            to extract per-population views.
        """
        if self._empty:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        assert self._dt_ms is not None, "Call update_temporal_parameters() before step()"

        if self._use_cpp:
            new_spikes = self._step_cpp()
        else:
            new_spikes = self._step_per_population()

        # Fill in-place so existing views into _last_spikes see the new data.
        self._last_spikes.copy_(new_spikes)
        return self._last_spikes

    def _step_cpp(self) -> torch.Tensor:
        """Fused C++ path: one kernel call for all neurons."""
        # For the batched path, enable optional features if ANY neuron has them.
        # Neurons without the feature have g_h_max=0 / g_T=0, so they contribute zero.
        any_noise = bool(self._enable_noise.any())
        any_ih = bool(self._enable_ih.any())
        any_t = bool(self._enable_t.any())

        spikes = _clif_step_cpp(
            # State tensors (modified in-place)
            V_soma=self.V_soma,
            g_E=self.g_E,
            g_I=self.g_I,
            g_nmda=self.g_nmda,
            g_GABA_B=self.g_GABA_B,
            g_adapt=self.g_adapt,
            ou_noise=self.ou_noise,
            refractory=self.refractory,
            # Synaptic inputs
            g_ampa_input=self.g_ampa_input,
            g_nmda_input=self.g_nmda_input,
            g_gaba_a_input=self.g_gaba_a_input,
            g_gaba_b_input=self.g_gaba_b_input,
            # Per-neuron parameter tensors
            g_E_decay=self.g_E_decay,
            g_I_decay=self.g_I_decay,
            g_nmda_decay=self.g_nmda_decay,
            g_GABA_B_decay=self.g_GABA_B_decay,
            adapt_decay=self.adapt_decay,
            V_soma_decay=self.V_soma_decay,
            g_L=self.g_L,
            g_L_scale=self.g_L_scale,
            v_threshold=self.v_threshold,
            adapt_increment=self.adapt_increment,
            tau_ref_per_neuron=self.tau_ref_per_neuron,
            # Per-neuron constants (all tensors now)
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
            dt_ms=self._dt_ms,
            # Noise (per-neuron seeds and params)
            enable_noise=any_noise,
            neuron_seeds=self._neuron_seeds if any_noise else torch.zeros(1, dtype=torch.int64, device=self.device),
            rng_timestep=int(self._rng_timestep[0].item()),
            ou_decay=self._ou_decay,
            ou_std=self._ou_std,
            # Gap junctions
            has_gap_junctions=self._has_any_gap_junctions,
            g_gap_input=self.g_gap_input,
            E_gap_reversal=self.E_gap_reversal,
            # T-channels (per-neuron; zero for neurons without)
            enable_t_channels=any_t,
            h_T=self.h_T,
            h_T_decay=self._h_T_decay,
            g_T=self._g_T,
            E_Ca=self._E_Ca,
            V_half_h_T=self._V_half_h_T,
            k_h_T=self._k_h_T,
            # I_h (HCN) (per-neuron; zero for neurons without)
            enable_ih=any_ih,
            h_gate=self.h_gate,
            h_decay=self._h_decay,
            g_h_max=self._g_h_max,
            E_h=self._E_h,
            V_half_h=self._V_half_h,
            k_h=self._k_h,
        )

        # Advance RNG timestep for all neurons
        if any_noise:
            self._rng_timestep.add_(2)
            # Keep individual modules' counters in sync
            for _key, neuron in self._entries:
                neuron._rng_timestep.add_(2)

        return spikes

    def _step_per_population(self) -> torch.Tensor:
        """Fallback: step each population individually (Python path).

        Used when C++ kernel is unavailable.
        Still benefits from aliased state (no copy needed).
        """
        all_spikes = torch.zeros(self.total_neurons, dtype=torch.bool, device=self.device)

        for key, neuron in self._entries:
            start, end = self.registry[key]
            s = slice(start, end)

            g_gap = self.g_gap_input[s] if self._has_any_gap_junctions else None
            e_gap = self.E_gap_reversal[s] if self._has_any_gap_junctions else None

            # Check if this population actually has non-zero gap input
            if g_gap is not None and not g_gap.any():
                g_gap = None
                e_gap = None

            spikes, _ = neuron.forward(
                g_ampa_input=self.g_ampa_input[s],
                g_nmda_input=self.g_nmda_input[s],
                g_gaba_a_input=self.g_gaba_a_input[s],
                g_gaba_b_input=self.g_gaba_b_input[s],
                g_gap_input=g_gap,
                E_gap_reversal=e_gap,
            )
            all_spikes[s] = spikes

        return all_spikes

    # =====================================================================
    # SPIKE EXTRACTION
    # =====================================================================

    def get_spikes(self, spikes_all: torch.Tensor, key: PopulationKey) -> torch.Tensor:
        """Extract a zero-copy spike view for a single population."""
        start, end = self.registry[key]
        return spikes_all[start:end]

    def get_membrane(self, key: PopulationKey) -> torch.Tensor:
        """Extract a zero-copy membrane voltage view for a single population."""
        start, end = self.registry[key]
        return self.V_soma[start:end]

    # =====================================================================
    # TEMPORAL PARAMETER UPDATES
    # =====================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Recompute decay factors when dt changes."""
        if self._empty:
            return

        self._dt_ms = dt_ms

        for key, neuron in self._entries:
            start, end = self.registry[key]
            s = slice(start, end)
            config = neuron.config

            # Compute decay tensors using per-neuron tau_mem (heterogeneous)
            self.g_E_decay[s] = decay_tensor(dt_ms, config.tau_E, self.device)
            self.g_I_decay[s] = decay_tensor(dt_ms, config.tau_I, self.device)
            self.g_nmda_decay[s] = decay_tensor(dt_ms, config.tau_nmda, self.device)
            self.g_GABA_B_decay[s] = decay_tensor(dt_ms, config.tau_GABA_B, self.device)
            self.adapt_decay[s] = torch.exp(-dt_ms / neuron.tau_adapt_ms)
            # V_soma_decay is per-neuron (heterogeneous tau_mem)
            self.V_soma_decay[s] = torch.exp(-dt_ms / neuron.tau_mem_ms)

            # Noise parameters (noise_std is per-neuron)
            ou_decay_val = math.exp(-dt_ms / config.noise_tau_ms)
            self._ou_decay[s] = ou_decay_val
            if neuron._has_noise:
                scale = math.sqrt(1.0 - ou_decay_val**2)
                self._ou_std[s] = neuron.noise_std * scale
            else:
                self._ou_std[s] = 0.0

            # I_h decay
            if config.enable_ih:
                self._h_decay[s] = math.exp(-dt_ms / config.tau_h_ms)

            # T-channel decay
            if config.enable_t_channels:
                self._h_T_decay[s] = math.exp(-dt_ms / config.tau_h_T_ms)

        # Also let individual modules update their own cached scalars
        # (needed for the fallback per-population path)
        for _key, neuron in self._entries:
            neuron.update_temporal_parameters(dt_ms)

    # =====================================================================
    # UTILITY
    # =====================================================================

    def is_batched(self, key: PopulationKey) -> bool:
        """Check if a population is in this batch."""
        return key in self.registry

    @property
    def population_keys(self) -> list[PopulationKey]:
        """All population keys in deterministic order."""
        return [key for key, _ in self._entries]
