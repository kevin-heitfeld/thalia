"""
DiagnosticsRecorder: Training-loop-integrated brain diagnostics.

Architecture
============
The recorder is completely decoupled from the simulation loop.  It never
drives its own loop; instead, call ``record(t, inputs, outputs)`` once per
timestep from whatever loop you own (training, evaluation, pre-training
diagnostic run).
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from thalia.brain.synapses import NeuromodulatorReceptor
from thalia.typing import BrainOutput, PopulationName, RegionName, SynapseId, SynapticInput

from .diagnostics_types import DiagnosticsConfig, RecorderSnapshot

if TYPE_CHECKING:
    from thalia.brain import Brain
    from thalia.brain.neurons import ConductanceLIF, TwoCompartmentLIF


# =============================================================================
# HELPERS
# =============================================================================


def _get_attr_fallback(obj: Any, *attrs: str) -> Optional[torch.Tensor]:
    """Return the first non-None attribute from *obj*, or ``None`` if all are absent.

    Used to transparently handle models that expose conductances under
    different names (e.g. ``g_E`` on :class:`ConductanceLIF` vs
    ``g_E_basal`` on :class:`TwoCompartmentLIF`).
    """
    for a in attrs:
        v = getattr(obj, a, None)
        if v is not None:
            return v
    return None


# =============================================================================
# DIAGNOSTICS RECORDER
# =============================================================================


class DiagnosticsRecorder:
    """Records brain activity for comprehensive diagnostics.

    Completely decoupled from the simulation loop: call ``record(t, inputs, outputs)``
    once per timestep from whatever loop you own, then ``analyze()`` to get a
    :class:`DiagnosticsReport`.

    See module docstring for a usage example.
    """

    def __init__(self, brain: Brain, config: DiagnosticsConfig) -> None:
        self.brain = brain
        self.config = config
        self.dt_ms = brain.dt_ms

        # These are fully set by _allocate_buffers(); declared here so Pylance
        # knows they belong to the instance (not defined outside __init__).
        self._gain_sample_step: int = 0
        self._cond_sample_step: int = 0
        self._voltages: Optional[np.ndarray] = None
        self._g_exc_samples: Optional[np.ndarray] = None
        self._g_inh_samples: Optional[np.ndarray] = None
        self._g_nmda_samples: Optional[np.ndarray] = None
        self._g_gaba_b_samples: Optional[np.ndarray] = None
        self._g_apical_samples: Optional[np.ndarray] = None  # g_E_apical for TwoCompartmentLIF only
        self._nm_concentration_history: np.ndarray  # allocated in _allocate_buffers

        self._pop_config_types: Dict[Tuple[RegionName, PopulationName], str] = {}

        self._build_index()
        self._print_memory_estimate()
        self._allocate_buffers()

        self._recording_started = False
        self._n_recorded = 0

    # =========================================================================
    # INDEX CONSTRUCTION
    # =========================================================================

    def _build_index(self) -> None:
        """Build ordered population and tract indices from the brain."""
        # Ordered list of (region_name, pop_name) tuples.
        self._pop_keys: List[Tuple[RegionName, PopulationName]] = []
        for region_name, region in self.brain.regions.items():
            for pop_name in region.neuron_populations.keys():
                self._pop_keys.append((region_name, pop_name))
        self._pop_index: Dict[Tuple[RegionName, PopulationName], int] = {
            key: idx for idx, key in enumerate(self._pop_keys)
        }
        self._n_pops = len(self._pop_keys)

        # Population sizes
        self._pop_sizes = np.zeros(self._n_pops, dtype=np.int32)
        for idx, (rn, pn) in enumerate(self._pop_keys):
            pop_obj = self.brain.regions[rn].neuron_populations[pn]
            self._pop_sizes[idx] = pop_obj.n_neurons

        # Ordered region names
        self._region_keys: List[RegionName] = list(self.brain.regions.keys())
        self._region_index: Dict[RegionName, int] = {k: i for i, k in enumerate(self._region_keys)}
        self._n_regions = len(self._region_keys)

        # Region → population indices
        self._region_pop_indices: Dict[RegionName, List[int]] = defaultdict(list)
        for idx, (rn, _) in enumerate(self._pop_keys):
            self._region_pop_indices[rn].append(idx)

        # Axonal tracts
        self._tract_keys: List[SynapseId] = list(self.brain.axonal_tracts.keys())
        self._tract_index: Dict[SynapseId, int] = {
            key: idx for idx, key in enumerate(self._tract_keys)
        }
        self._n_tracts = len(self._tract_keys)

        # STP modules — ordered list of (region, synapse_id) for trajectory sampling
        self._stp_keys: List[Tuple[RegionName, SynapseId]] = []
        self._stp_configs: List[Tuple[float, float, float]] = []
        for rn, region in self.brain.regions.items():
            for syn_id, stp_mod in region.stp_modules.items():
                self._stp_keys.append((rn, syn_id))
                cfg = stp_mod.config
                self._stp_configs.append((float(cfg.U), float(cfg.tau_d), float(cfg.tau_f)))
        self._n_stp = len(self._stp_keys)

        # NeuromodulatorReceptor submodules — one entry per receptor instance
        # across all regions, identified by "region_name/module.attr.path".
        self._nm_receptor_keys: List[Tuple[RegionName, str]] = []  # (region_name, dotted_attr_path)
        for rn, region in self.brain.regions.items():
            for mod_name, mod in region.named_modules():
                if isinstance(mod, NeuromodulatorReceptor) and mod_name:  # skip region itself (mod_name="")
                    self._nm_receptor_keys.append((rn, mod_name))
        self._n_nm_receptors = len(self._nm_receptor_keys)

        # Neuromodulator source populations — (region_name, pop_name_str) tuples for
        # the health check: "is any neuromodulator source firing?"
        self._nm_source_pop_keys: List[Tuple[RegionName, PopulationName]] = []
        for rn, region in self.brain.regions.items():
            nm_outputs = getattr(region, "neuromodulator_outputs", None)
            if nm_outputs:
                for pop_name in nm_outputs.values():
                    self._nm_source_pop_keys.append((rn, str(pop_name)))

        # ── Static brain metadata captured once for snapshot serialisation ──────────
        # Population polarity (StrEnum → str so it is JSON-safe)
        self._pop_polarities: Dict[Tuple[RegionName, PopulationName], str] = {}
        for rn, region in self.brain.regions.items():
            pols = region._population_polarities
            for pn in region.neuron_populations.keys():
                pol = pols.get(pn)
                self._pop_polarities[(rn, pn)] = str(pol) if pol is not None else "any"

        # Axonal delay per tract (ms), same order as _tract_keys
        self._tract_delay_ms: List[float] = [
            self.brain.axonal_tracts[sid].spec.delay_ms
            for sid in self._tract_keys
        ]

        # Homeostatic target firing rate (Hz) per population
        self._homeostasis_target_hz: Dict[Tuple[RegionName, PopulationName], float] = {}
        for rn, region in self.brain.regions.items():
            homeostasis = region._homeostasis
            for pn in region.neuron_populations.keys():
                hs = homeostasis.get(pn, None)
                if hs is not None:
                    if hs.target_firing_rate > 0:
                        self._homeostasis_target_hz[(rn, pn)] = hs.target_firing_rate * 1000.0

        # Fixed random neuron samples per population (seed=42 for reproducibility)
        rng = np.random.default_rng(seed=42)
        V = self.config.voltage_sample_size
        C = self.config.conductance_sample_size
        self._v_sample_idx: List[np.ndarray] = []
        self._c_sample_idx: List[np.ndarray] = []
        for size in self._pop_sizes:
            n_v = min(V, int(size))
            self._v_sample_idx.append(
                rng.choice(int(size), size=n_v, replace=False) if n_v > 0 else np.array([], dtype=int)
            )
            n_c = min(C, int(size))
            self._c_sample_idx.append(
                rng.choice(int(size), size=n_c, replace=False) if n_c > 0 else np.array([], dtype=int)
            )

    def _print_memory_estimate(self) -> None:
        """Print a pre-flight memory estimate when total allocation exceeds 10 MB.

        Called after ``_build_index()`` and before ``_allocate_buffers()``.
        """
        T = self.config.n_timesteps
        P = self._n_pops
        R = self._n_regions
        V = self.config.voltage_sample_size
        C = self.config.conductance_sample_size
        ci = self.config.conductance_sample_interval_steps
        gi_steps = max(1, int(self.config.gain_sample_interval_ms / self.dt_ms))
        n_gain = max(1, T // gi_steps)
        n_cond = max(1, T // ci)
        total_neurons = int(self._pop_sizes.sum())

        # Four bytes per element (float32 / int32).
        B = 4
        rows: list[tuple[str, int]] = [
            ("pop spike counts", T * P * B),
            ("per-neuron spike counts", total_neurons * B),
            ("region spike counts", T * R * B),
            ("tract sent", T * self._n_tracts * B),
            ("gain history", n_gain * P * B),
            ("STP history", n_gain * self._n_stp * B),
            ("NM history", n_gain * max(1, self._n_nm_receptors) * B),
            ("voltages", T * P * V * B),
            ("conductances (×5)", n_cond * P * C * B * 5),
            ("spike flat buffers", 2 * max(64, T // 10) * P * B),
        ]

        total_bytes = sum(b for _, b in rows)
        _MB = 1024 ** 2
        if total_bytes < 10 * _MB:
            return

        def _fmt(b: int) -> str:
            if b >= _MB:
                return f"{b / _MB:7.1f} MB"
            return f"{b / 1024:7.1f} KB"

        print(f"\n{'═'*80}")
        print(f"  DiagnosticsRecorder buffer estimate  "
              f"(T={T}, P={P}, R={R}):")
        print(f"{'═'*80}\n")
        for label, nbytes in rows:
            if nbytes > 0:
                print(f"    {label:<30s}: {_fmt(nbytes)}")
        print(f"    {'─' * 42}")
        print(f"    {'Total':<30s}: {_fmt(total_bytes)}")

    def _allocate_buffers(self) -> None:
        """Pre-allocate recording buffers."""
        T = self.config.n_timesteps
        P = self._n_pops
        R = self._n_regions
        V = self.config.voltage_sample_size
        C = self.config.conductance_sample_size
        self._allocate_spike_buffers(T, P, R)
        self._allocate_state_buffers(T, P, C, V)
        self._allocate_trajectory_buffers(T, P)

    def _allocate_spike_buffers(self, T: int, P: int, R: int) -> None:
        """Allocate spike-count, spike-time, and tract-transmission buffers."""
        # Population-level spike counts [T × P]
        self._pop_spike_counts = np.zeros((T, P), dtype=np.int32)

        # Per-neuron cumulative spike counts — accumulated in record() for every population.
        # Memory cost: sum(n_neurons) × 4 bytes — negligible vs voltage buffers.
        self._per_neuron_spike_counts: List[np.ndarray] = [
            np.zeros(int(sz), dtype=np.int32) for sz in self._pop_sizes
        ]

        # Region-level total spike counts [T × R] (for cross-correlation/coherence)
        self._region_spike_counts = np.zeros((T, R), dtype=np.int32)

        # Axonal tract: spikes sent from source at each timestep [T × n_tracts]
        self._tract_sent = np.zeros((T, self._n_tracts), dtype=np.int32)

        # Per-neuron spike times: (region, pop) → list-of-lists
        # Populated lazily in analyze() via _build_spike_times().
        self._spike_times: Dict[Tuple[str, str], List[List[int]]] = {}

        # Flat spike-time accumulation buffers — avoids per-neuron Python loop
        # in the hot path.  _spike_times is built once in analyze().
        initial_cap = max(64, T // 10)
        self._spike_flat_nidx: List[np.ndarray] = [
            np.empty(initial_cap, dtype=np.int32) for _ in range(P)
        ]
        self._spike_flat_ts: List[np.ndarray] = [
            np.empty(initial_cap, dtype=np.int32) for _ in range(P)
        ]
        self._spike_flat_n: np.ndarray = np.zeros(P, dtype=np.int64)

    def _allocate_state_buffers(self, T: int, P: int, C: int, V: int) -> None:
        """Allocate voltage and conductance sample buffers."""
        # Voltage traces [T × P × V]
        self._voltages = np.full((T, P, V), np.nan, dtype=np.float32)

        # Conductance samples – every conductance_sample_interval_steps timesteps
        ci = self.config.conductance_sample_interval_steps
        n_cond = max(1, T // ci)
        self._g_exc_samples  = np.full((n_cond, P, C), np.nan, dtype=np.float32)
        self._g_inh_samples  = np.full((n_cond, P, C), np.nan, dtype=np.float32)
        self._g_nmda_samples = np.full((n_cond, P, C), np.nan, dtype=np.float32)
        self._g_gaba_b_samples = np.full((n_cond, P, C), np.nan, dtype=np.float32)
        self._g_apical_samples = np.full((n_cond, P, C), np.nan, dtype=np.float32)
        self._cond_sample_step = 0

    def _allocate_trajectory_buffers(self, T: int, P: int) -> None:
        """Allocate homeostatic gain, STP efficacy, and neuromodulator history buffers."""
        # Homeostatic gain traces – every gain_sample_interval_ms timesteps (both modes)
        gi_steps = max(1, int(self.config.gain_sample_interval_ms / self.dt_ms))
        n_gain = max(1, T // gi_steps)
        self._g_L_scale_history = np.full((n_gain, P), np.nan, dtype=np.float32)
        self._gain_sample_step = 0
        self._gain_sample_times: List[int] = []

        # STP efficacy (x·u) trajectories — same sampling schedule as gain
        self._stp_efficacy_history = np.full((n_gain, self._n_stp), np.nan, dtype=np.float32)

        # Neuromodulator receptor concentration histories — same sampling schedule as gain
        # Shape: [n_gain, n_nm_receptors].  Mean concentration per receptor instance.
        self._nm_concentration_history = np.full(
            (n_gain, max(1, self._n_nm_receptors)), np.nan, dtype=np.float32
        )

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self) -> None:
        """Re-zero all recording buffers in-place (faster than re-allocating).

        Call before starting a new recording window.  Re-uses the same
        pre-allocated arrays so memory stays stable across repeated runs.
        """
        self._pop_spike_counts.fill(0)
        for arr in self._per_neuron_spike_counts:
            arr.fill(0)
        self._region_spike_counts.fill(0)
        self._tract_sent.fill(0)
        self._spike_times.clear()
        self._pop_config_types.clear()
        self._spike_flat_n.fill(0)
        self._gain_sample_times.clear()
        self._gain_sample_step = 0
        if self._voltages is not None:
            self._voltages.fill(np.nan)
        if self._g_exc_samples is not None:
            self._g_exc_samples.fill(np.nan)
        if self._g_inh_samples is not None:
            self._g_inh_samples.fill(np.nan)
        if self._g_nmda_samples is not None:
            self._g_nmda_samples.fill(np.nan)
        if self._g_gaba_b_samples is not None:
            self._g_gaba_b_samples.fill(np.nan)
        if self._g_apical_samples is not None:
            self._g_apical_samples.fill(np.nan)
        self._cond_sample_step = 0
        self._g_L_scale_history.fill(np.nan)
        self._stp_efficacy_history.fill(np.nan)
        self._nm_concentration_history.fill(np.nan)
        self._recording_started = False
        self._n_recorded = 0

    # =========================================================================
    # RECORD  — the single integration point
    # =========================================================================

    def record(self, timestep: int, inputs: SynapticInput, outputs: BrainOutput) -> None:
        """Record one timestep of brain output.

        Call this immediately after ``brain.forward()`` inside your loop.

        Args:
            timestep: Zero-based timestep index (must be < config.n_timesteps).
            inputs: Dict of synaptic inputs to the brain at this timestep.
            outputs: Dict ``{region_name: {pop_name: spike_tensor}}`` returned
                by ``brain.forward()``.
        """
        self._recording_started = True
        t = timestep

        if t >= self.config.n_timesteps:
            warnings.warn(
                f"DiagnosticsRecorder: timestep {t} exceeds n_timesteps "
                f"({self.config.n_timesteps}).  Ignoring.",
                stacklevel=2,
            )
            return

        # --- Spike counts per population & per region ----------------------------------
        for region_name, region_outputs in outputs.items():
            region_idx = self._region_index.get(region_name)
            region_total = 0

            for pop_name, pop_spikes in region_outputs.items():
                key = (region_name, pop_name)
                pop_idx = self._pop_index.get(key)
                if pop_idx is None:
                    continue

                n_active = int(pop_spikes.sum().item())
                self._pop_spike_counts[t, pop_idx] = n_active
                region_total += n_active

                if n_active > 0:
                    spike_arr = pop_spikes.cpu().numpy()
                    # Flat-buffer accumulation — one numpy slice per pop per step.
                    # _spike_times is built from these buffers in analyze().
                    nidx = np.where(spike_arr)[0].astype(np.int32)
                    n_new = len(nidx)
                    cnt = int(self._spike_flat_n[pop_idx])
                    cap = len(self._spike_flat_nidx[pop_idx])
                    if cnt + n_new > cap:
                        new_cap = max(cnt + n_new, cap * 2)
                        self._spike_flat_nidx[pop_idx] = np.resize(
                            self._spike_flat_nidx[pop_idx], new_cap
                        )
                        self._spike_flat_ts[pop_idx] = np.resize(
                            self._spike_flat_ts[pop_idx], new_cap
                        )
                    self._spike_flat_nidx[pop_idx][cnt:cnt + n_new] = nidx
                    self._spike_flat_ts[pop_idx][cnt:cnt + n_new] = t
                    self._spike_flat_n[pop_idx] = cnt + n_new
                    # Per-neuron cumulative counts — enables correct
                    # fraction_silent / fraction_hyperactive.
                    self._per_neuron_spike_counts[pop_idx] += spike_arr.astype(np.int32)

            if region_idx is not None:
                self._region_spike_counts[t, region_idx] = region_total

        # --- Axonal tract transmissions ------------------------------------------------
        for synapse_id, tract_idx in self._tract_index.items():
            src_spikes = outputs.get(synapse_id.source_region, {}).get(
                synapse_id.source_population, None
            )
            if src_spikes is not None:
                self._tract_sent[t, tract_idx] = int(src_spikes.sum().item())

        # --- Internal state sampling ---------------------------------------------------
        gi_steps = max(1, int(self.config.gain_sample_interval_ms / self.dt_ms))
        if t % gi_steps == 0:
            self._sample_gains(t)

        self._sample_voltages(t)
        ci = self.config.conductance_sample_interval_steps
        if ci > 0 and t % ci == 0:
            self._sample_conductances(t)

        self._n_recorded = t + 1

    def _sample_gains(self, timestep: int) -> None:
        """Sample homeostatic g_L_scale and STP efficacy for all populations/synapses."""
        step = self._gain_sample_step
        if step >= self._g_L_scale_history.shape[0]:
            return

        for idx, (rn, pn) in enumerate(self._pop_keys):
            pop_obj = self.brain.regions[rn].neuron_populations[pn]
            if hasattr(pop_obj, "g_L_scale") and pop_obj.g_L_scale is not None:
                val = pop_obj.g_L_scale
                self._g_L_scale_history[step, idx] = float(
                    val.mean().item() if isinstance(val, torch.Tensor) else val
                )

        # STP x·u efficacy
        for stp_idx, (rn, syn_id) in enumerate(self._stp_keys):
            stp_mod = self.brain.regions[rn].get_stp_module(syn_id)
            assert stp_mod is not None
            self._stp_efficacy_history[step, stp_idx] = float(
                (stp_mod.x * stp_mod.u).mean().item()
            )

        # Neuromodulator receptor concentrations
        if self._n_nm_receptors > 0:
            for nm_idx, (rn, mod_name) in enumerate(self._nm_receptor_keys):
                try:
                    obj: Any = self.brain.regions[rn]
                    for part in mod_name.split("."):
                        obj = getattr(obj, part)
                    conc: torch.Tensor = obj.concentration
                    self._nm_concentration_history[step, nm_idx] = float(
                        conc.mean().item()
                    )
                except (AttributeError, KeyError):
                    pass

        self._gain_sample_times.append(timestep)
        self._gain_sample_step += 1

    def _sample_voltages(self, timestep: int) -> None:
        """Sample membrane voltages for fixed neurons in each population."""
        if self._voltages is None:
            raise ValueError("_voltages buffer not allocated")
        t = timestep
        if t >= self._voltages.shape[0]:
            return

        for idx, (rn, pn) in enumerate(self._pop_keys):
            if rn not in self.brain.regions:
                raise ValueError(f"Region '{rn}' not found in brain")
            if pn not in self.brain.regions[rn].neuron_populations:
                raise ValueError(f"Population '{pn}' not found in region '{rn}'")
            pop_obj: Union[ConductanceLIF, TwoCompartmentLIF] = self.brain.regions[rn].neuron_populations[pn]
            v_idx = self._v_sample_idx[idx]
            if len(v_idx) == 0:
                continue
            vals = pop_obj.V_soma[torch.from_numpy(v_idx).long().to(pop_obj.V_soma.device)]
            self._voltages[t, idx, : len(v_idx)] = vals.cpu().numpy()

    def _sample_conductances(self, timestep: int) -> None:
        """Sample g_E, g_I, g_nmda, g_gaba_b for fixed neurons."""
        assert self._g_exc_samples is not None, "conductance buffers not allocated"
        assert self._g_inh_samples is not None
        assert self._g_nmda_samples is not None
        assert self._g_gaba_b_samples is not None
        step = self._cond_sample_step
        if step >= self._g_exc_samples.shape[0]:
            return

        for idx, (rn, pn) in enumerate(self._pop_keys):
            c_idx = self._c_sample_idx[idx]
            if len(c_idx) == 0:
                continue

            dev_idx = torch.from_numpy(c_idx).long()

            pop_obj = self.brain.regions[rn].neuron_populations[pn]

            # ConductanceLIF: g_E / g_I / g_nmda
            # TwoCompartmentLIF: g_E_basal / g_I_basal / g_nmda_basal
            g_exc  = _get_attr_fallback(pop_obj, "g_E",    "g_E_basal")
            g_inh  = _get_attr_fallback(pop_obj, "g_I",    "g_I_basal")
            g_nmda = _get_attr_fallback(pop_obj, "g_nmda", "g_nmda_basal")
            if g_exc is not None:
                dev_idx_dev = dev_idx.to(g_exc.device)
                self._g_exc_samples[step, idx, : len(c_idx)] = (
                    g_exc[dev_idx_dev].cpu().numpy()
                )
            if g_inh is not None:
                dev_idx_dev = dev_idx.to(g_inh.device)
                self._g_inh_samples[step, idx, : len(c_idx)] = (
                    g_inh[dev_idx_dev].cpu().numpy()
                )
            if g_nmda is not None:
                dev_idx_dev = dev_idx.to(g_nmda.device)
                self._g_nmda_samples[step, idx, : len(c_idx)] = (
                    g_nmda[dev_idx_dev].cpu().numpy()
                )
            # GABA_B (slow inhibition; drives theta, up/down states, burst termination)
            g_gaba_b = _get_attr_fallback(pop_obj, "g_gaba_b", "g_gaba_b_basal")
            if g_gaba_b is not None and self._g_gaba_b_samples is not None:
                dev_idx_dev = dev_idx.to(g_gaba_b.device)
                self._g_gaba_b_samples[step, idx, : len(c_idx)] = (
                    g_gaba_b[dev_idx_dev].cpu().numpy()
                )

            # Apical AMPA (TwoCompartmentLIF only; NaN for ConductanceLIF populations)
            g_exc_apical = getattr(pop_obj, "g_E_apical", None)
            if g_exc_apical is not None and self._g_apical_samples is not None:
                dev_idx_dev = dev_idx.to(g_exc_apical.device)
                self._g_apical_samples[step, idx, : len(c_idx)] = (
                    g_exc_apical[dev_idx_dev].cpu().numpy()
                )

        self._cond_sample_step += 1

    # =========================================================================
    # SPIKE-TIME POST-PROCESSING
    # =========================================================================

    def _build_spike_times(self) -> None:
        """Build the nested spike-times dict from flat accumulation buffers.

        Called once at the start of :meth:`analyze` so that all downstream
        consumers (analysis functions and plots) see a populated
        ``_spike_times`` dict without any change to their call sites.

        Concentrates the per-neuron Python loop in the cold path (once per
        simulation) instead of the hot path (once per timestep).
        """
        self._spike_times.clear()
        for pop_idx, key in enumerate(self._pop_keys):
            cnt = int(self._spike_flat_n[pop_idx])
            if cnt == 0:
                continue
            flat_nidx = self._spike_flat_nidx[pop_idx][:cnt].tolist()
            flat_ts   = self._spike_flat_ts[pop_idx][:cnt].tolist()
            n = int(self._pop_sizes[pop_idx])
            nested: List[List[int]] = [[] for _ in range(n)]
            for ni, ts in zip(flat_nidx, flat_ts):
                nested[ni].append(ts)
            self._spike_times[key] = nested

    # =========================================================================
    # SNAPSHOT — decouple analysis / plotting from live brain state
    # =========================================================================

    def _capture_stp_final_state(self) -> "Dict[str, Dict[str, float]]":
        """Read the current x / u tensors from all live STP modules.

        Called once inside :meth:`to_snapshot` so the snapshot carries the
        end-of-run STP state independent of the brain object.
        """
        result: Dict[str, Dict[str, float]] = {}
        for rn, syn_id in self._stp_keys:
            stp_mod = self.brain.regions[rn].get_stp_module(syn_id)
            assert stp_mod is not None
            result[str(syn_id)] = {
                "mean_x": float(stp_mod.x.mean().item()),
                "mean_u": float(stp_mod.u.mean().item()),
                "efficacy": float((stp_mod.x * stp_mod.u).mean().item()),
            }
        return result

    def _capture_tract_weight_stats(self) -> "Dict[str, Dict[str, float]]":
        """Snapshot per-tract weight statistics from the live brain."""
        result: Dict[str, Dict[str, float]] = {}
        for syn_id in self._tract_keys:
            region = self.brain.regions[syn_id.target_region]
            w = region.get_synaptic_weights(syn_id).data
            result[str(syn_id)] = {
                "mean": float(w.mean().item()),
                "n_nonzero": float((w != 0).sum().item()),
                "n_total": float(w.numel()),
            }
        return result

    def _capture_pop_neuron_params(self) -> "Dict[Tuple[str, str], Dict[str, float]]":
        """Snapshot per-population neuron biophysics from the live brain.

        Captures all standard ConductanceLIFConfig fields plus any specialized
        fields from subclasses (NorepinephrineNeuronConfig, SerotoninNeuronConfig,
        AcetylcholineNeuronConfig, etc.) so that the calibration advisor can
        reason about intrinsic parameters like i_h, SK, autoreceptors, and noise.
        """
        result: Dict[Tuple[str, str], Dict[str, float]] = {}
        # Extra fields to capture from specialized config subclasses.
        _EXTRA_FIELDS = (
            "i_h_conductance", "i_h_reversal",
            "sk_conductance", "sk_reversal", "ca_decay", "ca_influx_per_spike",
            "noise_std",
            "autoreceptor_conductance", "autoreceptor_tau_ms", "autoreceptor_gain",
            "gap_junction_strength",
            "uncertainty_to_current_gain", "serotonin_drive_gain",
        )
        for rn, pn in self._pop_keys:
            pop = self.brain.regions[rn].get_neuron_population(pn)
            assert pop is not None
            cfg = pop.config
            def _scalar(v: object) -> float:
                """Convert a scalar or per-neuron tensor to a single float."""
                if isinstance(v, torch.Tensor):
                    return float(v.mean().item())
                return float(v)

            params: Dict[str, float] = {
                "g_L": _scalar(cfg.g_L),
                "v_threshold": _scalar(cfg.v_threshold),
                "v_reset": float(cfg.v_reset),
                "E_L": float(cfg.E_L),
                "E_E": float(cfg.E_E),
                "E_I": float(cfg.E_I),
                "tau_E": float(cfg.tau_E),
                "tau_I": float(cfg.tau_I),
                "tau_ref": float(cfg.tau_ref),
                "tau_mem_ms": _scalar(cfg.tau_mem_ms),
                "adapt_increment": _scalar(cfg.adapt_increment),
                "tau_adapt": float(cfg.tau_adapt),
                "E_adapt": float(cfg.E_adapt),
                "tau_GABA_B": float(cfg.tau_GABA_B),
                "E_GABA_B": float(cfg.E_GABA_B),
                "tau_nmda": float(cfg.tau_nmda),
                "E_nmda": float(cfg.E_nmda),
            }
            # Capture config class name for type identification
            params["_config_class"] = hash(type(cfg).__name__) * 0  # store 0 as placeholder
            # Store the actual class name as a string field (keyed specially)
            for attr in _EXTRA_FIELDS:
                val = getattr(cfg, attr, None)
                if val is not None:
                    params[attr] = _scalar(val)
            # Store config class name for downstream classification
            params["_config_type_name"] = 0.0  # placeholder; actual name stored in _pop_config_types
            result[(rn, pn)] = params
        # Store config type names separately (not float-valued)
        for rn, pn in self._pop_keys:
            pop = self.brain.regions[rn].get_neuron_population(pn)
            assert pop is not None
            self._pop_config_types[(rn, pn)] = type(pop.config).__name__
        return result

    def to_snapshot(self) -> RecorderSnapshot:
        """Extract a :class:`RecorderSnapshot` from the current recorder state.

        The snapshot is a self-contained, serialisable copy of everything the
        analysis and plotting functions need.  It carries no references to the
        live ``Brain`` or any ``torch.Tensor`` objects, so it is safe to save
        to disk, share between processes, or use after the simulation has ended.

        Call :meth:`RecorderSnapshot.save` on the returned object to persist it.
        Call :meth:`RecorderSnapshot.load` to reload and re-run analysis later.
        """
        # Ensure spike times are populated before snapshotting.
        self._build_spike_times()
        return RecorderSnapshot(
            config=self.config,
            dt_ms=self.dt_ms,
            _pop_keys=list(self._pop_keys),
            _pop_index=dict(self._pop_index),
            _n_pops=self._n_pops,
            _pop_sizes=self._pop_sizes.copy(),
            _region_keys=list(self._region_keys),
            _region_index=dict(self._region_index),
            _n_regions=self._n_regions,
            _region_pop_indices={k: list(v) for k, v in self._region_pop_indices.items()},
            _tract_keys=list(self._tract_keys),
            _tract_index=dict(self._tract_index),
            _n_tracts=self._n_tracts,
            _stp_keys=list(self._stp_keys),
            _nm_receptor_keys=list(self._nm_receptor_keys),
            _n_nm_receptors=self._n_nm_receptors,
            _nm_source_pop_keys=list(self._nm_source_pop_keys),
            _v_sample_idx=[arr.copy() for arr in self._v_sample_idx],
            _c_sample_idx=[arr.copy() for arr in self._c_sample_idx],
            _n_recorded=self._n_recorded,
            _gain_sample_step=self._gain_sample_step,
            _cond_sample_step=self._cond_sample_step,
            _gain_sample_times=list(self._gain_sample_times),
            _pop_spike_counts=self._pop_spike_counts.copy(),
            _per_neuron_spike_counts=[arr.copy() for arr in self._per_neuron_spike_counts],
            _region_spike_counts=self._region_spike_counts.copy(),
            _tract_sent=self._tract_sent.copy(),
            _spike_times=dict(self._spike_times),  # shallow — lists are read-only after analyze()
            _voltages=self._voltages.copy() if self._voltages is not None else None,
            _g_exc_samples=self._g_exc_samples.copy() if self._g_exc_samples is not None else None,
            _g_inh_samples=self._g_inh_samples.copy() if self._g_inh_samples is not None else None,
            _g_nmda_samples=self._g_nmda_samples.copy() if self._g_nmda_samples is not None else None,
            _g_gaba_b_samples=self._g_gaba_b_samples.copy() if self._g_gaba_b_samples is not None else None,
            _g_apical_samples=self._g_apical_samples.copy() if self._g_apical_samples is not None else None,
            _g_L_scale_history=self._g_L_scale_history.copy(),
            _stp_efficacy_history=self._stp_efficacy_history.copy(),
            _nm_concentration_history=self._nm_concentration_history.copy(),
            _pop_polarities=dict(self._pop_polarities),
            _tract_delay_ms=list(self._tract_delay_ms),
            _homeostasis_target_hz=dict(self._homeostasis_target_hz),
            _stp_configs=list(self._stp_configs),
            _stp_final_state=self._capture_stp_final_state(),
            _tract_weight_stats=self._capture_tract_weight_stats(),
            _pop_neuron_params=self._capture_pop_neuron_params(),
            _pop_config_types=self._pop_config_types.copy(),
        )
