"""
DiagnosticsRecorder: Training-loop-integrated brain diagnostics.

Architecture
============
The recorder is completely decoupled from the simulation loop.  It never
drives its own loop; instead, call ``record(t, outputs)`` once per
timestep from whatever loop you own (training, evaluation, pre-training
diagnostic run):

    recorder = DiagnosticsRecorder(brain, config)

    for t in range(n_timesteps):
        outputs = brain.forward(inputs)
        recorder.record(t, outputs)        # ← plug into any loop

    report = recorder.analyze()
    recorder.print_report(report)
    recorder.save(report, "data/diagnostics")
    recorder.plot(report, "data/diagnostics")

Modes
=====
- ``"full"``  – records spike times, voltages, conductances, STP state.
  Intended for short (≤10 s) pre/post-training diagnostic runs.
- ``"stats"`` – records only population-level spike counts and homeostatic
  gains.  Use this inside long training loops to keep memory overhead low.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch

from thalia.typing import BrainOutput, PopulationName, RegionName, SynapseId

if TYPE_CHECKING:
    from thalia.brain import DynamicBrain
    from thalia.components.neurons import ConductanceLIF, TwoCompartmentLIF


# =============================================================================
# BIOLOGICAL REFERENCE FIRING RATES
# Each entry: (region_substring, population_substring, (min_hz, max_hz))
# The first matching rule wins (most specific first).
# =============================================================================

_BIO_RANGES: List[Tuple[str, str, Tuple[float, float]]] = [
    # ----- Cerebellum -----
    ("", "purkinje",             (40.0, 100.0)),
    ("", "inferior_olive",       (0.3,   3.0)),
    ("", "dcn",                  (10.0, 100.0)),
    ("cerebellum", "granule",    (0.1,   5.0)),
    # ----- Thalamus -----
    ("thalamus", "relay",        (5.0,  40.0)),
    ("thalamus", "trn",          (5.0,  80.0)),
    # ----- Cortical pyramidal cells (layer order: most specific first) -----
    # cortex_sensory receives continuous 30 Hz thalamic drive (active stimulation state);
    # L23 fires 5–15 Hz during sensory processing (Sakata & Harris 2009).
    ("cortex_sensory", "l23_pyr",  (0.1,  15.0)),
    ("", "l23_pyr",              (0.1,   3.0)),
    ("", "l4_sst_pred",          (5.0,  25.0)),
    ("", "l4_pyr",               (1.0,  10.0)),
    ("", "l5_pyr",               (2.0,  15.0)),
    ("", "l6a_pyr",              (1.0,   8.0)),
    ("", "l6b_pyr",              (1.0,   8.0)),
    # ----- Cortical interneurons -----
    ("", "_pv",                  (10.0, 70.0)),
    ("", "_sst",                 (5.0,  25.0)),
    # VIP interneurons fire 20–50 Hz in active brain states (Dipoppa et al. 2018);
    # 30 Hz ceiling reflects active-state operation (all motor-cortex VIP at 23–30 Hz).
    ("", "_vip",                 (2.0,  30.0)),
    # ----- Hippocampus -----
    # DG inhibitory subtypes: DG principal fires at 0.1–1 Hz, but its interneurons
    # receive direct EC input and fire at ~2–5 Hz despite sparse principal activity.
    # Small populations (20–30 neurons) also have high variance in 500 ms windows.
    # floor=0 prevents false CRITICAL; (0,5) matches observed 2–2.4 Hz rates.
    # These must come BEFORE the generic "dg" pattern.
    ("hippocampus", "dg_inhibitory_olm",          (0.0,  5.0)),
    ("hippocampus", "dg_inhibitory_bistratified",  (0.0,  5.0)),
    ("hippocampus", "dg",        (0.1,   1.0)),
    ("hippocampus", "ca3",       (1.0,   5.0)),
    # CA2 has only 3 OLM and 2 bistratified neurons; at ~0.67 Hz target firing,
    # stochastic silence in a 500 ms window is expected — floor=0 prevents false CRITICAL.
    # These specific entries must come BEFORE the generic "ca2" pattern.
    ("hippocampus", "ca2_inhibitory_olm",          (0.0,  5.0)),
    ("hippocampus", "ca2_inhibitory_bistratified",  (0.0,  5.0)),
    ("hippocampus", "ca2",       (1.0,   5.0)),
    ("hippocampus", "ca1",       (1.0,   5.0)),
    ("hippocampus", "_olm",      (5.0,  15.0)),
    ("hippocampus", "_bistratified", (5.0, 20.0)),
    # ----- Striatum -----
    ("striatum", "d1",           (0.1,   5.0)),
    ("striatum", "d2",           (0.1,   5.0)),
    ("striatum", "fsi",          (10.0, 50.0)),
    ("striatum", "tan",          (5.0,  10.0)),
    # ----- Basal ganglia -----
    ("globus_pallidus", "prototypic",  (30.0, 80.0)),
    ("globus_pallidus", "arkypallidal", (5.0, 20.0)),
    ("subthalamic", "stn",       (10.0, 40.0)),
    ("substantia_nigra", "vta_feedback", (30.0, 80.0)),   # SNr
    # ----- Dopaminergic / neuromodulatory -----
    ("substantia_nigra_compacta", "da", (2.0,  8.0)),
    ("vta", "da_mesolimbic",     (2.0,   8.0)),
    ("vta", "da_mesocortical",   (2.0,   8.0)),
    ("locus_coeruleus", "ne",    (1.0,   5.0)),
    ("dorsal_raphe", "serotonin", (0.5,  3.0)),
    ("nucleus_basalis", "ach",   (2.0,  15.0)),
    # ----- Medial septum -----
    ("medial_septum", "gaba",    (5.0,  15.0)),
    ("medial_septum", "ach",     (5.0,  15.0)),
    # ----- Limbic / other -----
    ("prefrontal", "executive",  (0.5,  5.0)),
    ("basolateral_amygdala", "principal", (1.0, 5.0)),
    ("central_amygdala", "",     (1.0,  10.0)),
    ("entorhinal", "",           (1.0,  10.0)),
    ("lateral_habenula", "principal", (5.0, 20.0)),
    ("rostromedial", "gaba",     (5.0,  30.0)),
    # ----- Generic GABA interneurons -----
    ("", "gaba",                 (5.0,  40.0)),
]

_EEG_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


def _bio_range(region: str, pop: str) -> Optional[Tuple[float, float]]:
    """Return (min_hz, max_hz) for a population, or None if unknown."""
    r, p = region.lower(), pop.lower()
    for reg_pat, pop_pat, rng in _BIO_RANGES:
        if reg_pat in r and pop_pat in p:
            return rng
    return None


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class DiagnosticsConfig:
    """Configuration for the DiagnosticsRecorder.

    Args:
        n_timesteps: Number of timesteps to record.  Used to pre-allocate
            buffers.  Must be set before calling ``reset()`` / ``record()``.
        dt_ms: Simulation timestep in milliseconds.
        mode: ``"full"`` to record spike times, voltages, conductances, and STP
            state; ``"stats"`` to record only spike counts and gains (lower
            memory, suitable for training loops).
        voltage_sample_size: Number of neurons to sample per population for
            voltage traces (``full`` mode only).
        conductance_sample_size: Number of neurons to sample for conductance
            traces (``full`` mode only).
        conductance_sample_interval_ms: How frequently (in ms) to snapshot
            conductances.
        gain_sample_interval_ms: How frequently (in ms) to snapshot homeostatic
            gains.  Applies in both modes.
        rate_bin_ms: Bin width (ms) for population firing-rate estimation and
            FFT analysis.
        coherence_n_pairs: Maximum number of region pairs to compute coherence
            for (avoids O(n²) scaling with many regions).
    """

    n_timesteps: int
    dt_ms: float = 1.0
    mode: str = "full"

    # Sampling resolution (full mode)
    voltage_sample_size: int = 8
    conductance_sample_size: int = 8
    conductance_sample_interval_ms: int = 1
    gain_sample_interval_ms: int = 10

    # Analysis parameters
    rate_bin_ms: float = 10.0
    coherence_n_pairs: int = 30


# =============================================================================
# RESULT DATA STRUCTURES
# =============================================================================


@dataclass
class PopulationStats:
    """Complete statistics for one population over the recorded window."""

    region_name: RegionName
    population_name: PopulationName
    n_neurons: int

    # Firing rate
    mean_fr_hz: float         # Mean firing rate across all neurons (Hz)
    std_fr_hz: float          # Std across neurons (Hz)
    fraction_silent: float    # Fraction of neurons with zero spikes
    fraction_hyperactive: float  # Fraction firing > 50 Hz
    total_spikes: int

    # ISI statistics (full mode; NaN otherwise)
    isi_mean_ms: float        # Mean ISI
    isi_cv: float             # Coefficient of variation (irregularity; ~1 = Poisson)
    fraction_bursting: float  # Fraction of ISIs < 10 ms (burst criterion)

    # Per-neuron FR histogram (for CDF plots)
    fr_histogram: np.ndarray       # [20] bins from 0 to max_fr_hz
    fr_histogram_edges: np.ndarray # [21] bin edges in Hz

    # Biological reference
    bio_range_hz: Optional[Tuple[float, float]]  # (min, max) or None

    @property
    def is_silent(self) -> bool:
        return self.fraction_silent > 0.99

    @property
    def is_hyperactive(self) -> bool:
        return self.fraction_hyperactive > 0.10

    @property
    def bio_plausibility(self) -> str:
        """'ok', 'low', 'high', or 'unknown'."""
        if self.bio_range_hz is None:
            return "unknown"
        lo, hi = self.bio_range_hz
        if self.mean_fr_hz < lo * 0.5:
            return "low"
        if self.mean_fr_hz > hi * 2.0:
            return "high"
        return "ok"


@dataclass
class RegionStats:
    """Statistics aggregated across all populations in a region."""

    region_name: RegionName
    populations: Dict[PopulationName, PopulationStats]
    mean_fr_hz: float
    total_spikes: int

    # E/I balance (conductance-based where available, else NaN)
    mean_g_exc: float
    mean_g_inh: float

    @property
    def ei_ratio(self) -> float:
        if self.mean_g_inh > 0:
            return self.mean_g_exc / self.mean_g_inh
        return float("nan")

    @property
    def is_active(self) -> bool:
        return self.mean_fr_hz > 0.1  # > 0.1 Hz average

    @property
    def has_pathological_populations(self) -> bool:
        for p in self.populations.values():
            if p.is_silent or p.is_hyperactive:
                return True
        return False


@dataclass
class OscillatoryStats:
    """Spectral and coherence analysis."""

    # Per-region power spectra (region → band → normalised power)
    region_band_power: Dict[RegionName, Dict[str, float]]

    # Per-region dominant frequency (Hz) and band name
    region_dominant_freq: Dict[RegionName, float]
    region_dominant_band: Dict[RegionName, str]

    # Cross-regional coherence in each EEG band
    # coherence_matrices[band][i, j] = coherence between region i and j
    coherence_theta: np.ndarray   # [n_regions × n_regions]
    region_order: List[RegionName]  # Row/column labels for coherence matrix

    # Global oscillation (from weighted sum of all regions)
    global_dominant_freq_hz: float
    global_band_power: Dict[str, float]


@dataclass
class ConnectivityStats:
    """Axonal tract transmission statistics."""

    @dataclass
    class TractStats:
        synapse_id: SynapseId
        spikes_sent: int
        transmission_ratio: float  # Fraction of timesteps source was active
        is_functional: bool
        # Cross-correlation delay verification (full mode; NaN otherwise)
        measured_delay_ms: float   # Peak lag of source→target cross-corr
        expected_delay_ms: float   # From axonal tract spec

    tracts: List[TractStats]
    n_functional: int
    n_broken: List["ConnectivityStats.TractStats"]


@dataclass
class HomeostaticStats:
    """Homeostatic gain and STP state over time."""

    # g_L_scale trajectories per population: key = "region:pop" → 1D array of gains
    gain_trajectories: Dict[str, np.ndarray]
    gain_sample_times_ms: np.ndarray  # Timestep-to-ms for x-axis

    # STP state at end of recording: synapse_id_str → {"mean_x": ..., "mean_u": ...}
    stp_final_state: Dict[str, Dict[str, float]]


@dataclass
class HealthReport:
    """Per-population and global health assessment."""

    # Global
    is_healthy: bool
    stability_score: float          # 0–1
    critical_issues: List[str]
    warnings: List[str]

    # Per-population biological plausibility
    population_status: Dict[str, str]  # "region:pop" → "ok" | "low" | "high" | "unknown"

    # Summary counts
    n_populations_ok: int
    n_populations_low: int
    n_populations_high: int
    n_populations_unknown: int


@dataclass
class DiagnosticsReport:
    """Complete diagnostics report produced by DiagnosticsRecorder.analyze()."""

    # Meta
    timestamp: float
    simulation_time_ms: float
    n_timesteps: int
    mode: Literal["full", "stats"]

    # Core data
    regions: Dict[RegionName, RegionStats]
    oscillations: OscillatoryStats
    connectivity: ConnectivityStats
    homeostasis: HomeostaticStats
    health: HealthReport

    # Raw traces kept for plotting (populated only in "full" mode)
    # shape: [T, n_pops] – spike counts per ms per population
    raw_spike_counts: Optional[np.ndarray] = None
    # shape: [T_cond, n_pops, sample_size] – sampled voltages
    raw_voltages: Optional[np.ndarray] = None
    # Timestamps for voltage/conductance samples
    voltage_sample_times_ms: Optional[np.ndarray] = None
    conductance_sample_times_ms: Optional[np.ndarray] = None
    # g_exc / g_inh traces
    raw_g_exc: Optional[np.ndarray] = None
    raw_g_inh: Optional[np.ndarray] = None
    # Population-level firing rates (binned): [n_bins, n_pops]
    pop_rate_binned: Optional[np.ndarray] = None
    # Population and region index
    pop_keys: Optional[List[Tuple[str, str]]] = None
    region_keys: Optional[List[str]] = None


# =============================================================================
# DIAGNOSTICS RECORDER
# =============================================================================


class DiagnosticsRecorder:
    """Records brain activity for comprehensive diagnostics.

    Completely decoupled from the simulation loop: call ``record(t, outputs)``
    once per timestep from whatever loop you own, then ``analyze()`` to get a
    :class:`DiagnosticsReport`.

    See module docstring for a usage example.
    """

    def __init__(self, brain: DynamicBrain, config: DiagnosticsConfig) -> None:
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

        self._build_index()
        self._allocate_buffers()
        self._recording_started = False
        self._n_recorded = 0

    # =========================================================================
    # INDEX CONSTRUCTION
    # =========================================================================

    def _build_index(self) -> None:
        """Build ordered population and tract indices from the brain."""
        # Ordered list of (region_name, pop_name) tuples.
        self._pop_keys: List[Tuple[str, str]] = []
        for region_name, region in self.brain.regions.items():
            for pop_name in region.neuron_populations.keys():
                self._pop_keys.append((region_name, pop_name))
        self._pop_index: Dict[Tuple[str, str], int] = {
            key: idx for idx, key in enumerate(self._pop_keys)
        }
        self._n_pops = len(self._pop_keys)

        # Population sizes
        self._pop_sizes = np.zeros(self._n_pops, dtype=np.int32)
        for idx, (rn, pn) in enumerate(self._pop_keys):
            pop_obj = self.brain.regions[rn].neuron_populations[pn]
            self._pop_sizes[idx] = pop_obj.n_neurons

        # Ordered region names
        self._region_keys: List[str] = list(self.brain.regions.keys())
        self._region_index: Dict[str, int] = {k: i for i, k in enumerate(self._region_keys)}
        self._n_regions = len(self._region_keys)

        # Region → population indices
        self._region_pop_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, (rn, _) in enumerate(self._pop_keys):
            self._region_pop_indices[rn].append(idx)

        # Axonal tracts
        self._tract_keys: List[SynapseId] = list(self.brain.axonal_tracts.keys())
        self._tract_index: Dict[SynapseId, int] = {
            key: idx for idx, key in enumerate(self._tract_keys)
        }
        self._n_tracts = len(self._tract_keys)

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

    def _allocate_buffers(self) -> None:
        """Pre-allocate recording buffers."""
        T = self.config.n_timesteps
        P = self._n_pops
        R = self._n_regions
        V = self.config.voltage_sample_size
        C = self.config.conductance_sample_size

        # Population-level spike counts [T × P]
        self._pop_spike_counts = np.zeros((T, P), dtype=np.int32)

        # Region-level total spike counts [T × R] (for cross-correlation/coherence)
        self._region_spike_counts = np.zeros((T, R), dtype=np.int32)

        # Axonal tract: spikes sent from source at each timestep [T × n_tracts]
        self._tract_sent = np.zeros((T, self._n_tracts), dtype=np.int32)

        # Per-neuron spike times: (region, pop) → list-of-lists
        # Initialised lazily on first spike
        self._spike_times: Dict[Tuple[str, str], List[List[int]]] = {}

        if self.config.mode == "full":
            # Voltage traces [T × P × V]
            self._voltages = np.full((T, P, V), np.nan, dtype=np.float32)

            # Conductance samples – every conductance_sample_interval_ms timesteps
            ci = self.config.conductance_sample_interval_ms
            n_cond = max(1, T // ci)
            self._g_exc_samples  = np.full((n_cond, P, C), np.nan, dtype=np.float32)
            self._g_inh_samples  = np.full((n_cond, P, C), np.nan, dtype=np.float32)
            self._g_nmda_samples = np.full((n_cond, P, C), np.nan, dtype=np.float32)
            self._cond_sample_step = 0
        else:
            self._voltages = None
            self._g_exc_samples = self._g_inh_samples = self._g_nmda_samples = None
            self._cond_sample_step = 0

        # Homeostatic gain traces – every gain_sample_interval_ms timesteps (both modes)
        gi = self.config.gain_sample_interval_ms
        n_gain = max(1, T // gi)
        self._g_L_scale_history = np.full((n_gain, P), np.nan, dtype=np.float32)
        self._gain_sample_step = 0
        self._gain_sample_times: List[int] = []

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self) -> None:
        """Clear all recording buffers (call before a new recording window)."""
        self._alloc_buffers_fresh()
        self._recording_started = False
        self._n_recorded = 0

    def _alloc_buffers_fresh(self) -> None:
        """Re-zero all buffers in-place without re-allocating (faster than re-creating)."""
        self._pop_spike_counts.fill(0)
        self._region_spike_counts.fill(0)
        self._tract_sent.fill(0)
        self._spike_times.clear()
        self._gain_sample_times.clear()
        self._gain_sample_step = 0
        if self.config.mode == "full":
            assert self._voltages is not None
            assert self._g_exc_samples is not None
            assert self._g_inh_samples is not None
            assert self._g_nmda_samples is not None
            self._voltages.fill(np.nan)
            self._g_exc_samples.fill(np.nan)
            self._g_inh_samples.fill(np.nan)
            self._g_nmda_samples.fill(np.nan)
            self._cond_sample_step = 0
        self._g_L_scale_history.fill(np.nan)

    # =========================================================================
    # RECORD  — the single integration point
    # =========================================================================

    def record(self, timestep: int, outputs: BrainOutput) -> None:
        """Record one timestep of brain output.

        Call this immediately after ``brain.forward()`` inside your loop.

        Args:
            timestep: Zero-based timestep index (must be < config.n_timesteps).
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

                # Per-neuron spike times (full mode only)
                if self.config.mode == "full" and n_active > 0:
                    if key not in self._spike_times:
                        n = int(self._pop_sizes[pop_idx])
                        self._spike_times[key] = [[] for _ in range(n)]
                    spiking = torch.where(pop_spikes)[0].tolist()
                    for nidx in spiking:
                        self._spike_times[key][nidx].append(t)

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
        gi = self.config.gain_sample_interval_ms
        if gi > 0 and t % gi == 0:
            self._sample_gains(t)

        if self.config.mode == "full":
            self._sample_voltages(t)
            ci = self.config.conductance_sample_interval_ms
            if ci > 0 and t % ci == 0:
                self._sample_conductances(t)

        self._n_recorded = t + 1

    def _sample_gains(self, timestep: int) -> None:
        """Sample homeostatic g_L_scale for all populations."""
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

        self._gain_sample_times.append(timestep)
        self._gain_sample_step += 1

    def _sample_voltages(self, timestep: int) -> None:
        """Sample membrane voltages for fixed neurons in each population."""
        assert self._voltages is not None, "_voltages buffer not allocated (mode != 'full'?)"
        t = timestep
        if t >= self._voltages.shape[0]:
            return

        for idx, (rn, pn) in enumerate(self._pop_keys):
            assert rn in self.brain.regions, f"Region '{rn}' not found in brain"
            assert pn in self.brain.regions[rn].neuron_populations, f"Population '{pn}' not found in region '{rn}'"
            pop_obj: Union[ConductanceLIF, TwoCompartmentLIF] = self.brain.regions[rn].neuron_populations[pn]
            v_idx = self._v_sample_idx[idx]
            if len(v_idx) == 0:
                continue
            with torch.no_grad():
                vals = pop_obj.V_soma[torch.from_numpy(v_idx).long().to(pop_obj.V_soma.device)]
                self._voltages[t, idx, : len(v_idx)] = vals.cpu().numpy()

    def _sample_conductances(self, timestep: int) -> None:
        """Sample g_E, g_I, g_nmda for fixed neurons."""
        assert self._g_exc_samples is not None, "conductance buffers not allocated (mode != 'full'?)"
        assert self._g_inh_samples is not None
        assert self._g_nmda_samples is not None
        step = self._cond_sample_step
        if step >= self._g_exc_samples.shape[0]:
            return

        for idx, (rn, pn) in enumerate(self._pop_keys):
            pop_obj = self.brain.regions[rn].neuron_populations[pn]
            c_idx = self._c_sample_idx[idx]
            if len(c_idx) == 0:
                continue

            dev_idx = torch.from_numpy(c_idx).long()

            with torch.no_grad():
                # ConductanceLIF: g_E / g_I / g_nmda
                # TwoCompartmentLIF: g_E_basal / g_I_basal / g_nmda_basal
                g_exc  = getattr(pop_obj, "g_E",    None)
                if g_exc is None:
                    g_exc = getattr(pop_obj, "g_E_basal", None)
                g_inh  = getattr(pop_obj, "g_I",    None)
                if g_inh is None:
                    g_inh = getattr(pop_obj, "g_I_basal", None)
                g_nmda = getattr(pop_obj, "g_nmda", None)
                if g_nmda is None:
                    g_nmda = getattr(pop_obj, "g_nmda_basal", None)
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

        self._cond_sample_step += 1

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    def analyze(self) -> DiagnosticsReport:
        """Compute and return a complete :class:`DiagnosticsReport`."""
        T = self._n_recorded or self.config.n_timesteps

        # Population-level stats
        pop_stats: Dict[Tuple[str, str], PopulationStats] = {}
        for idx, (rn, pn) in enumerate(self._pop_keys):
            pop_stats[(rn, pn)] = self._compute_population_stats(idx, T)

        # Region-level stats
        region_stats: Dict[str, RegionStats] = {}
        for rn in self._region_keys:
            region_stats[rn] = self._compute_region_stats(rn, pop_stats)

        # Binned population rates for FFT/coherence
        bin_steps = max(1, int(self.config.rate_bin_ms / self.dt_ms))
        n_bins = T // bin_steps
        pop_rate_binned = np.zeros((n_bins, self._n_pops), dtype=np.float32)
        for b in range(n_bins):
            start, end = b * bin_steps, (b + 1) * bin_steps
            # spikes per bin / n_neurons → firing rate per neuron
            pop_rate_binned[b] = (
                self._pop_spike_counts[start:end].sum(axis=0)
                / np.maximum(self._pop_sizes, 1)
            )

        # Region binned rates (sum across populations)
        region_rate_binned = np.zeros((n_bins, self._n_regions), dtype=np.float32)
        for r_idx, rn in enumerate(self._region_keys):
            p_indices = self._region_pop_indices[rn]
            region_rate_binned[:, r_idx] = pop_rate_binned[:, p_indices].sum(axis=1)

        oscillations = self._compute_oscillatory_stats(
            pop_rate_binned, region_rate_binned, n_bins, T
        )
        connectivity = self._compute_connectivity_stats(T)
        homeostasis = self._compute_homeostatic_stats()
        health = self._assess_health(pop_stats, region_stats)

        report = DiagnosticsReport(
            timestamp=time.time(),
            simulation_time_ms=T * self.dt_ms,
            n_timesteps=T,
            mode=self.config.mode,
            regions=region_stats,
            oscillations=oscillations,
            connectivity=connectivity,
            homeostasis=homeostasis,
            health=health,
            pop_keys=list(self._pop_keys),
            region_keys=list(self._region_keys),
            pop_rate_binned=pop_rate_binned,
            raw_spike_counts=self._pop_spike_counts[:T].copy(),
        )

        if self.config.mode == "full":
            assert self._voltages is not None
            assert self._g_exc_samples is not None
            assert self._g_inh_samples is not None
            report.raw_voltages = self._voltages[:T].copy()
            report.voltage_sample_times_ms = np.arange(T, dtype=np.float32) * self.dt_ms
            if self._cond_sample_step > 0:
                ci = self.config.conductance_sample_interval_ms
                report.conductance_sample_times_ms = (
                    np.arange(0, T, ci, dtype=np.float32) * self.dt_ms
                )[: self._cond_sample_step]
                report.raw_g_exc = self._g_exc_samples[: self._cond_sample_step].copy()
                report.raw_g_inh = self._g_inh_samples[: self._cond_sample_step].copy()

        return report

    # ------------------------------------------------------------------ helpers

    def _compute_population_stats(self, pop_idx: int, T: int) -> PopulationStats:
        """Compute statistics for a single population."""
        rn, pn = self._pop_keys[pop_idx]
        n_neurons = int(self._pop_sizes[pop_idx])
        counts = self._pop_spike_counts[:T, pop_idx]  # spikes per timestep
        total_spikes = int(counts.sum())

        # Per-neuron firing rates
        sim_s = T * self.dt_ms / 1000.0
        key = (rn, pn)

        if self.config.mode == "full" and key in self._spike_times:
            spike_counts_per_neuron = np.array(
                [len(times) for times in self._spike_times[key]], dtype=np.float32
            )
            # Account for neurons with no spikes
            if len(spike_counts_per_neuron) < n_neurons:
                pad = np.zeros(n_neurons - len(spike_counts_per_neuron), dtype=np.float32)
                spike_counts_per_neuron = np.concatenate([spike_counts_per_neuron, pad])
        else:
            # In stats mode: all neurons treated as contributing equally
            mean_spikes = total_spikes / max(n_neurons, 1)
            spike_counts_per_neuron = np.full(n_neurons, mean_spikes, dtype=np.float32)

        fr_per_neuron_hz = spike_counts_per_neuron / max(sim_s, 1e-9)
        mean_fr_hz = float(fr_per_neuron_hz.mean())
        std_fr_hz = float(fr_per_neuron_hz.std())
        fraction_silent = float((fr_per_neuron_hz < 0.01).mean())
        fraction_hyperactive = float((fr_per_neuron_hz > 50.0).mean())

        # FR histogram
        max_fr = max(float(fr_per_neuron_hz.max()), 1.0)
        hist, edges = np.histogram(fr_per_neuron_hz, bins=20, range=(0.0, max_fr))

        # ISI statistics
        isi_mean_ms, isi_cv, frac_burst = np.nan, np.nan, np.nan
        if self.config.mode == "full" and key in self._spike_times:
            all_isis: List[float] = []
            for times in self._spike_times[key]:
                if len(times) >= 2:
                    isis = np.diff(times) * self.dt_ms  # ms
                    all_isis.extend(isis.tolist())
            if all_isis:
                arr = np.array(all_isis, dtype=np.float32)
                isi_mean_ms = float(arr.mean())
                isi_cv = float(arr.std() / arr.mean()) if arr.mean() > 0 else np.nan
                frac_burst = float((arr < 10.0).mean())

        return PopulationStats(
            region_name=rn,
            population_name=pn,
            n_neurons=n_neurons,
            mean_fr_hz=mean_fr_hz,
            std_fr_hz=std_fr_hz,
            fraction_silent=fraction_silent,
            fraction_hyperactive=fraction_hyperactive,
            total_spikes=total_spikes,
            isi_mean_ms=isi_mean_ms,
            isi_cv=isi_cv,
            fraction_bursting=frac_burst,
            fr_histogram=hist.astype(np.float32),
            fr_histogram_edges=edges.astype(np.float32),
            bio_range_hz=_bio_range(rn, pn),
        )

    def _compute_region_stats(
        self,
        region_name: str,
        pop_stats: Dict[Tuple[str, str], PopulationStats],
    ) -> RegionStats:
        """Aggregate population stats into a RegionStats."""
        pops = {
            pn: pop_stats[(rn, pn)]
            for rn, pn in self._pop_keys
            if rn == region_name
        }
        mean_fr = float(np.mean([p.mean_fr_hz for p in pops.values()])) if pops else 0.0
        total_spikes = sum(p.total_spikes for p in pops.values())

        # E/I balance: mean conductances across sampled neurons in this region
        mean_g_exc, mean_g_inh = np.nan, np.nan
        if self.config.mode == "full" and self._g_exc_samples is not None and self._g_inh_samples is not None:
            p_indices = self._region_pop_indices[region_name]
            exc_vals = self._g_exc_samples[:self._cond_sample_step, p_indices, :].flatten()
            inh_vals = self._g_inh_samples[:self._cond_sample_step, p_indices, :].flatten()
            exc_valid = exc_vals[~np.isnan(exc_vals)]
            inh_valid = inh_vals[~np.isnan(inh_vals)]
            if len(exc_valid) > 0:
                mean_g_exc = float(exc_valid.mean())
            if len(inh_valid) > 0:
                mean_g_inh = float(inh_valid.mean())

        return RegionStats(
            region_name=region_name,
            populations=pops,
            mean_fr_hz=mean_fr,
            total_spikes=total_spikes,
            mean_g_exc=mean_g_exc,
            mean_g_inh=mean_g_inh,
        )

    def _band_power(
        self, power: np.ndarray, freqs: np.ndarray, f_min: float, f_max: float
    ) -> float:
        mask = (freqs >= f_min) & (freqs <= f_max)
        if not mask.any():
            return 0.0
        return float(power[mask].sum())

    def _coherence(
        self,
        x: np.ndarray,
        y: np.ndarray,
        f_min: float,
        f_max: float,
    ) -> float:
        """Magnitude-squared coherence estimate in frequency band [f_min, f_max] Hz."""
        n = len(x)
        dt_s = self.config.rate_bin_ms / 1000.0
        Fx = np.fft.rfft(x - x.mean())
        Fy = np.fft.rfft(y - y.mean())
        freqs = np.fft.rfftfreq(n, d=dt_s)
        mask = (freqs >= f_min) & (freqs <= f_max)
        if not mask.any():
            return 0.0
        Pxy = np.abs((Fx * np.conj(Fy))[mask].mean())
        Pxx = np.abs((Fx * np.conj(Fx))[mask].mean())
        Pyy = np.abs((Fy * np.conj(Fy))[mask].mean())
        denom = np.sqrt(Pxx * Pyy)
        return float(Pxy / denom) if denom > 1e-12 else 0.0

    def _compute_oscillatory_stats(
        self,
        pop_rate_binned: np.ndarray,
        region_rate_binned: np.ndarray,
        n_bins: int,
        T: int,
    ) -> OscillatoryStats:
        """Compute per-region spectra and cross-regional coherence."""
        dt_s = self.config.rate_bin_ms / 1000.0
        region_band_power: Dict[str, Dict[str, float]] = {}
        region_dominant_freq: Dict[str, float] = {}
        region_dominant_band: Dict[str, str] = {}

        for r_idx, rn in enumerate(self._region_keys):
            trace = region_rate_binned[:, r_idx].astype(np.float64)
            if trace.sum() < 1e-9 or n_bins < 8:
                region_band_power[rn] = {b: 0.0 for b in _EEG_BANDS}
                region_dominant_freq[rn] = 0.0
                region_dominant_band[rn] = "none"
                continue

            fft = np.fft.rfft(trace - trace.mean())
            freqs = np.fft.rfftfreq(n_bins, d=dt_s)
            power = np.abs(fft) ** 2

            # Band powers (normalised)
            raw_bp = {b: self._band_power(power, freqs, f1, f2) for b, (f1, f2) in _EEG_BANDS.items()}
            total = sum(raw_bp.values()) or 1.0
            norm_bp = {b: v / total for b, v in raw_bp.items()}
            region_band_power[rn] = norm_bp

            # Dominant frequency (skip DC)
            dom_idx = int(np.argmax(power[1:])) + 1 if len(power) > 1 else 0
            dom_freq = float(freqs[dom_idx])
            region_dominant_freq[rn] = dom_freq

            # Band the dominant frequency falls in (for consistent labeling)
            dom_band = "gamma"  # default if above all defined bands
            for b, (f1, f2) in _EEG_BANDS.items():
                if f1 <= dom_freq < f2:
                    dom_band = b
                    break
            region_dominant_band[rn] = dom_band

        # Global (sum of all region traces)
        global_trace = region_rate_binned.sum(axis=1).astype(np.float64)
        global_band_power: Dict[str, float] = {}
        global_dominant_freq = 0.0
        if global_trace.sum() > 1e-9 and n_bins >= 8:
            fft = np.fft.rfft(global_trace - global_trace.mean())
            freqs = np.fft.rfftfreq(n_bins, d=dt_s)
            power = np.abs(fft) ** 2
            raw_bp = {b: self._band_power(power, freqs, f1, f2) for b, (f1, f2) in _EEG_BANDS.items()}
            total = sum(raw_bp.values()) or 1.0
            global_band_power = {b: v / total for b, v in raw_bp.items()}
            dom_idx = int(np.argmax(power[1:])) + 1 if len(power) > 1 else 0
            global_dominant_freq = float(freqs[dom_idx])
        else:
            global_band_power = {b: 0.0 for b in _EEG_BANDS}

        # Cross-regional coherence (theta band, limited to coherence_n_pairs)
        n_r = self._n_regions
        coh_theta = np.zeros((n_r, n_r), dtype=np.float32)
        np.fill_diagonal(coh_theta, 1.0)

        # Select the most active region pairs to compute
        region_activity = region_rate_binned.sum(axis=0)  # [n_r]
        active_r = np.where(region_activity > 0)[0]
        computed = 0
        max_pairs = self.config.coherence_n_pairs
        for i in range(len(active_r)):
            for j in range(i + 1, len(active_r)):
                if computed >= max_pairs:
                    break
                ri, rj = active_r[i], active_r[j]
                c = self._coherence(
                    region_rate_binned[:, ri].astype(np.float64),
                    region_rate_binned[:, rj].astype(np.float64),
                    4.0, 8.0,  # theta
                )
                coh_theta[ri, rj] = c
                coh_theta[rj, ri] = c
                computed += 1

        return OscillatoryStats(
            region_band_power=region_band_power,
            region_dominant_freq=region_dominant_freq,
            region_dominant_band=region_dominant_band,
            coherence_theta=coh_theta,
            region_order=list(self._region_keys),
            global_dominant_freq_hz=global_dominant_freq,
            global_band_power=global_band_power,
        )

    def _compute_connectivity_stats(self, T: int) -> ConnectivityStats:
        """Analyse axonal tract transmission and verify delays."""
        tracts: List[ConnectivityStats.TractStats] = []

        for tract_idx, synapse_id in enumerate(self._tract_keys):
            sent = self._tract_sent[:T, tract_idx]
            total_sent = int(sent.sum())
            transmission_ratio = float((sent > 0).mean())
            is_functional = transmission_ratio > 0.01

            # Expected delay from tract spec
            tract_obj = self.brain.axonal_tracts[synapse_id]
            expected_delay_ms = float(tract_obj.spec.delay_ms)

            # Measured delay: cross-correlation between source and target populations
            measured_delay_ms = np.nan
            if self.config.mode == "full" and is_functional:
                tgt_key = (synapse_id.target_region, synapse_id.target_population)
                tgt_idx = self._pop_index.get(tgt_key)
                if tgt_idx is not None:
                    src = sent.astype(np.float64)
                    tgt = self._pop_spike_counts[:T, tgt_idx].astype(np.float64)
                    if src.std() > 0 and tgt.std() > 0:
                        xcorr = np.correlate(
                            tgt - tgt.mean(), src - src.mean(), mode="full"
                        )
                        lag_range = int(min(100, max(expected_delay_ms * 3, 10) / self.dt_ms))
                        center = len(src) - 1
                        search_lo = max(0, center)
                        search_hi = min(len(xcorr), center + lag_range)
                        if search_hi > search_lo:
                            peak = np.argmax(xcorr[search_lo:search_hi])
                            measured_delay_ms = float(peak) * self.dt_ms

            tracts.append(
                ConnectivityStats.TractStats(
                    synapse_id=synapse_id,
                    spikes_sent=total_sent,
                    transmission_ratio=transmission_ratio,
                    is_functional=is_functional,
                    measured_delay_ms=measured_delay_ms,
                    expected_delay_ms=expected_delay_ms,
                )
            )

        broken = [t for t in tracts if not t.is_functional]
        return ConnectivityStats(
            tracts=tracts,
            n_functional=len(tracts) - len(broken),
            n_broken=broken,
        )

    def _compute_homeostatic_stats(self) -> HomeostaticStats:
        """Summarise homeostatic gain trajectories and STP final state."""
        n_steps = self._gain_sample_step
        sample_times = np.array(self._gain_sample_times, dtype=np.float32) * self.dt_ms

        gain_trajectories: Dict[str, np.ndarray] = {}
        for idx, (rn, pn) in enumerate(self._pop_keys):
            vals = self._g_L_scale_history[:n_steps, idx]
            if not np.all(np.isnan(vals)) and not np.all(vals == 0):
                gain_trajectories[f"{rn}:{pn}"] = vals.copy()

        # STP final state snapshot
        stp_final: Dict[str, Dict[str, float]] = {}
        for region in self.brain.regions.values():
            if not hasattr(region, "stp_modules"):
                continue
            for syn_id, stp_mod in region.stp_modules.items():
                if hasattr(stp_mod, "x") and hasattr(stp_mod, "u"):
                    key = str(syn_id)
                    with torch.no_grad():
                        stp_final[key] = {
                            "mean_x": float(stp_mod.x.mean().item()),
                            "mean_u": float(stp_mod.u.mean().item()),
                            "efficacy": float((stp_mod.x * stp_mod.u).mean().item()),
                        }

        return HomeostaticStats(
            gain_trajectories=gain_trajectories,
            gain_sample_times_ms=sample_times,
            stp_final_state=stp_final,
        )

    def _assess_health(
        self,
        pop_stats: Dict[Tuple[str, str], PopulationStats],
        region_stats: Dict[str, RegionStats],
    ) -> HealthReport:
        """Assess overall brain health with per-population biological plausibility."""
        critical: List[str] = []
        warnings: List[str] = []
        population_status: Dict[str, str] = {}
        n_ok = n_low = n_high = n_unknown = 0

        for (rn, pn), ps in pop_stats.items():
            status = ps.bio_plausibility
            population_status[f"{rn}:{pn}"] = status
            if status == "ok":
                n_ok += 1
            elif status == "low":
                n_low += 1
            elif status == "high":
                n_high += 1
            else:
                n_unknown += 1

            # Critical: completely silent (no spikes at all)
            # Only flag as critical if the population is *expected* to fire (bio_range_hz[0] > 0).
            # Populations with a lower bound of 0 Hz (e.g. DG bistratified target 0–1 Hz) are
            # not required to fire in any given 500 ms window — silence is within target.
            if ps.total_spikes == 0:
                if ps.bio_range_hz is not None and ps.bio_range_hz[0] > 0:
                    critical.append(f"SILENT: {rn}:{pn} (expected {ps.bio_range_hz[0]:.0f}–{ps.bio_range_hz[1]:.0f} Hz)")
                elif ps.bio_range_hz is None:
                    warnings.append(f"Silent: {rn}:{pn} — no spikes recorded")

            # Critical: far outside biological range
            if status == "low" and ps.bio_range_hz is not None:
                lo, _ = ps.bio_range_hz
                if ps.mean_fr_hz < lo * 0.2:
                    critical.append(
                        f"SEVERELY LOW: {rn}:{pn} = {ps.mean_fr_hz:.1f} Hz "
                        f"(target {lo:.0f}–{ps.bio_range_hz[1]:.0f} Hz)"
                    )
                else:
                    warnings.append(
                        f"Low FR: {rn}:{pn} = {ps.mean_fr_hz:.1f} Hz "
                        f"(target {lo:.0f}–{ps.bio_range_hz[1]:.0f} Hz)"
                    )
            elif status == "high" and ps.bio_range_hz is not None:
                _, hi = ps.bio_range_hz
                if ps.mean_fr_hz > hi * 5.0:
                    critical.append(
                        f"HYPERACTIVE: {rn}:{pn} = {ps.mean_fr_hz:.0f} Hz "
                        f"(target {ps.bio_range_hz[0]:.0f}–{hi:.0f} Hz)"
                    )
                else:
                    warnings.append(
                        f"High FR: {rn}:{pn} = {ps.mean_fr_hz:.1f} Hz "
                        f"(target {ps.bio_range_hz[0]:.0f}–{hi:.0f} Hz)"
                    )

        # Check E/I balance
        for rn, rs in region_stats.items():
            if not np.isnan(rs.mean_g_exc) and not np.isnan(rs.mean_g_inh):
                if rs.mean_g_inh > 0:
                    ratio = rs.mean_g_exc / rs.mean_g_inh
                    if ratio > 5.0:
                        warnings.append(
                            f"E/I imbalance: {rn} g_exc/g_inh = {ratio:.1f} (excitation dominated)"
                        )
                    elif ratio < 0.1:
                        warnings.append(
                            f"E/I imbalance: {rn} g_exc/g_inh = {ratio:.2f} (inhibition dominated)"
                        )

        # Check homeostatic gain collapse
        for idx, (rn, pn) in enumerate(self._pop_keys):
            n_steps = self._gain_sample_step
            if n_steps > 0:
                final_gain = self._g_L_scale_history[n_steps - 1, idx]
                if not np.isnan(final_gain) and final_gain < 0.3:
                    critical.append(f"GAIN COLLAPSED: {rn}:{pn} g_L_scale = {final_gain:.3f}")

        n_crit = len(critical)
        n_warn = len(warnings)
        stability = max(0.0, 1.0 - n_crit * 0.3 - n_warn * 0.05)
        is_healthy = n_crit == 0 and stability > 0.7

        return HealthReport(
            is_healthy=is_healthy,
            stability_score=stability,
            critical_issues=critical,
            warnings=warnings,
            population_status=population_status,
            n_populations_ok=n_ok,
            n_populations_low=n_low,
            n_populations_high=n_high,
            n_populations_unknown=n_unknown,
        )

    # =========================================================================
    # REPORTING
    # =========================================================================

    def print_report(self, report: DiagnosticsReport, detailed: bool = True) -> None:
        """Print a formatted text report to stdout."""
        w = 80

        def sep(char: str = "=") -> str:
            return char * w

        print(f"\n{sep()}")
        print("THALIA BRAIN DIAGNOSTICS REPORT")
        print(sep())
        print(f"  Simulation time : {report.simulation_time_ms:.0f} ms  ({report.n_timesteps} timesteps)")
        print(f"  Mode            : {report.mode}")
        print(f"  Regions         : {len(report.regions)}")
        n_total = sum(len(r.populations) for r in report.regions.values())
        print(f"  Populations     : {n_total}")

        # ── Health ────────────────────────────────────────────────────────────
        print(f"\n{sep('─')}")
        print("HEALTH")
        print(sep("─"))
        h = report.health
        icon = "✓ HEALTHY" if h.is_healthy else "✗ ISSUES DETECTED"
        print(f"  {icon}")
        print(f"  Stability score     : {h.stability_score:.2f}")
        print(
            f"  Biological check    : "
            f"{h.n_populations_ok} ok  |  {h.n_populations_low} low  |  "
            f"{h.n_populations_high} high  |  {h.n_populations_unknown} unknown"
        )

        if h.critical_issues:
            print(f"\n  🔴 CRITICAL ({len(h.critical_issues)}):")
            for iss in h.critical_issues:
                print(f"    • {iss}")
        if h.warnings:
            print(f"\n  ⚠  WARNINGS ({len(h.warnings)}):")
            for w_msg in h.warnings:
                print(f"    • {w_msg}")
        if not h.critical_issues and not h.warnings:
            print("  ✓ No issues")

        # ── Region activity ───────────────────────────────────────────────────
        print(f"\n{sep('─')}")
        print("REGION ACTIVITY")
        print(sep("─"))
        active = [rn for rn, rs in report.regions.items() if rs.is_active]
        silent = [rn for rn, rs in report.regions.items() if not rs.is_active]
        print(f"  Active: {len(active)}/{len(report.regions)}")
        if silent:
            print(f"  Silent: {', '.join(silent)}")
        print()

        for rn, rs in report.regions.items():
            icon = "✓" if rs.is_active else "✗"
            ei_str = f"  E/I={rs.ei_ratio:.2f}" if not np.isnan(rs.ei_ratio) else ""
            print(f"  {icon} {rn}  {rs.mean_fr_hz:.2f} Hz avg | {rs.total_spikes:,} spikes{ei_str}")

            if detailed:
                for pn, ps in rs.populations.items():
                    bio_str = ""
                    if ps.bio_range_hz is not None:
                        lo, hi = ps.bio_range_hz
                        bio_str = f"  [target {lo:.0f}–{hi:.0f} Hz]"
                    status_icon = {"ok": "✓", "low": "⚠", "high": "⚠", "unknown": " "}[ps.bio_plausibility]
                    isi_str = ""
                    if not np.isnan(ps.isi_cv):
                        isi_str = f"  CV={ps.isi_cv:.2f}"
                    print(
                        f"    {status_icon} {pn:30s} {ps.mean_fr_hz:6.2f} Hz  "
                        f"({ps.n_neurons} neurons, {ps.total_spikes:,} spikes){isi_str}{bio_str}"
                    )

        # ── Oscillations ──────────────────────────────────────────────────────
        print(f"\n{sep('─')}")
        print("OSCILLATORY DYNAMICS")
        print(sep("─"))
        osc = report.oscillations
        print(f"  Global dominant: {osc.global_dominant_freq_hz:.1f} Hz")
        print("  Global band power:")
        for band, pwr in osc.global_band_power.items():
            bar = "█" * int(pwr * 30)
            print(f"    {band:>6s}  {bar} {pwr:.3f}")

        if detailed:
            print("\n  Per-region dominant frequency:")
            for rn in report.region_keys or []:
                freq = osc.region_dominant_freq.get(rn, 0.0)
                dom_band = osc.region_dominant_band.get(rn, "?")
                if freq > 0:
                    print(f"    {rn:35s}  {freq:6.1f} Hz  ({dom_band})")
                else:
                    print(f"    {rn:35s}   silent")

        # ── Connectivity ──────────────────────────────────────────────────────
        print(f"\n{sep('─')}")
        print("CONNECTIVITY")
        print(sep("─"))
        conn = report.connectivity
        print(f"  Functional axonal tracts: {conn.n_functional}/{len(conn.tracts)}")
        if conn.n_broken:
            print("  Non-functional tracts:")
            for bt in conn.n_broken:
                print(f"    ✗ {bt.synapse_id}  ({bt.transmission_ratio*100:.1f}%)")
        if detailed:
            print()
            for ts in conn.tracts:
                delay_str = ""
                if not np.isnan(ts.measured_delay_ms):
                    diff = ts.measured_delay_ms - ts.expected_delay_ms
                    delay_str = (
                        f"  delay: measured={ts.measured_delay_ms:.0f}ms "
                        f"expected={ts.expected_delay_ms:.0f}ms (Δ{diff:+.0f}ms)"
                    )
                icon = "✓" if ts.is_functional else "✗"
                print(
                    f"  {icon} {str(ts.synapse_id):65s}"
                    f"  {ts.transmission_ratio*100:5.1f}%  "
                    f"({ts.spikes_sent:,} spikes){delay_str}"
                )

        # ── Homeostasis ───────────────────────────────────────────────────────
        print(f"\n{sep('─')}")
        print("HOMEOSTATIC GAINS (g_L_scale)")
        print(sep("─"))
        hs = report.homeostasis
        if not hs.gain_trajectories:
            print("  No gain data recorded")
        else:
            print(f"  {'Population':<35s}  {'Initial':>8s}  {'Final':>8s}  {'Change':>8s}")
            print(f"  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*8}")
            for key in sorted(hs.gain_trajectories):
                traj = hs.gain_trajectories[key]
                valid = traj[~np.isnan(traj)]
                if len(valid) < 2:
                    continue
                init, final = valid[0], valid[-1]
                pct = (final - init) / init * 100 if init > 0 else 0
                status = "⚠" if final < 0.3 or final > 1.9 else " "
                print(f"  {status} {key:<35s}  {init:8.3f}  {final:8.3f}  {pct:+7.1f}%")

        if hs.stp_final_state:
            print(f"\n  STP state at end of recording (mean efficacy = u·x):")
            for key, state in sorted(hs.stp_final_state.items()):
                print(
                    f"    {key:<60s}  x={state['mean_x']:.3f}  u={state['mean_u']:.3f}"
                    f"  eff={state['efficacy']:.3f}"
                )

        print(f"\n{sep()}\n")

    # =========================================================================
    # SAVE
    # =========================================================================

    def save(self, report: DiagnosticsReport, output_dir: str) -> None:
        """Save report summary (JSON) and raw traces (npz) to ``output_dir``."""
        os.makedirs(output_dir, exist_ok=True)

        # JSON summary
        def _clean(v: Any) -> Any:
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return None
            if isinstance(v, np.floating):
                return float(v)
            if isinstance(v, np.integer):
                return int(v)
            return v

        summary: Dict[str, Any] = {
            "timestamp": report.timestamp,
            "simulation_time_ms": report.simulation_time_ms,
            "n_timesteps": report.n_timesteps,
            "mode": report.mode,
            "is_healthy": report.health.is_healthy,
            "stability_score": _clean(report.health.stability_score),
            "critical_issues": report.health.critical_issues,
            "warnings": report.health.warnings,
            "population_count": {
                "ok": report.health.n_populations_ok,
                "low": report.health.n_populations_low,
                "high": report.health.n_populations_high,
                "unknown": report.health.n_populations_unknown,
            },
            "region_firing_rates_hz": {
                rn: _clean(rs.mean_fr_hz) for rn, rs in report.regions.items()
            },
            "population_firing_rates_hz": {
                f"{rn}:{pn}": _clean(ps.mean_fr_hz)
                for rn, rs in report.regions.items()
                for pn, ps in rs.populations.items()
            },
            "stp_final_state": report.homeostasis.stp_final_state,
            "global_dominant_freq_hz": _clean(report.oscillations.global_dominant_freq_hz),
            "global_band_power": {k: _clean(v) for k, v in report.oscillations.global_band_power.items()},
        }

        json_path = os.path.join(output_dir, "diagnostics_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved JSON summary  → {json_path}")

        # NPZ traces
        traces: Dict[str, np.ndarray] = {}
        if report.raw_spike_counts is not None:
            traces["pop_spike_counts"] = report.raw_spike_counts
        if report.pop_rate_binned is not None:
            traces["pop_rate_binned"] = report.pop_rate_binned
        if report.raw_voltages is not None:
            traces["voltages"] = report.raw_voltages
        if report.raw_g_exc is not None:
            traces["g_exc"] = report.raw_g_exc
            traces["g_inh"] = report.raw_g_inh
        for key, traj in report.homeostasis.gain_trajectories.items():
            traces[f"gain_{key.replace(':', '_')}"] = traj
        if traces:
            npz_path = os.path.join(output_dir, "diagnostics_traces.npz")
            np.savez_compressed(npz_path, **traces)
            print(f"  ✓ Saved NPZ traces   → {npz_path}")

    # =========================================================================
    # PLOTTING
    # =========================================================================

    def plot(self, report: DiagnosticsReport, output_dir: str) -> None:
        """Generate and save diagnostic plots to ``output_dir``.

        Requires ``matplotlib``.  Silently skips if it is not installed.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed; skipping diagnostic plots.", stacklevel=2)
            return

        os.makedirs(output_dir, exist_ok=True)
        pop_keys = self._pop_keys

        # Colour per population: green=ok, orange=low/high, red=silent/hyperactive
        def pop_color(ps: PopulationStats) -> str:
            if ps.total_spikes == 0:
                return "#e74c3c"
            if ps.bio_plausibility == "ok":
                return "#2ecc71"
            if ps.bio_plausibility in ("low", "high"):
                return "#f39c12"
            return "#95a5a6"

        # --- 1: Firing rates overview -------------------------------------------
        fig, ax = plt.subplots(figsize=(14, max(6, self._n_pops * 0.2)))
        labels, values, colors = [], [], []
        for rn, rs in report.regions.items():
            for pn, ps in rs.populations.items():
                labels.append(f"{rn}:{pn}")
                values.append(ps.mean_fr_hz)
                colors.append(pop_color(ps))

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, color=colors, height=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Firing rate (Hz)")
        ax.set_title("Population Firing Rates  (green=bio-ok, orange=out-of-range, red=silent)")
        ax.axvline(0, color="k", linewidth=0.5)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "01_firing_rates.png"), dpi=150)
        plt.close(fig)
        print(f"  ✓ 01_firing_rates.png")

        # --- 2: Population rate heatmap -----------------------------------------
        if report.pop_rate_binned is not None:
            pr = report.pop_rate_binned  # [n_bins, n_pops]
            # Convert to Hz
            pr_hz = pr * (1000.0 / self.config.rate_bin_ms)
            t_axis = np.arange(pr_hz.shape[0]) * self.config.rate_bin_ms

            fig, ax = plt.subplots(figsize=(16, max(6, self._n_pops * 0.18)))
            vmax = float(np.percentile(pr_hz, 99)) or 1.0
            im = ax.imshow(
                pr_hz.T,
                aspect="auto",
                origin="lower",
                interpolation="none",
                extent=[t_axis[0], t_axis[-1], 0, self._n_pops],
                cmap="hot",
                vmin=0,
                vmax=vmax,
            )
            ax.set_yticks(np.arange(self._n_pops) + 0.5)
            ax.set_yticklabels([f"{rn}:{pn}" for rn, pn in pop_keys], fontsize=5)
            ax.set_xlabel("Time (ms)")
            ax.set_title("Population Firing Rate Heatmap (Hz)")
            plt.colorbar(im, ax=ax, label="Firing rate (Hz)")
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, "02_population_heatmap.png"), dpi=120)
            plt.close(fig)
            print(f"  ✓ 02_population_heatmap.png")

        # --- 3: Per-region spectra -----------------------------------------------
        osc = report.oscillations
        n_plots = len(report.region_keys or [])
        if n_plots > 0:
            ncols = 4
            nrows = (n_plots + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 2.5))
            axes = np.array(axes).flatten()
            for ax_i, rn in enumerate(report.region_keys or []):
                ax = axes[ax_i]
                bp = osc.region_band_power.get(rn, {})
                bands = list(bp.keys())
                powers = [bp[b] for b in bands]
                colors_bar = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
                ax.bar(bands, powers, color=colors_bar[: len(bands)])
                ax.set_title(rn, fontsize=7, pad=2)
                ax.set_ylim(0, 1)
                ax.tick_params(labelsize=6)
                dom_freq = osc.region_dominant_freq.get(rn, 0.0)
                if dom_freq > 0:
                    ax.text(0.98, 0.95, f"{dom_freq:.1f} Hz", ha="right", va="top",
                            transform=ax.transAxes, fontsize=7)
            for ax_i in range(n_plots, len(axes)):
                axes[ax_i].set_visible(False)
            fig.suptitle("Per-Region Power Spectra", fontsize=10)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, "03_region_spectra.png"), dpi=150)
            plt.close(fig)
            print(f"  ✓ 03_region_spectra.png")

        # --- 4: Cross-regional coherence (theta) --------------------------------
        coh = osc.coherence_theta
        n_r = len(osc.region_order)
        fig, ax = plt.subplots(figsize=(max(8, n_r * 0.5), max(6, n_r * 0.5)))
        im = ax.imshow(coh, vmin=0, vmax=1, cmap="YlOrRd", aspect="equal")
        ax.set_xticks(np.arange(n_r))
        ax.set_xticklabels(osc.region_order, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(n_r))
        ax.set_yticklabels(osc.region_order, fontsize=6)
        ax.set_title("Cross-Regional Theta Coherence (4–8 Hz)")
        plt.colorbar(im, ax=ax, label="Coherence")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "04_coherence_theta.png"), dpi=150)
        plt.close(fig)
        print(f"  ✓ 04_coherence_theta.png")

        # --- 5: ISI distributions for key populations ----------------------------
        if self.config.mode == "full":
            key_pops = [
                k for k in self._spike_times
                if any(
                    tag in k[1].lower()
                    for tag in ["relay", "ca1", "l23_pyr", "executive", "d1", "d2", "gaba", "stn"]
                )
            ][:12]

            if key_pops:
                ncols = min(4, len(key_pops))
                nrows = (len(key_pops) + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
                axes = np.array(axes).flatten()
                for ax_i, key in enumerate(key_pops):
                    ax = axes[ax_i]
                    all_isis: List[float] = []
                    for times in self._spike_times[key]:
                        if len(times) >= 2:
                            all_isis.extend(np.diff(times).tolist())
                    if all_isis:
                        isis_ms = np.array(all_isis, dtype=np.float32) * self.dt_ms
                        ax.hist(
                            isis_ms, bins=50, range=(0, float(np.percentile(isis_ms, 98))),
                            color="#3498db", edgecolor="none", density=True
                        )
                        cv = np.std(isis_ms) / np.mean(isis_ms) if np.mean(isis_ms) > 0 else 0
                        ax.set_title(f"{key[0]}:{key[1]}\nCV={cv:.2f}", fontsize=7)
                        ax.set_xlabel("ISI (ms)", fontsize=7)
                        ax.tick_params(labelsize=6)
                for ax_i in range(len(key_pops), len(axes)):
                    axes[ax_i].set_visible(False)
                fig.suptitle("Inter-Spike Interval Distributions", fontsize=10)
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, "05_isi_distributions.png"), dpi=150)
                plt.close(fig)
                print(f"  ✓ 05_isi_distributions.png")

        # --- 6: Homeostatic gain trajectories ------------------------------------
        hs = report.homeostasis
        if hs.gain_trajectories and len(hs.gain_sample_times_ms) > 1:
            t_ms = hs.gain_sample_times_ms
            fig, ax = plt.subplots(figsize=(14, 5))
            cmap = plt.get_cmap("tab20")
            for i, (key, traj) in enumerate(sorted(hs.gain_trajectories.items())):
                valid = ~np.isnan(traj)
                if valid.sum() < 2:
                    continue
                color = cmap(i % 20)
                ax.plot(t_ms[valid], traj[valid], linewidth=0.8, alpha=0.7, color=color, label=key)
            ax.axhline(0.3, color="red", linestyle="--", linewidth=1.0, label="Collapse threshold")
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("g_L_scale")
            ax.set_title("Homeostatic Gain Trajectories")
            ax.set_ylim(0, 2.1)
            ax.legend(fontsize=5, ncol=3, loc="upper right")
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, "06_homeostatic_gains.png"), dpi=150)
            plt.close(fig)
            print(f"  ✓ 06_homeostatic_gains.png")

        # --- 7: Voltage traces for sample neurons --------------------------------
        if self.config.mode == "full" and report.raw_voltages is not None:
            key_pop_names = ["relay", "ca1", "l23_pyr", "l4_pyr", "executive", "d1"]
            sampled = []
            for rn, pn in pop_keys:
                if any(tag in pn.lower() for tag in key_pop_names) and len(sampled) < 6:
                    idx = self._pop_index[(rn, pn)]
                    sampled.append((rn, pn, idx))

            if sampled:
                fig, axes = plt.subplots(len(sampled), 1, figsize=(16, len(sampled) * 1.8), sharex=True)
                if len(sampled) == 1:
                    axes = [axes]
                t_ms = np.arange(report.n_timesteps, dtype=np.float32) * self.dt_ms
                for ax, (rn, pn, pidx) in zip(axes, sampled):
                    v = report.raw_voltages[:, pidx, :]  # [T, sample_size]
                    n_valid = int(self._pop_sizes[pidx])
                    n_show = min(8, n_valid, v.shape[1])
                    for ni in range(n_show):
                        col = plt.get_cmap("tab10")(ni)
                        y = v[:, ni]
                        valid_mask = ~np.isnan(y)
                        if valid_mask.sum() > 1:
                            ax.plot(t_ms[valid_mask], y[valid_mask], linewidth=0.5, alpha=0.7, color=col)
                    ax.set_ylabel(f"{pn}\n({rn})", fontsize=6)
                    ax.tick_params(labelsize=6)
                axes[-1].set_xlabel("Time (ms)")
                fig.suptitle("Sample Neuron Voltage Traces", fontsize=10)
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, "07_voltage_traces.png"), dpi=150)
                plt.close(fig)
                print(f"  ✓ 07_voltage_traces.png")

        # --- 8: Raster plots for key regions ------------------------------------
        if self.config.mode == "full" and self._spike_times:
            raster_regions = [
                "thalamus", "hippocampus", "cortex_sensory",
                "prefrontal", "striatum", "medial_septum",
            ]
            existing = [rn for rn in raster_regions if rn in self.brain.regions][:6]
            if existing:
                fig, axes = plt.subplots(
                    len(existing), 1,
                    figsize=(16, len(existing) * 2.5),
                    sharex=True,
                )
                if len(existing) == 1:
                    axes = [axes]
                t_ms_max = report.n_timesteps * self.dt_ms
                for ax, rn in zip(axes, existing):
                    y_offset = 0
                    region = self.brain.regions[rn]
                    pop_colors_raster = plt.get_cmap("tab10")
                    for pi, pn in enumerate(region.neuron_populations.keys()):
                        key = (rn, pn)
                        if key not in self._spike_times:
                            continue
                        n_neurons = len(self._spike_times[key])
                        n_show = min(80, n_neurons)
                        col = pop_colors_raster(pi % 10)
                        for ni in range(n_show):
                            times = self._spike_times[key][ni]
                            if times:
                                t_vals = np.array(times, dtype=np.float32) * self.dt_ms
                                ax.scatter(
                                    t_vals, np.full_like(t_vals, y_offset + ni),
                                    s=1.0, c=[col], marker="|", linewidths=0.5
                                )
                        ax.axhline(y_offset + n_show, color="gray", linewidth=0.3)
                        ax.text(
                            t_ms_max * 1.002, y_offset + n_show / 2, pn,
                            fontsize=6, va="center"
                        )
                        y_offset += n_show + 5
                    ax.set_ylabel(rn, fontsize=8)
                    ax.set_xlim(0, t_ms_max)
                    ax.tick_params(labelsize=6)
                axes[-1].set_xlabel("Time (ms)")
                fig.suptitle("Spike Rasters (sample neurons)", fontsize=10)
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, "08_raster_plots.png"), dpi=130)
                plt.close(fig)
                print(f"  ✓ 08_raster_plots.png")

        # --- 9: E/I conductance phase portrait ----------------------------------
        if self.config.mode == "full" and self._g_exc_samples is not None and self._g_inh_samples is not None:
            exc_valid = not np.all(np.isnan(self._g_exc_samples))
            inh_valid = not np.all(np.isnan(self._g_inh_samples))
            if exc_valid and inh_valid:
                key_ei_pops = ["relay", "ca1", "l23_pyr", "l4_pyr", "executive"]
                sampled_ei: List[Tuple[str, str, int]] = []
                for rn, pn in pop_keys:
                    if any(t in pn for t in key_ei_pops) and len(sampled_ei) < 6:
                        idx = self._pop_index[(rn, pn)]
                        sampled_ei.append((rn, pn, idx))

                if sampled_ei:
                    fig, axes = plt.subplots(
                        1, len(sampled_ei),
                        figsize=(len(sampled_ei) * 3, 3),
                    )
                    if len(sampled_ei) == 1:
                        axes = [axes]
                    n_cond = self._cond_sample_step
                    for ax, (rn, pn, pidx) in zip(axes, sampled_ei):
                        g_e = self._g_exc_samples[:n_cond, pidx, :].flatten()
                        g_i = self._g_inh_samples[:n_cond, pidx, :].flatten()
                        valid = ~(np.isnan(g_e) | np.isnan(g_i))
                        if valid.sum() > 5:
                            ax.scatter(g_e[valid], g_i[valid], s=2, alpha=0.4, c="#3498db")
                        ax.set_xlabel("g_exc", fontsize=7)
                        ax.set_ylabel("g_inh", fontsize=7)
                        ax.set_title(f"{pn}\n({rn})", fontsize=7)
                        ax.tick_params(labelsize=6)
                    fig.suptitle("E/I Conductance Phase Portraits", fontsize=10)
                    plt.tight_layout()
                    fig.savefig(os.path.join(output_dir, "09_ei_phase_portrait.png"), dpi=150)
                    plt.close(fig)
                    print(f"  ✓ 09_ei_phase_portrait.png")

        print(f"\n  All plots saved to: {output_dir}")
