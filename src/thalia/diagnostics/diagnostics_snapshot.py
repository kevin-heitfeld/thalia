"""Recorder snapshot — complete serialisable state of a diagnostics recording.

Pure data module: no simulation dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from thalia.typing import SynapseId

from .diagnostics_config import DiagnosticsConfig


@dataclass
class RunningStats:
    """Full-run statistics accumulated across the entire simulation.

    Only present when :attr:`DiagnosticsConfig.window_size` is set, providing
    context beyond the retained window (e.g. total duration, full-run
    per-neuron spike counts for lifetime fraction-silent tracking).
    """

    total_n_recorded: int
    total_per_neuron_spike_counts: List[np.ndarray]


@dataclass
class RecorderSnapshot:
    """Complete, serialisable snapshot of a :class:`DiagnosticsRecorder`'s recorded state."""

    # ── Config ─────────────────────────────────────────────────────────────
    config: DiagnosticsConfig
    dt_ms: float

    # ── Index metadata ──────────────────────────────────────────────────────
    _pop_keys: List[Tuple[str, str]]
    _pop_index: Dict[Tuple[str, str], int]
    _n_pops: int
    _pop_sizes: np.ndarray  # int32 [n_pops]

    _region_keys: List[str]
    _region_index: Dict[str, int]
    _n_regions: int
    _region_pop_indices: Dict[str, List[int]]

    _tract_keys: List[SynapseId]
    _tract_index: Dict[SynapseId, int]
    _n_tracts: int

    _stp_keys: List[Tuple[str, SynapseId]]
    _nm_receptor_keys: List[Tuple[str, str]]
    _n_nm_receptors: int
    _nm_source_pop_keys: List[Tuple[str, str]]

    _v_sample_idx: List[np.ndarray]
    _c_sample_idx: List[np.ndarray]

    # ── Recording counters ──────────────────────────────────────────────────
    _n_recorded: int
    _gain_sample_step: int
    _cond_sample_step: int
    _gain_sample_times: List[int]

    # ── Spike buffers ───────────────────────────────────────────────────────
    _pop_spike_counts: np.ndarray            # int32 [T, n_pops]
    _per_neuron_spike_counts: List[np.ndarray]  # int32 [n_neurons] per pop
    _region_spike_counts: np.ndarray         # int32 [T, n_regions]
    _tract_sent: np.ndarray                  # int32 [T, n_tracts]
    _spike_times: Dict[Tuple[str, str], List[List[int]]]

    # ── State sample buffers ───────────────
    _voltages: Optional[np.ndarray]          # float32 [T, n_pops, V]
    _g_exc_samples: Optional[np.ndarray]     # float32 [n_cond, n_pops, C]
    _g_inh_samples: Optional[np.ndarray]
    _g_nmda_samples: Optional[np.ndarray]
    _g_gaba_b_samples: Optional[np.ndarray]
    _g_apical_samples: Optional[np.ndarray]

    # ── Trajectory buffers ──────────────────────────────────────────────────
    _g_L_scale_history: np.ndarray           # float32 [n_gain, n_pops]
    _stp_efficacy_history: np.ndarray        # float32 [n_gain, n_stp]
    _nm_concentration_history: np.ndarray    # float32 [n_gain, n_nm_receptors]

    # ── Static brain metadata (snapshotted at recording time) ────────────────
    _pop_polarities: Dict[Tuple[str, str], str]
    _tract_delay_ms: List[float]
    _homeostasis_target_hz: Dict[Tuple[str, str], float]
    _stp_configs: List[Tuple[float, float, float]]
    _stp_final_state: Dict[str, Dict[str, float]]
    _tract_weight_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    _pop_neuron_params: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)
    _pop_config_types: Dict[Tuple[str, str], str] = field(default_factory=dict)

    # ── Learning / training buffers ──────────────────────────────────────
    _learning_keys: List[Tuple[str, str]] = field(default_factory=list)
    _weight_dist_history: Optional[np.ndarray] = None
    _weight_update_magnitude_history: Optional[np.ndarray] = None
    _eligibility_mean_history: Optional[np.ndarray] = None
    _eligibility_ltp_ltd_ratio_history: Optional[np.ndarray] = None
    _bcm_theta_history: Optional[np.ndarray] = None
    _bcm_keys: List[int] = field(default_factory=list)
    _homeostatic_correction_rate: Optional[np.ndarray] = None
    _popvec_snapshots: Optional[np.ndarray] = None
    _popvec_snapshot_times: List[int] = field(default_factory=list)

    # ── Two-compartment dendritic data (TwoCompartmentLIF only) ───────────
    _v_dend_samples: Optional[np.ndarray] = None     # float32 [T, n_pops, V]
    _g_plateau_samples: Optional[np.ndarray] = None  # float32 [n_cond, n_pops, C]

    # ── Streaming / windowed recording context ────────────────────────────
    _running_stats: Optional[RunningStats] = None
