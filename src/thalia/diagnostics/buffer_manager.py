"""Buffer management: allocation, reset, and windowed-recording helpers.

Centralises all mutable recording state that was previously scattered
across ``DiagnosticsRecorder.__init__``, ``_allocate_*_buffers()``,
``reset()``, and the circular-buffer helper methods.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .diagnostics_config import DiagnosticsConfig
from .population_index import PopulationIndex


class BufferManager:
    """Owns every recording buffer and the cursors that drive them.

    The :class:`DiagnosticsRecorder` writes into these buffers during
    ``record()``; the :mod:`snapshot_builder` reads them when constructing
    a :class:`RecorderSnapshot`.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        index: PopulationIndex,
        config: DiagnosticsConfig,
        dt_ms: float,
    ) -> None:
        self._windowed: bool = config.window_size is not None
        self._window_size: Optional[int] = config.window_size

        T = config.n_timesteps
        P = index.n_pops
        R = index.n_regions
        V = config.voltage_sample_size
        C = config.conductance_sample_size
        W = self._window_size if self._windowed else T

        # ── Counters / cursors ───────────────────────────────────────────
        self.write_cursor: int = 0
        self.total_n_recorded: int = 0
        self.n_recorded: int = 0
        self.recording_started: bool = False

        self.gain_sample_step: int = 0
        self.cond_sample_step: int = 0
        self.gain_sample_times: List[int] = []

        # ── Spike buffers ────────────────────────────────────────────────
        self.pop_spike_counts: np.ndarray = np.zeros((W, P), dtype=np.int32)
        self.per_neuron_spike_counts: List[np.ndarray] = [
            np.zeros(int(sz), dtype=np.int32) for sz in index.pop_sizes
        ]
        self.region_spike_counts: np.ndarray = np.zeros((W, R), dtype=np.int32)
        self.tract_sent: np.ndarray = np.zeros(
            (W, index.n_tracts), dtype=np.int32
        )
        self.spike_times: Dict[Tuple[str, str], List[List[int]]] = {}

        initial_cap = max(64, T // 10)
        self.spike_flat_nidx: List[np.ndarray] = [
            np.empty(initial_cap, dtype=np.int32) for _ in range(P)
        ]
        self.spike_flat_ts: List[np.ndarray] = [
            np.empty(initial_cap, dtype=np.int32) for _ in range(P)
        ]
        self.spike_flat_n: np.ndarray = np.zeros(P, dtype=np.int64)

        # ── State sample buffers ─────────────────────────────────────────
        self.voltages: np.ndarray = np.full(
            (W, P, V), np.nan, dtype=np.float32
        )

        ci = config.conductance_sample_interval_steps
        n_cond = max(1, W // ci)
        self.g_exc_samples: np.ndarray = np.full(
            (n_cond, P, C), np.nan, dtype=np.float32
        )
        self.g_inh_samples: np.ndarray = np.full(
            (n_cond, P, C), np.nan, dtype=np.float32
        )
        self.g_nmda_samples: np.ndarray = np.full(
            (n_cond, P, C), np.nan, dtype=np.float32
        )
        self.g_gaba_b_samples: np.ndarray = np.full(
            (n_cond, P, C), np.nan, dtype=np.float32
        )
        self.g_apical_samples: np.ndarray = np.full(
            (n_cond, P, C), np.nan, dtype=np.float32
        )
        # Two-compartment extra: dendritic voltage (every timestep, like voltages)
        # and NMDA plateau conductance (sampled like other conductances).
        self.v_dend: np.ndarray = np.full(
            (W, P, V), np.nan, dtype=np.float32
        )
        self.g_plateau_samples: np.ndarray = np.full(
            (n_cond, P, C), np.nan, dtype=np.float32
        )

        # ── Trajectory buffers ───────────────────────────────────────────
        gi_steps = max(1, int(config.gain_sample_interval_ms / dt_ms))
        n_gain = max(1, T // gi_steps)

        self.g_L_scale_history: np.ndarray = np.full(
            (n_gain, P), np.nan, dtype=np.float32
        )
        self.stp_efficacy_history: np.ndarray = np.full(
            (n_gain, index.n_stp), np.nan, dtype=np.float32
        )
        self.nm_concentration_history: np.ndarray = np.full(
            (n_gain, max(1, index.n_nm_receptors)), np.nan, dtype=np.float32
        )

        # ── Learning trajectory buffers ──────────────────────────────────
        self.weight_dist_history: Optional[np.ndarray] = None
        self.weight_update_magnitude_history: Optional[np.ndarray] = None
        self.eligibility_mean_history: Optional[np.ndarray] = None
        self.eligibility_ltp_ltd_ratio_history: Optional[np.ndarray] = None
        self.prev_weight_snapshots: Optional[List[Optional[np.ndarray]]] = None
        self.bcm_theta_history: Optional[np.ndarray] = None

        if index.n_learning > 0:
            self.weight_dist_history = np.full(
                (n_gain, index.n_learning, 5), np.nan, dtype=np.float32
            )
            self.weight_update_magnitude_history = np.full(
                (n_gain, index.n_learning), np.nan, dtype=np.float32
            )
            self.eligibility_mean_history = np.full(
                (n_gain, index.n_learning), np.nan, dtype=np.float32
            )
            self.eligibility_ltp_ltd_ratio_history = np.full(
                (n_gain, index.n_learning), np.nan, dtype=np.float32
            )
            self.prev_weight_snapshots = [None] * index.n_learning
        if index.n_bcm > 0:
            self.bcm_theta_history = np.full(
                (n_gain, index.n_bcm), np.nan, dtype=np.float32
            )

        self.homeostatic_correction_rate: np.ndarray = np.full(
            (n_gain, P), np.nan, dtype=np.float32
        )

        # Population vector snapshots for representational stability
        popvec_interval = max(
            50, int(500.0 / config.gain_sample_interval_ms)
        )
        max_snapshots = max(1, n_gain // popvec_interval) + 1
        self.popvec_snapshots: np.ndarray = np.full(
            (max_snapshots, P), np.nan, dtype=np.float32
        )
        self.popvec_snapshot_times: List[int] = []
        self.popvec_interval: int = popvec_interval

        # Side-channel for neuron config class names (set during snapshot build)
        self.pop_config_types: Dict[Tuple[str, str], str] = {}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, n_learning: int) -> None:
        """Zero all buffers in-place. Call before starting a new recording."""
        self.pop_spike_counts.fill(0)
        for arr in self.per_neuron_spike_counts:
            arr.fill(0)
        self.region_spike_counts.fill(0)
        self.tract_sent.fill(0)
        self.spike_times.clear()
        self.pop_config_types.clear()
        self.spike_flat_n.fill(0)
        self.gain_sample_times.clear()
        self.gain_sample_step = 0

        self.voltages.fill(np.nan)
        self.g_exc_samples.fill(np.nan)
        self.g_inh_samples.fill(np.nan)
        self.g_nmda_samples.fill(np.nan)
        self.g_gaba_b_samples.fill(np.nan)
        self.g_apical_samples.fill(np.nan)
        self.v_dend.fill(np.nan)
        self.g_plateau_samples.fill(np.nan)
        self.cond_sample_step = 0

        self.g_L_scale_history.fill(np.nan)
        self.stp_efficacy_history.fill(np.nan)
        self.nm_concentration_history.fill(np.nan)

        self.recording_started = False
        self.n_recorded = 0
        self.write_cursor = 0
        self.total_n_recorded = 0

        if self.weight_dist_history is not None:
            self.weight_dist_history.fill(np.nan)
        if self.weight_update_magnitude_history is not None:
            self.weight_update_magnitude_history.fill(np.nan)
        if self.eligibility_mean_history is not None:
            self.eligibility_mean_history.fill(np.nan)
        if self.eligibility_ltp_ltd_ratio_history is not None:
            self.eligibility_ltp_ltd_ratio_history.fill(np.nan)
        if self.bcm_theta_history is not None:
            self.bcm_theta_history.fill(np.nan)
        if self.prev_weight_snapshots is not None:
            self.prev_weight_snapshots = [None] * n_learning
        self.homeostatic_correction_rate.fill(np.nan)
        self.popvec_snapshots.fill(np.nan)
        self.popvec_snapshot_times.clear()

    # ------------------------------------------------------------------
    # Windowed-recording helpers
    # ------------------------------------------------------------------

    def compute_write_index(self, timestep: int, n_timesteps: int) -> Optional[int]:
        """Return the buffer row for *timestep*, or ``None`` to skip.

        In windowed mode, returns the circular cursor.  In full-recording
        mode, returns *timestep* (or ``None`` if it exceeds the buffer).
        """
        if self._windowed:
            return self.write_cursor
        if timestep >= n_timesteps:
            return None
        return timestep

    def advance_cursor(self) -> None:
        """Advance the circular write cursor and periodically trim spike buffers."""
        if not self._windowed:
            return
        assert self._window_size is not None
        self.write_cursor = (self.write_cursor + 1) % self._window_size
        self.total_n_recorded += 1
        if self.total_n_recorded % self._window_size == 0:
            self.trim_spike_flat_buffers()

    def trim_spike_flat_buffers(self) -> None:
        """Remove spike events older than the current window."""
        assert self._window_size is not None
        cutoff = self.total_n_recorded - self._window_size
        if cutoff <= 0:
            return
        n_pops = len(self.spike_flat_n)
        for pop_idx in range(n_pops):
            cnt = int(self.spike_flat_n[pop_idx])
            if cnt == 0:
                continue
            ts = self.spike_flat_ts[pop_idx][:cnt]
            keep_from = int(np.searchsorted(ts, cutoff, side="left"))
            if keep_from == 0:
                continue
            keep_count = cnt - keep_from
            if keep_count > 0:
                self.spike_flat_nidx[pop_idx][:keep_count] = (
                    self.spike_flat_nidx[pop_idx][keep_from:cnt]
                )
                self.spike_flat_ts[pop_idx][:keep_count] = (
                    self.spike_flat_ts[pop_idx][keep_from:cnt]
                )
            self.spike_flat_n[pop_idx] = keep_count

    def build_spike_times(
        self,
        pop_keys: List[Tuple[str, str]],
        pop_sizes: np.ndarray,
    ) -> None:
        """Convert flat spike-time accumulators into nested per-neuron lists.

        In windowed mode, spike events outside the current window are discarded
        and remaining timestamps are renumbered to a zero-based range.
        """
        self.spike_times.clear()

        if self._window_size is not None:
            window_start = max(0, self.total_n_recorded - self._window_size)
        else:
            window_start = 0

        for pop_idx, key in enumerate(pop_keys):
            cnt = int(self.spike_flat_n[pop_idx])
            if cnt == 0:
                continue
            flat_nidx = self.spike_flat_nidx[pop_idx][:cnt]
            flat_ts = self.spike_flat_ts[pop_idx][:cnt]

            if self._window_size is not None and window_start > 0:
                mask = flat_ts >= window_start
                flat_nidx = flat_nidx[mask]
                flat_ts = flat_ts[mask] - window_start
            elif self._window_size is not None:
                flat_ts = flat_ts.copy()

            n = int(pop_sizes[pop_idx])
            nested: List[List[int]] = [[] for _ in range(n)]
            for ni, ts in zip(flat_nidx.tolist(), flat_ts.tolist()):
                nested[ni].append(ts)
            self.spike_times[key] = nested

    def rebuild_per_neuron_spike_counts_from_window(
        self, pop_sizes: np.ndarray
    ) -> List[np.ndarray]:
        """Recompute per-neuron spike counts from spike events within the window."""
        assert self._window_size is not None
        cutoff = max(0, self.total_n_recorded - self._window_size)
        n_pops = len(self.spike_flat_n)
        result: List[np.ndarray] = []
        for pop_idx in range(n_pops):
            pop_size = int(pop_sizes[pop_idx])
            counts = np.zeros(pop_size, dtype=np.int32)
            cnt = int(self.spike_flat_n[pop_idx])
            if cnt > 0:
                nidx = self.spike_flat_nidx[pop_idx][:cnt]
                ts = self.spike_flat_ts[pop_idx][:cnt]
                mask = ts >= cutoff
                if mask.any():
                    np.add.at(counts, nidx[mask], 1)
            result.append(counts)
        return result
