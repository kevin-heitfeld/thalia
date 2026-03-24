"""Snapshot construction: decouples analysis / plotting from live brain state.

Extracts the ``to_snapshot()`` assembly logic and the three ``_capture_*``
methods that were previously embedded in :class:`DiagnosticsRecorder`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import torch

from .buffer_manager import BufferManager
from .circular_buffer import linearise_circular
from .diagnostics_config import DiagnosticsConfig
from .diagnostics_snapshot import RecorderSnapshot, RunningStats
from .population_index import PopulationIndex

if TYPE_CHECKING:
    from thalia.brain import Brain


# =====================================================================
# Helpers
# =====================================================================

def _copy_or_none(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    return arr.copy() if arr is not None else None


# =====================================================================
# Brain-state capture (requires live Brain reference)
# =====================================================================


def capture_stp_final_state(
    brain: Brain,
    index: PopulationIndex,
) -> Dict[str, Dict[str, float]]:
    """Read the current x / u tensors from all live STP modules."""
    result: Dict[str, Dict[str, float]] = {}
    for rn, syn_id in index.stp_keys:
        stp_mod = brain.regions[rn].get_stp_module(syn_id)
        assert stp_mod is not None
        result[str(syn_id)] = {
            "mean_x": float(stp_mod.x.mean().item()),
            "mean_u": float(stp_mod.u.mean().item()),
            "efficacy": float((stp_mod.x * stp_mod.u).mean().item()),
        }
    return result


def capture_tract_weight_stats(
    brain: Brain,
    index: PopulationIndex,
) -> Dict[str, Dict[str, float]]:
    """Snapshot per-tract weight statistics from the live brain."""
    result: Dict[str, Dict[str, float]] = {}
    for syn_id in index.tract_keys:
        region = brain.regions[syn_id.target_region]
        w = region.get_synaptic_weights(syn_id).data
        result[str(syn_id)] = {
            "mean": float(w.mean().item()),
            "n_nonzero": float((w != 0).sum().item()),
            "n_total": float(w.numel()),
        }
    return result


def capture_pop_neuron_params(
    brain: Brain,
    index: PopulationIndex,
) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], Dict[Tuple[str, str], str]]:
    """Snapshot per-population neuron biophysics from the live brain.

    Returns ``(pop_neuron_params, pop_config_types)`` — the second dict
    maps ``(region, pop)`` to the neuron config class name.
    """
    _EXTRA_FIELDS = (
        "i_h_conductance", "i_h_reversal",
        "sk_conductance", "sk_reversal", "ca_decay", "ca_influx_per_spike",
        "noise_std",
        "autoreceptor_conductance", "autoreceptor_tau_ms", "autoreceptor_gain",
        "gap_junction_strength",
        "uncertainty_to_current_gain", "serotonin_drive_gain",
        # TwoCompartmentLIF dendritic parameters
        "bap_amplitude", "theta_Ca", "g_Ca_spike", "tau_Ca_ms",
        "g_c", "C_d", "g_L_d",
        "enable_nmda_plateau", "nmda_plateau_threshold",
        "v_dend_plateau_threshold", "g_nmda_plateau", "tau_plateau_ms",
    )

    result: Dict[Tuple[str, str], Dict[str, float]] = {}
    config_types: Dict[Tuple[str, str], str] = {}

    for rn, pn in index.pop_keys:
        pop = brain.regions[rn].get_neuron_population(pn)
        assert pop is not None
        cfg = pop.config

        def _scalar(v: object) -> float:
            if isinstance(v, torch.Tensor):
                return float(v.mean().item())
            return float(v)

        params: Dict[str, float] = {
            "g_L": _scalar(cfg.g_L),
            "v_threshold": _scalar(cfg.v_threshold),
            "v_reset": _scalar(cfg.v_reset),
            "E_E": _scalar(cfg.E_E),
            "E_I": _scalar(cfg.E_I),
            "tau_E": _scalar(cfg.tau_E),
            "tau_I": _scalar(cfg.tau_I),
            "tau_ref": _scalar(cfg.tau_ref),
            "tau_mem_ms": _scalar(cfg.tau_mem_ms),
            "adapt_increment": _scalar(cfg.adapt_increment),
            "tau_adapt_ms": _scalar(cfg.tau_adapt_ms),
            "E_adapt": _scalar(cfg.E_adapt),
            "tau_GABA_B": _scalar(cfg.tau_GABA_B),
            "E_GABA_B": _scalar(cfg.E_GABA_B),
            "tau_nmda": _scalar(cfg.tau_nmda),
            "E_nmda": _scalar(cfg.E_nmda),
        }
        params["_config_class"] = hash(type(cfg).__name__) * 0  # placeholder
        for attr in _EXTRA_FIELDS:
            val = getattr(cfg, attr, None)
            if val is not None:
                params[attr] = _scalar(val)
        params["_config_type_name"] = 0.0  # placeholder

        result[(rn, pn)] = params
        config_types[(rn, pn)] = type(cfg).__name__

    return result, config_types


# =====================================================================
# Main snapshot assembly
# =====================================================================


def build_snapshot(
    brain: Brain,
    index: PopulationIndex,
    buffers: BufferManager,
    config: DiagnosticsConfig,
    dt_ms: float,
) -> RecorderSnapshot:
    """Assemble a :class:`RecorderSnapshot` from the current recorder state.

    The snapshot is a self-contained, serialisable copy that carries no
    references to the live ``Brain`` or any ``torch.Tensor`` objects.
    """
    windowed = buffers._windowed

    # Trim spike flat buffers once more before building spike times.
    if windowed:
        buffers.trim_spike_flat_buffers()

    buffers.build_spike_times(index.pop_keys, index.pop_sizes)

    # ── Compute effective sizes ───────────────────────────────────────
    cursor = 0
    filled = False
    if windowed:
        assert buffers._window_size is not None
        effective_n = min(buffers._window_size, buffers.total_n_recorded)
        filled = buffers.total_n_recorded >= buffers._window_size
        cursor = buffers.write_cursor
    else:
        effective_n = buffers.n_recorded

    # ── Time-indexed circular buffers → linearised copies ─────────────
    if windowed:
        pop_spike_counts = linearise_circular(
            buffers.pop_spike_counts, cursor, filled
        )
        region_spike_counts = linearise_circular(
            buffers.region_spike_counts, cursor, filled
        )
        tract_sent = linearise_circular(buffers.tract_sent, cursor, filled)
        voltages: Optional[np.ndarray] = linearise_circular(
            buffers.voltages, cursor, filled
        )
        v_dend: Optional[np.ndarray] = linearise_circular(
            buffers.v_dend, cursor, filled
        )
    else:
        pop_spike_counts = buffers.pop_spike_counts.copy()
        region_spike_counts = buffers.region_spike_counts.copy()
        tract_sent = buffers.tract_sent.copy()
        voltages = buffers.voltages.copy()
        v_dend = buffers.v_dend.copy()

    # ── Conductance buffers (own circular cursor) ─────────────────────
    if windowed:
        cond_cap = buffers.g_exc_samples.shape[0]
        cond_filled = buffers.cond_sample_step >= cond_cap
        cond_cursor = buffers.cond_sample_step % cond_cap
        g_exc = linearise_circular(
            buffers.g_exc_samples, cond_cursor, cond_filled
        )
        g_inh = linearise_circular(
            buffers.g_inh_samples, cond_cursor, cond_filled
        )
        g_nmda = linearise_circular(
            buffers.g_nmda_samples, cond_cursor, cond_filled
        )
        g_gaba_b = linearise_circular(
            buffers.g_gaba_b_samples, cond_cursor, cond_filled
        )
        g_apical = linearise_circular(
            buffers.g_apical_samples, cond_cursor, cond_filled
        )
        g_plateau = linearise_circular(
            buffers.g_plateau_samples, cond_cursor, cond_filled
        )
        effective_cond_step = min(cond_cap, buffers.cond_sample_step)
    else:
        g_exc = buffers.g_exc_samples.copy()
        g_inh = buffers.g_inh_samples.copy()
        g_nmda = buffers.g_nmda_samples.copy()
        g_gaba_b = buffers.g_gaba_b_samples.copy()
        g_apical = buffers.g_apical_samples.copy()
        g_plateau = buffers.g_plateau_samples.copy()
        effective_cond_step = buffers.cond_sample_step

    # ── Per-neuron spike counts ───────────────────────────────────────
    if windowed:
        per_neuron = buffers.rebuild_per_neuron_spike_counts_from_window(
            index.pop_sizes
        )
    else:
        per_neuron = [arr.copy() for arr in buffers.per_neuron_spike_counts]

    # ── Running stats (windowed mode only) ────────────────────────────
    running_stats: Optional[RunningStats] = None
    if windowed:
        running_stats = RunningStats(
            total_n_recorded=buffers.total_n_recorded,
            total_per_neuron_spike_counts=[
                arr.copy() for arr in buffers.per_neuron_spike_counts
            ],
        )

    # ── Brain-state captures (live tensor reads) ──────────────────────
    pop_neuron_params, pop_config_types = capture_pop_neuron_params(
        brain, index
    )
    buffers.pop_config_types = pop_config_types

    return RecorderSnapshot(
        config=config,
        dt_ms=dt_ms,
        _pop_keys=list(index.pop_keys),
        _pop_index=dict(index.pop_index),
        _n_pops=index.n_pops,
        _pop_sizes=index.pop_sizes.copy(),
        _region_keys=list(index.region_keys),
        _region_index=dict(index.region_index),
        _n_regions=index.n_regions,
        _region_pop_indices={
            k: list(v) for k, v in index.region_pop_indices.items()
        },
        _tract_keys=list(index.tract_keys),
        _tract_index=dict(index.tract_index),
        _n_tracts=index.n_tracts,
        _stp_keys=list(index.stp_keys),
        _nm_receptor_keys=list(index.nm_receptor_keys),
        _n_nm_receptors=index.n_nm_receptors,
        _nm_source_pop_keys=list(index.nm_source_pop_keys),
        _v_sample_idx=[arr.copy() for arr in index.v_sample_idx],
        _c_sample_idx=[arr.copy() for arr in index.c_sample_idx],
        _n_recorded=effective_n,
        _gain_sample_step=buffers.gain_sample_step,
        _cond_sample_step=effective_cond_step,
        _gain_sample_times=list(buffers.gain_sample_times),
        _pop_spike_counts=pop_spike_counts,
        _per_neuron_spike_counts=per_neuron,
        _region_spike_counts=region_spike_counts,
        _tract_sent=tract_sent,
        _spike_times=dict(buffers.spike_times),
        _voltages=voltages,
        _g_exc_samples=g_exc,
        _g_inh_samples=g_inh,
        _g_nmda_samples=g_nmda,
        _g_gaba_b_samples=g_gaba_b,
        _g_apical_samples=g_apical,
        _v_dend_samples=v_dend,
        _g_plateau_samples=g_plateau,
        _g_L_scale_history=buffers.g_L_scale_history.copy(),
        _stp_efficacy_history=buffers.stp_efficacy_history.copy(),
        _nm_concentration_history=buffers.nm_concentration_history.copy(),
        _pop_polarities=dict(index.pop_polarities),
        _tract_delay_ms=list(index.tract_delay_ms),
        _homeostasis_target_hz=dict(index.homeostasis_target_hz),
        _stp_configs=list(index.stp_configs),
        _stp_final_state=capture_stp_final_state(brain, index),
        _tract_weight_stats=capture_tract_weight_stats(brain, index),
        _pop_neuron_params=pop_neuron_params,
        _pop_config_types=pop_config_types,
        _learning_keys=[
            (rn, str(sid)) for rn, sid in index.learning_keys
        ],
        _weight_dist_history=_copy_or_none(buffers.weight_dist_history),
        _weight_update_magnitude_history=_copy_or_none(
            buffers.weight_update_magnitude_history
        ),
        _eligibility_mean_history=_copy_or_none(
            buffers.eligibility_mean_history
        ),
        _eligibility_ltp_ltd_ratio_history=_copy_or_none(
            buffers.eligibility_ltp_ltd_ratio_history
        ),
        _bcm_theta_history=_copy_or_none(buffers.bcm_theta_history),
        _bcm_keys=list(index.bcm_keys),
        _homeostatic_correction_rate=_copy_or_none(
            buffers.homeostatic_correction_rate
        ),
        _popvec_snapshots=_copy_or_none(buffers.popvec_snapshots),
        _popvec_snapshot_times=list(buffers.popvec_snapshot_times),
        _running_stats=running_stats,
    )
