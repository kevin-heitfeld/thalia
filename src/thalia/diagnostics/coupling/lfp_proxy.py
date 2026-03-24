"""Region-level LFP proxy construction — current-based or spike-rate fallback.

Provides a single entry point :func:`build_region_lfp_proxy` that returns a
1-D time series at full timestep resolution for a given region.

**Current-based proxy** (preferred, when conductance + voltage data exist):
    ``LFP(t) ≈ Σ_pop w_pop · [ g_exc(t)·(E_E − V(t))
                               + g_nmda(t)·(E_nmda − V(t))
                               + g_gaba_a(t)·(E_I − V(t))
                               + g_gaba_b(t)·(E_GABA_B − V(t)) ]``

This captures synaptic/dendritic current contributions that are missed by
spike-rate smoothing (Mazzoni et al. 2015; Einevoll et al. 2013).

**Spike-rate fallback** (when conductances are unavailable or under-sampled):
    Gaussian-smoothed aggregate spike rate (σ ≈ 5 ms), zero-centred.
    Standard practice for spike-based LFP proxies in computational neuroscience.

The ``method`` return value indicates which approach was used so that
downstream consumers can report provenance.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d as sp_gaussian_filter1d

from thalia.diagnostics.analysis_region import extract_region_reversal_potentials
from thalia.diagnostics.diagnostics_snapshot import RecorderSnapshot

LfpMethod = Literal["current", "spike_rate"]


def _build_current_lfp(
    rec: RecorderSnapshot,
    region_name: str,
    T: int,
) -> Optional[np.ndarray]:
    """Attempt to build a current-based LFP proxy for *region_name*.

    Returns ``None`` if any required data (conductances, voltages, reversal
    potentials) is missing or if the conductance sampling interval is too
    coarse (> 2 timesteps — i.e. so far apart that interpolation would
    dominate the signal).
    """
    # Guard: all required arrays must exist.
    if (
        rec._g_exc_samples is None
        or rec._g_inh_samples is None
        or rec._voltages is None
    ):
        return None

    # Guard: conductances must be sampled frequently enough.
    ci = rec.config.conductance_sample_interval_steps
    if ci > 2:
        return None

    n_cond = rec._cond_sample_step
    if n_cond < 10:
        return None

    # Reversal potentials — region-weighted means.
    E_E, E_nmda, E_I, E_GABA_B = extract_region_reversal_potentials(rec, region_name)
    if np.isnan(E_E) or np.isnan(E_I):
        return None

    p_indices = rec._region_pop_indices.get(region_name, [])
    if not p_indices:
        return None

    # Neuron-count weights for population aggregation.
    pop_sizes = np.array([float(rec._pop_sizes[pi]) for pi in p_indices], dtype=np.float64)
    total_neurons = pop_sizes.sum()
    if total_neurons < 1:
        return None
    pop_weights = pop_sizes / total_neurons

    # Determine usable timesteps: min of T and conductance coverage.
    # When ci=1, n_cond ≈ T.  When ci=2, n_cond ≈ T/2.
    n_usable = min(T, n_cond * ci)

    # Build the aggregated synaptic current time series.
    lfp = np.zeros(n_usable, dtype=np.float64)

    for local_idx, pi in enumerate(p_indices):
        w = pop_weights[local_idx]

        # Voltage: mean across sampled neurons per timestep → [n_usable]
        v_pop = np.nanmean(rec._voltages[:n_usable, pi, :], axis=1).astype(np.float64)

        # Conductances: mean across sampled neurons per sample → [n_cond]
        g_exc = np.nanmean(rec._g_exc_samples[:n_cond, pi, :], axis=1).astype(np.float64)
        g_inh = np.nanmean(rec._g_inh_samples[:n_cond, pi, :], axis=1).astype(np.float64)

        g_nmda = (
            np.nanmean(rec._g_nmda_samples[:n_cond, pi, :], axis=1).astype(np.float64)
            if rec._g_nmda_samples is not None
            else np.zeros(n_cond, dtype=np.float64)
        )
        g_gaba_b = (
            np.nanmean(rec._g_gaba_b_samples[:n_cond, pi, :], axis=1).astype(np.float64)
            if rec._g_gaba_b_samples is not None
            else np.zeros(n_cond, dtype=np.float64)
        )

        # Up-sample conductances to full timestep resolution if ci > 1.
        if ci > 1:
            t_cond = np.arange(n_cond) * ci
            t_full = np.arange(n_usable)
            g_exc = np.interp(t_full, t_cond, g_exc)
            g_inh = np.interp(t_full, t_cond, g_inh)
            g_nmda = np.interp(t_full, t_cond, g_nmda)
            g_gaba_b = np.interp(t_full, t_cond, g_gaba_b)
        else:
            # ci == 1: truncate to n_usable
            g_exc = g_exc[:n_usable]
            g_inh = g_inh[:n_usable]
            g_nmda = g_nmda[:n_usable]
            g_gaba_b = g_gaba_b[:n_usable]

        # I_syn = g_exc·(E_E − V) + g_nmda·(E_nmda − V) + g_inh·(E_I − V) + g_gaba_b·(E_GABA_B − V)
        i_syn = g_exc * (E_E - v_pop)
        if not np.isnan(E_nmda):
            i_syn += g_nmda * (E_nmda - v_pop)
        i_syn += g_inh * (E_I - v_pop)
        if not np.isnan(E_GABA_B):
            i_syn += g_gaba_b * (E_GABA_B - v_pop)

        lfp += w * i_syn

    # Zero-centre (remove DC offset) for spectral analysis.
    lfp -= lfp.mean()

    # Pad to full T if conductance coverage was shorter.
    if n_usable < T:
        padded = np.zeros(T, dtype=np.float64)
        padded[:n_usable] = lfp
        return padded

    return lfp


def _build_spike_rate_lfp(
    rec: RecorderSnapshot,
    region_name: str,
    T: int,
    sigma_ms: float = 5.0,
) -> Optional[np.ndarray]:
    """Build a Gaussian-smoothed spike-rate LFP proxy for *region_name*.

    Returns ``None`` if the region has no neurons.
    """
    p_indices = rec._region_pop_indices.get(region_name, [])
    if not p_indices:
        return None

    total_neurons = int(sum(rec._pop_sizes[i] for i in p_indices))
    if total_neurons == 0:
        return None

    raw_counts = rec._pop_spike_counts[:T, p_indices].sum(axis=1).astype(np.float64)
    rate_hz = raw_counts / (total_neurons * rec.dt_ms / 1000.0)
    rate_hz -= rate_hz.mean()
    sigma_steps = sigma_ms / rec.dt_ms
    return sp_gaussian_filter1d(rate_hz, sigma=sigma_steps)


def build_region_lfp_proxy(
    rec: RecorderSnapshot,
    region_name: str,
    T: int,
    sigma_ms: float = 5.0,
) -> Tuple[Optional[np.ndarray], LfpMethod]:
    """Build the best available LFP proxy for *region_name*.

    Attempts a current-based proxy first; falls back to spike-rate smoothing.

    Parameters
    ----------
    rec : RecorderSnapshot
        The recorded diagnostics snapshot.
    region_name : str
        Region to build the proxy for.
    T : int
        Number of recorded timesteps.
    sigma_ms : float
        Gaussian smoothing σ for the spike-rate fallback (ms).

    Returns
    -------
    (signal, method)
        ``signal`` is a 1-D float64 array of length *T* (zero-centred),
        or ``None`` if the region has no data.
        ``method`` is ``"current"`` or ``"spike_rate"``.
    """
    current_lfp = _build_current_lfp(rec, region_name, T)
    if current_lfp is not None:
        return current_lfp, "current"

    spike_lfp = _build_spike_rate_lfp(rec, region_name, T, sigma_ms)
    return spike_lfp, "spike_rate"


def build_all_region_lfp_proxies(
    rec: RecorderSnapshot,
    T: int,
    sigma_ms: float = 5.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, LfpMethod]]:
    """Build LFP proxies for all regions, returning cached dicts.

    Returns
    -------
    (lfp_cache, method_cache)
        ``lfp_cache`` maps region name → 1-D float64 LFP array.
        ``method_cache`` maps region name → ``"current"`` or ``"spike_rate"``.
        Regions with no data are omitted from both dicts.
    """
    lfp_cache: Dict[str, np.ndarray] = {}
    method_cache: Dict[str, LfpMethod] = {}

    for rn in rec._region_keys:
        signal, method = build_region_lfp_proxy(rec, rn, T, sigma_ms)
        if signal is not None:
            lfp_cache[rn] = signal
            method_cache[rn] = method

    return lfp_cache, method_cache
