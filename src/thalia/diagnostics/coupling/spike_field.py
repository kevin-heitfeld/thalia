"""General spike-field coherence — PLV and PPC across all regions and EEG bands.

Computes phase-locking value (PLV) and pairwise phase consistency
(PPC; Vinck et al. 2010) for every population against its region's LFP proxy,
using the region's dominant EEG band where defined.

The LFP proxy for a region is the total spike count across all populations in
that region, band-passed to the target frequency range.  This is standard
practice in computational neuroscience when no extracellular potential model
is available (Mazzoni et al. 2015).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter as sp_butter
from scipy.signal import filtfilt as sp_filtfilt
from scipy.signal import hilbert as sp_hilbert

from thalia.diagnostics.bio_firing_ranges import expected_dominant_band
from thalia.diagnostics.bio_spectral_specs import EEG_BANDS
from thalia.diagnostics.coupling.lfp_proxy import build_region_lfp_proxy
from thalia.diagnostics.diagnostics_metrics import SpikeFieldResult
from thalia.diagnostics.diagnostics_snapshot import RecorderSnapshot

# Minimum simulation duration (ms) for meaningful phase estimation.
_MIN_DURATION_MS: float = 500.0
# Minimum spike count to attempt PLV compute (avoids noise-dominated estimates).
_MIN_SPIKES: int = 20


def _compute_ppc(plv: float, n: int) -> float:
    """Pairwise phase consistency (Vinck et al. 2010): unbiased PLV² estimator."""
    if n < 2:
        return float("nan")
    return (plv * plv * n - 1.0) / (n - 1.0)


def _pick_band(
    region: str,
    region_band_power: Optional[Dict[str, Dict[str, float]]],
) -> Optional[str]:
    """Choose the EEG band for SFC analysis of *region*.

    Uses the region's expected dominant band from ``RegionSpec`` when defined.
    Falls back to the band with highest relative power from the already-computed
    spectral analysis.  Returns ``None`` if no band can be determined.
    """
    dom = expected_dominant_band(region)
    if dom is not None:
        return dom
    # Fallback: highest relative power band from pre-computed results.
    if region_band_power is not None and region in region_band_power:
        bp = region_band_power[region]
        if bp:
            return max(bp, key=bp.get)  # type: ignore[arg-type]
    return None


def compute_spike_field_coherence(
    rec: RecorderSnapshot,
    T: int,
    region_band_power: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[Tuple[str, str], SpikeFieldResult]:
    """Compute spike-field PLV/PPC for each population against its region's LFP proxy.

    For each region, the LFP proxy is the total spike count across all populations,
    band-passed to the target EEG band.  For each population, spike times are
    extracted and their phases relative to the band-passed LFP proxy are used to
    compute PLV and PPC.

    Parameters
    ----------
    rec : RecorderSnapshot
        The recorded diagnostics snapshot.
    T : int
        Number of recorded timesteps.
    region_band_power : dict, optional
        Pre-computed ``region_band_power`` (from spectral analysis) used as a
        fallback when a region has no expected dominant band.

    Returns
    -------
    dict
        Mapping ``(region_name, population_name)`` → :class:`SpikeFieldResult`.
    """
    results: Dict[Tuple[str, str], SpikeFieldResult] = {}

    if T * rec.dt_ms < _MIN_DURATION_MS:
        return results

    fs = 1000.0 / rec.dt_ms
    nyq = fs / 2.0

    for rn in rec._region_keys:
        band_name = _pick_band(rn, region_band_power)
        if band_name is None or band_name not in EEG_BANDS:
            continue

        f_lo, f_hi = EEG_BANDS[band_name]
        if nyq <= f_hi:
            continue

        # ── Build region LFP proxy (current-based preferred) ──────────
        pop_indices = rec._region_pop_indices[rn]
        if not pop_indices:
            continue
        lfp_proxy, _method = build_region_lfp_proxy(rec, rn, T)
        if lfp_proxy is None:
            continue
        if np.abs(lfp_proxy).sum() < 1e-12:
            continue

        # ── Band-pass filter and extract analytic phase ────────────────
        try:
            b, a = sp_butter(4, [f_lo / nyq, f_hi / nyq], btype="bandpass")
            phase_signal = np.angle(sp_hilbert(sp_filtfilt(b, a, lfp_proxy)))
        except ValueError:
            continue

        # ── Compute PLV/PPC per population ─────────────────────────────
        for pi in pop_indices:
            rn_pop, pn_pop = rec._pop_keys[pi]
            key = (rn_pop, pn_pop)

            # Gather spike timesteps for this population.
            spike_phases: List[float] = []
            if key in rec._spike_times:
                for neuron_times in rec._spike_times[key]:
                    for t_sp in neuron_times:
                        if 0 <= t_sp < T:
                            spike_phases.append(phase_signal[t_sp])

            n = len(spike_phases)
            if n < _MIN_SPIKES:
                continue

            phases_arr = np.array(spike_phases)
            plv = float(np.abs(np.mean(np.exp(1j * phases_arr))))
            ppc = _compute_ppc(plv, n)
            results[key] = SpikeFieldResult(plv=plv, ppc=ppc, n_spikes=n, band=band_name)

    return results
