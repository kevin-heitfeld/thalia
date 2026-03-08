"""Thalamic health checks — TRN gating and relay burst mode."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

from .diagnostics_types import (
    HealthCategory,
    HealthIssue,
    OscillatoryStats,
)
from .region_tags import THALAMIC_TAGS, matches_any
from .sensory_patterns import WAKING_PATTERNS

if TYPE_CHECKING:
    from .diagnostics_recorder import DiagnosticsRecorder



def check_trn_relay_gating(
    rec: "DiagnosticsRecorder",
    issues: List[HealthIssue],
) -> None:
    """Check TRN–relay inverse firing-rate relationship in thalamic regions.

    A healthy thalamus under moderate drive shows TRN and relay firing in
    opposite directions: when TRN is active (mean FR ≥ 30 Hz), relay cells
    should be suppressed (gating / burst mode), giving a negative Pearson
    correlation between binned population spike counts.  A positive correlation
    (corr > 0.2) indicates that fast lateral TRN inhibition is absent or
    overwhelmed.

    Binning: 50 ms windows, requiring ≥ 10 bins (≥ 500 ms of recording).

    References: Jones 2007 "The Thalamus"; Pinault & Deschênes 1992.
    """
    T = rec._n_recorded or rec.config.n_timesteps
    bin_steps = max(1, int(50.0 / rec.dt_ms))  # 50 ms per bin
    n_bins = T // bin_steps
    if n_bins < 10:
        return  # need ≥ 500 ms for a meaningful correlation estimate

    for rn, region in rec.brain.regions.items():
        if not matches_any(rn, THALAMIC_TAGS):
            continue
        trn_indices = [
            rec._pop_index[(rn, pn)]
            for pn in region.neuron_populations.keys()
            if "trn" in pn.lower() and (rn, pn) in rec._pop_index
        ]
        relay_indices = [
            rec._pop_index[(rn, pn)]
            for pn in region.neuron_populations.keys()
            if "relay" in pn.lower() and (rn, pn) in rec._pop_index
        ]
        if not trn_indices or not relay_indices:
            continue

        pop_counts = rec._pop_spike_counts[:T]  # [T × P]
        trn_rate = (
            pop_counts[:n_bins * bin_steps][:, trn_indices]
            .reshape(n_bins, bin_steps, -1)
            .sum(axis=(1, 2))
            .astype(np.float64)
        )
        relay_rate = (
            pop_counts[:n_bins * bin_steps][:, relay_indices]
            .reshape(n_bins, bin_steps, -1)
            .sum(axis=(1, 2))
            .astype(np.float64)
        )

        # Mean TRN firing rate in Hz (spikes per second per neuron).
        trn_total_neurons = sum(int(rec._pop_sizes[i]) for i in trn_indices)
        bin_duration_s = bin_steps * rec.dt_ms / 1000.0
        trn_mean_hz = float(trn_rate.mean()) / max(1, trn_total_neurons) / bin_duration_s
        if trn_mean_hz < 30.0:
            continue  # TRN not sufficiently active to exert gating

        if trn_rate.std() < 1e-6 or relay_rate.std() < 1e-6:
            continue  # degenerate constant signal — correlation undefined

        corr = float(np.corrcoef(trn_rate, relay_rate)[0, 1])
        if corr > 0.2:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.THALAMUS,
                region=rn,
                message=(
                    f"TRN–relay gating absent: {rn}  "
                    f"corr(TRN, relay)={corr:.2f}  "
                    f"TRN mean FR ≈ {trn_mean_hz:.0f} Hz  "
                    f"(expected corr < 0 when TRN > 30 Hz; "
                    f"relay should be suppressed — check fast GABA-A from TRN; "
                    f"Jones 2007)"
                ),
            ))


def check_relay_burst_mode(
    oscillations: OscillatoryStats,
    issues: List[HealthIssue],
    sensory_pattern: str = "",
) -> None:
    """Check whether thalamic relay cells show LTS burst mode.

    Uses the short-ISI fraction (ISI < 15 ms) stored in
    :attr:`~.OscillatoryStats.relay_burst_mode`.  A fraction ≥ 5 % indicates
    detectable T-channel low-threshold spike (LTS) burst activity
    (McCormick & Huguenard 1992).  A near-zero fraction in isolation (no
    other burst-mode source) suggests the T-current is absent or blocked.

    Severity depends on the input pattern:
    - Under waking-state stimulation (``_WAKING_STATE_PATTERNS``): burst mode
      ≥ 5 % → **warning** (pathological hyperpolarisation; excessive TRN
      inhibition will block sensory relay).
    - Under background / none / unknown patterns: burst mode is noted as
      ``info`` only (LTS may be expected during non-driven intervals).

    This check only fires when spike time data was recorded (full mode) and
    at least one relay region was found.  It emits an ``info``-level note
    for tonic regions so the diagnostic report always shows the value.
    """
    if not oscillations.relay_burst_mode:
        return

    is_waking = sensory_pattern in WAKING_PATTERNS

    for rn, frac in sorted(oscillations.relay_burst_mode.items()):
        if frac >= 0.05:
            if is_waking:
                issues.append(HealthIssue(
                    severity="warning",
                    category=HealthCategory.THALAMUS,
                    region=rn,
                    message=(
                        f"Thalamic relay in burst mode during '{sensory_pattern}' stimulation: "
                        f"{rn}  short-ISI fraction={frac:.3f}  "
                        f"(≥ 5 % ISIs < 15 ms → T-channel LTS active under waking-state drive; "
                        f"excessive TRN inhibition or insufficient depolarising input — "
                        f"McCormick & Huguenard 1992)"
                    ),
                ))
            else:
                issues.append(HealthIssue(
                    severity="info",
                    category=HealthCategory.THALAMUS,
                    region=rn,
                    message=(
                        f"Relay burst mode (LTS) active: {rn}  "
                        f"short-ISI fraction={frac:.3f}  "
                        f"(≥ 5 % ISIs < 15 ms → T-channel bursting detected; "
                        f"McCormick & Huguenard 1992)"
                    ),
                ))
        else:
            issues.append(HealthIssue(
                severity="info",
                category=HealthCategory.THALAMUS,
                region=rn,
                message=(
                    f"Relay tonic mode: {rn}  "
                    f"short-ISI fraction={frac:.3f}  "
                    f"(< 5 % ISIs < 15 ms → no LTS burst mode detected; "
                    f"normal under sustained drive, unexpected under rest/background)"
                ),
            ))
