"""Hippocampal health checks — spike-theta PLV, SWR coupling, and HFO validation.

All checks operate on :class:`~.OscillatoryStats` fields populated by
:func:`~.analysis_oscillations.compute_oscillatory_stats`.  No live brain
state is accessed here.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .diagnostics_types import (
    HealthCategory,
    HealthIssue,
    HealthThresholds,
    OscillatoryStats,
    PopulationStats,
)


def check_hippocampal_theta_plv(
    oscillations: OscillatoryStats,
    issues: List[HealthIssue],
    config: HealthThresholds,
) -> None:
    """Check CA1 pyramidal spike–theta phase-locking value (PLV).

    CA1 principal cells in a healthy hippocampal network preferentially fire
    on the descending phase of the theta oscillation (O'Keefe & Recce 1993;
    Buzsáki 2002).  The PLV |mean(exp(j·θ_spike))| quantifies this coupling.

    Expected ranges:
    * CA1 pyramidal: PLV ≈ 0.15–0.40 (healthy theta phase locking)
    * PLV < ``config.plv_theta_crit`` → CRITICAL (coupling effectively absent;
      medial-septum theta drive likely missing)
    * PLV < ``config.plv_theta_warn`` → warning (sub-clinical coupling loss)
    * PLV > ``config.plv_theta_high`` → warning (near-perfect locking is
      seizure-like synchrony)

    When the medial septum GABA reference signal is absent, the region's own
    spike train is used as the phase reference (circular).  Such results are
    flagged with a ``[fallback ref: circular]`` note and should be interpreted
    with caution.

    References: O'Keefe & Recce 1993 *Hippocampus*; Buzsáki 2002 *Neuron*.
    """
    for rn, plv_val in oscillations.plv_theta.items():
        if np.isnan(plv_val):
            continue
        fallback_note = (
            " [fallback ref: circular, PLV inflated — MS GABA signal absent]"
            if oscillations.plv_theta_used_fallback.get(rn, False)
            else ""
        )
        if plv_val < config.plv_theta_crit:
            issues.append(HealthIssue(
                severity="critical",
                category=HealthCategory.OSCILLATIONS,
                region=rn,
                message=(
                    f"CRITICALLY weak spike-theta coupling: {rn}  PLV={plv_val:.3f} "
                    f"(expected 0.15\u20130.40 for CA1 pyramidal) "
                    f"\u2014 medial-septum to hippocampus theta drive may be absent"
                    f"{fallback_note}"
                ),
            ))
        elif plv_val < config.plv_theta_warn:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                region=rn,
                message=(
                    f"Weak spike-theta coupling: {rn}  PLV={plv_val:.3f} "
                    f"(expected 0.15\u20130.40 for CA1 pyramidal) "
                    f"\u2014 check medial-septum to hippocampus theta drive"
                    f"{fallback_note}"
                ),
            ))
        elif plv_val > config.plv_theta_high:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                region=rn,
                message=(
                    f"Excessive spike-theta lock-in: {rn}  PLV={plv_val:.3f} "
                    f"(>{config.plv_theta_high}) \u2014 near-perfect phase locking "
                    f"is seizure-like synchrony{fallback_note}"
                ),
            ))
        elif fallback_note:
            # PLV is in healthy range but the reference was circular — inform.
            issues.append(HealthIssue(
                severity="info",
                category=HealthCategory.OSCILLATIONS,
                region=rn,
                message=f"Spike-theta PLV in range: {rn}  PLV={plv_val:.3f}{fallback_note}",
            ))


def check_swr_hfo_coupling(
    oscillations: OscillatoryStats,
    issues: List[HealthIssue],
) -> None:
    """Validate CA1 HFO as genuine sharp-wave ripples via CA3→CA1 coupling.

    CA1 high-frequency oscillations (HFO, 100–250 Hz) are the hallmark of
    hippocampal sharp-wave ripples (SWRs), but identical-looking HFO power can
    arise from gamma oscillations or noise.  The distinguishing criterion is
    whether there is a corresponding CA3 sharp-wave transient 10–30 ms earlier
    (the physiological CA3→CA1 ripple initiation latency; Buzsáki 2015).

    * HFO present but CA3→CA1 cross-correlation < 0.05 → warning:
      the HFO is not driven by CA3; may be gamma or noise.
    * HFO + adequate CA3→CA1 coupling → info: genuine SWR cascade.

    Reference: Buzsáki 2015 *Neuron*.
    """
    for rn, coupling in oscillations.swr_ca3_ca1_coupling.items():
        xcorr = coupling.get("ca3_ca1_xcorr_peak", float("nan"))
        lag_ms = coupling.get("ca3_ca1_lag_ms", float("nan"))
        hfo_pwr = oscillations.hfo_band_power.get(rn, 0.0)
        if np.isnan(xcorr):
            continue
        if hfo_pwr > 0.02 and xcorr < 0.05:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                region=rn,
                message=(
                    f"CA1 HFO without CA3 drive: {rn}  "
                    f"HFO={hfo_pwr:.3f}  CA3\u2192CA1 xcorr={xcorr:.3f}  "
                    f"(expected xcorr \u2265 0.05 at 10\u201330 ms lag for genuine SWRs; "
                    f"Buzs\u00e1ki 2015) \u2014 CA1 HFO may be noise or gamma oscillation"
                ),
            ))
        elif xcorr >= 0.05 and hfo_pwr > 0.01:
            issues.append(HealthIssue(
                severity="info",
                category=HealthCategory.OSCILLATIONS,
                region=rn,
                message=(
                    f"SWR temporal coupling: {rn}  "
                    f"CA3\u2192CA1 xcorr={xcorr:.3f}  lag={lag_ms:.0f} ms  "
                    f"HFO={hfo_pwr:.3f} \u2014 genuine sharp-wave ripple cascade detected"
                ),
            ))


def check_theta_sequence(
    oscillations: OscillatoryStats,
    issues: List[HealthIssue],
) -> None:
    """Check that CA3 pyramidal activity precedes CA1 at theta-sequence timescales.

    In healthy hippocampal networks, the Schaffer-collateral projection from CA3
    to CA1 drives a consistent 5\u201330 ms feedforward latency within each theta
    cycle, producing compressed temporal sequences of place-cell activity
    (Foster & Wilson 2007; Dragoi & Buzs\u00e1ki 2006).  Any of the following
    indicate disrupted Schaffer-collateral timing:

    * ``xcorr_peak < 0.05`` \u2014 CA3 population fluctuations are not predictive of
      CA1 activity at the expected theta-sequence lags.  Possible causes:
      missing or disconnected CA3\u2192CA1 synapses, or CA3 and CA1 driven by
      independent sources.
    * ``peak_lag_ms \u2264 0`` \u2014 CA1 leads or is simultaneous with CA3; the causal
      feedforward order is reversed or absent.

    Healthy range: ``xcorr_peak \u2265 0.05`` at ``peak_lag_ms \u2208 [5, 30]`` ms.

    Reference: Foster & Wilson 2007 *Nature*; Dragoi & Buzs\u00e1ki 2006 *Cell*.
    """
    for rn, seq in oscillations.ca3_ca1_theta_sequence.items():
        xcorr = seq.get("xcorr_peak", float("nan"))
        lag_ms = seq.get("peak_lag_ms", float("nan"))
        if np.isnan(xcorr):
            continue

        if xcorr < 0.05:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                region=rn,
                message=(
                    f"Weak CA3\u2192CA1 theta-sequence coupling: {rn}  "
                    f"xcorr_peak={xcorr:.3f} (expected \u2265 0.05)  peak_lag={lag_ms:.1f} ms  "
                    f"\u2014 Schaffer-collateral feedforward timing disrupted; "
                    f"CA3 not driving CA1 within theta-cycle windows "
                    f"(Foster & Wilson 2007; Dragoi & Buzs\u00e1ki 2006)"
                ),
            ))
        elif not np.isnan(lag_ms) and lag_ms <= 0.0:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                region=rn,
                message=(
                    f"Reversed CA3\u2192CA1 theta-sequence order: {rn}  "
                    f"xcorr_peak={xcorr:.3f}  peak_lag={lag_ms:.1f} ms (\u2264 0)  "
                    f"\u2014 CA1 leads CA3; expected CA3 to precede CA1 by 5\u201330 ms "
                    f"via Schaffer collaterals (Foster & Wilson 2007)"
                ),
            ))


def check_medial_septum_theta_pacemaker(
    pop_stats: "Dict[Tuple[str, str], PopulationStats]",
    issues: List[HealthIssue],
) -> None:
    """Check that medial-septum populations are firing within their theta-pacemaker range.

    The medial septum (MS) is the primary theta rhythm generator; its GABA and
    cholinergic (ACh) populations must fire at 5–15 Hz to drive hippocampal
    theta oscillations via the septohippocampal pathway (Freund & Buzsáki 1996;
    Buzsáki 2002).  Silent or severely under-firing MS populations reliably
    predict absent or degraded hippocampal theta — the root cause of any
    hippocampal theta PLV failure.

    Thresholds:
    * MS population silent (no spikes) → CRITICAL: theta pacemaker offline.
    * MS population firing < 2 Hz → warning: severely below pacemaker range.

    Reference: Freund & Buzsáki 1996 *Prog Neurobiol*; Buzsáki 2002 *Neuron*.
    """
    for (rn, pn), ps in pop_stats.items():
        if "septum" not in rn.lower() and "medial_septum" not in rn.lower():
            continue
        pop_key = f"{rn}:{pn}"

        if ps.total_spikes == 0:
            issues.append(HealthIssue(
                severity="critical",
                category=HealthCategory.OSCILLATIONS,
                population=pop_key,
                region=rn,
                message=(
                    f"MEDIAL SEPTUM SILENT: {rn}:{pn} — theta pacemaker offline; "
                    f"MS GABA/ACh neurons must fire at 5\u201315 Hz to drive "
                    f"hippocampal theta (Freund & Buzs\u00e1ki 1996)"
                ),
            ))
        elif not np.isnan(ps.mean_fr_hz) and ps.mean_fr_hz < 2.0:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                population=pop_key,
                region=rn,
                message=(
                    f"Medial septum near-silent: {rn}:{pn}  FR={ps.mean_fr_hz:.1f} Hz "
                    f"(expected 5\u201315 Hz) \u2014 insufficient drive for hippocampal "
                    f"theta rhythm; check septohippocampal pathway weights "
                    f"(Freund & Buzs\u00e1ki 1996)"
                ),
            ))
