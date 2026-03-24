"""Compare diagnostics runs — structured comparison with regression detection.

Produces a :class:`ComparisonReport` with per-metric deltas, issue diffs, and
regression flags suitable for CI integration and programmatic analysis.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Set, Tuple

from .comparison_report import ComparisonReport, IssueDiff, MetricDelta

# ── Bio ranges (optional — graceful fallback if unavailable) ─────────────────
try:
    from .bio_ranges import bio_range as _bio_range

    def _get_bio_range(pop_key: str) -> Optional[Tuple[float, float]]:
        parts = pop_key.split(":", 1)
        if len(parts) == 2:
            return _bio_range(parts[0], parts[1])
        return None

except ImportError:

    def _get_bio_range(pop_key: str) -> Optional[Tuple[float, float]]:
        return None


# =============================================================================
# STRUCTURED COMPARISON
# =============================================================================


def _extract_messages(data: Dict[str, Any], severity: str) -> Set[str]:
    """Extract issue message strings from structured health JSON."""
    health = data.get("health", {})
    by_sev = health.get("issues_by_severity", {})
    return {i["message"] for i in by_sev.get(severity, [])}


def _build_issue_diff(
    old_data: Dict[str, Any], new_data: Dict[str, Any], severity: str
) -> IssueDiff:
    old_msgs = _extract_messages(old_data, severity)
    new_msgs = _extract_messages(new_data, severity)
    return IssueDiff(
        severity=severity,
        resolved=sorted(old_msgs - new_msgs),
        added=sorted(new_msgs - old_msgs),
        unchanged=sorted(old_msgs & new_msgs),
    )


def _build_metric_deltas(
    old_data: Dict[str, Any],
    new_data: Dict[str, Any],
    json_key: str,
    *,
    with_bio_range: bool = False,
) -> Dict[str, MetricDelta]:
    """Build MetricDelta dicts from a top-level JSON key holding {name: float}."""
    old_vals: Dict[str, float] = old_data.get(json_key, {})
    new_vals: Dict[str, float] = new_data.get(json_key, {})
    all_keys = sorted(set(old_vals.keys()) | set(new_vals.keys()))

    result: Dict[str, MetricDelta] = {}
    for k in all_keys:
        bio = _get_bio_range(k) if with_bio_range else None
        result[k] = MetricDelta(
            key=k,
            old_value=old_vals.get(k),
            new_value=new_vals.get(k),
            bio_range=bio,
        )
    return result


def compare_reports(
    old_data: Dict[str, Any],
    new_data: Dict[str, Any],
) -> ComparisonReport:
    """Produce a structured :class:`ComparisonReport` from two JSON report dicts.

    Parameters
    ----------
    old_data:
        Parsed JSON from the older ``diagnostics_report.json``.
    new_data:
        Parsed JSON from the newer ``diagnostics_report.json``.

    Returns
    -------
    ComparisonReport
        Structured result with per-metric deltas, issue diffs, and
        regression flags.
    """
    pop_deltas = _build_metric_deltas(
        old_data, new_data, "population_firing_rates_hz", with_bio_range=True,
    )
    region_deltas = _build_metric_deltas(
        old_data, new_data, "region_firing_rates_hz",
    )

    # Global oscillatory band power
    old_band = old_data.get("global_band_power", {})
    new_band = new_data.get("global_band_power", {})
    osc_deltas: Dict[str, MetricDelta] = {}
    for band in sorted(set(old_band.keys()) | set(new_band.keys())):
        osc_deltas[band] = MetricDelta(
            key=band,
            old_value=old_band.get(band),
            new_value=new_band.get(band),
        )
    # Include global dominant frequency
    old_dom = old_data.get("global_dominant_freq_hz")
    new_dom = new_data.get("global_dominant_freq_hz")
    if old_dom is not None or new_dom is not None:
        osc_deltas["dominant_freq_hz"] = MetricDelta(
            key="dominant_freq_hz",
            old_value=old_dom,
            new_value=new_dom,
        )

    # Aperiodic (1/f) exponent per region
    old_aper = old_data.get("region_aperiodic_exponent", {})
    new_aper = new_data.get("region_aperiodic_exponent", {})
    for rn in sorted(set(old_aper.keys()) | set(new_aper.keys())):
        osc_deltas[f"aperiodic_χ:{rn}"] = MetricDelta(
            key=f"aperiodic_χ:{rn}",
            old_value=old_aper.get(rn),
            new_value=new_aper.get(rn),
        )

    # Neuromodulator peak concentrations
    nm_deltas = _build_metric_deltas(
        old_data, new_data, "neuromodulator_peak_conc",
    )

    # Connectivity jitter deltas
    conn_jitter_deltas = _build_metric_deltas(
        old_data, new_data, "connectivity_jitter_ms",
    )

    # Inferred brain state transition
    old_state = old_data.get("inferred_brain_state", "unknown")
    new_state = new_data.get("inferred_brain_state", "unknown")

    # Learning STDP timing deltas
    learning_deltas: Dict[str, MetricDelta] = {}
    old_learn = old_data.get("learning", {})
    new_learn = new_data.get("learning", {})
    old_timing = old_learn.get("stdp_timing", {})
    new_timing = new_learn.get("stdp_timing", {})
    for key in sorted(set(old_timing.keys()) | set(new_timing.keys())):
        old_ltp = old_timing.get(key, {}).get("ltp_fraction")
        new_ltp = new_timing.get(key, {}).get("ltp_fraction")
        learning_deltas[f"stdp_ltp%:{key}"] = MetricDelta(
            key=f"stdp_ltp%:{key}",
            old_value=old_ltp,
            new_value=new_ltp,
        )
    old_da = old_learn.get("da_eligibility_alignment", {})
    new_da = new_learn.get("da_eligibility_alignment", {})
    for key in sorted(set(old_da.keys()) | set(new_da.keys())):
        learning_deltas[f"da_elig:{key}"] = MetricDelta(
            key=f"da_elig:{key}",
            old_value=old_da.get(key),
            new_value=new_da.get(key),
        )

    return ComparisonReport(
        critical_issues=_build_issue_diff(old_data, new_data, "critical"),
        warnings=_build_issue_diff(old_data, new_data, "warning"),
        population_deltas=pop_deltas,
        region_deltas=region_deltas,
        oscillation_deltas=osc_deltas,
        neuromodulator_deltas=nm_deltas,
        connectivity_deltas=conn_jitter_deltas,
        learning_deltas=learning_deltas,
        brain_state_old=old_state,
        brain_state_new=new_state,
    )


# =============================================================================
# RUN FROM FILESYSTEM
# =============================================================================


def run_comparison(
    base_dir: str,
    current_stamp: str,
) -> Optional[ComparisonReport]:
    """Compare the current run with the most recent previous run, if one exists.

    Parameters
    ----------
    base_dir:
        Top-level directory containing timestamped run folders.
    current_stamp:
        The timestamp string of the current run (e.g. ``"2026-03-21T132336"``).

    Returns
    -------
    ComparisonReport or None
        Structured comparison, or ``None`` if no previous run is available.
    """
    stamp_re = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{6}$")
    try:
        entries = sorted(
            (d for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d)) and stamp_re.match(d)),
            reverse=True,
        )
    except OSError:
        return None

    if len(entries) < 2 or entries[0] != current_stamp:
        return None

    new_stamp, old_stamp = entries[0], entries[1]
    old_report = os.path.join(base_dir, old_stamp, "diagnostics_report.json")
    new_report = os.path.join(base_dir, new_stamp, "diagnostics_report.json")

    if not os.path.exists(old_report) or not os.path.exists(new_report):
        return None

    with open(old_report, encoding="utf-8") as f:
        old_data = json.load(f)
    with open(new_report, encoding="utf-8") as f:
        new_data = json.load(f)

    return compare_reports(old_data, new_data)
