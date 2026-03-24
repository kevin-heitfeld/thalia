"""Tests for structured comparison report (§4.7)."""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from thalia.diagnostics.comparison import compare_reports
from thalia.diagnostics.comparison_report import IssueDiff, MetricDelta
from thalia.diagnostics.comparison_text import format_comparison_text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_report(
    *,
    pop_rates: Dict[str, float] | None = None,
    region_rates: Dict[str, float] | None = None,
    criticals: list[str] | None = None,
    warnings: list[str] | None = None,
    band_power: Dict[str, float] | None = None,
    dominant_freq: float | None = None,
    nm_conc: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Build a minimal diagnostics-report-shaped dict."""
    health: Dict[str, Any] = {"issues_by_severity": {}}
    if criticals:
        health["issues_by_severity"]["critical"] = [
            {"message": m} for m in criticals
        ]
    if warnings:
        health["issues_by_severity"]["warning"] = [
            {"message": m} for m in warnings
        ]
    d: Dict[str, Any] = {"health": health}
    if pop_rates is not None:
        d["population_firing_rates_hz"] = pop_rates
    if region_rates is not None:
        d["region_firing_rates_hz"] = region_rates
    if band_power is not None:
        d["global_band_power"] = band_power
    if dominant_freq is not None:
        d["global_dominant_freq_hz"] = dominant_freq
    if nm_conc is not None:
        d["neuromodulator_peak_conc"] = nm_conc
    return d


# =====================================================================
# MetricDelta
# =====================================================================


class TestMetricDelta:

    def test_delta_both_present(self) -> None:
        m = MetricDelta(key="x", old_value=1.0, new_value=3.0)
        assert m.delta == 2.0

    def test_delta_missing_old(self) -> None:
        m = MetricDelta(key="x", old_value=None, new_value=3.0)
        assert m.delta is None

    def test_transition_ok(self) -> None:
        m = MetricDelta(key="x", old_value=5.0, new_value=6.0, bio_range=(1.0, 10.0))
        assert m.transition == "ok"

    def test_transition_fixed(self) -> None:
        m = MetricDelta(key="x", old_value=0.1, new_value=5.0, bio_range=(1.0, 10.0))
        assert m.transition == "fixed"

    def test_transition_regressed(self) -> None:
        m = MetricDelta(key="x", old_value=5.0, new_value=20.0, bio_range=(1.0, 10.0))
        assert m.transition == "regressed"

    def test_transition_still_oob(self) -> None:
        m = MetricDelta(key="x", old_value=0.1, new_value=0.2, bio_range=(1.0, 10.0))
        assert m.transition == "still_oob"

    def test_transition_new(self) -> None:
        m = MetricDelta(key="x", old_value=None, new_value=5.0)
        assert m.transition == "new"

    def test_transition_removed(self) -> None:
        m = MetricDelta(key="x", old_value=5.0, new_value=None)
        assert m.transition == "removed"

    def test_to_dict_roundtrips_json(self) -> None:
        m = MetricDelta(key="x", old_value=1.0, new_value=2.0, bio_range=(0.5, 3.0))
        d = m.to_dict()
        s = json.dumps(d)
        assert json.loads(s) == d


# =====================================================================
# IssueDiff
# =====================================================================


class TestIssueDiff:

    def test_counts(self) -> None:
        d = IssueDiff(
            severity="critical",
            resolved=["a", "b"],
            added=["c"],
            unchanged=["d"],
        )
        assert d.old_count == 3  # resolved + unchanged
        assert d.new_count == 2  # added + unchanged

    def test_has_new_issues(self) -> None:
        d = IssueDiff(severity="warning", resolved=[], added=["x"], unchanged=[])
        assert d.has_new_issues

    def test_no_new_issues(self) -> None:
        d = IssueDiff(severity="warning", resolved=["y"], added=[], unchanged=[])
        assert not d.has_new_issues


# =====================================================================
# compare_reports
# =====================================================================


class TestCompareReports:

    def test_issue_diffs(self) -> None:
        old = _make_report(criticals=["dead region X", "silent pop Y"], warnings=["w1"])
        new = _make_report(criticals=["silent pop Y", "new problem"], warnings=[])
        r = compare_reports(old, new)
        assert "dead region X" in r.critical_issues.resolved
        assert "new problem" in r.critical_issues.added
        assert "silent pop Y" in r.critical_issues.unchanged
        assert "w1" in r.warnings.resolved

    def test_population_deltas(self) -> None:
        old = _make_report(pop_rates={"r:a": 5.0, "r:b": 10.0})
        new = _make_report(pop_rates={"r:a": 7.0, "r:c": 3.0})
        r = compare_reports(old, new)
        assert r.population_deltas["r:a"].delta == pytest.approx(2.0)
        assert r.population_deltas["r:b"].new_value is None  # removed
        assert r.population_deltas["r:c"].old_value is None  # new

    def test_region_deltas(self) -> None:
        old = _make_report(region_rates={"cortex": 4.0})
        new = _make_report(region_rates={"cortex": 6.0})
        r = compare_reports(old, new)
        assert r.region_deltas["cortex"].delta == pytest.approx(2.0)

    def test_oscillation_deltas(self) -> None:
        old = _make_report(band_power={"theta": 0.3}, dominant_freq=8.0)
        new = _make_report(band_power={"theta": 0.5, "gamma": 0.1}, dominant_freq=10.0)
        r = compare_reports(old, new)
        assert r.oscillation_deltas["theta"].delta == pytest.approx(0.2)
        assert r.oscillation_deltas["gamma"].old_value is None
        assert r.oscillation_deltas["dominant_freq_hz"].delta == pytest.approx(2.0)

    def test_neuromodulator_deltas(self) -> None:
        old = _make_report(nm_conc={"vta/dopamine": 0.5})
        new = _make_report(nm_conc={"vta/dopamine": 0.8})
        r = compare_reports(old, new)
        assert r.neuromodulator_deltas["vta/dopamine"].delta == pytest.approx(0.3)

    def test_has_regressions_critical(self) -> None:
        old = _make_report()
        new = _make_report(criticals=["bad thing"])
        r = compare_reports(old, new)
        assert r.has_regressions

    def test_no_regressions(self) -> None:
        old = _make_report(pop_rates={"r:a": 5.0})
        new = _make_report(pop_rates={"r:a": 6.0})
        r = compare_reports(old, new)
        assert not r.has_regressions

    def test_to_dict_roundtrips_json(self) -> None:
        old = _make_report(pop_rates={"r:a": 5.0}, criticals=["issue1"])
        new = _make_report(pop_rates={"r:a": 7.0}, warnings=["w1"])
        r = compare_reports(old, new)
        d = r.to_dict()
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["has_regressions"] is False
        assert parsed["critical_issues"]["old_count"] == 1
        assert parsed["critical_issues"]["new_count"] == 0

    def test_empty_reports(self) -> None:
        r = compare_reports({}, {})
        assert r.n_ok == 0
        assert r.n_regressed == 0
        assert not r.has_regressions


# =====================================================================
# format_comparison_text
# =====================================================================


class TestFormatText:

    def test_contains_heading(self) -> None:
        old = _make_report(pop_rates={"r:a": 5.0})
        new = _make_report(pop_rates={"r:a": 6.0})
        r = compare_reports(old, new)
        text = format_comparison_text(r, heading="TEST HEADING")
        assert "TEST HEADING" in text

    def test_no_heading(self) -> None:
        r = compare_reports({}, {})
        text = format_comparison_text(r, heading=None)
        assert "═" not in text

    def test_regression_warning(self) -> None:
        old = _make_report(criticals=[])
        new = _make_report(criticals=["new problem"])
        r = compare_reports(old, new)
        text = format_comparison_text(r)
        assert "REGRESSIONS DETECTED" in text

    def test_filtered_only_changed(self) -> None:
        old = _make_report(pop_rates={"r:a": 5.0, "r:b": 5.0})
        new = _make_report(pop_rates={"r:a": 5.01, "r:b": 10.0})
        r = compare_reports(old, new)
        text = format_comparison_text(r, only_changed=True)
        assert "r:b" in text
        # r:a delta is 0.01 < 0.05 threshold, should be filtered
        lines = [l for l in text.split("\n") if "r:a" in l]
        assert len(lines) == 0
