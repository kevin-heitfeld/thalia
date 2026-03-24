"""Human-readable text formatting for :class:`ComparisonReport`."""

from __future__ import annotations

from typing import Optional

from .comparison_report import ComparisonReport, MetricDelta


def _rate_str(rate: Optional[float]) -> str:
    if rate is None:
        return "  N/A  "
    return f"{rate:7.2f}"


def _delta_str(d: MetricDelta) -> str:
    if d.delta is None:
        return "      "
    sign = "+" if d.delta >= 0 else ""
    return f"{sign}{d.delta:6.2f}"


def _status_icon(d: MetricDelta) -> str:
    ok = d.new_in_range
    if ok is None:
        return " "
    return "✓" if ok else "✗"


def _transition_icon(d: MetricDelta) -> str:
    t = d.transition
    if t == "fixed":
        return "★"
    if t == "regressed":
        return "▼"
    if t == "still_oob":
        return "·"
    return " "


def format_comparison_text(
    report: ComparisonReport,
    *,
    heading: Optional[str] = "COMPARISON WITH PREVIOUS RUN",
    only_changed: bool = False,
    only_oob: bool = False,
) -> str:
    """Render a :class:`ComparisonReport` as a human-readable text table.

    Parameters
    ----------
    report:
        Structured comparison to format.
    heading:
        Optional heading. Pass ``None`` to suppress.
    only_changed:
        Only show populations whose rate changed by ≥0.05 Hz.
    only_oob:
        Only show populations currently out of biological range.
    """
    lines: list[str] = []

    if heading is not None:
        lines.append("")
        lines.append("═" * 100)
        lines.append(heading)
        lines.append("═" * 100)

    # ── Issue summary ────────────────────────────────────────────────────
    ci = report.critical_issues
    wi = report.warnings
    lines.append(f"  Critical issues : {ci.old_count} → {ci.new_count}")
    lines.append(f"  Warnings        : {wi.old_count} → {wi.new_count}")
    if report.brain_state_old != report.brain_state_new:
        lines.append(
            f"  Brain state     : {report.brain_state_old} → {report.brain_state_new}"
        )

    for diff, label in [(ci, "critical"), (wi, "warning")]:
        if diff.resolved:
            lines.append(f"\n  Resolved {label} issues ({len(diff.resolved)}):")
            for msg in diff.resolved:
                lines.append(f"    ★ {msg[:100]}")
        if diff.added:
            lines.append(f"\n  New {label} issues ({len(diff.added)}):")
            for msg in diff.added:
                lines.append(f"    ▼ {msg[:100]}")

    # ── Population firing rate table ─────────────────────────────────────
    lines.append(
        f"\n  {'Population':<50s}  {'Old':>7s}  {'New':>7s}  "
        f"{'Delta':>7s}  {'Range':>12s}  St  Tr"
    )
    lines.append(
        f"  {'─' * 50}  {'─' * 7}  {'─' * 7}  "
        f"{'─' * 7}  {'─' * 12}  ──  ──"
    )

    for pop_key in sorted(report.population_deltas):
        d = report.population_deltas[pop_key]

        if only_changed and d.delta is not None and abs(d.delta) < 0.05:
            continue
        if only_oob and d.new_in_range is True:
            continue

        range_str = ""
        if d.bio_range is not None:
            range_str = f"{d.bio_range[0]:5.1f}–{d.bio_range[1]:.1f}"

        lines.append(
            f"  {pop_key:<50s}  "
            f"{_rate_str(d.old_value)}  "
            f"{_rate_str(d.new_value)}  "
            f"{_delta_str(d)}  "
            f"{range_str:>12s}  "
            f"{_status_icon(d):>2s}  {_transition_icon(d):>2s}"
        )

    lines.append(
        f"\n  Summary: {report.n_ok} OK, {report.n_fixed} fixed (★), "
        f"{report.n_regressed} regressed (▼), {report.n_still_oob} still OOB (·)"
    )

    if report.has_regressions:
        lines.append("  ⚠ REGRESSIONS DETECTED")

    lines.append("")
    return "\n".join(lines)
