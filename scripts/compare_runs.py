"""Compare two diagnostic runs and show what improved, worsened, or stayed the same.

Usage:
    python scripts/compare_runs.py data/diagnostics/report_old.json data/diagnostics/report_new.json
    python scripts/compare_runs.py run03.json run04.json --only-changed
    python scripts/compare_runs.py run03.json run04.json --only-oob
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ── Bio ranges (duplicated subset for standalone use) ────────────────────────
# When running inside the thalia package, we import the canonical ranges.
# When running standalone, we fall back to a minimal built-in table.
try:
    from thalia.diagnostics.bio_ranges import bio_range as _bio_range

    def _get_bio_range(pop_key: str) -> Optional[Tuple[float, float]]:
        parts = pop_key.split(":", 1)
        if len(parts) == 2:
            return _bio_range(parts[0], parts[1])
        return None

except ImportError:
    def _get_bio_range(pop_key: str) -> Optional[Tuple[float, float]]:
        return None


def _in_range(rate: Optional[float], bio: Optional[Tuple[float, float]]) -> Optional[bool]:
    if rate is None or bio is None:
        return None
    return bio[0] <= rate <= bio[1]


def _rate_str(rate: Optional[float]) -> str:
    if rate is None:
        return "  N/A  "
    return f"{rate:7.2f}"


def _delta_str(old: Optional[float], new: Optional[float]) -> str:
    if old is None or new is None:
        return "      "
    d = new - old
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:6.2f}"


def _status_icon(rate: Optional[float], bio: Optional[Tuple[float, float]]) -> str:
    ok = _in_range(rate, bio)
    if ok is None:
        return " "
    return "✓" if ok else "✗"


def _transition_icon(
    old_rate: Optional[float],
    new_rate: Optional[float],
    bio: Optional[Tuple[float, float]],
) -> str:
    old_ok = _in_range(old_rate, bio)
    new_ok = _in_range(new_rate, bio)
    if old_ok is None or new_ok is None:
        return " "
    if not old_ok and new_ok:
        return "★"  # fixed
    if old_ok and not new_ok:
        return "▼"  # regressed
    if not old_ok and not new_ok:
        return "·"  # still broken
    return " "  # was OK, still OK


def compare(
    old: Dict[str, Any],
    new: Dict[str, Any],
    *,
    only_changed: bool = False,
    only_oob: bool = False,
) -> None:
    old_rates: Dict[str, float] = old.get("population_firing_rates_hz", {})
    new_rates: Dict[str, float] = new.get("population_firing_rates_hz", {})

    all_pops = sorted(set(old_rates.keys()) | set(new_rates.keys()))

    # ── Header ───────────────────────────────────────────────────────────
    print()
    print("═" * 110)
    print("DIAGNOSTIC RUN COMPARISON")
    print("═" * 110)

    old_score = old.get("stability_score")
    new_score = new.get("stability_score")
    old_crit = len(old.get("critical_issues", []))
    new_crit = len(new.get("critical_issues", []))
    old_warn = len(old.get("warnings", []))
    new_warn = len(new.get("warnings", []))

    if old_score is not None and new_score is not None:
        print(f"  Stability score : {old_score:.3f} → {new_score:.3f}  "
            f"({'improved' if new_score > old_score else 'worsened' if new_score < old_score else 'unchanged'})")
    else:
        print(f"  Stability score : {old_score} → {new_score}")
    print(f"  Critical issues : {old_crit} → {new_crit}")
    print(f"  Warnings        : {old_warn} → {new_warn}")

    # ── Resolved / new critical issues ───────────────────────────────────
    old_crits = set(old.get("critical_issues", []))
    new_crits = set(new.get("critical_issues", []))
    resolved = old_crits - new_crits
    added = new_crits - old_crits
    if resolved:
        print(f"\n  Resolved critical issues ({len(resolved)}):")
        for issue in sorted(resolved):
            print(f"    ★ {issue[:100]}")
    if added:
        print(f"\n  New critical issues ({len(added)}):")
        for issue in sorted(added):
            print(f"    ▼ {issue[:100]}")

    # ── Population table ─────────────────────────────────────────────────
    print(f"\n  {'Population':<50s}  {'Old':>7s}  {'New':>7s}  {'Delta':>7s}  {'Range':>12s}  St  Tr")
    print(f"  {'─' * 50}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 12}  ──  ──")

    n_fixed = 0
    n_regressed = 0
    n_still_oob = 0
    n_ok = 0

    for pop in all_pops:
        old_r = old_rates.get(pop)
        new_r = new_rates.get(pop)
        bio = _get_bio_range(pop)

        # Filter flags
        if only_changed and old_r is not None and new_r is not None:
            if abs(old_r - new_r) < 0.05:
                continue
        if only_oob:
            if _in_range(new_r, bio) is True:
                continue

        st = _status_icon(new_r, bio)
        tr = _transition_icon(old_r, new_r, bio)

        range_str = ""
        if bio is not None:
            range_str = f"{bio[0]:5.1f}–{bio[1]:.1f}"

        print(
            f"  {pop:<50s}  "
            f"{_rate_str(old_r)}  "
            f"{_rate_str(new_r)}  "
            f"{_delta_str(old_r, new_r)}  "
            f"{range_str:>12s}  "
            f"{st:>2s}  {tr:>2s}"
        )

        # Count transitions
        old_ok = _in_range(old_r, bio)
        new_ok = _in_range(new_r, bio)
        if old_ok is False and new_ok is True:
            n_fixed += 1
        elif old_ok is True and new_ok is False:
            n_regressed += 1
        elif old_ok is False and new_ok is False:
            n_still_oob += 1
        elif new_ok is True:
            n_ok += 1

    print(f"\n  Summary: {n_ok} OK, {n_fixed} fixed (★), {n_regressed} regressed (▼), {n_still_oob} still OOB (·)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two Thalia diagnostic JSON reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("old", help="Path to the older diagnostics_report.json")
    parser.add_argument("new", help="Path to the newer diagnostics_report.json")
    parser.add_argument(
        "--only-changed", action="store_true",
        help="Only show populations whose rate changed by ≥0.05 Hz",
    )
    parser.add_argument(
        "--only-oob", action="store_true",
        help="Only show populations that are currently out of biological range",
    )
    args = parser.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)

    if not old_path.exists():
        print(f"Error: {old_path} not found", file=sys.stderr)
        sys.exit(1)
    if not new_path.exists():
        print(f"Error: {new_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(old_path, encoding="utf-8") as f:
        old_data = json.load(f)
    with open(new_path, encoding="utf-8") as f:
        new_data = json.load(f)

    compare(old_data, new_data, only_changed=args.only_changed, only_oob=args.only_oob)
