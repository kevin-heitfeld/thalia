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

from thalia.diagnostics.comparison import compare_reports
from thalia.diagnostics.comparison_text import format_comparison_text


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--exit-on-regression", action="store_true",
        help="Exit with code 1 if regressions are detected (for CI)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

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

    report = compare_reports(old_data, new_data)
    print(format_comparison_text(
        report, only_changed=args.only_changed, only_oob=args.only_oob,
    ))

    if args.exit_on_regression and report.has_regressions:
        sys.exit(1)
