"""Automatic failure triage — isolate CRITICAL-SILENT regions in RegionTestRunner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .diagnostics_types import DiagnosticsReport
from .region_test_runner import RegionTestRunner

if TYPE_CHECKING:
    from thalia.brain import Brain


def run_triage(
    brain: "Brain",
    report: DiagnosticsReport,
    *,
    dt_ms: float | None = None,
    drive_rate_hz: float = 15.0,
) -> None:
    """Isolate every CRITICAL-SILENT region in a :class:`RegionTestRunner` and print results.

    A region is considered CRITICAL-SILENT when at least one of its populations
    has a critical-severity issue whose message begins with ``"SILENT:"``.  Each
    such region is run standalone with Poisson drive on every population so the
    caller can distinguish *intrinsically silent* (still silent in isolation)
    from *upstream-starved* (fires when driven directly).

    Parameters
    ----------
    brain:
        The built brain whose regions should be triaged.
    report:
        The ``DiagnosticsReport`` produced by the preceding recording run.
    dt_ms:
        Simulation timestep for the isolated runner.  Defaults to
        ``brain.dt_ms`` when not specified.
    drive_rate_hz:
        Poisson drive rate applied to every population of each silent region.
        Default 15 Hz matches the original driver behaviour.
    """
    _dt_ms = dt_ms if dt_ms is not None else brain.dt_ms

    # Collect regions with at least one CRITICAL SILENT issue.
    silent_regions: set[str] = set()
    for issue in report.health.all_issues:
        if issue.severity == "critical" and issue.message.startswith("SILENT:") and issue.region:
            silent_regions.add(issue.region)

    print(f"\n{'═' * 80}")
    print("TRIAGE")
    print(f"{'═' * 80}")

    if not silent_regions:
        print("\n  No CRITICAL-SILENT regions found.  Nothing to triage.\n")
        return

    print(
        f"\n  {len(silent_regions)} CRITICAL-SILENT region(s) will be run in isolation "
        f"with {drive_rate_hz:.0f} Hz Poisson drive on every population.\n"
        f"  Interpretation:\n"
        f"    - Still silent in isolation → intrinsic parameter failure (check weights / drive).\n"
        f"    - Fires in isolation         → upstream starvation in the full brain.\n"
    )

    for region_name in sorted(silent_regions):
        print(f"  {'─' * 60}")
        print(f"  Triaging: {region_name}")
        print(f"  {'─' * 60}")
        try:
            runner = RegionTestRunner.from_preset(
                "default", region_name, dt_ms=_dt_ms
            )
            region_obj = brain.regions[region_name]
            for pop_name in region_obj.neuron_populations:
                runner.add_poisson_input(
                    target_population=str(pop_name),
                    rate_hz=drive_rate_hz,
                    n_input=100,
                    connectivity=0.20,
                )
            result = runner.run(duration_ms=1000.0, warmup_ms=500.0)
            result.print()
        except Exception as exc:  # noqa: BLE001
            print(f"    ERROR during triage of '{region_name}': {exc}\n")
