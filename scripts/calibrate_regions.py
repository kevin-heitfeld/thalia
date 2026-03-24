"""
Isolated Region Calibration — run problematic regions independently.

Uses RegionTestRunner with preset-extracted inputs to rapidly test
individual regions without building the full 22-region brain.

Usage:
    python scripts/calibrate_regions.py                    # all problematic regions
    python scripts/calibrate_regions.py cortex_sensory     # specific region
    python scripts/calibrate_regions.py --list             # list available regions
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from thalia.diagnostics.region_test_runner import RegionTestRunner

from scripts.tee_writer import TeeWriter


PROBLEMATIC_REGIONS: List[str] = [
    "cortex_sensory",
    "cortex_association",
    "prefrontal_cortex",
    "thalamus_sensory",
    "thalamus_association",
    "thalamus_md",
    "hippocampus",
    "striatum",
]

# Realistic firing rates for source populations that use raw float weights
# (not ConductanceScaledSpec) and would otherwise default to 5 Hz.
RATE_OVERRIDES = {
    "globus_pallidus_interna:principal": 75.0,  # GPi tonic inhibition
    "globus_pallidus_externa:prototypic": 75.0,  # GPe tonic
    "medial_septum:gaba": 8.0,                   # Theta pacemaker
}


def _run_region(region_name: str) -> None:
    """Build a runner from preset, add all preset inputs, run, and print results."""
    print(f"\n{'=' * 70}")
    print(f"  CALIBRATING: {region_name}")
    print(f"{'=' * 70}")

    runner = RegionTestRunner.from_preset("default", region_name)
    runner.add_preset_inputs("default", rate_overrides=RATE_OVERRIDES)

    print(f"  Registered {len(runner._poisson_inputs)} Poisson input sources:")
    for inp in runner._poisson_inputs:
        sid = inp.synapse_id
        print(f"    {sid.source_population:>30s} -> {sid.target_population:<25s}  "
              f"{inp.rate_hz:>5.1f} Hz  n={inp.n_input:>4d}  conn={inp.connectivity:.2f}  "
              f"receptor={sid.receptor_type.name}")

    result = runner.run(duration_ms=3000.0, warmup_ms=2000.0)
    result.print()


def main() -> None:
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            from thalia.brain import BrainBuilder
            builder = BrainBuilder.preset_builder("default")
            for name in sorted(builder._region_specs.keys()):
                print(f"  {name}")
            return

        regions = sys.argv[1:]
    else:
        regions = PROBLEMATIC_REGIONS

    run_stamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    output_dir = Path("data") / "calibration_runs"
    os.makedirs(output_dir, exist_ok=True)

    for region_name in regions:
        output_file_name = os.path.join(output_dir, f"{run_stamp}_{region_name}.txt")
        with open(output_file_name, "w", encoding="utf-8") as output_file:
            TeeWriter.patch_stdout_and_stderr(output_file)
            _run_region(region_name)
            TeeWriter.restore_original_stdout_and_stderr()


if __name__ == "__main__":
    main()
