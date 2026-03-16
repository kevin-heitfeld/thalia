"""Cortical laminar cascade timing health checks."""

from __future__ import annotations

from typing import List

import numpy as np

from .diagnostics_types import (
    HealthCategory,
    HealthIssue,
    OscillatoryStats,
)


def check_laminar_cascade(
    oscillations: OscillatoryStats,
    issues: List[HealthIssue],
) -> None:
    """Check that cortical laminar cascade follows the expected feedforward order.

    The canonical thalamocortical feedforward relay is L4 → L2/3 → L5/L6
    (Thomson & Bannister 2003; Sakata & Harris 2009).  First-spike latencies
    measured from thalamic relay volleys should therefore satisfy:

        l4_lat_ms < l23_lat_ms  (L4 activates before L2/3)
        l4_lat_ms < l5_lat_ms   (L4 activates before L5)

    Two pathological patterns are flagged:

    * **Reversed L2/3 → L4 order** (l23_lat_ms ≤ l4_lat_ms): L2/3 fires
      before or simultaneously with L4, indicating that top-down feedback or
      horizontal connections are dominating over thalamocortical drive.

    * **Collapsed L4 → L5 cascade** (l5_lat_ms ≤ l4_lat_ms): L5 fires
      before or simultaneously with L4, suggesting runaway recurrent excitation
      or a missing L4 → L2/3 → L5 relay.

    Issues are not raised when latency data is absent (no thalamic volleys detected,
    or layer populations not present).
    """
    for rn, lats in oscillations.laminar_cascade_latencies.items():
        l4  = lats.get("l4_lat_ms",  float("nan"))
        l23 = lats.get("l23_lat_ms", float("nan"))
        l5  = lats.get("l5_lat_ms",  float("nan"))

        if not np.isnan(l23) and not np.isnan(l4) and l23 <= l4:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.LAMINAR,
                region=rn,
                message=(
                    f"Reversed laminar cascade: {rn}  "
                    f"L2/3 latency ({l23:.1f} ms) \u2264 L4 latency ({l4:.1f} ms) \u2014 "
                    f"expected L4 first (thalamocortical feedforward; Thomson & Bannister 2003)"
                ),
            ))

        if not np.isnan(l5) and not np.isnan(l4) and l5 <= l4:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.LAMINAR,
                region=rn,
                message=(
                    f"Collapsed laminar cascade: {rn}  "
                    f"L5 latency ({l5:.1f} ms) \u2264 L4 latency ({l4:.1f} ms) \u2014 "
                    f"expected L4 \u2192 L2/3 \u2192 L5 progression (Sakata & Harris 2009)"
                ),
            ))
