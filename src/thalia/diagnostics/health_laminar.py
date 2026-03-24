"""Cortical laminar cascade timing and functional role health checks."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .diagnostics_report import HealthCategory, HealthIssue
from .health_context import HealthCheckContext
from .region_tags import (
    CORTICAL_TAGS,
    L4_TAGS,
    L23_TAGS,
    L5_TAGS,
    L6_TAGS,
    PYRAMIDAL_TAGS,
    matches_any,
)


def check_laminar_cascade(ctx: HealthCheckContext) -> None:
    """Check that cortical laminar cascade follows the expected feedforward order.

    The canonical thalamocortical feedforward relay is L4 → L2/3 → L5/L6.
    First-spike latencies measured from thalamic relay volleys should therefore satisfy:

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
    oscillations, issues = ctx.oscillations, ctx.issues
    for rn, lats in oscillations.laminar_cascade_latencies.items():
        l4  = lats.l4_lat_ms
        l23 = lats.l23_lat_ms
        l5  = lats.l5_lat_ms

        if not np.isnan(l23) and not np.isnan(l4) and l23 <= l4:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.LAMINAR,
                region=rn,
                message=(
                    f"Reversed laminar cascade: {rn}  "
                    f"L2/3 latency ({l23:.1f} ms) \u2264 L4 latency ({l4:.1f} ms) \u2014 "
                    f"expected L4 first (thalamocortical feedforward)"
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


# ── Layer tag → friendly name mapping ─────────────────────────────────────────

_LAYER_TAG_MAP: Dict[str, Tuple[str, frozenset[str]]] = {
    "L4":  ("L4",  L4_TAGS),
    "L2/3": ("L2/3", L23_TAGS),
    "L5":  ("L5",  L5_TAGS),
    "L6":  ("L6",  L6_TAGS),
}

# Minimum set of layers required for a canonical cortical column.
_REQUIRED_LAYERS = ("L4", "L2/3", "L5")


def _classify_layer(pop_name: str) -> str | None:
    """Return the canonical layer label for *pop_name*, or ``None``."""
    pn_lower = pop_name.lower()
    for label, (_, tags) in _LAYER_TAG_MAP.items():
        if any(t in pn_lower for t in tags):
            return label
    return None


def check_cortical_column_roles(ctx: HealthCheckContext) -> None:
    """Validate functional roles of cortical layers within each column.

    Goes beyond the cascade-timing check in :func:`check_laminar_cascade` and
    verifies that each cortical region's layers fulfil their canonical
    functional assignments:

    1. **Layer completeness** — canonical cortical columns should contain at
       least L4 (input), L2/3 (integration), and L5 (output) populations.
       Missing layers are flagged as info.

    2. **L4→L2/3 feedforward delay** — the thalamocortical relay reaches L4
       first; L2/3 activation should follow with a ~1–2 ms delay
       (Constantinople & Bruno 2013).  Delays < 0.3 ms suggest near-
       simultaneous drive (bypassing L4) and > 5 ms suggests a weak or absent
       L4→L2/3 projection.

    3. **L5 output layer rate** — L5 pyramidal neurons should carry the
       primary cortical output to subcortical targets.  Their mean firing rate
       should not be dramatically lower than L2/3 pyramidal rates.  If L5 fires
       at < 25 % of L2/3's rate, the column's output drive is abnormally weak.
    """
    issues = ctx.issues
    pop_stats = ctx.pop_stats
    rec = ctx.rec
    oscillations = ctx.oscillations
    config = ctx.thresholds.regional

    # ── Identify cortical regions and their layer populations ─────────────
    # region → {layer_label → [(region, pop_name), ...]}
    region_layers: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    for rn, pn in rec._pop_keys:
        if not matches_any(rn, CORTICAL_TAGS):
            continue
        layer = _classify_layer(pn)
        if layer is None:
            continue
        region_layers.setdefault(rn, {}).setdefault(layer, []).append((rn, pn))

    for rn, layers in region_layers.items():
        # ── 1. Layer completeness ─────────────────────────────────────────
        missing = [lbl for lbl in _REQUIRED_LAYERS if lbl not in layers]
        if missing:
            issues.append(HealthIssue(
                severity="info",
                category=HealthCategory.LAMINAR,
                region=rn,
                message=(
                    f"Incomplete cortical column: {rn}  "
                    f"missing layer(s) {', '.join(missing)} — "
                    f"canonical columns require L4 (input), L2/3 (integration), "
                    f"L5 (output)"
                ),
            ))

        # ── 2. L4→L2/3 feedforward delay ─────────────────────────────────
        lats = oscillations.laminar_cascade_latencies.get(rn)
        if lats is not None:
            l4 = lats.l4_lat_ms
            l23 = lats.l23_lat_ms
            if not np.isnan(l4) and not np.isnan(l23) and l23 > l4:
                delay = l23 - l4
                if delay < config.l4_l23_delay_min_ms:
                    issues.append(HealthIssue(
                        severity="warning",
                        category=HealthCategory.LAMINAR,
                        region=rn,
                        message=(
                            f"L4\u2192L2/3 delay too short: {rn}  "
                            f"{delay:.2f} ms (expected ~1\u20132 ms) — "
                            f"near-simultaneous activation suggests thalamic "
                            f"input bypassing L4"
                        ),
                    ))
                elif delay > config.l4_l23_delay_max_ms:
                    issues.append(HealthIssue(
                        severity="warning",
                        category=HealthCategory.LAMINAR,
                        region=rn,
                        message=(
                            f"L4\u2192L2/3 delay too long: {rn}  "
                            f"{delay:.1f} ms (expected ~1\u20132 ms) — "
                            f"weak or absent L4\u2192L2/3 feedforward projection"
                        ),
                    ))

        # ── 3. L5 output layer firing rate ────────────────────────────────
        # Compare L5 pyramidal firing rate against L2/3 pyramidal rate.
        l5_pyr_rates = _pyramidal_rates(layers.get("L5", []), pop_stats)
        l23_pyr_rates = _pyramidal_rates(layers.get("L2/3", []), pop_stats)

        if l5_pyr_rates and l23_pyr_rates:
            l5_mean = np.mean(l5_pyr_rates)
            l23_mean = np.mean(l23_pyr_rates)
            if l23_mean > 0.0 and l5_mean < config.l5_output_rate_ratio * l23_mean:
                issues.append(HealthIssue(
                    severity="warning",
                    category=HealthCategory.LAMINAR,
                    region=rn,
                    message=(
                        f"Weak L5 output: {rn}  "
                        f"L5 pyramidal rate ({l5_mean:.2f} Hz) < 25% of "
                        f"L2/3 pyramidal rate ({l23_mean:.2f} Hz) — "
                        f"cortical output drive abnormally low "
                        f"(de Kock & Sakmann 2009)"
                    ),
                ))


def _pyramidal_rates(
    pop_keys: List[Tuple[str, str]],
    pop_stats: Dict[Tuple[str, str], object],
) -> List[float]:
    """Return mean firing rates for pyramidal populations in *pop_keys*."""
    rates: List[float] = []
    for key in pop_keys:
        _, pn = key
        if not matches_any(pn, PYRAMIDAL_TAGS):
            continue
        ps = pop_stats.get(key)
        if ps is not None and ps.mean_fr_hz > 0.0:
            rates.append(ps.mean_fr_hz)
    return rates
