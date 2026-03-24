"""Connectivity health checks — broken tracts, axonal delays, multi-hop latency."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .diagnostics_report import HealthCategory, HealthIssue
from .health_context import HealthCheckContext


def check_connectivity(ctx: HealthCheckContext) -> None:
    """Check broken tracts, axonal delay accuracy, and reversed connectivity."""
    connectivity, rec, issues = ctx.connectivity, ctx.rec, ctx.issues
    config = ctx.thresholds.connectivity
    # Check broken (non-functional) axonal tracts.
    for ts in connectivity.n_broken:
        issues.append(HealthIssue(severity="critical", category=HealthCategory.CONNECTIVITY,
            region=ts.synapse_id.source_region,
            message=f"BROKEN TRACT: {ts.synapse_id}  "
                    f"transmission_ratio={ts.transmission_ratio * 100:.1f}%  "
                    f"({ts.spikes_sent:,} spikes sent) — downstream region will be starved"))

    # Check axonal delay accuracy and reversed connectivity
    for ts in connectivity.tracts:
        if not np.isnan(ts.measured_delay_ms):
            if ts.measured_delay_ms < 0:
                # Anti-causal cross-correlation peaks are expected in recurrent
                # feedback loops (cortex→LC→cortex, EC→HPC→EC, corticothalamic)
                # where the downstream region fires in anticipation of the next
                # upstream cycle.  These are not genuine wiring bugs.
                issues.append(HealthIssue(severity="info", category=HealthCategory.CONNECTIVITY,
                    region=ts.synapse_id.source_region,
                    message=f"Possible reversed connectivity: {ts.synapse_id}  "
                            f"anti-causal peak at -{ts.measured_delay_ms:.0f} ms — "
                            f"verify source/target are not swapped (may be feedback-loop artifact)"))
            else:
                diff = abs(ts.measured_delay_ms - ts.expected_delay_ms)
                # Floor raised from 5×dt (~0.5 ms) to 15 ms: CC-based delay
                # estimation has ~10–15 ms noise for sparse spike trains
                # (< 5 Hz, < 20 spikes in a 3-second run); a sub-15 ms mismatch
                # is indistinguishable from measurement noise at these rates.
                tolerance = max(config.delay_noise_floor_ms, ts.expected_delay_ms * config.delay_tolerance_fraction)
                if diff > tolerance:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.CONNECTIVITY,
                        region=ts.synapse_id.source_region,
                        message=f"Delay mismatch: {ts.synapse_id}  "
                                f"measured={ts.measured_delay_ms:.0f} ms  "
                                f"expected={ts.expected_delay_ms:.0f} ms  "
                                f"(\u0394{ts.measured_delay_ms - ts.expected_delay_ms:+.0f} ms)"))

            # Jitter check — high jitter indicates unreliable temporal precision.
            if not np.isnan(ts.transmission_jitter_ms) and ts.transmission_jitter_ms > config.jitter_high_ms:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.CONNECTIVITY,
                    region=ts.synapse_id.source_region,
                    message=f"High transmission jitter: {ts.synapse_id}  "
                            f"jitter={ts.transmission_jitter_ms:.1f} ms  "
                            f"(threshold={config.jitter_high_ms:.0f} ms)"))

    # Multi-hop latency check.
    # Build a region-level directed graph from functional tracts that have a valid
    # (non-negative, non-NaN) measured delay.  Then BFS over 2-hop and 3-hop paths
    # and compare the cumulative measured latency against the cumulative expected
    # latency.
    #
    # For regions connected by multiple populations (e.g., multiple projection types)
    # keep the *minimum* measured and expected delays — the fastest pathway governs
    # when downstream spikes can first appear.
    _region_edges: Dict[str, Dict[str, Tuple[float, float]]] = {}  # {src: {tgt: (meas, exp)}}
    for ts in connectivity.tracts:
        if not ts.is_functional or np.isnan(ts.measured_delay_ms) or ts.measured_delay_ms < 0:
            continue
        src_r = ts.synapse_id.source_region
        tgt_r = ts.synapse_id.target_region
        if src_r == tgt_r:
            continue
        meas_d = ts.measured_delay_ms
        exp_d = ts.expected_delay_ms
        prev = _region_edges.get(src_r, {}).get(tgt_r)
        if prev is None or meas_d < prev[0]:
            _region_edges.setdefault(src_r, {})[tgt_r] = (meas_d, exp_d)

    # BFS: enumerate all simple paths of length 2 and 3 through the region graph.
    # Multi-hop tolerance: 90 % of cumulative expected delay, floor 35 ms.
    # CC-based delay estimation is noisy for sparse spike trains and slow-tau
    # populations; multi-hop paths accumulate this noise linearly (~10 ms per
    # hop).  35 ms floor (≈ 3 × single-hop noise) suppresses artifact chains
    # while still catching grossly mis-wired multi-hop paths (which would
    # deviate by > 50 ms from their expected cumulative latency).
    _multihop_tolerance_frac = config.multihop_tolerance_fraction
    _multihop_min_tolerance_ms = config.multihop_min_tolerance_ms
    for src_r, src_edges in _region_edges.items():
        for mid_r, (meas_ab, exp_ab) in src_edges.items():
            if mid_r not in _region_edges:
                continue
            for tgt_r, (meas_bc, exp_bc) in _region_edges[mid_r].items():
                if tgt_r == src_r:
                    continue
                cum_meas_2 = meas_ab + meas_bc
                cum_exp_2 = exp_ab + exp_bc
                tol_2 = max(_multihop_min_tolerance_ms, cum_exp_2 * _multihop_tolerance_frac)
                if abs(cum_meas_2 - cum_exp_2) > tol_2:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.CONNECTIVITY,
                        region=src_r,
                        message=f"Multi-hop delay mismatch (2-hop): "
                                f"{src_r}\u2192{mid_r}\u2192{tgt_r}  "
                                f"cumulative measured={cum_meas_2:.0f} ms  "
                                f"expected={cum_exp_2:.0f} ms  "
                                f"(\u0394{cum_meas_2 - cum_exp_2:+.0f} ms)"))
                # 3-hop extension
                if tgt_r not in _region_edges:
                    continue
                for tgt2_r, (meas_cd, exp_cd) in _region_edges[tgt_r].items():
                    if tgt2_r in (src_r, mid_r):
                        continue
                    cum_meas_3 = cum_meas_2 + meas_cd
                    cum_exp_3 = cum_exp_2 + exp_cd
                    tol_3 = max(_multihop_min_tolerance_ms, cum_exp_3 * _multihop_tolerance_frac)
                    if abs(cum_meas_3 - cum_exp_3) > tol_3:
                        issues.append(HealthIssue(severity="warning", category=HealthCategory.CONNECTIVITY,
                            region=src_r,
                            message=f"Multi-hop delay mismatch (3-hop): "
                                    f"{src_r}\u2192{mid_r}\u2192{tgt_r}\u2192{tgt2_r}  "
                                    f"cumulative measured={cum_meas_3:.0f} ms  "
                                    f"expected={cum_exp_3:.0f} ms  "
                                    f"(\u0394{cum_meas_3 - cum_exp_3:+.0f} ms)"))
