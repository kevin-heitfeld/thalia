"""Shared helpers for diagnostics plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
import numpy as np

from thalia.diagnostics.diagnostics_metrics import PopulationStats
from thalia.diagnostics.diagnostics_report import DiagnosticsReport

matplotlib.use("Agg")


@dataclass
class PlotConfig:
    """Scale factors for diagnostic figure sizes."""

    col_width: float = 4.0
    row_height: float = 3.0
    compact_row_height: float = 2.5
    trace_row_height: float = 1.8
    height_per_pop: float = 0.2
    timeline_width: float = 14.0
    timeline_height: float = 5.0
    max_fig_height: float = 40.0


def get_cmap(name: str, n: int | None = None):
    """Return a Matplotlib colormap using the non-deprecated colormaps registry."""
    cmap = matplotlib.colormaps[name]
    return cmap if n is None else cmap.resampled(n)


def pop_color(ps: PopulationStats) -> str:
    """Return a hex colour string summarising a population's health."""
    if ps.total_spikes == 0:
        return "#e74c3c"
    if ps.bio_plausibility == "ok":
        return "#2ecc71"
    if ps.bio_plausibility in ("low", "high"):
        return "#f39c12"
    return "#95a5a6"


def unhealthiness(ps: PopulationStats) -> float:
    """Score how unhealthy a population is (higher = worse)."""
    score = 0.0
    if ps.bio_range_hz is not None:
        lo, hi = ps.bio_range_hz
        mid = (lo + hi) / 2.0 if (lo + hi) > 0 else 1.0
        if ps.mean_fr_hz < lo:
            score += (lo - ps.mean_fr_hz) / mid
        elif ps.mean_fr_hz > hi:
            score += (ps.mean_fr_hz - hi) / mid
    if ps.total_spikes == 0:
        score += 5.0
    if not np.isnan(ps.pairwise_correlation) and ps.pairwise_correlation > 0.15:
        score += ps.pairwise_correlation * 3.0
    if ps.fraction_burst_events > 0.03:
        score += ps.fraction_burst_events * 10.0
    if not np.isnan(ps.sfa_index) and ps.sfa_index > 3.0:
        score += (ps.sfa_index - 3.0)
    return score


def rank_populations_by_health(report: DiagnosticsReport) -> List[Tuple[str, str]]:
    """Return all (region_name, pop_name) pairs sorted by unhealthiness descending."""
    items: List[Tuple[float, int, str, str]] = []
    for rs in report.regions.values():
        for ps in rs.populations.values():
            items.append((-unhealthiness(ps), -ps.total_spikes, ps.region_name, ps.population_name))
    items.sort()
    return [(rn, pn) for _, _, rn, pn in items]


def rank_regions_by_health(report: DiagnosticsReport) -> List[str]:
    """Return region names sorted by worst population unhealthiness descending."""
    region_scores: Dict[str, float] = {}
    for rs in report.regions.values():
        worst = max((unhealthiness(ps) for ps in rs.populations.values()), default=0.0)
        region_scores[rs.region_name] = worst
    return sorted(region_scores, key=lambda rn: (-region_scores[rn], rn))
