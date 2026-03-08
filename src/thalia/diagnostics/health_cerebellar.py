"""Cerebellar health checks — Purkinje-DCN coupling, IO synchrony, and mossy-fibre drive."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .bio_ranges import bio_range
from .diagnostics_types import (
    HealthCategory,
    HealthIssue,
    OscillatoryStats,
    PopulationStats,
)
from .region_tags import CEREBELLAR_TAGS, matches_any


def check_cerebellar_coupling(
    oscillations: OscillatoryStats,
    issues: List[HealthIssue],
) -> None:
    """Check Purkinje-DCN anti-correlation and IO gap-junction pairwise synchrony.

    Two independent cerebellar-circuit checks:

    * **Purkinje–DCN correlation**: The deep cerebellar nuclei (DCN) rebound
      during pauses in Purkinje cell inhibition, producing an anti-phase
      population-rate relationship (Heck et al. 2007).  A positive correlation
      (corr > 0.1) indicates that Purkinje inhibition is not suppressing DCN.

    * **IO pairwise synchrony**: Inferior olive neurons are electrically coupled
      via gap junctions and co-fire even at low rates (Llinás & Yarom 1981).
      Expected pairwise ρ > 0.3; low synchrony indicates absent or broken
      gap-junction coupling.

    References: Heck et al. 2007 *J Neurophysiol*; Llinás & Yarom 1981 *J Physiol*.
    """
    for rn_cb2, corr_pd in oscillations.purkinje_dcn_corr.items():
        if not np.isnan(corr_pd) and corr_pd > 0.1:
            issues.append(HealthIssue(severity="warning", category=HealthCategory.CEREBELLAR, region=rn_cb2,
                message=f"Purkinje-DCN not anti-correlated: {rn_cb2}  corr={corr_pd:.2f} "
                        f"(expected < 0; DCN should rebound during Purkinje silence)"))

    for rn_io2, corr_io2 in oscillations.io_pairwise_corr.items():
        if not np.isnan(corr_io2) and corr_io2 < 0.1:
            issues.append(HealthIssue(severity="warning", category=HealthCategory.CEREBELLAR, region=rn_io2,
                message=f"IO pairwise synchrony weak: {rn_io2}  corr={corr_io2:.2f} "
                        f"(expected >0.3; gap-junction-coupled IO neurons should co-fire)"))


def check_inferior_olive(
    pop_stats: Dict[Tuple[str, str], PopulationStats],
    issues: List[HealthIssue],
) -> None:
    """Check that every inferior-olive population is firing in its biological range.

    The inferior olive (IO) is the sole source of climbing-fibre error signals
    that drive Purkinje cell long-term depression and cerebellar motor learning.
    Normal IO discharge: **0.3–3 Hz** (complex spikes; Llinás & Yarom 1981;
    Marr–Albus–Ito theory).

    Two failure modes are flagged:

    * **Silent IO** (FR = 0 Hz, or below 0.3 Hz threshold) → CRITICAL.
      Without climbing-fibre input the cerebellar circuit cannot update its
      internal model regardless of how healthy its intrinsic dynamics appear.
    * **Hyperactive IO** (FR > 3 Hz) → warning.
      Sustained rates above 3 Hz are not physiologically supported by the
      IO pacemaker and indicate runaway excitation or missing inhibitory input.
    """
    _io_range = bio_range("", "inferior_olive") or (0.3, 3.0)
    _IO_LO, _IO_HI = _io_range

    for (rn, pn), ps in pop_stats.items():
        if pn != "inferior_olive":
            continue
        fr = ps.mean_fr_hz
        if np.isnan(fr) or fr < _IO_LO:
            issues.append(HealthIssue(
                severity="critical",
                category=HealthCategory.CEREBELLAR,
                population=f"{rn}:{pn}",
                region=rn,
                message=(
                    f"INFERIOR OLIVE SILENT: {rn}:{pn}  FR={fr:.2f} Hz "
                    f"(expected {_IO_LO}\u2013{_IO_HI} Hz)  "
                    f"\u2014 no climbing-fibre error signal; cerebellar learning disabled "
                    f"(Llin\u00e1s & Yarom 1981; Marr\u2013Albus\u2013Ito theory)"
                ),
            ))
        elif fr > _IO_HI:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.CEREBELLAR,
                population=f"{rn}:{pn}",
                region=rn,
                message=(
                    f"Inferior olive hyperactive: {rn}:{pn}  FR={fr:.2f} Hz "
                    f"> {_IO_HI} Hz  "
                    f"\u2014 IO pacemaker does not sustain rates above 3 Hz; "
                    f"check excitatory drive or missing GABAergic inhibition"
                ),
            ))


def check_mossy_fibre_granule_drive(
    pop_stats: Dict[Tuple[str, str], PopulationStats],
    issues: List[HealthIssue],
) -> None:
    """Check that mossy-fibre input drives granule cells in every cerebellar region.

    Granule cells are the obligatory relay between mossy-fibre afferents and
    Purkinje cells via parallel fibres.  If granule cells are silent while
    Purkinje cells are actively discharging, the Purkinje activity must be
    spontaneous (or driven by climbing fibres alone) rather than driven by
    the mossy-fibre pathway.  This indicates a missing or broken
    mossy-fibre → granule-cell projection (Eccles et al. 1967; Apps &
    Garwicz 2005).

    For each cerebellar region that has both a ``granule`` population and at
    least one ``purkinje`` population:

    * **Granule cells silent** (mean FR < lower bio-range bound, 0.1 Hz) **AND**
      **Purkinje cells active** (mean FR ≥ 1 Hz) →
      CRITICAL: "Mossy-fibre → granule coupling absent; Purkinje firing is
      spontaneous, not driven by mossy-fibre input."
    * **Granule cells silent** with no Purkinje data available →
      warning: granule cells are not receiving mossy-fibre drive.

    The 0.1 Hz lower bound is taken from ``bio_range(rn, "granule")``.
    Purkinje "in-range" threshold is 1 Hz — well below the normal 40–100 Hz
    tonic range so the check fires even if Purkinje spiking is reduced.

    References: Eccles et al. 1967 *The Cerebellum as a Neuronal Machine*;
    Apps & Garwicz 2005 *Nat Rev Neurosci*.
    """
    # Build per-region lookup: granule FR and mean Purkinje FR.
    granule_fr: Dict[str, float] = {}
    purkinje_fr: Dict[str, float] = {}

    for (rn, pn), ps in pop_stats.items():
        if not matches_any(rn, CEREBELLAR_TAGS):
            continue
        fr = ps.mean_fr_hz
        if np.isnan(fr):
            continue
        if "granule" in pn.lower():
            # Use the lowest granule FR if multiple sub-populations exist.
            if rn not in granule_fr or fr < granule_fr[rn]:
                granule_fr[rn] = fr
        elif "purkinje" in pn.lower():
            # Accumulate for mean across multiple Purkinje sub-populations.
            if rn not in purkinje_fr:
                purkinje_fr[rn] = fr
            else:
                purkinje_fr[rn] = (purkinje_fr[rn] + fr) / 2.0

    for rn, g_fr in granule_fr.items():
        spec_range = bio_range(rn, "granule")
        granule_lo = spec_range[0] if spec_range else 0.1

        if g_fr >= granule_lo:
            continue  # Granule cells are active — mossy-fibre drive present.

        pkj_fr = purkinje_fr.get(rn)

        if pkj_fr is not None and pkj_fr >= 1.0:
            issues.append(HealthIssue(
                severity="critical",
                category=HealthCategory.CEREBELLAR,
                region=rn,
                message=(
                    f"MOSSY-FIBRE DRIVE ABSENT: {rn}  "
                    f"granule FR={g_fr:.3f} Hz (< {granule_lo} Hz threshold)  "
                    f"Purkinje FR={pkj_fr:.1f} Hz  "
                    f"\u2014 Purkinje firing is spontaneous, not driven by mossy-fibre "
                    f"input via parallel fibres; check mossy-fibre \u2192 granule "
                    f"synaptic weights and connectivity "
                    f"(Apps & Garwicz 2005; Eccles et al. 1967)"
                ),
            ))
        else:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.CEREBELLAR,
                region=rn,
                message=(
                    f"Granule cells silent: {rn}  "
                    f"granule FR={g_fr:.3f} Hz (< {granule_lo} Hz threshold)  "
                    f"\u2014 mossy-fibre \u2192 granule drive absent; "
                    f"parallel-fibre EPSPs on Purkinje cells will be missing "
                    f"(Apps & Garwicz 2005)"
                ),
            ))
