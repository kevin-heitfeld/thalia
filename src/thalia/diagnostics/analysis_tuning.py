"""Tuning guidance for out-of-bio-range populations.

Computes an input budget per tract for each population that is firing outside
its biological range, then uses the analytical rate predictor to suggest which
parameter(s) to adjust and by how much.

Works entirely from a :class:`RecorderSnapshot` — no live ``Brain`` needed,
provided the snapshot was captured with weight stats and neuron params
(the default since these fields were added).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from thalia.brain.synapses import STPConfig
from thalia.typing import SynapseId

from .bio_ranges import bio_range
from .diagnostics_types import DiagnosticsReport, PopulationStats, RecorderSnapshot
from .rate_predictor import InputSpec, RatePrediction, predict_rate


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TractContribution:
    """Steady-state conductance budget for one incoming tract."""

    synapse_id_str: str
    source_region: str
    source_population: str
    receptor: str  # ampa, nmda, gaba_a, gaba_b
    n_source: int
    source_rate_hz: float
    weight_mean: float
    connectivity: float  # n_nonzero / n_total
    stp_efficacy: float
    g_ss: float  # steady-state conductance
    fraction: float  # fraction of total same-sign conductance


@dataclass
class TuningGuidance:
    """Tuning advice for a single out-of-range population."""

    region: str
    population: str
    observed_rate_hz: float
    bio_range_hz: Tuple[float, float]
    target_rate_hz: float  # midpoint of bio range

    # Input budget
    excitatory_tracts: List[TractContribution]
    inhibitory_tracts: List[TractContribution]
    g_E_total: float
    g_I_total: float

    # Analytical prediction from current inputs
    prediction: Optional[RatePrediction]

    # Suggestion
    direction: Literal["decrease", "increase"]
    dominant_tract: Optional[str]  # synapse_id_str of largest same-sign contributor
    suggested_weight_scale: Optional[float]


@dataclass
class TuningReport:
    """Tuning guidance for all out-of-range populations."""

    items: List[TuningGuidance] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _receptor_kind(receptor_type_str: str) -> str:
    """Map ReceptorType string to rate_predictor receptor kind."""
    rt = receptor_type_str.lower()
    if rt == "nmda":
        return "nmda"
    if rt in ("gaba_a", "gabaa"):
        return "gaba_a"
    if rt in ("gaba_b", "gabab"):
        return "gaba_b"
    return "ampa"


def _tau_for_receptor(receptor: str, params: Dict[str, float]) -> float:
    """Return decay time constant for a receptor from neuron params."""
    if receptor == "nmda":
        return params.get("tau_nmda", 100.0)
    if receptor == "gaba_b":
        return params.get("tau_GABA_B", 400.0)
    if receptor == "gaba_a":
        return params.get("tau_I", 10.0)
    return params.get("tau_E", 5.0)


def _decay_fraction(dt_ms: float, tau: float) -> float:
    return 1.0 - math.exp(-dt_ms / tau)


def _pop_rate(report: DiagnosticsReport, region: str, pop: str) -> float:
    """Look up observed mean firing rate for a population."""
    for rstat in report.regions.values():
        for pstat in rstat.populations.values():
            if pstat.region_name == region and pstat.population_name == pop:
                return pstat.mean_fr_hz
    return 0.0


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_tuning(
    snapshot: RecorderSnapshot,
    report: DiagnosticsReport,
    *,
    only_critical: bool = False,
) -> TuningReport:
    """Compute tuning guidance for all OOB populations.

    Parameters
    ----------
    snapshot :
        Recorder snapshot with ``_tract_weight_stats`` and ``_pop_neuron_params``.
    report :
        Diagnostics report with population firing rates and health issues.
    only_critical :
        If True, only analyse populations flagged as CRITICAL in health report.

    Returns
    -------
    TuningReport with per-population tuning guidance sorted by severity.
    """
    if not snapshot._tract_weight_stats or not snapshot._pop_neuron_params:
        return TuningReport()

    # Build lookup: (region, pop) → PopulationStats
    pop_stats: Dict[Tuple[str, str], PopulationStats] = {}
    for rstat in report.regions.values():
        for pstat in rstat.populations.values():
            pop_stats[(pstat.region_name, pstat.population_name)] = pstat

    # Build lookup: target (region, pop) → list of incoming tract SynapseIds
    incoming_tracts: Dict[Tuple[str, str], List[SynapseId]] = {}
    for syn_id in snapshot._tract_keys:
        key = (syn_id.target_region, syn_id.target_population)
        incoming_tracts.setdefault(key, []).append(syn_id)

    # Identify OOB populations
    items: List[TuningGuidance] = []

    for (rn, pn), pstat in pop_stats.items():
        br = bio_range(rn, pn)
        if br is None:
            continue

        lo, hi = br
        if lo <= pstat.mean_fr_hz <= hi:
            continue  # in range

        if only_critical:
            # Check if this pop has a critical health issue
            pop_key = f"{rn}:{pn}"
            has_critical = any(
                issue.severity == "critical" and issue.population == pop_key
                for issue in report.health.issues
            )
            if not has_critical:
                continue

        neuron_params = snapshot._pop_neuron_params.get((rn, pn))
        if neuron_params is None:
            continue


        target_rate = (lo + hi) / 2.0
        direction: Literal["decrease", "increase"] = (
            "decrease" if pstat.mean_fr_hz > hi else "increase"
        )

        # Build per-tract contributions
        exc_tracts: List[TractContribution] = []
        inh_tracts: List[TractContribution] = []
        input_specs: List[InputSpec] = []

        for syn_id in incoming_tracts.get((rn, pn), []):
            sid_str = str(syn_id)
            wstats = snapshot._tract_weight_stats.get(sid_str)
            if wstats is None:
                continue

            src_rn = syn_id.source_region
            src_pn = syn_id.source_population
            src_key = (src_rn, src_pn)
            src_stats = pop_stats.get(src_key)
            src_rate = src_stats.mean_fr_hz if src_stats else 0.0
            n_source = src_stats.n_neurons if src_stats else 0

            receptor = _receptor_kind(str(syn_id.receptor_type))
            weight_mean = wstats["mean"]
            n_total = int(wstats["n_total"])
            n_nonzero = int(wstats["n_nonzero"])
            connectivity = n_nonzero / n_total if n_total > 0 else 0.0

            # STP efficacy lookup
            stp_eff = 1.0
            stp_state = snapshot._stp_final_state.get(sid_str)
            if stp_state:
                stp_eff = stp_state.get("efficacy", 1.0)

            # Compute steady-state conductance
            tau = _tau_for_receptor(receptor, neuron_params)
            decay_f = _decay_fraction(snapshot.dt_ms, tau)
            n_eff = n_source * connectivity
            per_step = n_eff * (src_rate / 1000.0) * snapshot.dt_ms * abs(weight_mean) * stp_eff
            g_ss = per_step / decay_f if decay_f > 0 else 0.0

            tc = TractContribution(
                synapse_id_str=sid_str,
                source_region=src_rn,
                source_population=src_pn,
                receptor=receptor,
                n_source=n_source,
                source_rate_hz=src_rate,
                weight_mean=weight_mean,
                connectivity=connectivity,
                stp_efficacy=stp_eff,
                g_ss=g_ss,
                fraction=0.0,  # filled below
            )

            if receptor in ("ampa", "nmda"):
                exc_tracts.append(tc)
            else:
                inh_tracts.append(tc)

            # Build InputSpec for prediction
            stp_cfg = None
            # Find matching STP config if available
            for stp_idx, (_stp_rn, stp_sid) in enumerate(snapshot._stp_keys):
                if stp_sid == syn_id:
                    U, tau_d, tau_f = snapshot._stp_configs[stp_idx]
                    stp_cfg = STPConfig(U=U, tau_d=tau_d, tau_f=tau_f)
                    break

            input_specs.append(InputSpec(
                n=n_source,
                rate_hz=src_rate,
                weight_mean=abs(weight_mean),
                connectivity=connectivity,
                receptor=receptor,
                stp=stp_cfg,
                label=f"{src_rn}:{src_pn}",
            ))

        # Compute fraction of total
        g_E_total = sum(t.g_ss for t in exc_tracts)
        g_I_total = sum(t.g_ss for t in inh_tracts)
        for t in exc_tracts:
            t.fraction = t.g_ss / g_E_total if g_E_total > 0 else 0.0
        for t in inh_tracts:
            t.fraction = t.g_ss / g_I_total if g_I_total > 0 else 0.0

        # Sort by g_ss descending
        exc_tracts.sort(key=lambda t: t.g_ss, reverse=True)
        inh_tracts.sort(key=lambda t: t.g_ss, reverse=True)

        # Run analytical prediction
        prediction = None
        if input_specs:
            prediction = predict_rate(
                g_L=neuron_params["g_L"],
                v_threshold=neuron_params["v_threshold"],
                v_reset=neuron_params["v_reset"],
                E_L=neuron_params["E_L"],
                E_E=neuron_params["E_E"],
                E_I=neuron_params["E_I"],
                E_adapt=neuron_params.get("E_adapt", -0.5),
                tau_E=neuron_params["tau_E"],
                tau_I=neuron_params["tau_I"],
                tau_nmda=neuron_params.get("tau_nmda", 100.0),
                tau_gaba_b=neuron_params.get("tau_GABA_B", 400.0),
                tau_ref=neuron_params["tau_ref"],
                adapt_increment=neuron_params["adapt_increment"],
                tau_adapt=neuron_params["tau_adapt"],
                dt_ms=snapshot.dt_ms,
                inputs=input_specs,
            )

        # Identify dominant tract and suggest scaling
        if direction == "decrease":
            # Too fast — dominant excitatory tract should be scaled down
            dom = exc_tracts[0] if exc_tracts else None
        else:
            # Too slow — dominant inhibitory tract should be scaled down,
            # or dominant excitatory should be scaled up
            if inh_tracts and g_I_total > 0:
                dom = inh_tracts[0]
            elif exc_tracts:
                dom = exc_tracts[0]
            else:
                dom = None

        suggested_scale = None
        if dom and prediction and prediction.rate_hz > 0 and target_rate > 0:
            # Rough heuristic: scale weight to move rate toward target
            # Use ratio of target/observed as first-order approximation
            ratio = target_rate / max(pstat.mean_fr_hz, 0.1)
            if direction == "decrease" and dom in exc_tracts:
                suggested_scale = ratio  # scale down excitation
            elif direction == "increase" and dom in inh_tracts:
                suggested_scale = ratio  # scale down inhibition
            elif direction == "increase" and dom in exc_tracts:
                suggested_scale = 1.0 / ratio if ratio > 0 else None  # scale up excitation

        items.append(TuningGuidance(
            region=rn,
            population=pn,
            observed_rate_hz=pstat.mean_fr_hz,
            bio_range_hz=br,
            target_rate_hz=target_rate,
            excitatory_tracts=exc_tracts,
            inhibitory_tracts=inh_tracts,
            g_E_total=g_E_total,
            g_I_total=g_I_total,
            prediction=prediction,
            direction=direction,
            dominant_tract=dom.synapse_id_str if dom else None,
            suggested_weight_scale=suggested_scale,
        ))

    # Sort: largest absolute deviation from bio range first
    def _deviation(g: TuningGuidance) -> float:
        lo, hi = g.bio_range_hz
        if g.observed_rate_hz > hi:
            return g.observed_rate_hz - hi
        return lo - g.observed_rate_hz

    items.sort(key=_deviation, reverse=True)
    return TuningReport(items=items)


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

def print_tuning_report(tuning: TuningReport) -> None:
    """Print tuning guidance to stdout."""
    if not tuning.items:
        return

    w = 80
    print(f"\n{'═' * w}")
    print("TUNING GUIDANCE — Out-of-Range Populations")
    print(f"{'═' * w}")
    print(f"  {len(tuning.items)} population(s) outside biological range\n")

    for g in tuning.items:
        lo, hi = g.bio_range_hz
        arrow = "▲" if g.direction == "increase" else "▼"
        print(f"  {'─' * (w - 4)}")
        print(f"  {g.region}:{g.population}")
        print(f"    Observed: {g.observed_rate_hz:>7.2f} Hz   "
              f"Bio range: [{lo:.0f}, {hi:.0f}] Hz   "
              f"Target: {g.target_rate_hz:.0f} Hz  {arrow}")

        # Input budget table
        all_tracts = g.excitatory_tracts + g.inhibitory_tracts
        if all_tracts:
            print(f"\n    {'Source':<30} {'Recv':<6} {'Rate':>6} {'w_mean':>9} "
                  f"{'conn':>5} {'STP':>5} {'g_ss':>8} {'%':>5}")
            print(f"    {'-'*30} {'-'*6} {'-'*6} {'-'*9} {'-'*5} {'-'*5} {'-'*8} {'-'*5}")

            for t in all_tracts:
                src = f"{t.source_region}:{t.source_population}"
                if len(src) > 30:
                    src = src[:27] + "..."
                pct = t.fraction * 100
                sign = "E" if t.receptor in ("ampa", "nmda") else "I"
                print(f"    {src:<30} {t.receptor:<6} {t.source_rate_hz:>5.1f}Hz "
                      f"{t.weight_mean:>9.5f} {t.connectivity:>5.2f} "
                      f"{t.stp_efficacy:>5.3f} {t.g_ss:>8.5f} {pct:>4.0f}%{sign}")

            print(f"\n    g_E_total = {g.g_E_total:.5f}   g_I_total = {g.g_I_total:.5f}")

        # Analytical prediction
        if g.prediction:
            p = g.prediction
            print(f"    V_inf = {p.V_inf:.3f}  (threshold = {p.v_threshold})")
            if p.g_adapt > 0:
                print(f"    V_inf (with adapt) = {p.V_inf_with_adapt:.3f}")
            print(f"    Predicted rate = {p.rate_hz:.1f} Hz  ({p.regime})")

        # Suggestion
        if g.dominant_tract and g.suggested_weight_scale is not None:
            # Parse the synapse_id_str for readable output
            dom_label = g.dominant_tract
            action = "Scale weight" if g.direction == "decrease" else "Scale weight"
            print(f"\n    → Suggestion: {action} on [{dom_label}] × {g.suggested_weight_scale:.3f}")
        elif g.direction == "increase" and not g.excitatory_tracts:
            print(f"\n    → No excitatory inputs found — check region connectivity setup")
        elif g.direction == "decrease" and not g.inhibitory_tracts:
            print(f"\n    → No inhibitory inputs found — add inhibition or reduce excitatory weights")

        print()

    print(f"{'═' * w}\n")
