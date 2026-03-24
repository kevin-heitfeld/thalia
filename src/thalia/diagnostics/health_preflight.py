"""Pre-flight validation for spiking neural networks.

Checks run BEFORE any simulation that catch configuration errors which would
prevent learning or produce invalid dynamics.  Call :func:`validate_brain`
immediately after :meth:`~thalia.brain.BrainBuilder.build` and fix all
CRITICAL issues before starting a run — otherwise wasted compute is guaranteed.

Checks implemented
------------------
**Neuron parameters** (per population):

1. V_reset < V_threshold — neurons must be able to spike after reset
2. tau_ref ≥ dt — refractory period must span at least one simulation step
3. tau_mem_ms in [5, 200] ms — biologically plausible membrane time constant
4. E_E > 0 — excitatory reversal above rest so AMPA/NMDA are actually depolarising
5. E_I < 0 — inhibitory reversal below rest so GABA_A is hyperpolarising
6. E_GABA_B ≤ E_I — slow metabotropic inhibition must be at least as hyperpolarising
   as fast ionotropic inhibition
7. g_L > 0 — non-positive leak is non-physical (infinite time constant)
8. V_threshold > 0 — threshold must be above resting potential

**Weight initialisation** (per synapse):

9.  All-zero weight matrix → dead synapse, no signal can propagate
10. ≥95 % of weights at w_max → only LTD from step 1
11. Weights outside [w_min, w_max] → clamp will fire immediately

**Axonal tract wiring** (per tract):

12. delay_ms ≥ 0.1 ms — near-zero delay may violate STDP causality
13. Source region/population exists
14. Target region/population exists

**Short-term plasticity** (per STP module):

15. U in (0, 1]
16. tau_d > 0
17. tau_f ≥ 0

**STDP–delay compatibility** (per STDP synapse):

18. tau_plus ≥ axonal delay — if the LTP window closes before the pre-spike
    arrives, LTP is structurally impossible

**Network topology** (global):

19. Regions with no incoming connections — will only fire from noise
20. Regions that are terminal sinks (no outgoing tracts) — informational only
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from thalia.brain.neurons import ConductanceLIF
from thalia.learning.strategies import STDPStrategy

from .diagnostics_report import HealthCategory, HealthIssue

if TYPE_CHECKING:
    from thalia.brain import Brain


# ---------------------------------------------------------------------------
# Biologically plausible parameter ranges (normalised units, E_L = 0)
# ---------------------------------------------------------------------------

# PV/FSI interneurons typically have tau_mem ~5–10 ms; the lowest reported
# values in the literature are ~4 ms (Rudy & McBain 2001).  Using 4.0 ms
# avoids false positives for fast-spiking cells whose heterogeneous
# distributions centre near 5 ms (mean can land at 4.7–5.0 ms).
_TAU_MEM_MIN_MS: float = 4.0
_TAU_MEM_MAX_MS: float = 200.0
_AXONAL_DELAY_MIN_MS: float = 0.1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _issue(
    issues: List[HealthIssue],
    severity: str,
    message: str,
    region: str | None = None,
    population: str | None = None,
) -> None:
    issues.append(
        HealthIssue(
            severity=severity,
            category=HealthCategory.PREFLIGHT,
            message=message,
            region=region,
            population=population,
        )
    )


# ---------------------------------------------------------------------------
# Check 1 – Neuron biophysical parameters
# ---------------------------------------------------------------------------


def _check_neuron_parameters(brain: Brain, issues: List[HealthIssue]) -> None:
    """Per-population biophysical sanity checks on ConductanceLIF neurons."""
    for region_name, region in brain.regions.items():
        for pop_name, neurons in region.neuron_populations.items():
            if not isinstance(neurons, ConductanceLIF):
                continue

            cfg = neurons.config
            pop_label = f"{region_name}:{pop_name}"

            # 1. Per-neuron: v_reset must be strictly below v_threshold.
            #    Both are per-neuron tensors; check element-wise.
            n_violations = int((neurons.v_reset >= neurons.v_threshold).sum().item())
            if n_violations > 0:
                _issue(
                    issues, "critical",
                    f"[{pop_label}] {n_violations}/{neurons.n_neurons} neurons have "
                    f"v_reset >= v_threshold: after a spike the membrane is reset to "
                    f"or above threshold — neuron will fire every step until depleted",
                    region=region_name, population=pop_label,
                )

            # 2. Refractory period must span at least one timestep.
            if cfg.tau_ref < brain.dt_ms:
                _issue(
                    issues, "critical",
                    f"[{pop_label}] tau_ref ({cfg.tau_ref:.2f} ms) < dt "
                    f"({brain.dt_ms:.2f} ms): refractory period is shorter than one "
                    f"simulation step — double-firing within a single step is possible",
                    region=region_name, population=pop_label,
                )

            # 3. Membrane time constant in biological range.
            #    tau_mem_ms may be a scalar float or a per-neuron tensor.
            if isinstance(cfg.tau_mem_ms, (int, float)):
                tau_vals = [float(cfg.tau_mem_ms)]
            else:
                tau_vals = [float(cfg.tau_mem_ms.mean().item())]

            for tau_ms in tau_vals:
                if tau_ms < _TAU_MEM_MIN_MS:
                    _issue(
                        issues, "warning",
                        f"[{pop_label}] tau_mem_ms ({tau_ms:.1f} ms) below biological "
                        f"minimum ({_TAU_MEM_MIN_MS} ms): dynamics are unrealistically "
                        f"fast — STDP time windows will be meaningless",
                        region=region_name, population=pop_label,
                    )
                elif tau_ms > _TAU_MEM_MAX_MS:
                    _issue(
                        issues, "warning",
                        f"[{pop_label}] tau_mem_ms ({tau_ms:.1f} ms) above biological "
                        f"maximum ({_TAU_MEM_MAX_MS} ms): dynamics are unrealistically "
                        f"slow — sparse spiking even under strong drive",
                        region=region_name, population=pop_label,
                    )

            # 4. Excitatory reversal must be above resting potential (E_L = 0).
            if cfg.E_E <= 0.0:
                _issue(
                    issues, "critical",
                    f"[{pop_label}] E_E ({cfg.E_E:.3f}) <= 0 (E_L): AMPA/NMDA drive "
                    f"will not depolarise neurons — excitatory inputs are inhibitory",
                    region=region_name, population=pop_label,
                )

            # 5. Inhibitory reversal must be below resting potential.
            if cfg.E_I >= 0.0:
                _issue(
                    issues, "warning",
                    f"[{pop_label}] E_I ({cfg.E_I:.3f}) >= 0 (E_L): GABA_A reversal "
                    f"at or above rest — inhibition is depolarising (shunting only)",
                    region=region_name, population=pop_label,
                )

            # 6. GABA_B must be at least as hyperpolarising as GABA_A.
            if cfg.E_GABA_B > cfg.E_I:
                _issue(
                    issues, "warning",
                    f"[{pop_label}] E_GABA_B ({cfg.E_GABA_B:.3f}) > E_I "
                    f"({cfg.E_I:.3f}): slow metabotropic inhibition is less "
                    f"hyperpolarising than fast ionotropic — GABA_B provides weaker "
                    f"inhibition than GABA_A, contrary to biology",
                    region=region_name, population=pop_label,
                )

            # 7. Leak conductance must be strictly positive.
            if neurons.g_L.min().item() <= 0.0:
                n_bad = int((neurons.g_L <= 0.0).sum().item())
                _issue(
                    issues, "critical",
                    f"[{pop_label}] {n_bad}/{neurons.n_neurons} neurons have g_L <= 0: "
                    f"zero or negative leak conductance → membrane potential diverges",
                    region=region_name, population=pop_label,
                )

            # 8. Threshold must be above resting potential.
            if neurons.v_threshold.min().item() <= 0.0:
                n_bad = int((neurons.v_threshold <= 0.0).sum().item())
                _issue(
                    issues, "critical",
                    f"[{pop_label}] {n_bad}/{neurons.n_neurons} neurons have "
                    f"v_threshold <= 0 (E_L): threshold at or below rest — spontaneous "
                    f"spiking from noise is certain",
                    region=region_name, population=pop_label,
                )


# ---------------------------------------------------------------------------
# Check 2 – Weight initialisation
# ---------------------------------------------------------------------------


def _check_weight_initialization(brain: Brain, issues: List[HealthIssue]) -> None:
    """Detect pathological weight-matrix starting states."""
    for region_name, region in brain.regions.items():
        w_min = float(getattr(region.config, "w_min", 0.0))
        w_max = float(getattr(region.config, "w_max", 1.0))

        for synapse_id, param in region.synaptic_weights.items():
            w = param.data
            label = str(synapse_id)

            connected_mask = w.abs() > 1e-9
            n_connected = int(connected_mask.sum().item())

            # 9. Entirely silent synapse — no signal, no plasticity.
            if n_connected == 0:
                _issue(
                    issues, "critical",
                    f"All weights are zero for {label}: synapse is dead — no signal "
                    f"can propagate and no learning is possible",
                    region=region_name,
                )
                continue

            w_connected = w[connected_mask]

            # 10. All weights pinned at ceiling.
            frac_at_max = float((w_connected >= w_max * 0.99).float().mean().item())
            if frac_at_max > 0.95:
                _issue(
                    issues, "warning",
                    f"{frac_at_max * 100:.0f}% of weights at w_max ({w_max:.3f}) for "
                    f"{label}: only LTD is possible from step 1 — weights will "
                    f"monotonically decrease until saturated at zero",
                    region=region_name,
                )

            # 11. Weights outside the configured [w_min, w_max] clamp range.
            oob_mask = (w_connected < w_min - 1e-6) | (w_connected > w_max + 1e-6)
            frac_oob = float(oob_mask.float().mean().item())
            if frac_oob > 0.01:
                w_actual_min = float(w_connected.min().item())
                w_actual_max = float(w_connected.max().item())
                _issue(
                    issues, "warning",
                    f"{frac_oob * 100:.1f}% of weights outside [{w_min:.3f}, "
                    f"{w_max:.3f}] for {label} (actual range "
                    f"[{w_actual_min:.4f}, {w_actual_max:.4f}]): weight clamp will "
                    f"immediately truncate a large fraction of the weight distribution",
                    region=region_name,
                )


# ---------------------------------------------------------------------------
# Check 3 – Axonal tract wiring
# ---------------------------------------------------------------------------


def _check_axonal_tracts(brain: Brain, issues: List[HealthIssue]) -> None:
    """Verify every axonal tract references existing regions/populations."""
    for synapse_id, tract in brain.axonal_tracts.items():
        label = str(synapse_id)

        # 12. Delay must be positive enough to be meaningful.
        if tract.spec.delay_ms < _AXONAL_DELAY_MIN_MS:
            _issue(
                issues, "warning",
                f"Axonal delay {tract.spec.delay_ms:.2f} ms for {label} is below "
                f"minimum ({_AXONAL_DELAY_MIN_MS} ms): near-instantaneous transmission "
                f"may violate STDP causality — pre-spike and post-spike appear "
                f"simultaneous",
                region=synapse_id.source_region,
            )

        # 13. Source region/population must exist.
        src_region = brain.get_region_by_name(synapse_id.source_region)
        if src_region is None:
            _issue(
                issues, "critical",
                f"Source region '{synapse_id.source_region}' for tract {label} does "
                f"not exist in the brain",
                region=synapse_id.source_region,
            )
            continue
        if synapse_id.source_population not in src_region.neuron_populations:
            _issue(
                issues, "critical",
                f"Source population '{synapse_id.source_population}' not found in "
                f"region '{synapse_id.source_region}' for tract {label}",
                region=synapse_id.source_region,
            )

        # 14. Target region/population must exist.
        tgt_region = brain.get_region_by_name(synapse_id.target_region)
        if tgt_region is None:
            _issue(
                issues, "critical",
                f"Target region '{synapse_id.target_region}' for tract {label} does "
                f"not exist in the brain",
                region=synapse_id.target_region,
            )
            continue
        if synapse_id.target_population not in tgt_region.neuron_populations:
            _issue(
                issues, "critical",
                f"Target population '{synapse_id.target_population}' not found in "
                f"region '{synapse_id.target_region}' for tract {label}",
                region=synapse_id.target_region,
            )


# ---------------------------------------------------------------------------
# Check 4 – STP parameter validity
# ---------------------------------------------------------------------------


def _check_stp_parameters(brain: Brain, issues: List[HealthIssue]) -> None:
    """Check Tsodyks-Markram STP parameters are physically valid."""
    for region_name, region in brain.regions.items():
        for synapse_id, stp_module in region.stp_modules.items():
            cfg = stp_module.config
            label = str(synapse_id)

            # 15. U must be a valid release probability.
            if not (0.0 < cfg.U <= 1.0):
                _issue(
                    issues, "critical",
                    f"STP U ({cfg.U:.3f}) outside (0, 1] for {label}: release "
                    f"probability must be strictly positive and at most 1.0",
                    region=region_name,
                )

            # 16. Depression recovery time constant must be positive.
            if cfg.tau_d <= 0.0:
                _issue(
                    issues, "critical",
                    f"STP tau_d ({cfg.tau_d:.1f} ms) must be > 0 for {label}",
                    region=region_name,
                )

            # 17. Facilitation time constant must be non-negative.
            if cfg.tau_f < 0.0:
                _issue(
                    issues, "critical",
                    f"STP tau_f ({cfg.tau_f:.1f} ms) must be >= 0 for {label}",
                    region=region_name,
                )


# ---------------------------------------------------------------------------
# Check 5 – STDP window vs axonal delay
# ---------------------------------------------------------------------------


def _check_stdp_vs_delay(brain: Brain, issues: List[HealthIssue]) -> None:
    """Warn when the STDP LTP window closes before the pre-spike arrives.

    STDP's pre-before-post LTP window is ~tau_plus wide.  If the axonal delay
    exceeds tau_plus, the postsynaptic neuron fires (or has decayed its
    eligibility trace to near-zero) before the presynaptic spike arrives at
    the synapse — making causal LTP structurally impossible.
    """
    for region_name, region in brain.regions.items():
        for synapse_id, strategy in region._learning_strategies.items():
            if not isinstance(strategy, STDPStrategy):
                continue

            tau_plus = strategy.config.tau_plus

            tract = brain.axonal_tracts[synapse_id] if synapse_id in brain.axonal_tracts else None
            if tract is None:
                continue

            delay_ms = tract.spec.delay_ms

            if tau_plus < delay_ms:
                _issue(
                    issues, "warning",
                    f"STDP tau_plus ({tau_plus:.1f} ms) < axonal delay "
                    f"({delay_ms:.1f} ms) for {synapse_id}: the LTP pre→post "
                    f"eligibility window closes before the pre-spike reaches the "
                    f"synapse — causal LTP is structurally impossible on this "
                    f"connection",
                    region=region_name,
                )


# ---------------------------------------------------------------------------
# Check 6 – Network topology
# ---------------------------------------------------------------------------


def _check_isolated_regions(brain: Brain, issues: List[HealthIssue]) -> None:
    """Identify regions with no incoming drive and terminal sinks."""
    # Collect all regions that appear as targets of axonal tracts or that
    # receive external inputs (registered as synaptic_weights whose source
    # region differs from the containing region).
    regions_with_input: set[str] = set()

    for synapse_id in brain.axonal_tracts:
        regions_with_input.add(synapse_id.target_region)

    for region_name, region in brain.regions.items():
        for sid in region.synaptic_weights:
            if sid.source_region != region_name:
                regions_with_input.add(region_name)
                break

    # 19. Regions with no incoming connections.
    for region_name, region in brain.regions.items():
        if region_name not in regions_with_input and len(region.neuron_populations) > 0:
            _issue(
                issues, "warning",
                f"Region '{region_name}' has no incoming connections (neither axonal "
                f"tracts nor external inputs): neurons will only fire from intrinsic "
                f"noise or I_h pacemaker currents",
                region=region_name,
            )

    # 20. Regions that are terminal sinks (informational only).
    #     Neuromodulator source regions (VTA, LC, DR, NB, SNc, …) have no
    #     axonal tracts — they broadcast via the NeuromodulatorHub instead.
    #     Flagging them as sinks would be a false positive.
    neuromod_publishers: set[str] = set()
    for channel_key in brain.neuromodulator_hub.registered_channels():
        for src in brain.neuromodulator_hub.source_regions_for_channel(channel_key):
            neuromod_publishers.add(src)

    regions_with_output: set[str] = set()
    for synapse_id in brain.axonal_tracts:
        regions_with_output.add(synapse_id.source_region)

    for region_name in brain.regions:
        if region_name not in regions_with_output and region_name not in neuromod_publishers:
            _issue(
                issues, "info",
                f"Region '{region_name}' has no outgoing axonal tracts: its activity "
                f"is a terminal sink and cannot drive downstream learning",
                region=region_name,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def validate_brain(brain: Brain) -> List[HealthIssue]:
    """Run all pre-flight validation checks on a built brain.

    Call this immediately after :meth:`~thalia.brain.BrainBuilder.build`,
    before starting any simulation loop.  Fix all CRITICAL issues first —
    they indicate configurations that guarantee broken learning or invalid
    dynamics.  WARNINGS indicate suboptimal but potentially functional
    configurations.  INFO items are informational observations.

    Args:
        brain: The constructed :class:`~thalia.brain.Brain` to validate.

    Returns:
        List of :class:`~thalia.diagnostics.HealthIssue` sorted by severity
        (critical → warning → info), then by region name.
    """
    issues: List[HealthIssue] = []

    _check_neuron_parameters(brain, issues)
    _check_weight_initialization(brain, issues)
    _check_axonal_tracts(brain, issues)
    _check_stp_parameters(brain, issues)
    _check_stdp_vs_delay(brain, issues)
    _check_isolated_regions(brain, issues)

    issues.sort(key=lambda i: (
        _SEVERITY_ORDER.get(i.severity, 3),
        i.region or "",
        i.message,
    ))

    return issues
