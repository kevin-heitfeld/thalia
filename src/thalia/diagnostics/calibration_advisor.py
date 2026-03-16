"""Calibration advisor: actionable parameter recommendations for OOB populations.

Extends the tuning guidance system with concrete, code-level recommendations
for fixing out-of-range firing rates.  Instead of just "scale weight × 0.85",
this module tells you:

- **What kind of issue** it is (intrinsic pacemaker, synaptic drive, adaptation,
  noise-driven regime, structural/missing connection)
- **Which parameter to change** with the current value and a calculated target
- **Where in the code** to make the change (source file path)
- **Why** this parameter matters (brief mechanism explanation)

Usage::

    from thalia.diagnostics.calibration_advisor import (
        compute_calibration_advice,
        print_calibration_advice,
    )

    advice = compute_calibration_advice(snapshot, report)
    print_calibration_advice(advice)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from .analysis_tuning import TuningGuidance, TuningReport
from .bio_ranges import bio_range
from .diagnostics_types import DiagnosticsReport, RecorderSnapshot


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

IssueKind = Literal[
    "intrinsic_pacemaker",   # Rate set by I_h / SK / noise (NE, 5-HT, ACh, DA)
    "adaptation",            # SFA or SK over/under-damping the rate
    "synaptic_drive",        # Rate determined by incoming synaptic weights
    "noise_driven",          # Subthreshold V_inf; rate from noise fluctuations
    "structural",            # Missing connections or populations
    "internal_drive",        # Region-internal drive mechanism (e.g. LC GABA scale)
]


@dataclass
class ParameterRecommendation:
    """A single recommended parameter change."""

    param_name: str
    """Parameter name as it appears in code (e.g. 'sk_conductance')."""

    current_value: float
    """Current value of the parameter."""

    recommended_value: float
    """Calculated target value."""

    source_file: str
    """Relative path to the source file where this parameter is defined."""

    mechanism: str
    """Brief explanation of how this parameter affects firing rate."""

    confidence: Literal["high", "medium", "low"]
    """Confidence in the recommendation based on the analytical model."""


@dataclass
class CalibrationAdvice:
    """Actionable calibration advice for one OOB population."""

    region: str
    population: str
    observed_rate_hz: float
    bio_range_hz: Tuple[float, float]
    target_rate_hz: float
    direction: Literal["decrease", "increase"]
    issue_kind: IssueKind
    issue_explanation: str
    """Human-readable explanation of why this population is out of range."""

    recommendations: List[ParameterRecommendation]
    """Ordered list of parameter changes to try (most impactful first)."""


@dataclass
class CalibrationReport:
    """Complete calibration advice for all OOB populations."""

    items: List[CalibrationAdvice] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pacemaker neuron classification
# ---------------------------------------------------------------------------

# Config class names that indicate pacemaker neuron types
_PACEMAKER_CONFIG_TYPES = {
    "NorepinephrineNeuronConfig",
    "SerotoninNeuronConfig",
    "AcetylcholineNeuronConfig",
}

# Region:population patterns for dopamine pacemakers (they use ConductanceLIFConfig
# but are constructed with pacemaker parameters in their region __init__)
_DA_PACEMAKER_POPS = {
    ("vta", "da_mesolimbic"),
    ("vta", "da_mesocortical"),
    ("substantia_nigra_compacta", "da"),
}

# Source file mapping for specialized neuron configs
_CONFIG_SOURCE_FILES = {
    "NorepinephrineNeuronConfig": "src/thalia/brain/neurons/norepinephrine_neuron.py",
    "SerotoninNeuronConfig": "src/thalia/brain/neurons/serotonin_neuron.py",
    "AcetylcholineNeuronConfig": "src/thalia/brain/neurons/acetylcholine_neuron.py",
}

# Region source file mapping (for region-level parameter overrides)
_REGION_SOURCE_FILES: Dict[str, str] = {
    "locus_coeruleus": "src/thalia/brain/regions/locus_coeruleus.py",
    "dorsal_raphe": "src/thalia/brain/regions/dorsal_raphe.py",
    "nucleus_basalis": "src/thalia/brain/regions/nucleus_basalis.py",
    "vta": "src/thalia/brain/regions/vta.py",
    "substantia_nigra_compacta": "src/thalia/brain/regions/substantia_nigra_compacta.py",
    "cortex_sensory": "src/thalia/brain/regions/cortex_sensory.py",
    "cortex_association": "src/thalia/brain/regions/cortex_association.py",
    "prefrontal_cortex": "src/thalia/brain/regions/prefrontal_cortex.py",
    "hippocampus": "src/thalia/brain/regions/hippocampus.py",
    "entorhinal_cortex": "src/thalia/brain/regions/entorhinal_cortex.py",
    "thalamus": "src/thalia/brain/regions/thalamus.py",
    "cerebellum": "src/thalia/brain/regions/cerebellum.py",
    "striatum": "src/thalia/brain/regions/striatum.py",
    "basolateral_amygdala": "src/thalia/brain/regions/basolateral_amygdala.py",
    "central_amygdala": "src/thalia/brain/regions/central_amygdala.py",
    "medial_septum": "src/thalia/brain/regions/medial_septum.py",
    "lateral_habenula": "src/thalia/brain/regions/lateral_habenula.py",
    "globus_pallidus_externa": "src/thalia/brain/regions/globus_pallidus_externa.py",
    "globus_pallidus_interna": "src/thalia/brain/regions/globus_pallidus_interna.py",
    "subthalamic_nucleus": "src/thalia/brain/regions/subthalamic_nucleus.py",
    "substantia_nigra": "src/thalia/brain/regions/substantia_nigra.py",
    "rostromedial_tegmentum": "src/thalia/brain/regions/rostromedial_tegmentum.py",
}

# Config file mapping for region configs with tunable defaults
_CONFIG_SOURCE_FILES_REGION: Dict[str, str] = {
    "hippocampus": "src/thalia/brain/configs/hippocampus.py",
    "cortex_sensory": "src/thalia/brain/configs/cortical_column.py",
    "cortex_association": "src/thalia/brain/configs/cortical_column.py",
    "prefrontal_cortex": "src/thalia/brain/configs/cortical_column.py",
    "entorhinal_cortex": "src/thalia/brain/configs/hippocampus.py",
}


# ---------------------------------------------------------------------------
# Analytical helpers
# ---------------------------------------------------------------------------

def _estimate_i_h_rate_contribution(i_h_conductance: float, i_h_reversal: float,
                                     g_L: float, E_L: float, v_threshold: float,
                                     tau_ref: float) -> float:
    """Estimate firing rate contribution from I_h pacemaker current.

    I_h acts as a constant depolarising conductance.  The steady-state
    membrane potential with only g_L and g_ih is:

        V_inf = (g_L * E_L + g_ih * E_ih) / (g_L + g_ih)

    If V_inf > v_threshold, the ISI is:

        T = tau_eff * ln((v_reset - V_inf) / (v_threshold - V_inf)) + tau_ref
        rate = 1000 / T

    Returns estimated Hz, or 0 if subthreshold.
    """
    g_total = g_L + i_h_conductance
    V_inf = (g_L * E_L + i_h_conductance * i_h_reversal) / g_total
    if V_inf <= v_threshold:
        return 0.0
    tau_eff = 1.0 / g_total
    ratio = (0.0 - V_inf) / (v_threshold - V_inf)  # v_reset ≈ 0
    if ratio <= 0:
        return 0.0
    T = tau_eff * math.log(ratio) + tau_ref
    return 1000.0 / T if T > 0 else 0.0


def _solve_i_h_for_target_rate(target_rate_hz: float, i_h_reversal: float,
                                g_L: float, E_L: float, v_threshold: float,
                                v_reset: float, tau_ref: float) -> float:
    """Solve for i_h_conductance that gives a target firing rate.

    From: rate = 1000 / (tau_eff * ln((v_reset - V_inf)/(v_threshold - V_inf)) + tau_ref)

    We solve numerically via bisection since the relationship is nonlinear.
    """
    if target_rate_hz <= 0:
        return 0.0

    def _rate_at_g_ih(g_ih: float) -> float:
        g_total = g_L + g_ih
        V_inf = (g_L * E_L + g_ih * i_h_reversal) / g_total
        if V_inf <= v_threshold:
            return 0.0
        tau_eff = 1.0 / g_total
        ratio = (v_reset - V_inf) / (v_threshold - V_inf)
        if ratio <= 0:
            return 0.0
        T = tau_eff * math.log(ratio) + tau_ref
        return 1000.0 / T if T > 0 else 0.0

    # Bisection in [0, 2.0]
    g_lo, g_hi = 0.0, 2.0
    for _ in range(100):
        g_mid = (g_lo + g_hi) / 2.0
        r = _rate_at_g_ih(g_mid)
        if r < target_rate_hz:
            g_lo = g_mid
        else:
            g_hi = g_mid
    return (g_lo + g_hi) / 2.0


def _solve_sk_for_rate_reduction(current_rate: float, target_rate: float,
                                  current_sk: float) -> float:
    """Estimate SK conductance to achieve target rate reduction.

    SK acts as a rate-dependent negative feedback: higher SK → more adaptation
    → lower sustained rate.  The relationship is approximately:

        rate ∝ 1 / (1 + α * sk_conductance)

    where α depends on calcium dynamics.  We use the ratio to estimate:

        new_sk ≈ current_sk * (current_rate / target_rate)

    This is a first-order approximation; the actual relationship is more complex.
    """
    if target_rate <= 0 or current_rate <= 0:
        return current_sk
    ratio = current_rate / target_rate
    return current_sk * ratio


def _solve_adapt_increment_for_rate(current_rate: float, target_rate: float,
                                     current_adapt: float, tau_adapt: float) -> float:
    """Estimate adapt_increment to achieve target rate.

    At equilibrium: g_adapt = rate * adapt_increment * tau_adapt / 1000
    The adaptation conductance pulls V_inf toward E_adapt, reducing the rate.

    Approximation: rate scales inversely with adapt_increment for moderate
    adaptation.  new_adapt ≈ current_adapt * (current_rate / target_rate).
    """
    if target_rate <= 0 or current_rate <= 0 or current_adapt <= 0:
        return current_adapt
    ratio = current_rate / target_rate
    return current_adapt * ratio


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _classify_population(
    region: str,
    population: str,
    params: Dict[str, float],
    config_type: str,
    tuning: Optional[TuningGuidance],
) -> IssueKind:
    """Classify the likely cause of an OOB firing rate."""
    # Check for pacemaker neuron types
    if config_type in _PACEMAKER_CONFIG_TYPES:
        return "intrinsic_pacemaker"
    if (region, population) in _DA_PACEMAKER_POPS:
        return "intrinsic_pacemaker"

    # Check for internal drive mechanisms (e.g. LC GABA driven by LP-filtered NE)
    if population == "gaba" and region in ("locus_coeruleus", "dorsal_raphe",
                                            "nucleus_basalis", "vta",
                                            "substantia_nigra_compacta"):
        return "internal_drive"

    # Check for adaptation issues
    adapt_increment = params.get("adapt_increment", 0.0)
    if adapt_increment > 0:
        # If the neuron has adaptation, it might be an adaptation tuning issue
        if tuning and tuning.direction == "decrease":
            # Rate too high despite adaptation → adaptation may be insufficient
            return "adaptation"

    # Check for noise-driven regime
    noise_std = params.get("noise_std", 0.0)
    if tuning and tuning.prediction:
        if tuning.prediction.regime == "noise-driven":
            return "noise_driven"
        if tuning.prediction.regime == "silent" and noise_std > 0:
            return "noise_driven"

    # Check for structural issues (no inputs)
    if tuning:
        has_exc = len(tuning.excitatory_tracts) > 0
        has_inh = len(tuning.inhibitory_tracts) > 0
        if not has_exc and tuning.direction == "increase":
            return "structural"
        if not has_inh and tuning.direction == "decrease":
            # Has excitation but no inhibition — might be structural
            return "structural"

    return "synaptic_drive"


def _explain_issue(kind: IssueKind, region: str, population: str,
                   direction: str, params: Dict[str, float],
                   config_type: str) -> str:
    """Generate a human-readable explanation of the issue."""
    pop_key = f"{region}:{population}"

    if kind == "intrinsic_pacemaker":
        ih = params.get("i_h_conductance", 0.0)
        sk = params.get("sk_conductance", 0.0)
        if direction == "decrease":
            return (f"{pop_key} is a pacemaker neuron firing too fast.  "
                    f"Rate is set by I_h drive (g_ih={ih:.3f}) vs SK adaptation "
                    f"(g_sk={sk:.3f}).  Reduce I_h or increase SK to slow it down.")
        else:
            return (f"{pop_key} is a pacemaker neuron firing too slowly.  "
                    f"Rate is set by I_h drive (g_ih={ih:.3f}) vs SK adaptation "
                    f"(g_sk={sk:.3f}).  Increase I_h or reduce SK to speed it up.")

    if kind == "internal_drive":
        return (f"{pop_key} is driven by an internal mechanism in the region "
                f"(e.g. LP-filtered primary neuron activity → GABA drive).  "
                f"Adjust the drive scale factor in the region's _compute_gaba_drive() method.")

    if kind == "adaptation":
        adapt = params.get("adapt_increment", 0.0)
        tau = params.get("tau_adapt", 100.0)
        if direction == "decrease":
            return (f"{pop_key} fires too fast despite adaptation "
                    f"(adapt_increment={adapt:.3f}, tau_adapt={tau:.0f}ms).  "
                    f"Increase adapt_increment or reduce v_threshold.")
        else:
            return (f"{pop_key} fires too slowly — adaptation may be too strong "
                    f"(adapt_increment={adapt:.3f}, tau_adapt={tau:.0f}ms).  "
                    f"Reduce adapt_increment or increase excitatory drive.")

    if kind == "noise_driven":
        noise = params.get("noise_std", 0.0)
        return (f"{pop_key} is in the noise-driven regime (V_inf < threshold).  "
                f"Rate depends on noise_std={noise:.3f} and distance to threshold.  "
                f"Adjust noise_std, v_threshold, or synaptic drive.")

    if kind == "structural":
        if direction == "increase":
            return (f"{pop_key} has no registered excitatory afferents (inter-region tracts).  "
                    f"Its firing is driven entirely by intra-region connections or intrinsic "
                    f"currents.  Check BrainBuilder connectivity or region wiring.")
        else:
            return (f"{pop_key} has no registered inhibitory afferents.  "
                    f"Add inhibitory connections or increase intrinsic adaptation.")

    # synaptic_drive
    if direction == "decrease":
        return (f"{pop_key} receives too much excitatory drive.  "
                f"Reduce the dominant excitatory synaptic weight or increase inhibition.")
    return (f"{pop_key} receives insufficient excitatory drive.  "
            f"Increase excitatory weights, connectivity, or reduce inhibition.")


def _generate_recommendations(
    kind: IssueKind,
    region: str,
    population: str,
    observed_rate: float,
    target_rate: float,
    direction: str,
    params: Dict[str, float],
    config_type: str,
    tuning: Optional[TuningGuidance],
) -> List[ParameterRecommendation]:
    """Generate concrete parameter change recommendations."""
    recs: List[ParameterRecommendation] = []

    config_src = _CONFIG_SOURCE_FILES.get(config_type, "")
    region_src = _REGION_SOURCE_FILES.get(region, "")
    config_region_src = _CONFIG_SOURCE_FILES_REGION.get(region, "")

    if kind == "intrinsic_pacemaker":
        i_h = params.get("i_h_conductance")
        sk = params.get("sk_conductance")
        i_h_rev = params.get("i_h_reversal", 0.75)
        g_L = params.get("g_L", 0.05)
        E_L = params.get("E_L", 0.0)
        v_thresh = params.get("v_threshold", 1.0)
        v_reset = params.get("v_reset", 0.0)
        tau_ref = params.get("tau_ref", 2.0)
        noise = params.get("noise_std", 0.0)
        autoreceptor = params.get("autoreceptor_conductance")
        src = config_src or region_src

        if direction == "decrease" and sk is not None:
            # Rate too high → increase SK
            new_sk = _solve_sk_for_rate_reduction(observed_rate, target_rate, sk)
            # Clamp to reasonable range and round
            new_sk = round(min(new_sk, sk * 3.0), 3)
            recs.append(ParameterRecommendation(
                param_name="sk_conductance",
                current_value=sk,
                recommended_value=new_sk,
                source_file=src,
                mechanism=(
                    f"SK (Ca²⁺-activated K⁺) channels provide negative feedback: "
                    f"each spike → Ca²⁺ influx → SK activation → hyperpolarising "
                    f"current.  Higher SK → stronger after-spike inhibition → lower rate."
                ),
                confidence="medium",
            ))

        if direction == "decrease" and i_h is not None:
            # Rate too high → decrease I_h
            new_ih = _solve_i_h_for_target_rate(
                target_rate, i_h_rev, g_L, E_L, v_thresh, v_reset, tau_ref
            )
            # The I_h solver doesn't account for SK or noise, so apply a scaling
            # correction: if noise drives part of the rate, I_h should be reduced less
            if noise > 0:
                new_ih = max(new_ih, i_h * 0.5)  # don't cut more than 50%
            new_ih = round(max(new_ih, 0.01), 3)
            recs.append(ParameterRecommendation(
                param_name="i_h_conductance",
                current_value=i_h,
                recommended_value=new_ih,
                source_file=src,
                mechanism=(
                    f"I_h (HCN) provides tonic depolarising drive for pacemaking.  "
                    f"Lower I_h → weaker depolarisation → slower tonic rate."
                ),
                confidence="medium",
            ))

        if direction == "increase" and i_h is not None:
            new_ih = _solve_i_h_for_target_rate(
                target_rate, i_h_rev, g_L, E_L, v_thresh, v_reset, tau_ref
            )
            new_ih = round(max(new_ih, i_h * 0.5), 3)
            recs.append(ParameterRecommendation(
                param_name="i_h_conductance",
                current_value=i_h,
                recommended_value=new_ih,
                source_file=src,
                mechanism=(
                    f"I_h (HCN) provides tonic depolarising drive for pacemaking.  "
                    f"Higher I_h → stronger depolarisation → faster tonic rate."
                ),
                confidence="medium",
            ))

        if direction == "increase" and sk is not None and sk > 0.005:
            new_sk = _solve_sk_for_rate_reduction(observed_rate, target_rate, sk)
            new_sk = round(max(new_sk, 0.005), 3)
            recs.append(ParameterRecommendation(
                param_name="sk_conductance",
                current_value=sk,
                recommended_value=new_sk,
                source_file=src,
                mechanism=(
                    f"SK provides negative feedback.  Lower SK → less after-spike "
                    f"inhibition → higher sustained rate."
                ),
                confidence="medium",
            ))

        if direction == "decrease" and autoreceptor is not None:
            # Increase autoreceptor for more self-inhibition
            ratio = observed_rate / max(target_rate, 0.1)
            new_auto = round(autoreceptor * ratio, 3)
            recs.append(ParameterRecommendation(
                param_name="autoreceptor_conductance",
                current_value=autoreceptor,
                recommended_value=new_auto,
                source_file=src,
                mechanism=(
                    f"Autoreceptor (5-HT1A / GIRK) provides slow self-inhibition "
                    f"proportional to recent spiking.  Higher conductance → stronger "
                    f"negative feedback → lower sustained rate."
                ),
                confidence="low",
            ))

    elif kind == "internal_drive":
        # Internal drive mechanism — recommend adjusting drive scale in region
        recs.append(ParameterRecommendation(
            param_name="gaba_drive_scale (in _compute_gaba_drive)",
            current_value=float("nan"),  # Unknown from snapshot
            recommended_value=float("nan"),
            source_file=region_src,
            mechanism=(
                f"The GABA population is driven by low-pass filtered primary neuron "
                f"activity.  The scale factor in _compute_gaba_drive() controls "
                f"subthreshold drive level.  Adjust to place V_inf just below threshold "
                f"for noise-driven firing at the target rate."
            ),
            confidence="low",
        ))
        # Also suggest noise_std adjustment
        noise = params.get("noise_std", 0.0)
        if noise > 0:
            if direction == "increase" and observed_rate < 1.0:
                # Nearly silent → drive is too weak, noise can't overcome gap
                new_noise = round(noise * 1.5, 3)
                recs.append(ParameterRecommendation(
                    param_name="noise_std",
                    current_value=noise,
                    recommended_value=new_noise,
                    source_file=region_src,
                    mechanism=(
                        f"In the noise-driven regime, firing rate depends on how often "
                        f"noise fluctuations cross threshold.  Higher noise_std → more "
                        f"threshold crossings → higher rate.  But first ensure the drive "
                        f"scale places V_inf close to threshold."
                    ),
                    confidence="low",
                ))

    elif kind == "adaptation":
        adapt = params.get("adapt_increment", 0.0)
        tau = params.get("tau_adapt", 100.0)
        src = config_region_src or region_src

        if direction == "decrease" and adapt > 0:
            new_adapt = _solve_adapt_increment_for_rate(
                observed_rate, target_rate, adapt, tau
            )
            new_adapt = round(min(new_adapt, adapt * 3.0), 3)
            recs.append(ParameterRecommendation(
                param_name="adapt_increment",
                current_value=adapt,
                recommended_value=new_adapt,
                source_file=src,
                mechanism=(
                    f"Spike-frequency adaptation: each spike adds {adapt:.3f} to an "
                    f"inhibitory conductance that decays with τ={tau:.0f} ms.  "
                    f"Higher adapt_increment → stronger rate limiting."
                ),
                confidence="high",
            ))

        elif direction == "increase" and adapt > 0:
            new_adapt = _solve_adapt_increment_for_rate(
                observed_rate, target_rate, adapt, tau
            )
            new_adapt = round(max(new_adapt, 0.01), 3)
            recs.append(ParameterRecommendation(
                param_name="adapt_increment",
                current_value=adapt,
                recommended_value=new_adapt,
                source_file=src,
                mechanism=(
                    f"Adaptation may be too strong, causing the neuron to self-silence.  "
                    f"Reducing adapt_increment allows more sustained firing."
                ),
                confidence="high",
            ))

        # Also suggest v_threshold change for adaptation-driven populations
        v_thresh = params.get("v_threshold", 1.0)
        if direction == "decrease":
            ratio = target_rate / max(observed_rate, 0.1)
            # Higher threshold → harder to reach → lower rate
            new_thresh = round(v_thresh / ratio, 3)
            new_thresh = min(new_thresh, v_thresh * 1.3)  # cap at +30%
            recs.append(ParameterRecommendation(
                param_name="v_threshold",
                current_value=v_thresh,
                recommended_value=new_thresh,
                source_file=src,
                mechanism=(
                    f"Higher threshold → larger voltage excursion needed per spike → "
                    f"longer ISI → lower rate.  Effective for moderate rate reductions."
                ),
                confidence="medium",
            ))

    elif kind == "noise_driven":
        noise = params.get("noise_std", 0.0)
        v_thresh = params.get("v_threshold", 1.0)
        src = config_region_src or region_src

        if direction == "decrease" and noise > 0:
            # Reduce noise or increase threshold distance
            ratio = target_rate / max(observed_rate, 0.1)
            new_noise = round(noise * math.sqrt(ratio), 4)  # rate ∝ noise² roughly
            recs.append(ParameterRecommendation(
                param_name="noise_std",
                current_value=noise,
                recommended_value=new_noise,
                source_file=src,
                mechanism=(
                    f"In the noise-driven regime, rate ∝ exp(-Δ²/2σ²) where Δ is the "
                    f"distance to threshold.  Reducing noise_std exponentially reduces "
                    f"threshold-crossing probability."
                ),
                confidence="low",
            ))

        if direction == "increase" and noise > 0:
            ratio = target_rate / max(observed_rate, 0.1)
            new_noise = round(noise * math.sqrt(ratio), 4)
            recs.append(ParameterRecommendation(
                param_name="noise_std",
                current_value=noise,
                recommended_value=new_noise,
                source_file=src,
                mechanism=(
                    f"In the noise-driven regime, increasing noise_std brings more "
                    f"fluctuations above threshold → higher rate."
                ),
                confidence="low",
            ))

        # Suggest v_threshold adjustment (applies in all regimes)
        if direction == "decrease":
            new_thresh = round(v_thresh * 1.05, 3)
            recs.append(ParameterRecommendation(
                param_name="v_threshold",
                current_value=v_thresh,
                recommended_value=new_thresh,
                source_file=src,
                mechanism="Raising threshold increases distance to spike → fewer noise crossings.",
                confidence="medium",
            ))

    elif kind == "structural":
        recs.append(ParameterRecommendation(
            param_name="(connectivity)",
            current_value=0.0,
            recommended_value=0.0,
            source_file=region_src,
            mechanism=(
                f"This population {'lacks excitatory afferents' if direction == 'increase' else 'lacks inhibitory afferents'}.  "
                f"Its rate is set entirely by intra-region connections or intrinsic currents.  "
                f"If the rate should be synaptically controlled, add inter-region tracts "
                f"in BrainBuilder.  Otherwise treat as an intrinsic parameter tuning issue — "
                f"adjust v_threshold, adapt_increment, noise_std, or tonic_drive in the region."
            ),
            confidence="low",
        ))
        # Still suggest intrinsic param changes as fallback
        adapt = params.get("adapt_increment", 0.0)
        v_thresh = params.get("v_threshold", 1.0)
        src = config_region_src or region_src
        if direction == "decrease" and adapt >= 0:
            new_adapt = max(adapt * 1.3, adapt + 0.05) if adapt > 0 else 0.05
            new_adapt = round(new_adapt, 3)
            recs.append(ParameterRecommendation(
                param_name="adapt_increment",
                current_value=adapt,
                recommended_value=new_adapt,
                source_file=src,
                mechanism="Adding or increasing adaptation provides rate-dependent negative feedback.",
                confidence="medium",
            ))

    elif kind == "synaptic_drive":
        # The existing TuningGuidance handles this; add complementary intrinsic suggestions
        if tuning and tuning.suggested_weight_scale is not None and tuning.dominant_tract:
            recs.append(ParameterRecommendation(
                param_name=f"weight ({tuning.dominant_tract})",
                current_value=1.0,
                recommended_value=round(tuning.suggested_weight_scale, 3),
                source_file=region_src,
                mechanism=(
                    f"Scale the synaptic weight on the dominant "
                    f"{'excitatory' if direction == 'decrease' else 'inhibitory'} tract."
                ),
                confidence="medium",
            ))

        # Also suggest intrinsic changes
        adapt = params.get("adapt_increment", 0.0)
        v_thresh = params.get("v_threshold", 1.0)
        src = config_region_src or region_src

        if direction == "decrease" and adapt >= 0:
            new_adapt = _solve_adapt_increment_for_rate(
                observed_rate, target_rate, adapt, params.get("tau_adapt", 100.0)
            ) if adapt > 0 else round(0.05 * (observed_rate / max(target_rate, 0.1)), 3)
            new_adapt = round(new_adapt, 3)
            recs.append(ParameterRecommendation(
                param_name="adapt_increment",
                current_value=adapt,
                recommended_value=new_adapt,
                source_file=src,
                mechanism="Spike-frequency adaptation dampens sustained firing rates.",
                confidence="medium" if adapt > 0 else "low",
            ))

        if direction == "increase":
            # Suggest lowering v_threshold to make it easier to spike
            new_thresh = round(v_thresh * (target_rate / max(observed_rate, 0.1)) ** 0.2, 3)
            new_thresh = max(new_thresh, v_thresh * 0.8)  # cap at -20%
            if new_thresh < v_thresh:
                recs.append(ParameterRecommendation(
                    param_name="v_threshold",
                    current_value=v_thresh,
                    recommended_value=new_thresh,
                    source_file=src,
                    mechanism="Lower threshold → easier to reach → higher rate.",
                    confidence="medium",
                ))

    return recs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_calibration_advice(
    snapshot: RecorderSnapshot,
    report: DiagnosticsReport,
    tuning: Optional[TuningReport] = None,
) -> CalibrationReport:
    """Compute actionable calibration advice for all OOB populations.

    Parameters
    ----------
    snapshot :
        Recorder snapshot with neuron params and config types.
    report :
        Diagnostics report with per-population firing rates.
    tuning :
        Optional pre-computed TuningReport.  If None, only intrinsic
        parameter recommendations are generated (no synaptic budget analysis).

    Returns
    -------
    CalibrationReport with per-population advice sorted by severity.
    """
    from .analysis_tuning import compute_tuning

    if tuning is None:
        tuning = compute_tuning(snapshot, report)

    # Build tuning lookup
    tuning_by_pop: Dict[Tuple[str, str], TuningGuidance] = {}
    for item in tuning.items:
        tuning_by_pop[(item.region, item.population)] = item

    # Build pop stats lookup
    pop_stats: Dict[Tuple[str, str], float] = {}
    for rstat in report.regions.values():
        for pstat in rstat.populations.values():
            pop_stats[(pstat.region_name, pstat.population_name)] = pstat.mean_fr_hz

    items: List[CalibrationAdvice] = []

    # Process all OOB populations (from tuning report + any with bio_range violations)
    oob_pops: set[Tuple[str, str]] = set()
    for key in tuning_by_pop:
        oob_pops.add(key)

    # Also check populations not in tuning (e.g. those without weight stats)
    for (rn, pn), rate in pop_stats.items():
        br = bio_range(rn, pn)
        if br is None:
            continue
        lo, hi = br
        if not (lo <= rate <= hi):
            oob_pops.add((rn, pn))

    for rn, pn in sorted(oob_pops):
        params = snapshot._pop_neuron_params.get((rn, pn))
        if params is None:
            continue

        br = bio_range(rn, pn)
        if br is None:
            continue
        lo, hi = br
        rate = pop_stats.get((rn, pn), 0.0)
        if lo <= rate <= hi:
            continue  # Now in range (shouldn't happen but guard)

        target = (lo + hi) / 2.0
        direction: Literal["decrease", "increase"] = "decrease" if rate > hi else "increase"

        config_type = snapshot._pop_config_types.get((rn, pn), "ConductanceLIFConfig")
        tg = tuning_by_pop.get((rn, pn))

        kind = _classify_population(rn, pn, params, config_type, tg)
        explanation = _explain_issue(kind, rn, pn, direction, params, config_type)
        recommendations = _generate_recommendations(
            kind, rn, pn, rate, target, direction, params, config_type, tg
        )

        items.append(CalibrationAdvice(
            region=rn,
            population=pn,
            observed_rate_hz=rate,
            bio_range_hz=(lo, hi),
            target_rate_hz=target,
            direction=direction,
            issue_kind=kind,
            issue_explanation=explanation,
            recommendations=recommendations,
        ))

    # Sort by absolute deviation from bio range (largest first)
    def _deviation(a: CalibrationAdvice) -> float:
        lo, hi = a.bio_range_hz
        if a.observed_rate_hz > hi:
            return a.observed_rate_hz - hi
        return lo - a.observed_rate_hz

    items.sort(key=_deviation, reverse=True)
    return CalibrationReport(items=items)


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

_CONFIDENCE_SYMBOLS = {"high": "●", "medium": "◐", "low": "○"}


def print_calibration_advice(advice: CalibrationReport) -> None:
    """Print actionable calibration advice to stdout."""
    if not advice.items:
        return

    w = 88
    print(f"\n{'═' * w}")
    print("CALIBRATION ADVISOR — Actionable Parameter Recommendations")
    print(f"{'═' * w}")
    print(f"  {len(advice.items)} population(s) need calibration")
    print(f"  Confidence: ● high  ◐ medium  ○ low\n")

    for a in advice.items:
        lo, hi = a.bio_range_hz
        arrow = "▲ need higher" if a.direction == "increase" else "▼ need lower"
        kind_label = a.issue_kind.replace("_", " ").title()

        print(f"  {'─' * (w - 4)}")
        print(f"  {a.region}:{a.population}  [{kind_label}]")
        print(f"    Rate: {a.observed_rate_hz:.2f} Hz → target [{lo:.0f}–{hi:.0f}] Hz "
              f"(midpoint {a.target_rate_hz:.0f} Hz)  {arrow}")
        print(f"    {a.issue_explanation}")

        if a.recommendations:
            print(f"\n    Recommended changes:")
            for i, rec in enumerate(a.recommendations, 1):
                sym = _CONFIDENCE_SYMBOLS.get(rec.confidence, "?")
                if math.isnan(rec.current_value) or math.isnan(rec.recommended_value):
                    val_str = "  (inspect code manually)"
                else:
                    val_str = f"  {rec.current_value:.4f} → {rec.recommended_value:.4f}"
                print(f"    {sym} {i}. {rec.param_name}{val_str}")
                if rec.source_file:
                    print(f"         File: {rec.source_file}")
                # Wrap mechanism text
                mech = rec.mechanism
                indent = "         "
                max_line = w - len(indent) - 2
                words = mech.split()
                line = indent
                for word in words:
                    if len(line) + len(word) + 1 > w - 2:
                        print(line)
                        line = indent + word
                    else:
                        line = line + " " + word if line.strip() else indent + word
                if line.strip():
                    print(line)
        else:
            print(f"\n    (No specific parameter recommendations available)")

        print()

    print(f"{'═' * w}\n")
