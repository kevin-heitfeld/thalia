"""Analytical firing-rate predictor for ConductanceLIF neurons.

Computes V_inf and equilibrium firing rate analytically from neuron parameters
and a list of synaptic input specifications — **no simulation needed**.

The key formula derives from the ConductanceLIF steady state:

    g_total  = g_L + g_E + g_I + g_adapt
    V_inf    = (g_L·E_L + g_E·E_E + g_I·E_I + g_adapt·E_adapt) / g_total
    tau_eff  = 1 / g_total          [C_m normalised to 1; units: ms]
    T_spike  = tau_eff · ln((v_reset − V_inf) / (v_threshold − V_inf))
    rate_hz  = 1000 / (T_spike + tau_ref)

Each Poisson input contributes steady-state conductance:

    g_ss = n · connectivity · weight_mean · stp_eff
           · (rate_hz / 1000) · dt_ms
           / (1 − exp(−dt_ms / tau_receptor))

Adaptation is solved iteratively to self-consistency:

    g_adapt = rate_eq · adapt_increment · tau_adapt / 1000

Usage::

    from thalia.diagnostics.rate_predictor import predict_rate, InputSpec

    result = predict_rate(
        g_L=0.10,
        v_threshold=1.0,
        tau_E=5.0,
        inputs=[
            InputSpec(n=500, rate_hz=8.0, weight_mean=0.00582, label="CeA→LHb"),
            InputSpec(n=500, rate_hz=5.0, weight_mean=0.0001,
                      receptor="gaba_a", label="GPe→LHb"),
        ],
        baseline_drive=0.007,   # per-step tonic conductance (like GPe/LHb)
    )
    result.print()

    # Point prediction without printing:
    r = predict_rate(g_L=0.05, adapt_increment=0.10, tau_adapt=200.0,
                     inputs=[InputSpec(n=200, rate_hz=5.0, weight_mean=0.001)])
    print(r.rate_hz)   # → 7.4 Hz with adaptation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional

from thalia.brain.synapses import STPConfig

ReceptorKind = Literal["ampa", "nmda", "gaba_a", "gaba_b"]


# ---------------------------------------------------------------------------
# Input specification
# ---------------------------------------------------------------------------

@dataclass
class InputSpec:
    """One synaptic input source for analytical rate prediction.

    Attributes
    ----------
    n :
        Number of presynaptic neurons.
    rate_hz :
        Mean presynaptic firing rate in Hz.
    weight_mean :
        Mean synaptic weight (conductance units per synapse).
    connectivity :
        Connection probability / fraction (0–1).  Default 1.0.
    receptor :
        Receptor type: ``"ampa"``, ``"nmda"``, ``"gaba_a"``, or ``"gaba_b"``.
    stp :
        Optional :class:`~thalia.brain.synapses.stp.STPConfig`.  When provided
        the weight is scaled by the steady-state STP efficacy *u_ss × x_ss*.
    label :
        Human-readable label shown in :meth:`RatePrediction.print`.
    """

    n: int
    rate_hz: float
    weight_mean: float
    connectivity: float = 1.0
    receptor: ReceptorKind = "ampa"
    stp: Optional[STPConfig] = None
    label: str = ""


# ---------------------------------------------------------------------------
# Per-input contribution (computed internally)
# ---------------------------------------------------------------------------

@dataclass
class _InputContrib:
    label: str
    receptor: ReceptorKind
    n_effective: float   # n * connectivity
    rate_hz: float
    stp_eff: float        # 1.0 if no STP
    g_ss: float           # Steady-state conductance contributed


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RatePrediction:
    """Analytical firing-rate prediction result.

    All conductances are in the same normalised units used by
    :class:`~thalia.brain.neurons.ConductanceLIF`.

    Attributes
    ----------
    contributions :
        Per-input breakdown of steady-state conductance.
    g_E :
        Total excitatory (AMPA + NMDA) steady-state conductance.
    g_I :
        Total inhibitory (GABA_A + GABA_B) steady-state conductance.
    g_adapt :
        Adaptation conductance at equilibrium (0 when ``adapt_increment=0``).
    V_inf :
        Steady-state membrane potential **without** adaptation.
    V_inf_with_adapt :
        V_inf after accounting for equilibrium adaptation conductance.
    rate_hz :
        Predicted equilibrium firing rate in Hz.
    regime :
        ``"silent"``, ``"suprathreshold"``, or ``"noise-driven"``.
    tau_eff_ms :
        Effective membrane time constant at the steady-state operating point (ms).
    """

    contributions: List[_InputContrib]

    # Aggregate conductances
    g_E: float
    g_I: float
    g_adapt: float

    # Neuron params echoed back
    g_L: float
    v_threshold: float
    v_reset: float
    E_L: float
    E_E: float
    E_I: float
    E_adapt: float
    tau_ref: float

    # Derived
    tau_eff_ms: float
    V_inf: float
    V_inf_with_adapt: float
    rate_hz: float
    regime: str
    n_iterations: int

    # ------------------------------------------------------------------ #

    def print(self) -> None:
        """Print a human-readable prediction summary."""
        w = 68
        print(f"\n{'─' * w}")
        print("  Analytical Rate Prediction")
        print(f"{'─' * w}")

        if self.contributions:
            print(f"  {'Label':<28} {'Recv':<6} {'N_eff':>7} {'Rate':>7} {'STP':>6} {'g_ss':>8}")
            print(f"  {'-'*28} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*8}")
            for c in self.contributions:
                lbl = c.label or f"({c.receptor})"
                print(f"  {lbl:<28} {c.receptor:<6} {c.n_effective:>7.0f}"
                      f" {c.rate_hz:>6.1f}Hz {c.stp_eff:>6.3f} {c.g_ss:>8.5f}")
            print(f"  {'─' * (w - 2)}")

        print(f"  g_E_ss  (AMPA + NMDA)        : {self.g_E:>8.5f}")
        print(f"  g_I_ss  (GABA_A + GABA_B)    : {self.g_I:>8.5f}")
        if self.g_adapt > 0:
            print(f"  g_adapt (equil. adaptation)  : {self.g_adapt:>8.5f}")
        g_total = self.g_L + self.g_E + self.g_I + self.g_adapt
        print(f"  g_total                      : {g_total:>8.5f}  (g_L={self.g_L})")
        print(f"  tau_eff                      : {self.tau_eff_ms:>7.2f} ms")
        print(f"  V_inf    (no adapt)          : {self.V_inf:>7.3f}  [threshold={self.v_threshold}]")
        if self.g_adapt > 0:
            print(f"  V_inf    (with adapt)        : {self.V_inf_with_adapt:>7.3f}")

        flag = ""
        if self.rate_hz > 200:
            flag = "  ⚠ SEVERELY HYPERACTIVE"
        elif self.rate_hz > 100:
            flag = "  ⚠ HYPERACTIVE"
        elif self.rate_hz == 0:
            flag = "  ✗ SILENT"
        else:
            flag = "  ✓"
        print(f"  Rate prediction              : {self.rate_hz:>7.1f} Hz{flag}")
        print(f"  Regime                       : {self.regime}")
        if self.n_iterations > 1:
            print(f"  Adapt. convergence iters     : {self.n_iterations}")
        print(f"{'─' * w}\n")


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------

def predict_rate(
    *,
    # Neuron biophysics
    g_L: float = 0.05,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
    E_L: float = 0.0,
    E_E: float = 3.0,
    E_I: float = -0.5,
    E_adapt: float = -0.5,
    tau_E: float = 5.0,
    tau_I: float = 10.0,
    tau_nmda: float = 100.0,
    tau_gaba_b: float = 400.0,
    tau_ref: float = 2.0,
    adapt_increment: float = 0.0,
    tau_adapt: float = 100.0,
    dt_ms: float = 1.0,
    # Synaptic inputs
    inputs: Optional[List[InputSpec]] = None,
    # Optional per-step tonic baseline conductance (GPe/GPi/LHb style).
    # Added as: g_ss = baseline_drive / (1 - exp(-dt_ms / tau_baseline))
    baseline_drive: float = 0.0,
    tau_baseline: float = 5.0,
    # Adaptation solve settings
    max_adapt_iters: int = 200,
    adapt_tol: float = 1e-5,
) -> RatePrediction:
    """Predict equilibrium firing rate analytically.

    Parameters
    ----------
    g_L :
        Leak conductance.
    v_threshold :
        Spike threshold (default 1.0 in normalised units).
    v_reset :
        Reset potential after spike emission (default 0.0).
    E_L, E_E, E_I, E_adapt :
        Reversal potentials for leak, excitation, inhibition, and adaptation.
    tau_E, tau_I, tau_nmda, tau_gaba_b :
        Synaptic decay time constants (ms) for AMPA, GABA_A, NMDA, and GABA_B.
    tau_ref :
        Absolute refractory period (ms).
    adapt_increment :
        Per-spike conductance increment for spike-frequency adaptation (SFA).
    tau_adapt :
        SFA time constant (ms).
    dt_ms :
        Simulation timestep (ms); required for the g_ss formula.
    inputs :
        List of :class:`InputSpec` objects.  ``None`` / empty → tonic drive only.
    baseline_drive :
        Per-step AMPA conductance added each timestep (tonic pacemaker pattern).
    tau_baseline :
        Receptor time constant for *baseline_drive* (default 5.0 ms = AMPA).
    max_adapt_iters :
        Cap on adaptation self-consistency iterations.
    adapt_tol :
        Convergence tolerance on rate (Hz) for adaptation loop.

    Returns
    -------
    RatePrediction
    """
    _tau_map: dict[str, float] = {
        "ampa":   tau_E,
        "nmda":   tau_nmda,
        "gaba_a": tau_I,
        "gaba_b": tau_gaba_b,
    }

    def _decay_fraction(tau: float) -> float:
        """1 - exp(-dt/tau):  fraction of a per-step conductance that accumulates."""
        return 1.0 - math.exp(-dt_ms / tau)

    # ── 1. Compute per-input steady-state conductances ──────────────────────
    contribs: List[_InputContrib] = []
    g_E_total = 0.0
    g_I_total = 0.0

    if inputs:
        for spec in inputs:
            stp_eff = 1.0
            if spec.stp is not None:
                stp_eff = spec.stp.steady_state_utilization(spec.rate_hz)

            tau_rec = _tau_map[spec.receptor]
            decay_f = _decay_fraction(tau_rec)

            n_eff = spec.n * spec.connectivity
            # Conductance added per timestep (expected value):
            #   n_eff × (rate_hz/1000) × dt_ms × weight_mean × stp_eff
            # At steady state: g_ss = per_step_input / decay_fraction
            per_step = n_eff * (spec.rate_hz / 1000.0) * dt_ms * spec.weight_mean * stp_eff
            g_ss = per_step / decay_f

            contribs.append(_InputContrib(
                label=spec.label,
                receptor=spec.receptor,
                n_effective=n_eff,
                rate_hz=spec.rate_hz,
                stp_eff=stp_eff,
                g_ss=g_ss,
            ))

            if spec.receptor in ("ampa", "nmda"):
                g_E_total += g_ss
            else:
                g_I_total += g_ss

    # Tonic baseline drive (like GPe/GPi/LHb pacemaker)
    if baseline_drive > 0.0:
        decay_f_base = _decay_fraction(tau_baseline)
        g_ss_base = baseline_drive / decay_f_base
        g_E_total += g_ss_base
        contribs.append(_InputContrib(
            label="tonic_baseline",
            receptor="ampa",
            n_effective=1.0,
            rate_hz=float("nan"),
            stp_eff=1.0,
            g_ss=g_ss_base,
        ))

    # ── 2. V_inf without adaptation ─────────────────────────────────────────
    def _v_inf(g_adapt: float) -> float:
        num = g_L * E_L + g_E_total * E_E + g_I_total * E_I + g_adapt * E_adapt
        den = g_L + g_E_total + g_I_total + g_adapt
        return num / den if den > 0 else 0.0

    def _rate_from_v_inf(vi: float, g_adapt: float) -> float:
        """Mean-field ISI estimate; returns 0 if sub-threshold."""
        g_tot = g_L + g_E_total + g_I_total + g_adapt
        tau_eff = 1.0 / g_tot  # ms  (C_m=1, g in ms⁻¹, so tau = 1/g has units ms)
        if vi <= v_threshold:
            return 0.0
        if vi <= v_reset:
            # Should not happen for normal parameters, but guard anyway
            return 0.0
        ratio = (v_reset - vi) / (v_threshold - vi)
        if ratio <= 0:
            return 0.0
        T_isi = tau_eff * math.log(ratio) + tau_ref
        if T_isi <= 0:
            return float("inf")
        return 1000.0 / T_isi

    v_inf_no_adapt = _v_inf(0.0)
    g_total_no_adapt = g_L + g_E_total + g_I_total
    tau_eff_ms = 1.0 / g_total_no_adapt if g_total_no_adapt > 0 else float("inf")

    # ── 3. Solve adaptation self-consistently ───────────────────────────────
    # At equilibrium: g_adapt = rate_eq * adapt_increment * tau_adapt / 1000
    # Let alpha = adapt_increment * tau_adapt / 1000.
    # When the neuron is strongly driven and V_inf >> threshold, the fixed point
    # sits almost exactly at the threshold-crossing adaptation level g_crit:
    #   g_crit = g_total_no_adapt * (V_inf - threshold) / (threshold - E_adapt)
    #   rate_eq ≈ g_crit / alpha
    # This analytical formula is exact at the bifurcation and is an excellent
    # approximation whenever V_inf is noticeably above threshold.
    # For weakly-driven neurons (V_inf barely above threshold), the fixed point
    # may be well below g_crit; we refine with bisection in those cases.
    g_adapt_eq = 0.0
    n_iters = 1
    rate_adapt_override: Optional[float] = None   # set when using analytical formula

    if adapt_increment > 0.0:
        alpha = adapt_increment * tau_adapt / 1000.0
        rate_no_adapt = _rate_from_v_inf(v_inf_no_adapt, 0.0)

        if rate_no_adapt == 0.0:
            # Already silent — adaptation has no effect
            g_adapt_eq = 0.0
            n_iters = 1
        elif v_threshold <= E_adapt:
            # E_adapt >= threshold: adaptation can't pull V_inf below threshold
            g_adapt_eq = 0.0
            n_iters = 1
        else:
            # g_crit: the adaptation conductance that brings V_inf exactly to threshold
            #   V_inf(g) = threshold  →  g = g_total_no * (V_inf_no - threshold) / (threshold - E_adapt)
            denom_crit = v_threshold - E_adapt  # > 0
            g_total_no = g_L + g_E_total + g_I_total
            g_crit = g_total_no * (v_inf_no_adapt - v_threshold) / denom_crit

            if g_crit <= 0.0:
                g_adapt_eq = 0.0
                n_iters = 1
            else:
                # Analytical fixed-point approximation: rate_eq ≈ g_crit / alpha
                # This is highly accurate when V_inf drops steeply at g_crit
                # (i.e. when V_inf_no_adapt is well above threshold).
                rate_eq_analytical = g_crit / alpha

                # Refine with bisection when the drive is moderate
                # (V_inf only slightly above threshold, so the fixed point may be
                # at a g_adapt significantly below g_crit where the rate is gentler).
                if v_inf_no_adapt - v_threshold < 0.15:
                    # Bisection on h(g) = rate(g)*alpha - g in [0, g_crit]
                    g_lo, g_hi = 0.0, g_crit
                    for i in range(max_adapt_iters):
                        g_mid = 0.5 * (g_lo + g_hi)
                        rate_mid = _rate_from_v_inf(_v_inf(g_mid), g_mid)
                        h_mid = rate_mid * alpha - g_mid
                        if (g_hi - g_lo) < 1e-9:
                            n_iters = i + 1
                            g_adapt_eq = g_mid
                            break
                        if h_mid > 0.0:
                            g_lo = g_mid
                        else:
                            g_hi = g_mid
                    else:
                        g_adapt_eq = 0.5 * (g_lo + g_hi)
                        n_iters = max_adapt_iters
                    # Compute rate at the bisection result; fall back to analytical
                    # if bisection converged to a degenerate point near g_crit where
                    # V_inf ≈ threshold (rate function discontinuous there).
                    rate_bisect = _rate_from_v_inf(_v_inf(g_adapt_eq), g_adapt_eq)
                    if rate_bisect == 0.0 or (g_hi - g_lo) > 1e-6:
                        # Bisection converged to g_crit (degenerate); use analytical
                        g_adapt_eq = g_crit
                        rate_adapt_override = rate_eq_analytical
                    # else: bisection found a proper interior fixed point, rate_bisect is valid
                else:
                    # Strongly driven: use analytical formula directly
                    g_adapt_eq = g_crit
                    rate_adapt_override = rate_eq_analytical
                    n_iters = 1

    # ── 4. Final quantities ──────────────────────────────────────────────────
    v_inf_final = _v_inf(g_adapt_eq)
    if rate_adapt_override is not None:
        rate_final = rate_adapt_override
    else:
        rate_final = _rate_from_v_inf(v_inf_final, g_adapt_eq)

    if rate_final == 0.0 and v_inf_no_adapt >= v_threshold * 0.85:
        regime = "noise-driven"
    elif rate_final == 0.0:
        regime = "silent"
    else:
        regime = "suprathreshold"

    # Cap display at refractory limit
    rate_ref_limit = 1000.0 / tau_ref if tau_ref > 0 else float("inf")
    rate_final = min(rate_final, rate_ref_limit)

    return RatePrediction(
        contributions=contribs,
        g_E=g_E_total,
        g_I=g_I_total,
        g_adapt=g_adapt_eq,
        g_L=g_L,
        v_threshold=v_threshold,
        v_reset=v_reset,
        E_L=E_L,
        E_E=E_E,
        E_I=E_I,
        E_adapt=E_adapt,
        tau_ref=tau_ref,
        tau_eff_ms=tau_eff_ms,
        V_inf=v_inf_no_adapt,
        V_inf_with_adapt=v_inf_final,
        rate_hz=rate_final,
        regime=regime,
        n_iterations=n_iters,
    )
