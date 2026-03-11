"""Analytical rate predictor — quick calibration examples.

Run with:
    python scripts/predict_rate.py

Demonstrates how to use predict_rate() and stp_table() to diagnose and fix
firing-rate problems without running a full simulation.
"""

from thalia.brain.synapses.stp import STPConfig
from thalia.diagnostics import InputSpec, predict_rate, stp_eq, stp_table

# ── LHb: why was it hyperactive in _06, and does _07 fix it? ─────────────────
print("\n" + "=" * 68)
print("  LATERAL HABENULA  (CeA drive)")
print("=" * 68)

# _06 params: no adaptation
lhb_inputs = [
    InputSpec(n=500, rate_hz=8.0, weight_mean=0.00582, label="CeA→LHb (excit)"),
    InputSpec(n=500, rate_hz=5.0, weight_mean=0.00050, receptor="gaba_a", label="GPe→LHb (inhib)"),
]

print("\n[_06] No adaptation — should show ~66 Hz hyperactivity:")
predict_rate(g_L=0.10, v_threshold=1.0, tau_E=5.0, inputs=lhb_inputs).print()

print("[_07] adapt_increment=0.10 — target 5–20 Hz:")
predict_rate(
    g_L=0.10, v_threshold=1.0, tau_E=5.0,
    adapt_increment=0.10, tau_adapt=100.0, E_adapt=-0.5,
    inputs=lhb_inputs,
).print()

# ── LC:gaba — why was it hyperactive and does the drive fix work? ─────────────
print("=" * 68)
print("  LOCUS COERULEUS — GABA interneurons")
print("=" * 68)

print("\n[_06] gaba_drive = 0.010 + activity*0.5 (activity≈1.0 → drive=0.51):")
predict_rate(
    g_L=0.05, v_threshold=1.0, tau_E=5.0,
    baseline_drive=0.51, tau_baseline=5.0,
).print()

print("[_07] gaba_drive = 0.004 + activity*0.004 (activity≈1.0 → drive=0.008):")
predict_rate(
    g_L=0.05, v_threshold=1.0, tau_E=5.0,
    baseline_drive=0.008, tau_baseline=5.0,
).print()

# ── GPe arkypallidal — why was it near-silent? ───────────────────────────────
print("=" * 68)
print("  GPe ARKYPALLIDAL")
print("=" * 68)

stp_gpe = STPConfig(U=0.50, tau_d=800, tau_f=20)
print(f"\nSTN→GPe STP at 20 Hz (the STN rate in _06 = 19.84 Hz):")
stp_eq(U=0.50, tau_d=800, tau_f=20, rate_hz=20.0).print()
print("→ eff=0.056: the STN synapse is ~18× weaker than the nominal weight!\n")

print("[_06] baseline factor=0.5 (sub-threshold):")
predict_rate(
    g_L=0.05, v_threshold=1.0, tau_E=5.0,
    baseline_drive=0.020,
    inputs=[
        InputSpec(n=200, rate_hz=20.0, weight_mean=0.00050, label="STN→GPe", stp=stp_gpe),
    ],
).print()

print("[_07] baseline factor=0.9 (target 10–20 Hz):")
predict_rate(
    g_L=0.05, v_threshold=1.0, tau_E=5.0,
    baseline_drive=0.036,
    inputs=[
        InputSpec(n=200, rate_hz=20.0, weight_mean=0.00050, label="STN→GPe", stp=stp_gpe),
    ],
).print()

# ── STP depletion survey ──────────────────────────────────────────────────────
print("=" * 68)
print("  STP DEPLETION SURVEY — STN→GPe (U=0.5, tau_d=800)")
print("=" * 68)
stp_table(U=0.50, tau_d=800, tau_f=20)

print("=" * 68)
print("  STP SURVEY — corticostriatal (typical U=0.25, tau_d=200)")
print("=" * 68)
stp_table(U=0.25, tau_d=200, tau_f=50)
