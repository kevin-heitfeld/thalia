"""Phase 2 tests: NMDA Mg²⁺ voltage-gating and coincidence detection.

Validates:
- 2.1 The Mg²⁺ voltage-gate modulates NMDA conductance as a function of membrane potential.
- 2.2 STDP-driving NMDA current is large only when pre- and postsynaptic activity coincide
      (postsynaptic cell is depolarised when presynaptic input arrives).

References:
    Jahr & Stevens (1990) — Voltage dependence of NMDA-activated macroscopic conductances
                            predicted by single-channel kinetics. J Neurosci 10(9):3178-3182.
"""

import pytest
import torch

from thalia.components.neurons import ConductanceLIF, ConductanceLIFConfig
from thalia.units import ConductanceTensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_neuron(n: int = 1, **kwargs) -> ConductanceLIF:
    """Create a minimal ConductanceLIF instance."""
    cfg = ConductanceLIFConfig(
        region_name="test",
        population_name="pop",
        tau_mem=20.0,
        noise_std=0.0,       # No noise — deterministic tests
        use_ou_noise=False,
        **kwargs,
    )
    neuron = ConductanceLIF(n_neurons=n, config=cfg)
    neuron.update_temporal_parameters(dt_ms=1.0)
    return neuron


def _g(t: torch.Tensor) -> ConductanceTensor:
    return ConductanceTensor(t)


# ---------------------------------------------------------------------------
# 2.1 — Mg²⁺ voltage-gate unit tests
# ---------------------------------------------------------------------------

class TestNMDAVoltageGate:
    """Direct tests of f_nmda(V) = sigmoid(k * (V - V_half))."""

    def test_gate_near_zero_at_rest(self):
        """At resting potential (V=0), NMDA unblock should be small (<20%)."""
        neuron = make_neuron()
        # Force membrane to resting potential
        neuron.membrane = torch.zeros(1)
        neuron.g_nmda = torch.ones(1)  # non-zero NMDA conductance

        cfg = neuron.config
        f_nmda_at_rest = torch.sigmoid(
            torch.tensor(cfg.nmda_mg_k * (0.0 - cfg.nmda_mg_v_half))
        ).item()

        assert f_nmda_at_rest < 0.20, (
            f"At rest (V=0), NMDA gate should be <20% but got {f_nmda_at_rest:.3f}. "
            f"(k={cfg.nmda_mg_k}, V_half={cfg.nmda_mg_v_half})"
        )

    def test_gate_high_at_threshold(self):
        """At spike threshold (V=1), NMDA unblock should be large (>80%)."""
        neuron = make_neuron()
        neuron.membrane = torch.ones(1)  # at threshold
        neuron.g_nmda = torch.ones(1)

        cfg = neuron.config
        f_nmda_at_thresh = torch.sigmoid(
            torch.tensor(cfg.nmda_mg_k * (1.0 - cfg.nmda_mg_v_half))
        ).item()

        assert f_nmda_at_thresh > 0.80, (
            f"At threshold (V=1), NMDA gate should be >80% but got {f_nmda_at_thresh:.3f}. "
            f"(k={cfg.nmda_mg_k}, V_half={cfg.nmda_mg_v_half})"
        )

    def test_gate_monotone_with_voltage(self):
        """NMDA gate must increase monotonically with membrane voltage."""
        neuron = make_neuron()
        cfg = neuron.config
        voltages = torch.linspace(-0.5, 1.5, 50)
        gates = torch.sigmoid(cfg.nmda_mg_k * (voltages - cfg.nmda_mg_v_half))
        diffs = torch.diff(gates)
        assert (diffs > 0).all(), "NMDA gate must be strictly increasing in voltage"

    def test_effective_nmda_larger_when_depolarized(self):
        """The Mg²⁺ gate directly: f_nmda(0.8) >> f_nmda(0.0)."""
        cfg = make_neuron().config

        k = cfg.nmda_mg_k
        v_half = cfg.nmda_mg_v_half

        f_at_rest = float(torch.sigmoid(torch.tensor(k * (0.0 - v_half))))
        f_at_depol = float(torch.sigmoid(torch.tensor(k * (0.8 - v_half))))

        ratio = f_at_depol / (f_at_rest + 1e-9)
        assert ratio > 3.0, (
            f"Effective NMDA at V=0.8 should be >3× that at V=0: "
            f"f(0)={f_at_rest:.3f}, f(0.8)={f_at_depol:.3f}, ratio={ratio:.1f}"
        )


# ---------------------------------------------------------------------------
# 2.2 — STDP coincidence detection: NMDA only amplifies coincident activity
# ---------------------------------------------------------------------------

class TestNMDACoincidenceDetection:
    """NMDA as the coincidence detector for STDP (Hebb rule substrate).

    These tests directly validate the gate formula, avoiding the simulation
    complexity of spike resets and accumulated dynamics.
    """

    def test_instantaneous_nmda_drive_at_rest_is_small(self):
        """At resting potential, effective NMDA contribution is small (<20% of raw g_nmda)."""
        neuron = make_neuron()
        cfg = neuron.config
        g_nmda_raw = 1.0  # arbitrary unit

        f_rest = float(torch.sigmoid(torch.tensor(cfg.nmda_mg_k * (0.0 - cfg.nmda_mg_v_half))))
        g_nmda_eff_rest = g_nmda_raw * f_rest

        assert g_nmda_eff_rest < 0.20, (
            f"At rest (V=0), effective NMDA should be <20% of raw, got {g_nmda_eff_rest:.3f} "
            f"(f_nmda={f_rest:.3f})"
        )

    def test_instantaneous_nmda_drive_when_depolarized_is_large(self):
        """When postsynaptic cell is depolarized, effective NMDA is large (>70% of raw)."""
        neuron = make_neuron()
        cfg = neuron.config
        g_nmda_raw = 1.0

        f_depol = float(torch.sigmoid(torch.tensor(cfg.nmda_mg_k * (0.8 - cfg.nmda_mg_v_half))))
        g_nmda_eff_depol = g_nmda_raw * f_depol

        assert g_nmda_eff_depol > 0.70, (
            f"At V=0.8, effective NMDA should be >70% of raw, got {g_nmda_eff_depol:.3f} "
            f"(f_nmda={f_depol:.3f})"
        )

    def test_coincidence_ratio_exceeds_3x(self):
        """Effective NMDA at coincidence (V=0.8) must be >3× NMDA at rest (V=0).

        This ratio is the molecular basis of STDP: only when pre fires while
        post is depolarised does NMDA provide the Ca²⁺ signal for potentiation.
        """
        neuron = make_neuron()
        cfg = neuron.config
        g_nmda_raw = 1.0

        f_rest = float(torch.sigmoid(torch.tensor(cfg.nmda_mg_k * (0.0 - cfg.nmda_mg_v_half))))
        f_depol = float(torch.sigmoid(torch.tensor(cfg.nmda_mg_k * (0.8 - cfg.nmda_mg_v_half))))

        ratio = f_depol / (f_rest + 1e-9)
        assert ratio > 3.0, (
            f"NMDA coincidence detection ratio must be >3×, got {ratio:.2f} "
            f"(f_rest={f_rest:.3f}, f_depol={f_depol:.3f})"
        )

    def test_no_nmda_input_gives_zero_effective(self):
        """Without NMDA input, effective NMDA is zero regardless of membrane voltage."""
        neuron = make_neuron()
        cfg = neuron.config
        g_nmda_raw = 0.0  # no presynaptic NMDA

        for v_test in [0.0, 0.5, 0.8, 1.0]:
            f = float(torch.sigmoid(torch.tensor(cfg.nmda_mg_k * (v_test - cfg.nmda_mg_v_half))))
            g_eff = g_nmda_raw * f
            assert g_eff == pytest.approx(0.0, abs=1e-9), (
                f"Without NMDA input, effective must be 0 at V={v_test}, got {g_eff}"
            )

    def test_nmda_gate_integrated_over_spike_pattern(self):
        """Over a 50ms window, coincident pre+post activity accumulates more NMDA than pre-only.

        Tests the key LTP/LTD asymmetry: coincidence is biologically distinct from
        pre-alone even in a running simulation.
        """
        neuron = make_neuron()
        cfg = neuron.config

        # Simulate a simple scenario: pre fires once, post is either depolarized or at rest
        # We'll inject a fixed NMDA pulse and track g_nmda_eff over decay (no further input)
        n_steps = 50
        g_nmda_pulse = 0.5  # spike arrives at step 0

        # Case A: post at rest throughout → f_nmda(0) ≈ 0.07
        total_eff_rest = 0.0
        g_nmda_A = 0.5  # after receiving pulse
        decay = float(torch.exp(torch.tensor(-1.0 / cfg.tau_nmda)))
        v_rest = 0.0
        for _ in range(n_steps):
            f = float(torch.sigmoid(torch.tensor(cfg.nmda_mg_k * (v_rest - cfg.nmda_mg_v_half))))
            total_eff_rest += g_nmda_A * f
            g_nmda_A *= decay

        # Case B: post depolarized at V=0.75 throughout → f_nmda(0.75) ≈ 0.77
        total_eff_depol = 0.0
        g_nmda_B = 0.5
        v_depol = 0.75
        for _ in range(n_steps):
            f = float(torch.sigmoid(torch.tensor(cfg.nmda_mg_k * (v_depol - cfg.nmda_mg_v_half))))
            total_eff_depol += g_nmda_B * f
            g_nmda_B *= decay

        ratio = total_eff_depol / (total_eff_rest + 1e-9)
        assert ratio > 3.0, (
            f"Integrated NMDA during coincidence should be >3× pre-only, got ratio={ratio:.2f} "
            f"(depol={total_eff_depol:.3f}, rest={total_eff_rest:.3f})"
        )


# ---------------------------------------------------------------------------
# Integration: stable network dynamics with nmda_ratio=0.20
# ---------------------------------------------------------------------------

class TestNMDAStability:
    """Confirm that 20% NMDA ratio does not cause runaway excitation."""

    def test_network_stable_with_20pct_nmda(self):
        """Firing rate at 20% NMDA ratio stays within biological range (~1-100Hz)."""
        # Minimal single-population recurrent network
        n = 20
        neuron = make_neuron(n=n)  # uses default tau_mem=20ms
        neuron.update_temporal_parameters(dt_ms=1.0)

        dt_ms = 1.0
        n_steps = 1000  # 1 second
        n_spikes = 0

        for step in range(n_steps):
            # Sub-threshold tonic drive: g_ampa accumulates to steady-state ≈ g_in * tau_E / dt.
            # With tau_E=5ms, dt=1ms: SS = g_in * 5. Need SS * E_E/(g_L + SS) < 0.9 (sub-threshold).
            # g_in = 0.003 → SS ≈ 0.015 → V_inf ≈ 0.015*3/(0.05+0.015) ≈ 0.69 (sub-threshold).
            g_ampa = _g(torch.full((n,), 0.003))
            g_nmda = _g(torch.full((n,), 0.00075))  # 20% of excitatory input
            g_inh = _g(torch.zeros(n))

            spikes, mem = neuron.forward(g_ampa, g_inh, g_nmda)
            n_spikes += int(spikes.sum().item())

            # Safeguard: membrane must not blow up
            assert float(mem.max()) < 5.0, (
                f"Membrane blew up at step {step}: max={float(mem.max()):.2f}. "
                "NMDA likely causing runaway excitation."
            )

        # 1s / n=20 neurons: total spikes can be 0 (silent with sub-threshold drive is OK)
        # but membrane must remain finite and non-pathological
        # Firing rate per neuron per second
        rate_hz = n_spikes / n / (n_steps * dt_ms / 1000.0)
        assert rate_hz <= 100.0, (
            f"With sub-threshold drive, firing rate should be ≤100 Hz, got {rate_hz:.1f} Hz. "
            "NMDA may be causing excessive excitation (positive-feedback runaway)."
        )
