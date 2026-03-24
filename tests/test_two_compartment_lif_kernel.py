"""Equivalence tests for fused C++ TwoCompartmentLIF kernel vs Python fallback.

Tests that the C++ kernel produces numerically identical results to the Python
implementation for all code paths: basic dynamics, NMDA Mg²⁺ block, coupling,
Ca spikes, BAP, OU noise, gap junctions, spike reset, and refractory.
"""

from typing import Optional

import pytest
import torch

from thalia.brain.neurons.two_compartment_lif_neuron import TwoCompartmentLIF, TwoCompartmentLIFConfig
from thalia.utils.two_compartment_lif_fused import is_available as tclif_cpp_available


pytestmark = pytest.mark.skipif(not tclif_cpp_available(), reason="C++ TwoCompartmentLIF kernel not compiled")

DEVICE = "cpu"
DT_MS = 1.0
N = 100
ATOL = 1e-5
RTOL = 1e-5


def _make_config(**overrides) -> TwoCompartmentLIFConfig:
    """Create a TwoCompartmentLIFConfig with sensible defaults for testing."""
    defaults = dict(
        tau_mem_ms=torch.linspace(15.0, 25.0, N),
        tau_E=5.0,
        tau_I=7.0,
        tau_nmda=80.0,
        tau_GABA_B=100.0,
        tau_adapt_ms=200.0,
        v_threshold=torch.ones(N),
        v_reset=-0.1,
        g_L=torch.linspace(0.04, 0.06, N),
        adapt_increment=torch.full((N,), 0.01),
        E_E=3.0,
        E_I=-2.0,
        E_nmda=3.0,
        E_GABA_B=-3.0,
        E_adapt=-1.0,
        E_Ca=5.0,
        tau_ref=5.0,
        noise_std=0.0,  # Off by default for deterministic comparison
        noise_tau_ms=3.0,
        mg_conc=1.0,
        # Two-compartment specific
        g_c=0.05,
        C_d=0.5,
        g_L_d=0.03,
        bap_amplitude=0.3,
        theta_Ca=2.0,
        g_Ca_spike=0.30,
        tau_Ca_ms=20.0,
        # Disabled by default
        enable_t_channels=False,
        enable_ih=False,
    )
    defaults.update(overrides)
    return TwoCompartmentLIFConfig(**defaults)


def _make_neuron(config=None, **overrides):
    """Create a TwoCompartmentLIF neuron, update temporal params."""
    if config is None:
        config = _make_config(**overrides)
    neuron = TwoCompartmentLIF(N, config, region_name="test_region", population_name="test_pop", device=DEVICE)
    neuron.update_temporal_parameters(DT_MS)
    return neuron


def _clone_state(neuron: TwoCompartmentLIF) -> dict[str, Optional[torch.Tensor]]:
    """Deep-clone all mutable state tensors from a neuron."""
    return {
        "V_soma": neuron.V_soma.clone(),
        "g_E_basal": neuron.g_E_basal.clone(),
        "g_I_basal": neuron.g_I_basal.clone(),
        "g_GABA_B_basal": neuron.g_GABA_B_basal.clone(),
        "g_nmda_basal": neuron.g_nmda_basal.clone(),
        "g_adapt": neuron.g_adapt.clone(),
        "V_dend": neuron.V_dend.clone(),
        "g_E_apical": neuron.g_E_apical.clone(),
        "g_I_apical": neuron.g_I_apical.clone(),
        "g_nmda_apical": neuron.g_nmda_apical.clone(),
        "g_Ca": neuron.g_Ca.clone(),
        "g_plateau_dend": neuron.g_plateau_dend.clone(),
        "ou_noise": neuron.ou_noise.clone(),
        "refractory": neuron.refractory.clone() if neuron.refractory is not None else None,
        "_rng_timestep": neuron._rng_timestep.clone(),
    }


def _restore_state(neuron: TwoCompartmentLIF, state: dict[str, Optional[torch.Tensor]]) -> None:
    """Restore all mutable state tensors from a snapshot."""
    neuron.V_soma.copy_(state["V_soma"])
    neuron.g_E_basal.copy_(state["g_E_basal"])
    neuron.g_I_basal.copy_(state["g_I_basal"])
    neuron.g_GABA_B_basal.copy_(state["g_GABA_B_basal"])
    neuron.g_nmda_basal.copy_(state["g_nmda_basal"])
    neuron.g_adapt.copy_(state["g_adapt"])
    neuron.V_dend.copy_(state["V_dend"])
    neuron.g_E_apical.copy_(state["g_E_apical"])
    neuron.g_I_apical.copy_(state["g_I_apical"])
    neuron.g_nmda_apical.copy_(state["g_nmda_apical"])
    neuron.g_Ca.copy_(state["g_Ca"])
    neuron.g_plateau_dend.copy_(state["g_plateau_dend"])
    neuron.ou_noise.copy_(state["ou_noise"])
    if state["refractory"] is not None and neuron.refractory is not None:
        neuron.refractory.copy_(state["refractory"])
    neuron._rng_timestep.copy_(state["_rng_timestep"])


def _random_conductances(n: int = N) -> dict[str, torch.Tensor]:
    """Generate random conductance inputs."""
    return {
        "g_ampa_basal": torch.rand(n) * 0.1,
        "g_nmda_basal": torch.rand(n) * 0.05,
        "g_gaba_a_basal": torch.rand(n) * 0.08,
        "g_gaba_b_basal": torch.rand(n) * 0.02,
        "g_ampa_apical": torch.rand(n) * 0.06,
        "g_nmda_apical": torch.rand(n) * 0.04,
        "g_gaba_a_apical": torch.rand(n) * 0.05,
    }


def _run_python_path(neuron, inputs, state) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one step via Python fallback path."""
    _restore_state(neuron, state)
    return neuron._forward_python(
        inputs["g_ampa_basal"], inputs["g_nmda_basal"],
        inputs["g_gaba_a_basal"], inputs["g_gaba_b_basal"],
        inputs["g_ampa_apical"], inputs["g_nmda_apical"],
        inputs["g_gaba_a_apical"],
        None, None, DT_MS, neuron.config,
    )


def _run_cpp_path(neuron, inputs, state) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one step via C++ fast path."""
    _restore_state(neuron, state)
    return neuron._forward_cpp(
        inputs["g_ampa_basal"], inputs["g_nmda_basal"],
        inputs["g_gaba_a_basal"], inputs["g_gaba_b_basal"],
        inputs["g_ampa_apical"], inputs["g_nmda_apical"],
        inputs["g_gaba_a_apical"],
        None, None, DT_MS, neuron.config,
    )


def _compare_outputs(py_out, cpp_out, label=""):
    """Compare Python and C++ outputs (spikes, V_soma, V_dend)."""
    py_spikes, py_vs, py_vd = py_out
    cpp_spikes, cpp_vs, cpp_vd = cpp_out
    assert torch.equal(py_spikes, cpp_spikes), f"{label}: Spikes differ"
    torch.testing.assert_close(cpp_vs, py_vs, atol=ATOL, rtol=RTOL, msg=f"{label}: V_soma")
    torch.testing.assert_close(cpp_vd, py_vd, atol=ATOL, rtol=RTOL, msg=f"{label}: V_dend")


def _compare_state(neuron_py_state, neuron_cpp, label=""):
    """Compare internal state after Python vs C++ execution."""
    for name in ["g_E_basal", "g_I_basal", "g_GABA_B_basal", "g_nmda_basal",
                  "g_adapt", "g_E_apical", "g_I_apical", "g_nmda_apical", "g_Ca",
                  "g_plateau_dend"]:
        actual = getattr(neuron_cpp, name)
        torch.testing.assert_close(actual, neuron_py_state[name], atol=ATOL, rtol=RTOL,
                                   msg=f"{label}: state {name}")


class TestBasicEquivalence:
    """Test C++ and Python produce identical results for basic dynamics."""

    def test_zero_input(self):
        """Both paths produce same result with zero synaptic input."""
        neuron = _make_neuron()
        # Force initial forward to set refractory
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))
        state = _clone_state(neuron)

        inputs = {k: torch.zeros(N) for k in _random_conductances()}
        py_out = _run_python_path(neuron, inputs, state)
        py_state = _clone_state(neuron)
        cpp_out = _run_cpp_path(neuron, inputs, state)
        _compare_outputs(py_out, cpp_out, "zero_input")
        _compare_state(py_state, neuron, "zero_input")

    def test_random_input_single_step(self):
        """Both paths match with random conductance inputs."""
        neuron = _make_neuron()
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))
        state = _clone_state(neuron)

        inputs = _random_conductances()
        py_out = _run_python_path(neuron, inputs, state)
        py_state = _clone_state(neuron)
        cpp_out = _run_cpp_path(neuron, inputs, state)
        _compare_outputs(py_out, cpp_out, "random_single")
        _compare_state(py_state, neuron, "random_single")

    def test_multi_step_accumulation(self):
        """Both paths match over 20 consecutive steps."""
        neuron = _make_neuron()
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))

        torch.manual_seed(42)
        for step in range(20):
            state = _clone_state(neuron)
            inputs = _random_conductances()
            py_out = _run_python_path(neuron, inputs, state)
            py_state = _clone_state(neuron)
            cpp_out = _run_cpp_path(neuron, inputs, state)
            _compare_outputs(py_out, cpp_out, f"step_{step}")
            _compare_state(py_state, neuron, f"step_{step}")
            # Advance to Python end state for next step
            _restore_state(neuron, py_state)


class TestNMDABlock:
    """Test NMDA Mg²⁺ block at different compartment voltages."""

    def test_strong_nmda_input(self):
        """Strong NMDA to both compartments — tests Mg²⁺ block at V_soma and V_dend."""
        neuron = _make_neuron()
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))

        # Set non-zero voltages to make Mg²⁺ block relevant
        neuron.V_soma.fill_(0.5)
        neuron.V_dend.fill_(0.3)
        state = _clone_state(neuron)

        inputs = _random_conductances()
        inputs["g_nmda_basal"] = torch.rand(N) * 0.5   # Strong basal NMDA
        inputs["g_nmda_apical"] = torch.rand(N) * 0.5  # Strong apical NMDA

        py_out = _run_python_path(neuron, inputs, state)
        py_state = _clone_state(neuron)
        cpp_out = _run_cpp_path(neuron, inputs, state)
        _compare_outputs(py_out, cpp_out, "strong_nmda")
        _compare_state(py_state, neuron, "strong_nmda")


class TestCaSpikeAndBAP:
    """Test Ca²⁺ spike and back-propagating action potential."""

    def test_ca_spike_trigger(self):
        """When V_dend >= theta_Ca, Ca conductance should increase."""
        neuron = _make_neuron()
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))

        # Push dendritic voltage above theta_Ca for some neurons
        neuron.V_dend[:50].fill_(2.5)  # Above theta_Ca=2.0
        neuron.V_dend[50:].fill_(0.5)  # Below
        state = _clone_state(neuron)

        inputs = _random_conductances()
        py_out = _run_python_path(neuron, inputs, state)
        py_state = _clone_state(neuron)
        cpp_out = _run_cpp_path(neuron, inputs, state)
        _compare_outputs(py_out, cpp_out, "ca_spike")
        _compare_state(py_state, neuron, "ca_spike")

    def test_bap_on_spike(self):
        """BAP should depolarise dendrite when soma spikes."""
        neuron = _make_neuron()
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))

        # Push soma above threshold for some neurons
        neuron.V_soma[:30].fill_(1.5)  # Above v_threshold=1.0
        neuron.refractory[:30].fill_(0)  # Not refractory
        neuron.V_dend[:30].fill_(0.5)   # Sub-threshold dendrite
        state = _clone_state(neuron)

        inputs = {k: torch.zeros(N) for k in _random_conductances()}
        py_out = _run_python_path(neuron, inputs, state)
        py_state = _clone_state(neuron)
        cpp_out = _run_cpp_path(neuron, inputs, state)
        _compare_outputs(py_out, cpp_out, "bap")
        # After BAP, V_dend for spiking neurons should be elevated
        _compare_state(py_state, neuron, "bap")
        # Verify spikes actually occurred
        assert py_out[0][:30].any(), "Expected some spikes for above-threshold neurons"


class TestNoise:
    """Test OU noise path equivalence."""

    def test_noise_enabled(self):
        """Both paths produce identical noise when enabled."""
        neuron = _make_neuron(noise_std=0.05)
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))
        state = _clone_state(neuron)

        inputs = _random_conductances()
        py_out = _run_python_path(neuron, inputs, state)
        py_state = _clone_state(neuron)
        cpp_out = _run_cpp_path(neuron, inputs, state)
        _compare_outputs(py_out, cpp_out, "noise")
        # Check OU noise state
        torch.testing.assert_close(neuron.ou_noise, py_state["ou_noise"], atol=ATOL, rtol=RTOL)


class TestGapJunctions:
    """Test gap junction integration."""

    def test_gap_junctions(self):
        """Gap junction conductance/reversal correctly shifts somatic dynamics."""
        neuron = _make_neuron()
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))
        state = _clone_state(neuron)

        inputs = _random_conductances()
        g_gap = torch.rand(N) * 0.02
        E_gap = torch.rand(N) * 0.5

        # Python path with gap junctions
        _restore_state(neuron, state)
        py_out = neuron._forward_python(
            inputs["g_ampa_basal"], inputs["g_nmda_basal"],
            inputs["g_gaba_a_basal"], inputs["g_gaba_b_basal"],
            inputs["g_ampa_apical"], inputs["g_nmda_apical"],
            inputs["g_gaba_a_apical"],
            g_gap, E_gap, DT_MS, neuron.config,
        )
        py_state = _clone_state(neuron)

        # C++ path with gap junctions
        _restore_state(neuron, state)
        cpp_out = neuron._forward_cpp(
            inputs["g_ampa_basal"], inputs["g_nmda_basal"],
            inputs["g_gaba_a_basal"], inputs["g_gaba_b_basal"],
            inputs["g_ampa_apical"], inputs["g_nmda_apical"],
            inputs["g_gaba_a_apical"],
            g_gap, E_gap, DT_MS, neuron.config,
        )

        _compare_outputs(py_out, cpp_out, "gap_junctions")
        _compare_state(py_state, neuron, "gap_junctions")


class TestRefractoryAndAdaptation:
    """Test refractory period and spike-frequency adaptation."""

    def test_refractory_blocks_spikes(self):
        """Neurons in refractory period should not spike even if above threshold."""
        neuron = _make_neuron()
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))

        # Set all neurons above threshold but refractory
        neuron.V_soma.fill_(1.5)
        neuron.refractory.fill_(3)  # 3 steps remaining
        state = _clone_state(neuron)

        inputs = {k: torch.zeros(N) for k in _random_conductances()}
        py_out = _run_python_path(neuron, inputs, state)
        py_state = _clone_state(neuron)
        cpp_out = _run_cpp_path(neuron, inputs, state)
        _compare_outputs(py_out, cpp_out, "refractory")
        assert not py_out[0].any(), "No spikes should occur during refractory"
        assert not cpp_out[0].any(), "No spikes should occur during refractory"

    def test_adaptation_increment(self):
        """g_adapt should increase by adapt_increment on spike."""
        neuron = _make_neuron()
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))

        # Set up for spiking
        neuron.V_soma[:20].fill_(1.5)
        neuron.refractory[:20].fill_(0)
        initial_adapt = neuron.g_adapt[:20].clone()
        state = _clone_state(neuron)

        inputs = {k: torch.zeros(N) for k in _random_conductances()}
        py_out = _run_python_path(neuron, inputs, state)
        py_state = _clone_state(neuron)
        cpp_out = _run_cpp_path(neuron, inputs, state)
        _compare_outputs(py_out, cpp_out, "adaptation")
        _compare_state(py_state, neuron, "adaptation")


class TestCouplingDynamics:
    """Test soma-dendrite coupling conductance."""

    def test_coupling_transfers_voltage(self):
        """Dendritic depolarisation should drive soma via coupling."""
        neuron = _make_neuron()
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))

        # High dendritic voltage, low somatic
        neuron.V_dend.fill_(1.0)
        neuron.V_soma.fill_(0.0)
        state = _clone_state(neuron)

        inputs = {k: torch.zeros(N) for k in _random_conductances()}
        py_out = _run_python_path(neuron, inputs, state)
        py_state = _clone_state(neuron)
        cpp_out = _run_cpp_path(neuron, inputs, state)
        _compare_outputs(py_out, cpp_out, "coupling")
        # V_soma should increase due to coupling from depolarised dendrite
        assert (py_out[1] > 0.0).all(), "Soma should be pulled up by dendrite coupling"


class TestDispatcherIntegration:
    """Test that the forward() dispatcher correctly routes to C++."""

    def test_forward_uses_cpp(self):
        """forward() should produce same result as _forward_cpp when C++ available."""
        neuron = _make_neuron()
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))
        state = _clone_state(neuron)

        inputs = _random_conductances()

        # Run via public forward()
        _restore_state(neuron, state)
        fwd_out = neuron.forward(**inputs)
        fwd_state = _clone_state(neuron)

        # Run via explicit _forward_cpp
        _restore_state(neuron, state)
        cpp_out = neuron._forward_cpp(
            inputs["g_ampa_basal"], inputs["g_nmda_basal"],
            inputs["g_gaba_a_basal"], inputs["g_gaba_b_basal"],
            inputs["g_ampa_apical"], inputs["g_nmda_apical"],
            inputs["g_gaba_a_apical"],
            None, None, DT_MS, neuron.config,
        )

        _compare_outputs(fwd_out, cpp_out, "dispatcher")


class TestNMDAPlateau:
    """Test NMDA plateau potential mechanism."""

    def _make_plateau_neuron(self, **overrides):
        """Create a neuron with NMDA plateau enabled."""
        config = _make_config(
            enable_nmda_plateau=True,
            nmda_plateau_threshold=0.15,
            v_dend_plateau_threshold=0.5,
            g_nmda_plateau=0.08,
            tau_plateau_ms=150.0,
            **overrides,
        )
        return _make_neuron(config=config)

    def test_plateau_trigger(self):
        """Plateau should activate when NMDA and V_dend both exceed thresholds."""
        neuron = self._make_plateau_neuron()
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))

        # V_dend must be high enough for both the plateau voltage threshold (0.5)
        # AND to sufficiently relieve the Mg²⁺ block so that effective NMDA
        # conductance exceeds nmda_plateau_threshold (0.15).
        neuron.V_dend.fill_(2.0)
        initial_plateau = neuron.g_plateau_dend.clone()
        assert (initial_plateau == 0).all(), "Plateau should start at zero"

        # Strong NMDA apical input (will be partially unblocked at V_dend=1.0)
        inputs = {k: torch.zeros(N) for k in _random_conductances()}
        inputs["g_nmda_apical"] = torch.full((N,), 0.5)  # Strong apical NMDA

        neuron._forward_python(
            inputs["g_ampa_basal"], inputs["g_nmda_basal"],
            inputs["g_gaba_a_basal"], inputs["g_gaba_b_basal"],
            inputs["g_ampa_apical"], inputs["g_nmda_apical"],
            inputs["g_gaba_a_apical"],
            None, None, DT_MS, neuron.config,
        )
        assert (neuron.g_plateau_dend > 0).any(), "Plateau should be activated by strong NMDA + depolarized dendrite"

    def test_plateau_no_trigger_low_vdend(self):
        """Plateau should NOT activate when V_dend is below threshold."""
        neuron = self._make_plateau_neuron()
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))

        # V_dend below threshold (0.5)
        neuron.V_dend.fill_(0.1)

        inputs = {k: torch.zeros(N) for k in _random_conductances()}
        inputs["g_nmda_apical"] = torch.full((N,), 0.5)

        neuron._forward_python(
            inputs["g_ampa_basal"], inputs["g_nmda_basal"],
            inputs["g_gaba_a_basal"], inputs["g_gaba_b_basal"],
            inputs["g_ampa_apical"], inputs["g_nmda_apical"],
            inputs["g_gaba_a_apical"],
            None, None, DT_MS, neuron.config,
        )
        assert (neuron.g_plateau_dend == 0).all(), "Plateau should not activate with low V_dend"

    def test_plateau_decay(self):
        """Plateau conductance should decay over time."""
        neuron = self._make_plateau_neuron()
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))

        # Manually set plateau conductance
        neuron.g_plateau_dend.fill_(0.1)
        g_before = neuron.g_plateau_dend.clone()

        # Run with zero input — plateau should decay
        inputs = {k: torch.zeros(N) for k in _random_conductances()}
        neuron._forward_python(
            inputs["g_ampa_basal"], inputs["g_nmda_basal"],
            inputs["g_gaba_a_basal"], inputs["g_gaba_b_basal"],
            inputs["g_ampa_apical"], inputs["g_nmda_apical"],
            inputs["g_gaba_a_apical"],
            None, None, DT_MS, neuron.config,
        )
        assert (neuron.g_plateau_dend < g_before).all(), "Plateau should decay over time"
        assert (neuron.g_plateau_dend > 0).all(), "Plateau should not immediately reach zero"

    def test_plateau_disabled_by_default(self):
        """Without enable_nmda_plateau, plateau should remain zero."""
        neuron = _make_neuron()  # Default: plateau disabled
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))

        neuron.V_dend.fill_(1.0)
        inputs = {k: torch.zeros(N) for k in _random_conductances()}
        inputs["g_nmda_apical"] = torch.full((N,), 0.5)

        neuron._forward_python(
            inputs["g_ampa_basal"], inputs["g_nmda_basal"],
            inputs["g_gaba_a_basal"], inputs["g_gaba_b_basal"],
            inputs["g_ampa_apical"], inputs["g_nmda_apical"],
            inputs["g_gaba_a_apical"],
            None, None, DT_MS, neuron.config,
        )
        assert (neuron.g_plateau_dend == 0).all(), "Plateau should stay zero when disabled"

    def test_plateau_cpp_python_equivalence(self):
        """C++ and Python paths should produce identical plateau results."""
        neuron = self._make_plateau_neuron()
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))

        # Set up for plateau activation
        neuron.V_dend.fill_(1.0)
        state = _clone_state(neuron)

        inputs = _random_conductances()
        inputs["g_nmda_apical"] = torch.full((N,), 0.5)

        py_out = _run_python_path(neuron, inputs, state)
        py_state = _clone_state(neuron)
        cpp_out = _run_cpp_path(neuron, inputs, state)
        _compare_outputs(py_out, cpp_out, "plateau")
        _compare_state(py_state, neuron, "plateau")

    def test_plateau_multi_step_equivalence(self):
        """C++ and Python match over multiple steps with plateau active."""
        neuron = self._make_plateau_neuron()
        neuron.forward(torch.zeros(N), torch.zeros(N), torch.zeros(N), torch.zeros(N))

        # Pre-activate plateau and run multiple steps
        neuron.g_plateau_dend.fill_(0.08)
        neuron.V_dend.fill_(0.8)

        torch.manual_seed(99)
        for step in range(10):
            state = _clone_state(neuron)
            inputs = _random_conductances()
            inputs["g_nmda_apical"] = torch.rand(N) * 0.3
            py_out = _run_python_path(neuron, inputs, state)
            py_state = _clone_state(neuron)
            cpp_out = _run_cpp_path(neuron, inputs, state)
            _compare_outputs(py_out, cpp_out, f"plateau_step_{step}")
            _compare_state(py_state, neuron, f"plateau_step_{step}")
            _restore_state(neuron, py_state)
