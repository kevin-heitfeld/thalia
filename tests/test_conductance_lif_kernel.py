"""Numerical equivalence tests: ConductanceLIF C++ kernel vs Python implementation.

Runs both paths on identical state and inputs, verifies they produce the same results.
"""

from __future__ import annotations

import pytest
import torch

from thalia.brain.neurons.conductance_lif_neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.utils.conductance_lif_fused import is_available as clif_cpp_available


pytestmark = pytest.mark.skipif(
    not clif_cpp_available(),
    reason="ConductanceLIF C++ kernel not available (compilation failed)",
)


def _make_neuron(
    n: int = 100,
    *,
    noise_std: float = 0.08,
    enable_ih: bool = False,
    enable_t_channels: bool = False,
    adapt_increment: float = 0.02,
    tau_E: float = 5.0,
    tau_I: float = 10.0,
    dt_ms: float = 1.0,
) -> ConductanceLIF:
    """Create a ConductanceLIF instance with specified parameters."""
    config = ConductanceLIFConfig(
        tau_mem_ms=20.0,
        v_reset=0.0,
        v_threshold=1.0,
        tau_ref=5.0,
        g_L=0.05,
        E_E=3.0,
        E_I=-0.5,
        tau_E=tau_E,
        tau_I=tau_I,
        tau_nmda=100.0,
        E_nmda=3.0,
        tau_GABA_B=400.0,
        E_GABA_B=-0.8,
        noise_std=noise_std,
        noise_tau_ms=3.0,
        adapt_increment=adapt_increment,
        enable_ih=enable_ih,
        enable_t_channels=enable_t_channels,
    )
    neuron = ConductanceLIF(
        n_neurons=n,
        config=config,
        region_name="test_region",
        population_name="test_pop",
        device="cpu",
    )
    neuron.update_temporal_parameters(dt_ms)
    # Initialize refractory (normally done on first forward)
    neuron.refractory = (neuron._u_refractory_init * neuron.tau_ref_per_neuron / dt_ms).to(torch.int32)
    return neuron


def _clone_state(neuron: ConductanceLIF) -> dict[str, torch.Tensor]:
    """Deep-clone all mutable state tensors from a neuron."""
    state = {
        "V_soma": neuron.V_soma.clone(),
        "g_E": neuron.g_E.clone(),
        "g_I": neuron.g_I.clone(),
        "g_nmda": neuron.g_nmda.clone(),
        "g_GABA_B": neuron.g_GABA_B.clone(),
        "g_adapt": neuron.g_adapt.clone(),
        "ou_noise": neuron.ou_noise.clone(),
        "refractory": neuron.refractory.clone(),
        "_rng_timestep": neuron._rng_timestep.clone(),
    }
    if neuron.h_gate is not None:
        state["h_gate"] = neuron.h_gate.clone()
    if neuron.h_T is not None:
        state["h_T"] = neuron.h_T.clone()
    return state


def _restore_state(neuron: ConductanceLIF, state: dict[str, torch.Tensor]) -> None:
    """Restore mutable state tensors to a neuron."""
    neuron.V_soma.copy_(state["V_soma"])
    neuron.g_E.copy_(state["g_E"])
    neuron.g_I.copy_(state["g_I"])
    neuron.g_nmda.copy_(state["g_nmda"])
    neuron.g_GABA_B.copy_(state["g_GABA_B"])
    neuron.g_adapt.copy_(state["g_adapt"])
    neuron.ou_noise.copy_(state["ou_noise"])
    neuron.refractory.copy_(state["refractory"])
    neuron._rng_timestep.copy_(state["_rng_timestep"])
    if "h_gate" in state:
        neuron.h_gate.copy_(state["h_gate"])
    if "h_T" in state:
        neuron.h_T.copy_(state["h_T"])


def _run_both_paths(
    neuron: ConductanceLIF,
    g_ampa: torch.Tensor | None = None,
    g_nmda: torch.Tensor | None = None,
    g_gaba_a: torch.Tensor | None = None,
    g_gaba_b: torch.Tensor | None = None,
    g_gap: torch.Tensor | None = None,
    E_gap: torch.Tensor | None = None,
) -> tuple[dict, dict]:
    """Run both C++ and Python paths from identical state, return results."""
    config = neuron.config
    dt_ms = neuron._dt_ms
    _z = neuron._zeros

    g_ampa_in   = g_ampa   if g_ampa   is not None else _z
    g_nmda_in   = g_nmda   if g_nmda   is not None else _z
    g_gaba_a_in = g_gaba_a if g_gaba_a is not None else _z
    g_gaba_b_in = g_gaba_b if g_gaba_b is not None else _z

    # Save state before running
    state_before = _clone_state(neuron)

    # ── Run C++ path ──
    spikes_cpp, V_cpp = neuron._forward_cpp(
        g_ampa_in.clone(), g_nmda_in.clone(), g_gaba_a_in.clone(), g_gaba_b_in.clone(),
        g_gap.clone() if g_gap is not None else None,
        E_gap.clone() if E_gap is not None else None,
        dt_ms, config,
    )
    cpp_result = {
        "spikes": spikes_cpp.clone(),
        "V_soma": neuron.V_soma.clone(),
        "g_E": neuron.g_E.clone(),
        "g_I": neuron.g_I.clone(),
        "g_nmda": neuron.g_nmda.clone(),
        "g_GABA_B": neuron.g_GABA_B.clone(),
        "g_adapt": neuron.g_adapt.clone(),
        "ou_noise": neuron.ou_noise.clone(),
        "refractory": neuron.refractory.clone(),
    }
    if neuron.h_gate is not None:
        cpp_result["h_gate"] = neuron.h_gate.clone()
    if neuron.h_T is not None:
        cpp_result["h_T"] = neuron.h_T.clone()

    # ── Restore state and run Python path ──
    _restore_state(neuron, state_before)

    additional = neuron._get_additional_conductances()
    spikes_py, V_py = neuron._forward_python(
        g_ampa_in.clone(), g_nmda_in.clone(), g_gaba_a_in.clone(), g_gaba_b_in.clone(),
        g_gap.clone() if g_gap is not None else None,
        E_gap.clone() if E_gap is not None else None,
        dt_ms, config, additional,
    )
    py_result = {
        "spikes": spikes_py.clone(),
        "V_soma": neuron.V_soma.clone(),
        "g_E": neuron.g_E.clone(),
        "g_I": neuron.g_I.clone(),
        "g_nmda": neuron.g_nmda.clone(),
        "g_GABA_B": neuron.g_GABA_B.clone(),
        "g_adapt": neuron.g_adapt.clone(),
        "ou_noise": neuron.ou_noise.clone(),
        "refractory": neuron.refractory.clone(),
    }
    if neuron.h_gate is not None:
        py_result["h_gate"] = neuron.h_gate.clone()
    if neuron.h_T is not None:
        py_result["h_T"] = neuron.h_T.clone()

    return cpp_result, py_result


def _assert_close(cpp: dict, py: dict, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    """Assert all tensors in both result dicts are close."""
    for key in py:
        c = cpp[key]
        p = py[key]
        if c.dtype == torch.bool:
            assert torch.equal(c, p), f"{key}: spikes differ — C++ has {c.sum().item()}, Python has {p.sum().item()}"
        elif c.dtype == torch.int32:
            assert torch.equal(c, p), f"{key}: int tensors differ"
        else:
            if not torch.allclose(c, p, atol=atol, rtol=rtol):
                diff = (c - p).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                pytest.fail(
                    f"{key}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} "
                    f"(atol={atol}, rtol={rtol})"
                )


# =========================================================================
# Test cases
# =========================================================================


class TestConductanceLIFKernelEquivalence:
    """Test that C++ kernel produces identical results to Python implementation."""

    def test_basic_no_input(self):
        """No synaptic input — pure decay step."""
        neuron = _make_neuron(50, noise_std=0.0)
        cpp, py = _run_both_paths(neuron)
        _assert_close(cpp, py)

    def test_excitatory_input_only(self):
        """AMPA excitation drives neurons toward threshold."""
        neuron = _make_neuron(100, noise_std=0.0)
        g_ampa = torch.rand(100) * 0.5
        cpp, py = _run_both_paths(neuron, g_ampa=g_ampa)
        _assert_close(cpp, py)

    def test_all_four_conductances(self):
        """All receptor types active simultaneously."""
        neuron = _make_neuron(100, noise_std=0.0)
        cpp, py = _run_both_paths(
            neuron,
            g_ampa=torch.rand(100) * 0.3,
            g_nmda=torch.rand(100) * 0.1,
            g_gaba_a=torch.rand(100) * 0.2,
            g_gaba_b=torch.rand(100) * 0.05,
        )
        _assert_close(cpp, py)

    def test_with_noise(self):
        """OU noise path produces identical results."""
        neuron = _make_neuron(200, noise_std=0.08)
        g_ampa = torch.rand(200) * 0.3
        cpp, py = _run_both_paths(neuron, g_ampa=g_ampa)
        _assert_close(cpp, py)

    def test_with_adaptation(self):
        """Spike-frequency adaptation after sufficient excitation."""
        neuron = _make_neuron(50, noise_std=0.0, adapt_increment=0.05)
        # Drive neurons hard to trigger spikes
        g_ampa = torch.ones(50) * 2.0
        cpp, py = _run_both_paths(neuron, g_ampa=g_ampa)
        _assert_close(cpp, py)

    def test_with_ih(self):
        """I_h (HCN pacemaker) enabled."""
        neuron = _make_neuron(50, noise_std=0.0, enable_ih=True)
        # Start hyperpolarized to activate I_h
        neuron.V_soma.fill_(-0.4)
        cpp, py = _run_both_paths(neuron)
        _assert_close(cpp, py)

    def test_with_t_channels(self):
        """T-type Ca2+ channels enabled."""
        neuron = _make_neuron(50, noise_std=0.0, enable_t_channels=True)
        # Hyperpolarize to de-inactivate T-channels
        neuron.V_soma.fill_(-0.5)
        cpp, py = _run_both_paths(neuron)
        _assert_close(cpp, py)

    def test_with_ih_and_t_channels(self):
        """Both I_h and T-channels enabled (thalamic relay neuron)."""
        neuron = _make_neuron(50, noise_std=0.0, enable_ih=True, enable_t_channels=True)
        neuron.V_soma.fill_(-0.3)
        g_gaba_a = torch.rand(50) * 0.3  # Inhibition to hyperpolarize
        cpp, py = _run_both_paths(neuron, g_gaba_a=g_gaba_a)
        _assert_close(cpp, py)

    def test_gap_junctions(self):
        """Gap junction inputs."""
        neuron = _make_neuron(50, noise_std=0.0)
        g_gap = torch.rand(50) * 0.1
        E_gap = torch.randn(50) * 0.2  # Dynamic reversal
        cpp, py = _run_both_paths(neuron, g_gap=g_gap, E_gap=E_gap)
        _assert_close(cpp, py)

    def test_multi_step_consistency(self):
        """Run multiple steps and verify state divergence stays zero."""
        neuron = _make_neuron(100, noise_std=0.08)
        for step in range(20):
            g_ampa = torch.rand(100) * 0.3 * (1.0 if step % 3 == 0 else 0.0)
            g_gaba_a = torch.rand(100) * 0.2 * (1.0 if step % 2 == 0 else 0.0)
            cpp, py = _run_both_paths(neuron, g_ampa=g_ampa, g_gaba_a=g_gaba_a)
            _assert_close(cpp, py, atol=1e-4, rtol=1e-4)

    def test_refractory_neurons(self):
        """Neurons in refractory period should not spike."""
        neuron = _make_neuron(50, noise_std=0.0)
        # Set some neurons as refractory
        neuron.refractory[:20] = 3  # 3 steps remaining
        # Drive above threshold
        g_ampa = torch.ones(50) * 5.0
        cpp, py = _run_both_paths(neuron, g_ampa=g_ampa)
        _assert_close(cpp, py)

    def test_small_population(self):
        """Very small population (edge case for parallelization)."""
        neuron = _make_neuron(3, noise_std=0.08)
        g_ampa = torch.rand(3) * 0.5
        cpp, py = _run_both_paths(neuron, g_ampa=g_ampa)
        _assert_close(cpp, py)

    def test_large_population(self):
        """Larger population to test parallel_for grain boundaries."""
        neuron = _make_neuron(1000, noise_std=0.08)
        g_ampa = torch.rand(1000) * 0.3
        g_nmda = torch.rand(1000) * 0.1
        cpp, py = _run_both_paths(neuron, g_ampa=g_ampa, g_nmda=g_nmda)
        _assert_close(cpp, py)

    def test_heterogeneous_time_constants(self):
        """Per-neuron tau_mem_ms diversity."""
        config = ConductanceLIFConfig(
            tau_mem_ms=torch.linspace(5.0, 50.0, 100),
            v_reset=0.0,
            v_threshold=1.0,
            tau_ref=5.0,
            g_L=0.05,
            E_E=3.0,
            E_I=-0.5,
            tau_E=3.0,
            tau_I=3.0,
            tau_nmda=100.0,
            E_nmda=3.0,
            tau_GABA_B=400.0,
            E_GABA_B=-0.8,
            noise_std=0.0,
            noise_tau_ms=3.0,
        )
        neuron = ConductanceLIF(
            n_neurons=100, config=config,
            region_name="test", population_name="hetero",
            device="cpu",
        )
        neuron.update_temporal_parameters(1.0)
        neuron.refractory = (neuron._u_refractory_init * neuron.tau_ref_per_neuron / 1.0).to(torch.int32)
        g_ampa = torch.rand(100) * 0.5
        cpp, py = _run_both_paths(neuron, g_ampa=g_ampa)
        _assert_close(cpp, py)
