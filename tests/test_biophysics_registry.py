"""Tests for BiophysicsRegistry: read-only introspection layer over brain populations."""

from __future__ import annotations

import torch.nn as nn

from thalia.brain.biophysics_registry import BiophysicsRegistry, _extract_population
from thalia.brain.neurons import (
    ConductanceLIF,
    ConductanceLIFConfig,
    TwoCompartmentLIF,
    TwoCompartmentLIFConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clif_config(**overrides: object) -> ConductanceLIFConfig:
    """Minimal ConductanceLIFConfig with sensible defaults."""
    defaults = dict(
        tau_mem_ms=20.0,
        v_reset=0.0,
        v_threshold=1.0,
        tau_ref=5.0,
        g_L=0.05,
        E_E=3.0,
        E_I=-0.5,
        tau_E=5.0,
        tau_I=10.0,
        tau_nmda=100.0,
        E_nmda=3.0,
        tau_GABA_B=400.0,
        E_GABA_B=-0.8,
        noise_std=0.08,
        noise_tau_ms=3.0,
    )
    defaults.update(overrides)
    return ConductanceLIFConfig(**defaults)  # type: ignore[arg-type]


def _make_tc_config(**overrides: object) -> TwoCompartmentLIFConfig:
    """Minimal TwoCompartmentLIFConfig with sensible defaults."""
    defaults = dict(
        tau_mem_ms=20.0,
        v_reset=0.0,
        v_threshold=1.0,
        tau_ref=5.0,
        g_L=0.05,
        E_E=3.0,
        E_I=-0.5,
        tau_E=5.0,
        tau_I=10.0,
        tau_nmda=100.0,
        E_nmda=3.0,
        tau_GABA_B=400.0,
        E_GABA_B=-0.8,
        noise_std=0.08,
        noise_tau_ms=3.0,
        # Two-compartment extras
        g_c=0.05,
        C_d=0.5,
        g_L_d=0.03,
        bap_amplitude=0.3,
        theta_Ca=2.0,
        g_Ca_spike=0.30,
        tau_Ca_ms=20.0,
    )
    defaults.update(overrides)
    return TwoCompartmentLIFConfig(**defaults)  # type: ignore[arg-type]


class FakeRegion(nn.Module):
    """Minimal region with real neuron populations for registry testing."""

    def __init__(self, populations: dict[str, nn.Module]) -> None:
        super().__init__()
        self.neuron_populations = nn.ModuleDict(populations)


def _build_registry_from_specs(
    specs: list[tuple[str, str, nn.Module]],
) -> BiophysicsRegistry:
    """Build a registry from (region_name, pop_name, neuron_module) triples."""
    entries = [_extract_population(r, p, n) for r, p, n in specs]
    return BiophysicsRegistry(entries)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPopulationBiophysicsExtraction:
    """Test extracting PopulationBiophysics from real neuron modules."""

    def test_extract_conductance_lif(self) -> None:
        config = _make_clif_config(tau_mem_ms=20.0, v_threshold=1.0, adapt_increment=0.02)
        neuron = ConductanceLIF(
            n_neurons=50, config=config,
            region_name="test_region", population_name="exc",
            device="cpu",
        )
        bio = _extract_population("test_region", "exc", neuron)

        assert bio.region == "test_region"
        assert bio.population == "exc"
        assert bio.n_neurons == 50
        assert bio.neuron_type == "ConductanceLIF"
        assert abs(bio.tau_mem_ms - 20.0) < 1.0  # heterogeneous → mean ≈ 20
        assert abs(bio.v_threshold - 1.0) < 0.5
        assert bio.E_E == 3.0
        assert bio.E_I == -0.5
        assert bio.tau_ref == 5.0
        # Two-compartment fields should be None
        assert bio.g_c is None
        assert bio.C_d is None

    def test_extract_two_compartment(self) -> None:
        config = _make_tc_config()
        neuron = TwoCompartmentLIF(
            n_neurons=30, config=config,
            region_name="cortex", population_name="L5_PYR",
            device="cpu",
        )
        bio = _extract_population("cortex", "L5_PYR", neuron)

        assert bio.neuron_type == "TwoCompartmentLIF"
        assert bio.n_neurons == 30
        assert bio.g_c == 0.05
        assert bio.C_d == 0.5
        assert bio.g_L_d == 0.03
        assert bio.bap_amplitude == 0.3
        assert bio.theta_Ca == 2.0

    def test_extract_with_ih_enabled(self) -> None:
        config = _make_clif_config(enable_ih=True, g_h_max=0.07)
        neuron = ConductanceLIF(
            n_neurons=20, config=config,
            region_name="thalamus", population_name="relay",
            device="cpu",
        )
        bio = _extract_population("thalamus", "relay", neuron)

        assert bio.enable_ih is True
        assert bio.g_h_max == 0.07
        assert bio.enable_t_channels is False
        assert bio.g_T is None

    def test_extract_with_t_channels(self) -> None:
        config = _make_clif_config(enable_t_channels=True, g_T=0.15)
        neuron = ConductanceLIF(
            n_neurons=20, config=config,
            region_name="thalamus", population_name="relay",
            device="cpu",
        )
        bio = _extract_population("thalamus", "relay", neuron)

        assert bio.enable_t_channels is True
        assert bio.g_T == 0.15


class TestBiophysicsRegistry:
    """Test the registry query API."""

    def _build_test_registry(self) -> BiophysicsRegistry:
        """Build a small registry with 3 populations."""
        n1 = ConductanceLIF(
            n_neurons=100, config=_make_clif_config(tau_mem_ms=20.0, v_threshold=1.0, adapt_increment=0.02),
            region_name="cortex", population_name="exc", device="cpu",
        )
        n2 = ConductanceLIF(
            n_neurons=25, config=_make_clif_config(tau_mem_ms=10.0, v_threshold=0.8, adapt_increment=0.0),
            region_name="cortex", population_name="inh", device="cpu",
        )
        n3 = TwoCompartmentLIF(
            n_neurons=40, config=_make_tc_config(tau_mem_ms=25.0, v_threshold=1.2),
            region_name="hippo", population_name="CA1", device="cpu",
        )
        return _build_registry_from_specs([
            ("cortex", "exc", n1),
            ("cortex", "inh", n2),
            ("hippo", "CA1", n3),
        ])

    def test_len_and_total_neurons(self) -> None:
        reg = self._build_test_registry()
        assert len(reg) == 3
        assert reg.total_neurons == 165  # 100 + 25 + 40

    def test_get_existing(self) -> None:
        reg = self._build_test_registry()
        bio = reg.get("cortex", "exc")
        assert bio.region == "cortex"
        assert bio.population == "exc"
        assert bio.n_neurons == 100

    def test_get_missing_raises(self) -> None:
        reg = self._build_test_registry()
        try:
            reg.get("nonexistent", "pop")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_all_populations(self) -> None:
        reg = self._build_test_registry()
        pops = reg.all_populations()
        assert len(pops) == 3
        names = [(p.region, p.population) for p in pops]
        assert ("cortex", "exc") in names
        assert ("cortex", "inh") in names
        assert ("hippo", "CA1") in names

    def test_compare(self) -> None:
        reg = self._build_test_registry()
        tau_map = reg.compare("tau_mem_ms")
        assert len(tau_map) == 3
        # Mean should be approximately the configured values
        assert abs(tau_map[("cortex", "exc")] - 20.0) < 2.0
        assert abs(tau_map[("cortex", "inh")] - 10.0) < 2.0
        assert abs(tau_map[("hippo", "CA1")] - 25.0) < 2.0

    def test_compare_optional_field(self) -> None:
        reg = self._build_test_registry()
        # g_c is only set for TwoCompartmentLIF populations
        gc_map = reg.compare("g_c")
        assert len(gc_map) == 1
        assert ("hippo", "CA1") in gc_map

    def test_populations_with_exact_match(self) -> None:
        reg = self._build_test_registry()
        tc = reg.populations_with(neuron_type="TwoCompartmentLIF")
        assert len(tc) == 1
        assert tc[0].population == "CA1"

    def test_populations_with_gt(self) -> None:
        reg = self._build_test_registry()
        big = reg.populations_with(n_neurons_gt=30)
        assert len(big) == 2  # cortex.exc=100, hippo.CA1=40

    def test_populations_with_lt(self) -> None:
        reg = self._build_test_registry()
        small = reg.populations_with(n_neurons_lt=30)
        assert len(small) == 1
        assert small[0].population == "inh"

    def test_populations_with_combined_criteria(self) -> None:
        reg = self._build_test_registry()
        result = reg.populations_with(
            neuron_type="ConductanceLIF",
            n_neurons_gte=50,
        )
        assert len(result) == 1
        assert result[0].population == "exc"

    def test_summary_table(self) -> None:
        reg = self._build_test_registry()
        table = reg.summary_table()
        assert "cortex" in table
        assert "hippo" in table
        assert "Total populations: 3" in table

    def test_repr(self) -> None:
        reg = self._build_test_registry()
        r = repr(reg)
        assert "3 populations" in r
        assert "165 neurons" in r

    def test_frozen_dataclass(self) -> None:
        """PopulationBiophysics should be immutable."""
        reg = self._build_test_registry()
        bio = reg.get("cortex", "exc")
        try:
            bio.tau_mem_ms = 999.0  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass


class TestFromBrain:
    """Test BiophysicsRegistry.from_brain with a mock brain-like object."""

    def test_from_brain_mock(self) -> None:
        """Verify from_brain correctly iterates brain.regions → neuron_populations."""
        n1 = ConductanceLIF(
            n_neurons=10, config=_make_clif_config(),
            region_name="R1", population_name="P1", device="cpu",
        )
        n2 = ConductanceLIF(
            n_neurons=20, config=_make_clif_config(tau_mem_ms=15.0),
            region_name="R2", population_name="P2", device="cpu",
        )

        class MockBrain:
            regions = nn.ModuleDict({
                "R1": FakeRegion({"P1": n1}),
                "R2": FakeRegion({"P2": n2}),
            })

        reg = BiophysicsRegistry.from_brain(MockBrain())
        assert len(reg) == 2
        assert reg.get("R1", "P1").n_neurons == 10
        assert reg.get("R2", "P2").n_neurons == 20
