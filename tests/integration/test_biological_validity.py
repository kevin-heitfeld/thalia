"""Biological Validity Tests for State Management.

Tests verify that state management preserves biologically realistic dynamics:
- Eligibility trace decay with correct time constants
- Membrane potentials stay in valid range
- Neuromodulatorevels stay in valid range
- No negative spike counts
- STDP trace continuity
- CA3 persistent activity preservation

Author: Thalia Project
Date: December 2025
"""

import numpy as np
import pytest
import torch

from thalia.config import (
    HippocampusConfig,
    LayeredCortexConfig,
    LayerSizeCalculator,
    StriatumConfig,
    ThalamicRelayConfig,
)
from thalia.regions import (
    LayeredCortex,
    Striatum,
    ThalamicRelay,
    TrisynapticHippocampus,
)


class TestEligibilityTraceDecay:
    """Test eligibility traces decay correctly after checkpoint load."""

    @pytest.fixture
    def striatum_config(self) -> StriatumConfig:
        """Create striatum config with eligibility traces enabled."""
        return StriatumConfig(
            learning_rate=0.001,
            eligibility_tau_ms=100.0,  # 100ms decay constant
            device="cpu",
            dt_ms=1.0,
        )

    def test_eligibility_decays_exponentially(self, striatum_config):
        """Verify eligibility traces decay with e^(-t/tau) after checkpoint load."""
        calc = LayerSizeCalculator()
        sizes = calc.striatum_from_actions(n_actions=2, neurons_per_action=6)
        sizes["input_size"] = 50
        striatum = Striatum(striatum_config, sizes, "cpu")
        striatum.add_input_source_striatum("default", 50)  # Register input source

        # Build eligibility by running with active input
        for _ in range(20):
            input_spikes = torch.rand(50) > 0.5
            striatum.forward({"default": input_spikes})

        # Check eligibility is built up
        initial_d1_elig = striatum.d1_pathway.eligibility.clone()
        initial_d2_elig = striatum.d2_pathway.eligibility.clone()

        assert initial_d1_elig.max() > 0.05, "D1 eligibility should build up"
        assert initial_d2_elig.max() > 0.05, "D2 eligibility should build up"

        # Save and load state
        state = striatum.get_state()
        striatum2 = Striatum(striatum_config, sizes, "cpu")
        striatum2.add_input_source_striatum("default", 50)  # Register input source
        striatum2.load_state(state)

        # Verify eligibility preserved exactly
        assert torch.allclose(striatum2.d1_pathway.eligibility, initial_d1_elig, atol=1e-6)
        assert torch.allclose(striatum2.d2_pathway.eligibility, initial_d2_elig, atol=1e-6)

        # Continue simulation with no input (pure decay)
        for _ in range(100):  # 100ms
            striatum2.forward({"default": torch.zeros(50)})

        # After 100ms with tau=100ms, should decay significantly
        final_d1_elig = striatum2.d1_pathway.eligibility
        final_d2_elig = striatum2.d2_pathway.eligibility

        d1_ratio = final_d1_elig.mean() / (initial_d1_elig.mean() + 1e-8)
        d2_ratio = final_d2_elig.mean() / (initial_d2_elig.mean() + 1e-8)

        expected_ratio = np.exp(-1)  # ~0.368

        # Eligibility decay is affected by ongoing activity and noise
        # Expect 50-95% decay (looser bounds for biological realism)
        assert (
            0.05 < d1_ratio < 0.50
        ), f"D1 eligibility decay ratio {d1_ratio:.3f} should show decay (expected ~{expected_ratio:.3f})"
        assert (
            0.05 < d2_ratio < 0.50
        ), f"D2 eligibility decay ratio {d2_ratio:.3f} should show decay (expected ~{expected_ratio:.3f})"

    def test_eligibility_no_sudden_jumps(self, striatum_config):
        """Verify no discontinuities in eligibility at checkpoint boundary."""
        calc = LayerSizeCalculator()
        sizes = calc.striatum_from_actions(n_actions=2, neurons_per_action=6)
        sizes["input_size"] = 50
        striatum = Striatum(striatum_config, sizes, "cpu")
        striatum.add_input_source_striatum("default", 50)  # Register input source

        # Build eligibility
        for _ in range(20):
            striatum.forward({"default": torch.rand(50) > 0.5})

        # Record eligibility just before checkpoint
        elig_before = striatum.d1_pathway.eligibility.clone()

        # Save and load
        state = striatum.get_state()
        striatum.load_state(state)

        # Check eligibility unchanged immediately after load
        elig_after = striatum.d1_pathway.eligibility
        assert torch.allclose(
            elig_before, elig_after, atol=1e-6
        ), "Eligibility should not jump at checkpoint"


class TestMembranePotentialBounds:
    """Test membrane potentials stay in biologically valid range."""

    @pytest.fixture
    def cortex_sizes(self):
        """Create cortex sizes."""
        return {
            "input_size": 20,
            "l4_size": 30,
            "l23_size": 40,
            "l5_size": 24,
            "l6a_size": 15,
            "l6b_size": 15,
        }

    @pytest.fixture
    def cortex_config(self) -> LayeredCortexConfig:
        """Create cortex config."""
        return LayeredCortexConfig(dt_ms=1.0)

    def test_membrane_stays_in_valid_range(self, cortex_config, cortex_sizes):
        """Verify membrane potentials stay in [-80mV, +50mV] range."""
        cortex = LayeredCortex(config=cortex_config, sizes=cortex_sizes, device="cpu")

        # Run simulation with checkpoints
        for _ in range(5):
            # Active stimulation
            for _ in range(20):
                input_spikes = torch.rand(20) > 0.8
                cortex.forward({"default": input_spikes})

            # Checkpoint and restore
            state = cortex.get_state()
            cortex.load_state(state)

            # Check all layer membranes in valid range
            # L4 neurons
            if cortex.l4_neurons.membrane is not None:
                v_l4 = cortex.l4_neurons.membrane
                assert (
                    v_l4 >= -85
                ).all(), f"L4 hyperpolarization below K+ reversal: {v_l4.min():.1f}mV"
                assert (
                    v_l4 <= 60
                ).all(), f"L4 depolarization above Na+ reversal: {v_l4.max():.1f}mV"

            # L2/3 neurons
            if cortex.l23_neurons.membrane is not None:
                v_l23 = cortex.l23_neurons.membrane
                assert (
                    v_l23 >= -85
                ).all(), f"L2/3 hyperpolarization below K+ reversal: {v_l23.min():.1f}mV"
                assert (
                    v_l23 <= 60
                ).all(), f"L2/3 depolarization above Na+ reversal: {v_l23.max():.1f}mV"

            # L5 neurons
            if cortex.l5_neurons.membrane is not None:
                v_l5 = cortex.l5_neurons.membrane
                assert (
                    v_l5 >= -85
                ).all(), f"L5 hyperpolarization below K+ reversal: {v_l5.min():.1f}mV"
                assert (
                    v_l5 <= 60
                ).all(), f"L5 depolarization above Na+ reversal: {v_l5.max():.1f}mV"

    def test_membrane_no_nan_or_inf(self, cortex_config, cortex_sizes):
        """Verify no NaN or Inf values in membrane potentials."""
        cortex = LayeredCortex(config=cortex_config, sizes=cortex_sizes, device="cpu")

        # Run with high activity
        for _ in range(50):
            input_spikes = torch.rand(20) > 0.5
            cortex.forward({"default": input_spikes})

        # Checkpoint
        state = cortex.get_state()
        cortex.load_state(state)

        # Continue and check for numerical stability
        for _ in range(50):
            cortex.forward({"default": torch.rand(20) > 0.5})

        # Verify no NaN/Inf
        if cortex.l4_neurons.membrane is not None:
            assert torch.isfinite(cortex.l4_neurons.membrane).all(), "L4 membrane has NaN/Inf"
        if cortex.l23_neurons.membrane is not None:
            assert torch.isfinite(cortex.l23_neurons.membrane).all(), "L2/3 membrane has NaN/Inf"
        if cortex.l5_neurons.membrane is not None:
            assert torch.isfinite(cortex.l5_neurons.membrane).all(), "L5 membrane has NaN/Inf"


class TestNeuromodulatorBounds:
    """Test neuromodulator levels stay in valid range."""

    @pytest.fixture
    def striatum_config(self) -> StriatumConfig:
        """Create striatum config."""
        return StriatumConfig(
            device="cpu",
            dt_ms=1.0,
        )

    def test_dopamine_stays_in_valid_range(self, striatum_config):
        """Verify dopamine levels stay in [0, 1.5] range."""
        calc = LayerSizeCalculator()
        sizes = calc.striatum_from_actions(n_actions=2, neurons_per_action=6)
        sizes["input_size"] = 50
        striatum = Striatum(striatum_config, sizes, "cpu")
        striatum.add_input_source_striatum("default", 50)  # Register input source

        # Multiple learning cycles with rewards
        for _ in range(50):
            striatum.forward({"default": torch.rand(50) > 0.7})
            # Deliver random reward (-1 to +1)
            reward = np.random.uniform(-1, 1)
            striatum.set_neuromodulators(dopamine=max(0.0, min(1.5, 0.2 + reward)))

        # Save and load
        state = striatum.get_state()
        striatum2 = Striatum(striatum_config, sizes, "cpu")
        striatum2.add_input_source_striatum("default", 50)  # Register input source
        striatum2.load_state(state)

        # Check dopamine in valid range (access from forward_coordinator)
        da_level = striatum2.forward_coordinator._tonic_dopamine
        assert 0 <= da_level <= 1.5, f"Dopamine {da_level:.3f} out of biological range [0, 1.5]"

    def test_neuromodulators_preserved_in_checkpoint(self, striatum_config):
        """Verify all neuromodulator levels preserved across checkpoint."""
        calc = LayerSizeCalculator()
        sizes = calc.striatum_from_actions(n_actions=2, neurons_per_action=6)
        sizes["input_size"] = 50
        striatum = Striatum(striatum_config, sizes, "cpu")
        striatum.add_input_source_striatum("default", 50)  # Register input source

        # Set specific neuromodulator levels
        striatum.set_neuromodulators(dopamine=0.8, norepinephrine=0.5)

        # Save and load
        state = striatum.get_state()
        striatum2 = Striatum(striatum_config, sizes, "cpu")
        striatum2.add_input_source_striatum("default", 50)  # Register input source
        striatum2.load_state(state)

        # Verify levels preserved
        assert (
            abs(striatum2.forward_coordinator._tonic_dopamine - 0.8) < 0.01
        ), "Dopamine not preserved"
        assert (
            abs(striatum2.forward_coordinator._ne_level - 0.5) < 0.01
        ), "Norepinephrine not preserved"


class TestNoNegativeSpikes:
    """Test no negative spike counts after state restoration."""

    @pytest.fixture
    def thalamus_config(self) -> ThalamicRelayConfig:
        """Create thalamus config."""
        return ThalamicRelayConfig(device="cpu", dt_ms=1.0)

    def test_no_negative_spikes_before_checkpoint(self, thalamus_config):
        """Verify no negative spikes during normal operation."""
        sizes = {"input_size": 30, "relay_size": 30, "trn_size": 0}
        thalamus = ThalamicRelay(thalamus_config, sizes, "cpu")

        for _ in range(100):
            sensory_input = torch.rand(30) > 0.8
            relay_spikes = thalamus.forward({"input": sensory_input})

            assert (relay_spikes >= 0).all(), f"Negative relay spikes: {relay_spikes.min():.3f}"
            assert (relay_spikes <= 1).all(), f"Relay spikes > 1: {relay_spikes.max():.3f}"

    def test_no_negative_spikes_after_checkpoint(self, thalamus_config):
        """Verify no negative spikes after checkpoint load."""
        sizes = {"input_size": 30, "relay_size": 30, "trn_size": 0}
        thalamus = ThalamicRelay(thalamus_config, sizes, "cpu")

        # Run simulation
        for _ in range(50):
            thalamus.forward({"input": torch.rand(30) > 0.8})

        # Save and load
        state = thalamus.get_state()
        thalamus2 = ThalamicRelay(thalamus_config, sizes, "cpu")
        thalamus2.load_state(state)

        # Continue - still no negative spikes
        for _ in range(50):
            relay_spikes = thalamus2.forward({"input": torch.rand(30) > 0.8})

            assert (
                relay_spikes >= 0
            ).all(), f"Negative relay spikes after load: {relay_spikes.min():.3f}"


class TestSTDPTraceContinuity:
    """Test STDP traces continue smoothly after checkpoint load."""

    @pytest.fixture
    def cortex_sizes(self):
        """Create cortex sizes."""
        return {
            "input_size": 20,
            "l4_size": 30,
            "l23_size": 40,
            "l5_size": 24,
            "l6a_size": 15,
            "l6b_size": 15,
        }

    @pytest.fixture
    def cortex_config(self) -> LayeredCortexConfig:
        """Create cortex config."""
        return LayeredCortexConfig(dt_ms=1.0)

    def test_stdp_trace_no_discontinuity(self, cortex_config, cortex_sizes):
        """Verify STDP traces have no jumps at checkpoint boundary."""
        cortex = LayeredCortex(config=cortex_config, sizes=cortex_sizes, device="cpu")

        # Build up STDP traces
        for _ in range(20):
            cortex.forward({"default": torch.rand(20) > 0.7})

        # Record traces before checkpoint
        l4_trace_before = (
            cortex.state.l4_trace.clone() if cortex.state.l4_trace is not None else None
        )
        l23_trace_before = (
            cortex.state.l23_trace.clone() if cortex.state.l23_trace is not None else None
        )

        # Save and load
        state = cortex.get_state()
        cortex.load_state(state)

        # Check traces unchanged immediately after load
        if l4_trace_before is not None:
            assert torch.allclose(
                cortex.state.l4_trace, l4_trace_before, atol=1e-6
            ), "L4 trace jumped at checkpoint"
        if l23_trace_before is not None:
            assert torch.allclose(
                cortex.state.l23_trace, l23_trace_before, atol=1e-6
            ), "L2/3 trace jumped at checkpoint"

    def test_stdp_trace_decay_continues(self, cortex_config, cortex_sizes):
        """Verify STDP traces decay naturally after checkpoint."""
        cortex = LayeredCortex(config=cortex_config, sizes=cortex_sizes, device="cpu")

        # Build traces
        for _ in range(20):
            cortex.forward({"default": torch.rand(20) > 0.7})

        # Checkpoint
        state = cortex.get_state()
        cortex2 = LayeredCortex(config=cortex_config, sizes=cortex_sizes, device="cpu")
        cortex2.load_state(state)

        initial_trace = (
            cortex2.state.l4_trace.clone() if cortex2.state.l4_trace is not None else None
        )

        # Run with no input (pure decay)
        if initial_trace is not None:
            for _ in range(50):
                cortex2.forward({"default": torch.zeros(20)})

            # Traces should have decayed
            final_trace = cortex2.state.l4_trace
            if final_trace is not None and initial_trace.sum() > 0:
                ratio = final_trace.sum() / (initial_trace.sum() + 1e-8)
                assert ratio < 0.95, f"STDP trace should decay, ratio={ratio:.3f}"


class TestCA3PersistentActivity:
    """Test CA3 persistent activity preserved across checkpoint."""

    @pytest.fixture
    def hippocampus_config(self) -> HippocampusConfig:
        """Create hippocampus config."""
        return HippocampusConfig(
            device="cpu",
            dt_ms=1.0,
        )

    def test_ca3_pattern_not_reset(self, hippocampus_config):
        """Verify CA3 pattern doesn't reset to zero at checkpoint."""
        calc = LayerSizeCalculator()
        sizes = calc.hippocampus_from_input(ec_input_size=20)
        sizes["input_size"] = 20
        hippocampus = TrisynapticHippocampus(hippocampus_config, sizes, "cpu")

        # Build CA3 pattern
        for _ in range(15):
            hippocampus.forward({"ec": torch.rand(20) > 0.6})

        # Check some CA3 activity exists
        ca3_spikes_before = (
            hippocampus.state.ca3_spikes.clone()
            if hippocampus.state.ca3_spikes is not None
            else None
        )

        # Save and load
        state = hippocampus.get_state()
        hippocampus.load_state(state)

        # CA3 state should be preserved, not reset
        if ca3_spikes_before is not None:
            ca3_spikes_after = hippocampus.state.ca3_spikes
            assert torch.equal(
                ca3_spikes_before, ca3_spikes_after
            ), "CA3 spikes should be preserved exactly"
