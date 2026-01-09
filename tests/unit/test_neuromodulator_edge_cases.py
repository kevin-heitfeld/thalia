"""
Edge case tests for neuromodulator handling.

Tests that regions robustly handle extreme and invalid neuromodulator values:
- Out-of-range values (negative, very large)
- NaN and Inf values
- Multi-modulator interactions
- Temporal stability with fluctuating values
- Learning stability

These tests are critical for biological plausibility and production robustness
in reinforcement learning scenarios where neuromodulators can fluctuate.

Author: Thalia Project
Date: December 22, 2025
Priority: P0 (Critical)
"""

import pytest
import torch

from thalia.regions.striatum import Striatum, StriatumConfig
from thalia.regions.hippocampus import Hippocampus, HippocampusConfig
from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


# =============================================================================
# Extreme Value Tests - Striatum (Dopamine)
# =============================================================================

@pytest.mark.parametrize("dopamine", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_striatum_handles_valid_dopamine_range(dopamine, device):
    """Test striatum handles valid dopamine values within biological range.

    Biological context: Dopamine should be in [-2, 2] range for biological
    plausibility. Values within this range should work without issues.
    """
    config = StriatumConfig(
        n_input=50,
        n_output=3,
        population_coding=True,
        neurons_per_action=4,
        device=str(device),
    )
    striatum = Striatum(config)
    striatum.set_neuromodulators(dopamine=dopamine)

    input_spikes = torch.rand(50, device=device) > 0.8

    # Should not crash
    output = striatum(input_spikes)

    # Should not produce NaN or Inf
    assert not torch.isnan(output).any(), f"NaN output with dopamine={dopamine}"
    assert not torch.isinf(output.float()).any(), f"Inf output with dopamine={dopamine}"

    # Output shape should be valid
    expected_size = config.n_output * config.neurons_per_action
    assert output.shape[0] == expected_size, \
        f"Expected output size {expected_size}, got {output.shape[0]}"


@pytest.mark.parametrize("dopamine", [-10.0, -3.0, 3.0, 10.0, 100.0])
def test_striatum_rejects_extreme_dopamine(dopamine, device):
    """Test striatum rejects out-of-range dopamine values.

    Extreme values outside [-2, 2] indicate upstream bugs or configuration
    errors and should be caught early with validation.
    """
    config = StriatumConfig(
        n_input=50,
        n_output=3,
        population_coding=True,
        neurons_per_action=4,
        device=str(device),
    )
    striatum = Striatum(config)

    # Should raise ValueError for out-of-range values
    with pytest.raises(ValueError, match="(?i)(invalid|dopamine|range)"):
        striatum.set_neuromodulators(dopamine=dopamine)


# =============================================================================
# Extreme Value Tests - Hippocampus (Acetylcholine)
# =============================================================================

@pytest.mark.parametrize("acetylcholine", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_hippocampus_handles_valid_acetylcholine_range(acetylcholine, device):
    """Test hippocampus handles valid acetylcholine values within biological range.

    Biological context: ACh modulates encoding vs retrieval in hippocampus.
    Values within [0, 2] range should work without issues.
    """
    config = HippocampusConfig(input_size=40, ca1_size=20, device=str(device),
    )
    hippocampus = Hippocampus(config)
    hippocampus.set_neuromodulators(acetylcholine=acetylcholine)

    input_spikes = torch.rand(40, device=device) > 0.8
    output = hippocampus(input_spikes)

    # Robustness checks
    assert not torch.isnan(output).any(), \
        f"NaN output with acetylcholine={acetylcholine}"
    assert not torch.isinf(output.float()).any(), \
        f"Inf output with acetylcholine={acetylcholine}"
    assert output.shape[0] == hippocampus.config.n_output


@pytest.mark.parametrize("acetylcholine", [-5.0, -1.0, 3.0, 5.0, 50.0])
def test_hippocampus_rejects_extreme_acetylcholine(acetylcholine, device):
    """Test hippocampus rejects out-of-range acetylcholine values.

    Biological context: ACh should be in [0, 2] range for biological plausibility.
    Values outside this range should be rejected early.
    """
    config = HippocampusConfig(input_size=40, ca1_size=20, device=str(device),
    )
    hippocampus = Hippocampus(config)

    # Should raise ValueError for out-of-range values
    with pytest.raises(ValueError, match="(?i)(invalid|acetylcholine|range)"):
        hippocampus.set_neuromodulators(acetylcholine=acetylcholine)


# =============================================================================
# Extreme Value Tests - Prefrontal (Norepinephrine)
# =============================================================================

@pytest.mark.parametrize("norepinephrine", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_prefrontal_handles_valid_norepinephrine_range(norepinephrine, device):
    """Test PFC handles valid norepinephrine values within biological range.

    Biological context: NE modulates arousal and working memory gating in PFC.
    Values within [0, 2] range should work without issues.
    """
    config = PrefrontalConfig(input_size=30, n_neurons=15, device=str(device),
    )
    prefrontal = Prefrontal(config)
    prefrontal.set_neuromodulators(norepinephrine=norepinephrine)

    input_spikes = torch.rand(30, device=device) > 0.8
    output = prefrontal(input_spikes)

    # Check stability
    assert not torch.isnan(output).any(), \
        f"NaN output with norepinephrine={norepinephrine}"
    assert not torch.isinf(output.float()).any(), \
        f"Inf output with norepinephrine={norepinephrine}"
    assert output.shape[0] == prefrontal.config.n_output


@pytest.mark.parametrize("norepinephrine", [-3.0, -1.0, 3.0, 5.0, 20.0])
def test_prefrontal_rejects_extreme_norepinephrine(norepinephrine, device):
    """Test PFC rejects out-of-range norepinephrine values.

    Biological context: NE should be in [0, 2] range for biological plausibility.
    Values outside this range should be rejected early.
    """
    config = PrefrontalConfig(input_size=30, n_neurons=15, device=str(device),
    )
    prefrontal = Prefrontal(config)

    # Should raise ValueError for out-of-range values
    with pytest.raises(ValueError, match="(?i)(invalid|norepinephrine|range)"):
        prefrontal.set_neuromodulators(norepinephrine=norepinephrine)


# =============================================================================
# Invalid Value Tests (NaN, Inf)
# =============================================================================

def test_striatum_rejects_nan_dopamine(device):
    """Test striatum rejects NaN dopamine with clear error.

    NaN should never be a valid neuromodulator value and should be caught
    early with a descriptive error message.
    """
    config = StriatumConfig(n_actions=3, neurons_per_action=10, input_sources={'default': 50}, device=str(device))
    striatum = Striatum(config)

    # Should raise clear error for NaN
    with pytest.raises(
        ValueError,
        match="(?i)(invalid|nan|neuromodulator|dopamine)"
    ):
        striatum.set_neuromodulators(dopamine=float('nan'))


def test_striatum_rejects_inf_dopamine(device):
    """Test striatum rejects Inf dopamine with clear error."""
    config = StriatumConfig(n_actions=3, neurons_per_action=10, input_sources={'default': 50}, device=str(device))
    striatum = Striatum(config)

    with pytest.raises(
        ValueError,
        match="(?i)(invalid|inf|neuromodulator|dopamine)"
    ):
        striatum.set_neuromodulators(dopamine=float('inf'))


def test_hippocampus_rejects_nan_acetylcholine(device):
    """Test hippocampus rejects NaN acetylcholine with clear error."""
    config = HippocampusConfig(input_size=40, ca1_size=20, device=str(device),
    )
    hippocampus = Hippocampus(config)

    with pytest.raises(
        ValueError,
        match="(?i)(invalid|nan|neuromodulator|acetylcholine)"
    ):
        hippocampus.set_neuromodulators(acetylcholine=float('nan'))


def test_prefrontal_rejects_nan_norepinephrine(device):
    """Test PFC rejects NaN norepinephrine with clear error."""
    config = PrefrontalConfig(input_size=50, n_neurons=30, device=str(device))
    pfc = Prefrontal(config)

    with pytest.raises(
        ValueError,
        match="(?i)(invalid|nan|neuromodulator|norepinephrine)"
    ):
        pfc.set_neuromodulators(norepinephrine=float('nan'))


# =============================================================================
# Multi-Modulator Interaction Tests
# =============================================================================

@pytest.mark.parametrize("dopamine,norepinephrine", [
    (0.0, 0.0),   # Both at minimum
    (1.0, 1.0),   # Both at standard level
    (2.0, 2.0),   # Both at maximum valid range
    (0.0, 2.0),   # Opposing extremes within range
    (2.0, 0.0),
    (-1.0, 1.0),  # DA negative (punishment), NE elevated
    (1.0, 0.5),   # Mixed moderate levels
])
def test_prefrontal_handles_valid_combined_modulators(dopamine, norepinephrine, device):
    """Test PFC handles multiple neuromodulator interactions within valid ranges.

    Biological context: DA and NE interact in PFC for working memory
    and cognitive control. System should handle all valid combinations.
    """
    config = PrefrontalConfig(input_size=50, n_neurons=30, device=str(device))
    pfc = Prefrontal(config)

    pfc.set_neuromodulators(
        dopamine=dopamine,
        norepinephrine=norepinephrine,
    )

    input_spikes = torch.rand(50, device=device) > 0.8
    output = pfc(input_spikes)

    # Robustness with combined modulators
    assert not torch.isnan(output).any(), \
        f"NaN with DA={dopamine}, NE={norepinephrine}"
    assert not torch.isinf(output.float()).any(), \
        f"Inf with DA={dopamine}, NE={norepinephrine}"


@pytest.mark.parametrize("dopamine,norepinephrine", [
    (5.0, 5.0),   # Both elevated beyond range
    (-3.0, -1.0), # Both below range
    (10.0, 0.0),  # DA way too high
    (0.0, 10.0),  # NE way too high
])
def test_prefrontal_rejects_invalid_combined_modulators(dopamine, norepinephrine, device):
    """Test PFC rejects invalid combinations of neuromodulators.

    Should reject any combination where at least one modulator is out of range.
    """
    config = PrefrontalConfig(input_size=50, n_neurons=30, device=str(device))
    pfc = Prefrontal(config)

    # Should raise ValueError for any out-of-range value
    with pytest.raises(ValueError, match="(?i)(invalid|dopamine|norepinephrine|range)"):
        pfc.set_neuromodulators(
            dopamine=dopamine,
            norepinephrine=norepinephrine,
        )


# =============================================================================
# Temporal Stability Tests
# =============================================================================

def test_striatum_stable_with_fluctuating_dopamine(device):
    """Test striatum remains stable with rapidly changing dopamine.

    Biological context: Dopamine fluctuates based on reward prediction
    errors in reinforcement learning. System should handle temporal
    variability without instability.
    """
    config = StriatumConfig(n_actions=3, neurons_per_action=10, input_sources={'default': 50}, device=str(device))
    striatum = Striatum(config)

    input_spikes = torch.rand(50, device=device) > 0.8

    # Vary dopamine across timesteps (simulating RPE fluctuations)
    dopamine_sequence = [0.0, 1.0, 0.5, -0.5, 2.0, 0.3, 0.8, 1.2, 0.1]

    for t, da in enumerate(dopamine_sequence):
        striatum.set_neuromodulators(dopamine=da)
        output = striatum(input_spikes)

        # Should not crash or produce invalid outputs
        assert not torch.isnan(output).any(), \
            f"NaN at timestep {t} with dopamine={da}"
        assert not torch.isinf(output.float()).any(), \
            f"Inf at timestep {t} with dopamine={da}"


def test_hippocampus_stable_with_fluctuating_acetylcholine(device):
    """Test hippocampus remains stable with changing acetylcholine.

    Biological context: ACh levels change during encoding vs retrieval
    phases. System should handle mode switching without instability.
    """
    config = HippocampusConfig(input_size=40, ca1_size=20, device=str(device),
    )
    hippocampus = Hippocampus(config)

    input_spikes = torch.rand(40, device=device) > 0.8

    # Vary ACh: high during encoding, low during retrieval
    ach_sequence = [1.0, 1.0, 0.8, 0.5, 0.2, 0.0, 0.0, 0.3, 0.9, 1.0]

    for t, ach in enumerate(ach_sequence):
        hippocampus.set_neuromodulators(acetylcholine=ach)
        output = hippocampus(input_spikes)

        assert not torch.isnan(output).any(), \
            f"NaN at timestep {t} with acetylcholine={ach}"
        assert not torch.isinf(output.float()).any(), \
            f"Inf at timestep {t} with acetylcholine={ach}"


# =============================================================================
# Learning Stability Tests
# =============================================================================

@pytest.mark.parametrize("modulator_value", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_striatum_learning_stable_with_valid_dopamine(modulator_value, device):
    """Test striatal learning doesn't diverge with valid dopamine range.

    Critical for biological plausibility: Learning should remain stable
    across the full valid dopamine range [-2, 2].
    """
    config = StriatumConfig(n_actions=3, neurons_per_action=10, input_sources={'default': 50}, device=str(device))
    striatum = Striatum(config)

    # Get initial weights if accessible
    initial_weights = {}
    if hasattr(striatum, 'synaptic_weights'):
        for source, weights in striatum.synaptic_weights.items():
            initial_weights[source] = weights.clone()

    # Set valid modulator
    striatum.set_neuromodulators(dopamine=modulator_value)

    # Run learning with strong consistent input
    input_spikes = torch.ones(50, device=device)
    for _ in range(10):
        striatum(input_spikes)

    # Check weights are still valid
    if hasattr(striatum, 'synaptic_weights'):
        for source, weights in striatum.synaptic_weights.items():
            assert not torch.isnan(weights).any(), \
                f"Learning produced NaN with dopamine={modulator_value}"
            assert not torch.isinf(weights).any(), \
                f"Learning produced Inf with dopamine={modulator_value}"
            # Weights should remain in reasonable range
            assert weights.min() >= -0.5, \
                f"Weights below valid range with dopamine={modulator_value}"
            assert weights.max() <= 1.5, \
                f"Weights above valid range with dopamine={modulator_value}"


@pytest.mark.parametrize("acetylcholine", [0.0, 0.5, 1.0, 1.5, 2.0])
def test_hippocampus_learning_stable_with_valid_acetylcholine(acetylcholine, device):
    """Test hippocampal learning remains stable with valid acetylcholine range.

    ACh modulates learning rate in hippocampus. Extreme values should
    saturate learning, not cause divergence.
    """
    config = HippocampusConfig(input_size=40, ca1_size=20, device=str(device),
    )
    hippocampus = Hippocampus(config)

    # Set extreme ACh
    hippocampus.set_neuromodulators(acetylcholine=acetylcholine)

    # Run learning
    input_spikes = torch.ones(40, device=device)
    for _ in range(10):
        output = hippocampus(input_spikes)

        # Output should remain valid throughout learning
        assert not torch.isnan(output).any(), \
            f"NaN during learning with acetylcholine={acetylcholine}"
        assert not torch.isinf(output.float()).any(), \
            f"Inf during learning with acetylcholine={acetylcholine}"

    # Check weights if accessible
    if hasattr(hippocampus, 'synaptic_weights'):
        for weights in hippocampus.synaptic_weights.values():
            assert not torch.isnan(weights).any(), "Learning produced NaN weights"
            assert not torch.isinf(weights).any(), "Learning produced Inf weights"


# =============================================================================
# Extended Stability Tests
# =============================================================================

def test_striatum_extended_run_with_valid_dopamine(device):
    """Test striatum stability over extended run with valid dopamine range.

    Longer test (100 steps) to catch delayed instabilities that might
    not appear in short tests. Uses maximum valid dopamine (2.0).
    """
    config = StriatumConfig(n_actions=3, neurons_per_action=10, input_sources={'default': 50}, device=str(device))
    striatum = Striatum(config)

    # Use maximum valid dopamine (e.g., large RPE)
    striatum.set_neuromodulators(dopamine=2.0)

    input_spikes = torch.rand(50, device=device) > 0.8

    for step in range(100):
        output = striatum(input_spikes)

        # Check every 25 steps for efficiency
        if step % 25 == 0:
            assert not torch.isnan(output).any(), \
                f"NaN at step {step}"
            assert not torch.isinf(output.float()).any(), \
                f"Inf at step {step}"

            # Check internal state if accessible
            if hasattr(striatum, 'neurons'):
                if hasattr(striatum.neurons, 'membrane') and striatum.neurons.membrane is not None:
                    membrane = striatum.neurons.membrane
                    assert not torch.isnan(membrane).any(), \
                        f"NaN in membrane at step {step}"
                    assert not torch.isinf(membrane).any(), \
                        f"Inf in membrane at step {step}"


def test_multi_region_neuromodulator_stability(device):
    """Test multiple regions with different neuromodulators simultaneously.

    Integration test: Multiple regions with different modulators should
    all remain stable when run together (simulating full brain).
    Uses maximum valid values for each neuromodulator.
    """
    # Create multiple regions
    striatum = Striatum(StriatumConfig(n_actions=3, neurons_per_action=10, input_sources={'default': 50}, device=str(device)))
    hippocampus = Hippocampus(HippocampusConfig(input_size=40, ca1_size=20, device=str(device)
    ))
    pfc = Prefrontal(PrefrontalConfig(input_size=50, n_neurons=30, device=str(device)))

    # Set different modulators for each (all within valid ranges)
    striatum.set_neuromodulators(dopamine=2.0)
    hippocampus.set_neuromodulators(acetylcholine=2.0)
    pfc.set_neuromodulators(dopamine=2.0, norepinephrine=2.0)

    # Run all regions
    for _ in range(20):
        striatum_out = striatum(torch.rand(50, device=device) > 0.8)
        hippo_out = hippocampus(torch.rand(40, device=device) > 0.8)
        pfc_out = pfc(torch.rand(50, device=device) > 0.8)

        # All outputs should be valid
        assert not torch.isnan(striatum_out).any()
        assert not torch.isnan(hippo_out).any()
        assert not torch.isnan(pfc_out).any()
