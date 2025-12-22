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

@pytest.mark.parametrize("dopamine", [-10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
def test_striatum_handles_extreme_dopamine(dopamine, device):
    """Test striatum handles out-of-range dopamine values.

    Biological context: Dopamine should be in [0, 1] range typically,
    but system should not crash with extreme values from reward prediction
    errors or other learning scenarios.
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


# =============================================================================
# Extreme Value Tests - Hippocampus (Acetylcholine)
# =============================================================================

@pytest.mark.parametrize("acetylcholine", [-5.0, 0.0, 1.0, 5.0, 50.0])
def test_hippocampus_handles_extreme_acetylcholine(acetylcholine, device):
    """Test hippocampus handles out-of-range acetylcholine values.

    Biological context: ACh modulates encoding vs retrieval in hippocampus.
    System should degrade gracefully with extreme values rather than crash.
    """
    config = HippocampusConfig(
        n_input=40,
        n_output=20,
        device=str(device),
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


# =============================================================================
# Extreme Value Tests - Prefrontal (Norepinephrine)
# =============================================================================

@pytest.mark.parametrize("norepinephrine", [-3.0, 0.0, 1.0, 3.0, 20.0])
def test_prefrontal_handles_extreme_norepinephrine(norepinephrine, device):
    """Test PFC handles out-of-range norepinephrine values.

    Biological context: NE modulates arousal and working memory gating in PFC.
    System should handle extreme values without instability.
    """
    config = PrefrontalConfig(
        n_input=50,
        n_output=30,
        device=str(device),
    )
    pfc = Prefrontal(config)
    pfc.set_neuromodulators(norepinephrine=norepinephrine)

    input_spikes = torch.rand(50, device=device) > 0.8
    output = pfc(input_spikes)

    assert not torch.isnan(output).any(), \
        f"NaN output with norepinephrine={norepinephrine}"
    assert not torch.isinf(output.float()).any(), \
        f"Inf output with norepinephrine={norepinephrine}"


# =============================================================================
# Invalid Value Tests (NaN, Inf)
# =============================================================================

@pytest.mark.xfail(reason="TODO: Add NaN/Inf validation to set_neuromodulators()")
def test_striatum_rejects_nan_dopamine(device):
    """Test striatum rejects NaN dopamine with clear error.

    NaN should never be a valid neuromodulator value and should be caught
    early with a descriptive error message.
    """
    config = StriatumConfig(n_input=50, n_output=3, device=str(device))
    striatum = Striatum(config)

    # Should raise clear error for NaN
    with pytest.raises(
        (ValueError, AssertionError),
        match="(?i)(invalid|nan|neuromodulator|dopamine)"
    ):
        striatum.set_neuromodulators(dopamine=float('nan'))


@pytest.mark.xfail(reason="TODO: Add NaN/Inf validation to set_neuromodulators()")
def test_striatum_rejects_inf_dopamine(device):
    """Test striatum rejects Inf dopamine with clear error."""
    config = StriatumConfig(n_input=50, n_output=3, device=str(device))
    striatum = Striatum(config)

    with pytest.raises(
        (ValueError, AssertionError),
        match="(?i)(invalid|inf|neuromodulator|dopamine)"
    ):
        striatum.set_neuromodulators(dopamine=float('inf'))


@pytest.mark.xfail(reason="TODO: Add NaN/Inf validation to set_neuromodulators()")
def test_hippocampus_rejects_nan_acetylcholine(device):
    """Test hippocampus rejects NaN acetylcholine with clear error."""
    config = HippocampusConfig(
        n_input=40,
        n_output=20,
        device=str(device),
    )
    hippocampus = Hippocampus(config)

    with pytest.raises(
        (ValueError, AssertionError),
        match="(?i)(invalid|nan|neuromodulator|acetylcholine)"
    ):
        hippocampus.set_neuromodulators(acetylcholine=float('nan'))


@pytest.mark.xfail(reason="TODO: Add NaN/Inf validation to set_neuromodulators()")
def test_prefrontal_rejects_nan_norepinephrine(device):
    """Test PFC rejects NaN norepinephrine with clear error."""
    config = PrefrontalConfig(n_input=50, n_output=30, device=str(device))
    pfc = Prefrontal(config)

    with pytest.raises(
        (ValueError, AssertionError),
        match="(?i)(invalid|nan|neuromodulator|norepinephrine)"
    ):
        pfc.set_neuromodulators(norepinephrine=float('nan'))


# =============================================================================
# Multi-Modulator Interaction Tests
# =============================================================================

@pytest.mark.parametrize("dopamine,norepinephrine", [
    (0.0, 0.0),   # Both at minimum
    (1.0, 1.0),   # Both at maximum (normal operating range)
    (0.0, 1.0),   # Opposing extremes
    (1.0, 0.0),
    (5.0, 5.0),   # Both elevated (stress-like state)
    (-1.0, -1.0), # Both below range
])
def test_prefrontal_handles_combined_modulators(dopamine, norepinephrine, device):
    """Test PFC handles multiple neuromodulator interactions.

    Biological context: DA and NE interact in PFC for working memory
    and cognitive control. System should handle all combinations without
    instability or crashes.
    """
    config = PrefrontalConfig(n_input=50, n_output=30, device=str(device))
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


# =============================================================================
# Temporal Stability Tests
# =============================================================================

def test_striatum_stable_with_fluctuating_dopamine(device):
    """Test striatum remains stable with rapidly changing dopamine.

    Biological context: Dopamine fluctuates based on reward prediction
    errors in reinforcement learning. System should handle temporal
    variability without instability.
    """
    config = StriatumConfig(n_input=50, n_output=3, device=str(device))
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
    config = HippocampusConfig(
        n_input=40,
        n_output=20,
        device=str(device),
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

@pytest.mark.parametrize("modulator_value", [-100.0, 0.0, 1.0, 100.0, 1000.0])
def test_striatum_learning_stable_with_extreme_dopamine(modulator_value, device):
    """Test striatal learning doesn't diverge with extreme dopamine.

    Critical for biological plausibility: Learning should saturate or clip,
    not produce exploding gradients or infinite weight changes.
    """
    config = StriatumConfig(n_input=50, n_output=3, device=str(device))
    striatum = Striatum(config)

    # Get initial weights if accessible
    initial_weights = {}
    if hasattr(striatum, 'synaptic_weights'):
        for source, weights in striatum.synaptic_weights.items():
            initial_weights[source] = weights.clone()

    # Set extreme modulator
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


@pytest.mark.parametrize("acetylcholine", [-10.0, 0.0, 1.0, 10.0])
def test_hippocampus_learning_stable_with_extreme_acetylcholine(acetylcholine, device):
    """Test hippocampal learning remains stable with extreme acetylcholine.

    ACh modulates learning rate in hippocampus. Extreme values should
    saturate learning, not cause divergence.
    """
    config = HippocampusConfig(
        n_input=40,
        n_output=20,
        device=str(device),
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

def test_striatum_extended_run_with_extreme_dopamine(device):
    """Test striatum stability over extended run with extreme dopamine.

    Longer test (100 steps) to catch delayed instabilities that might
    not appear in short tests.
    """
    config = StriatumConfig(n_input=50, n_output=3, device=str(device))
    striatum = Striatum(config)

    # Use extreme but realistic dopamine (e.g., large RPE)
    striatum.set_neuromodulators(dopamine=5.0)

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
    """
    # Create multiple regions
    striatum = Striatum(StriatumConfig(n_input=50, n_output=3, device=str(device)))
    hippocampus = Hippocampus(HippocampusConfig(
        n_input=40, n_output=20, device=str(device)
    ))
    pfc = Prefrontal(PrefrontalConfig(n_input=50, n_output=30, device=str(device)))

    # Set different modulators for each
    striatum.set_neuromodulators(dopamine=5.0)
    hippocampus.set_neuromodulators(acetylcholine=3.0)
    pfc.set_neuromodulators(dopamine=2.0, norepinephrine=4.0)

    # Run all regions
    for _ in range(20):
        striatum_out = striatum(torch.rand(50, device=device) > 0.8)
        hippo_out = hippocampus(torch.rand(40, device=device) > 0.8)
        pfc_out = pfc(torch.rand(50, device=device) > 0.8)

        # All outputs should be valid
        assert not torch.isnan(striatum_out).any()
        assert not torch.isnan(hippo_out).any()
        assert not torch.isnan(pfc_out).any()
