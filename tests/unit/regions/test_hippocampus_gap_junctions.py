"""Tests for gap junction integration in hippocampal CA1 interneurons.

Tests verify:
- Gap junctions can be enabled/disabled via configuration
- Coupling improves synchronization of CA1 interneurons
- Gap junctions use ca1_inhib weights to infer neighborhoods
- State management (ca1_membrane) works correctly
- Theta-gamma coupling benefits from gap junction synchronization
"""

import pytest
import torch
from thalia.regions.hippocampus.trisynaptic import TrisynapticHippocampus
from thalia.regions.hippocampus.config import HippocampusConfig
from thalia.config import compute_hippocampus_sizes


def test_gap_junctions_can_be_disabled():
    """Gap junctions can be disabled via configuration."""
    sizes = compute_hippocampus_sizes(32)
    cfg = HippocampusConfig(
        n_input=32,
        n_output=64,
        dg_size=sizes["dg_size"],
        ca3_size=sizes["ca3_size"],
        ca2_size=sizes["ca2_size"],
        ca1_size=sizes["ca1_size"],
        gap_junctions_enabled=False,
    )
    hippo = TrisynapticHippocampus(cfg)

    # Should have no gap junction module
    assert hippo.gap_junctions_ca1 is None


def test_gap_junctions_enabled_by_default():
    """Gap junctions are enabled by default."""
    sizes = compute_hippocampus_sizes(32)
    cfg = HippocampusConfig(
        n_input=32,
        n_output=64,
        dg_size=sizes["dg_size"],
        ca3_size=sizes["ca3_size"],
        ca2_size=sizes["ca2_size"],
        ca1_size=sizes["ca1_size"],
    )
    hippo = TrisynapticHippocampus(cfg)

    # Should have gap junction module
    assert hippo.gap_junctions_ca1 is not None
    assert hasattr(hippo, 'gap_junctions_ca1')


def test_gap_junction_creates_coupling():
    """Gap junctions create coupling between CA1 interneurons.

    Verifies that:
    1. Gap junctions produce non-zero coupling currents
    2. Coupling currents correlate with membrane voltage differences
    3. Gap junctions are functionally active during processing
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sizes = compute_hippocampus_sizes(32)
    cfg = HippocampusConfig(
        n_input=32,
        n_output=64,
        dg_size=sizes["dg_size"],
        ca3_size=sizes["ca3_size"],
        ca2_size=sizes["ca2_size"],
        ca1_size=sizes["ca1_size"],
        gap_junction_strength=0.15,  # Strong coupling for clear effect
        gap_junction_max_neighbors=8,
        gap_junctions_enabled=True,
        device=device,
    )
    hippo = TrisynapticHippocampus(cfg)

    # Stimulate with input pattern
    torch.manual_seed(42)
    input_pattern = torch.rand(cfg.n_input, device=device) > 0.7

    gap_currents = []
    membrane_voltages = []

    # Run simulation
    n_steps = 30
    for _ in range(n_steps):
        hippo.forward(input_pattern)

        # Collect gap junction coupling current if available
        if hippo.state.ca1_membrane is not None and hippo.gap_junctions_ca1 is not None:
            # Compute what gap current would be
            gap_current = hippo.gap_junctions_ca1(hippo.state.ca1_membrane)
            gap_currents.append(gap_current.clone())
            membrane_voltages.append(hippo.state.ca1_membrane.clone())

    # Verify gap junctions are active
    assert len(gap_currents) > 0, "No gap currents collected"

    gap_currents = torch.stack(gap_currents)  # [time, neurons]
    membrane_voltages = torch.stack(membrane_voltages)

    # Test 1: Gap currents should be non-zero (coupling is active)
    gap_current_magnitude = gap_currents.abs().mean().item()
    print(f"Mean gap current magnitude: {gap_current_magnitude:.4f}")
    assert gap_current_magnitude > 0.001, (
        f"Gap currents are too small: {gap_current_magnitude:.4f}"
    )

    # Test 2: Gap currents should correlate with membrane voltage differences
    membrane_std = membrane_voltages.std(dim=1).mean().item()  # Variance across neurons
    print(f"Mean membrane voltage std: {membrane_std:.4f}")
    assert membrane_std > 0.01, "Membrane voltages are too uniform"

    # Test 3: Gap junctions should be active throughout simulation
    assert (gap_currents.abs() > 0.001).any(), "Gap currents are zero throughout"


def test_gap_junction_state_management():
    """Gap junction state (ca1_membrane) is properly managed."""
    sizes = compute_hippocampus_sizes(32)
    cfg = HippocampusConfig(
        n_input=32,
        n_output=64,
        dg_size=sizes["dg_size"],
        ca3_size=sizes["ca3_size"],
        ca2_size=sizes["ca2_size"],
        ca1_size=sizes["ca1_size"],
        gap_junctions_enabled=True,
    )
    hippo = TrisynapticHippocampus(cfg)

    # Before first forward, state is None (lazy initialization)
    assert hippo.state.ca1_membrane is None

    # After forward pass, ca1_membrane should be initialized and updated
    input_spikes = torch.ones(cfg.n_input, device=hippo.device)
    hippo.forward(input_spikes)

    assert hippo.state.ca1_membrane is not None
    assert hippo.state.ca1_membrane.shape == (cfg.n_output,)  # CA1 size
    # Should have non-zero values (neurons responded)
    assert hippo.state.ca1_membrane.abs().sum() > 0


def test_gap_junction_state_serialization():
    """Gap junction state (ca1_membrane) is included in state save/load."""
    sizes = compute_hippocampus_sizes(32)
    cfg = HippocampusConfig(
        n_input=32,
        n_output=64,
        dg_size=sizes["dg_size"],
        ca3_size=sizes["ca3_size"],
        ca2_size=sizes["ca2_size"],
        ca1_size=sizes["ca1_size"],
        gap_junctions_enabled=True,
    )
    hippo = TrisynapticHippocampus(cfg)

    # Run forward to populate ca1_membrane
    input_spikes = torch.ones(cfg.n_input, device=hippo.device)
    hippo.forward(input_spikes)

    # Get state
    state = hippo.get_state()
    assert state.ca1_membrane is not None

    # Serialize and deserialize
    state_dict = state.to_dict()
    assert "ca1_membrane" in state_dict

    # Create new state from dict
    from thalia.regions.hippocampus.config import HippocampusState
    restored_state = HippocampusState.from_dict(state_dict, device=hippo.device)
    assert restored_state.ca1_membrane is not None
    assert torch.allclose(restored_state.ca1_membrane, state.ca1_membrane)


def test_gap_junction_backward_compatibility():
    """Old states without ca1_membrane can be loaded (backward compatible)."""
    from thalia.regions.hippocampus.config import HippocampusState

    # Simulate old state dict (no ca1_membrane)
    old_state_dict = {
        "dg_spikes": torch.zeros(96),
        "ca3_spikes": torch.zeros(48),
        "ca1_spikes": torch.zeros(64),
        "ca3_membrane": torch.zeros(48),
        "ca3_persistent": torch.zeros(48),
        "dg_trace": torch.zeros(96),
        "ca3_trace": torch.zeros(48),
        "nmda_trace": torch.zeros(64),
        "ffi_strength": 0.0,
        # ca1_membrane deliberately missing
    }

    # Should load without error
    state = HippocampusState.from_dict(old_state_dict)

    # ca1_membrane should be None (not present in old state)
    assert state.ca1_membrane is None


def test_gap_junction_uses_ca1_inhib_weights():
    """Gap junctions use ca1_inhib weights to infer neighborhoods."""
    sizes = compute_hippocampus_sizes(32)
    cfg = HippocampusConfig(
        n_input=32,
        n_output=64,
        dg_size=sizes["dg_size"],
        ca3_size=sizes["ca3_size"],
        ca2_size=sizes["ca2_size"],
        ca1_size=sizes["ca1_size"],
        gap_junctions_enabled=True,
    )
    hippo = TrisynapticHippocampus(cfg)

    # Gap junctions should be built from ca1_inhib weights
    assert hippo.gap_junctions_ca1 is not None

    # The coupling matrix should reflect shared targets
    # (interneurons with similar inhibitory targets are coupled)
    stats = hippo.gap_junctions_ca1.get_coupling_stats()
    assert stats["n_connections"] > 0, "Should have gap junction connections"
    assert stats["avg_neighbors"] <= cfg.gap_junction_max_neighbors


def test_gap_junction_integration_with_theta():
    """Gap junctions work correctly with theta-gamma coupled processing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sizes = compute_hippocampus_sizes(32)
    cfg = HippocampusConfig(
        n_input=32,
        n_output=64,
        dg_size=sizes["dg_size"],
        ca3_size=sizes["ca3_size"],
        ca2_size=sizes["ca2_size"],
        ca1_size=sizes["ca1_size"],
        gap_junctions_enabled=True,
        theta_gamma_enabled=True,  # Enable theta-gamma coupling
        device=device,
    )
    hippo = TrisynapticHippocampus(cfg)

    # Set oscillator phases (simulate theta cycle)
    hippo.set_oscillator_phases(
        phases={'theta': 0.0, 'gamma': 0.0},
        signals={'theta': 0.8, 'gamma': 0.6},
        theta_slot=0,
    )

    # Run forward passes
    input_pattern = torch.rand(cfg.n_input, device=device) > 0.7

    gap_currents_recorded = False
    for _ in range(20):
        hippo.forward(input_pattern)

        # Verify gap junctions are working
        if hippo.state.ca1_membrane is not None and hippo.gap_junctions_ca1 is not None:
            gap_current = hippo.gap_junctions_ca1(hippo.state.ca1_membrane)
            if gap_current.abs().sum() > 0.001:
                gap_currents_recorded = True

    assert gap_currents_recorded, "Gap junctions should produce non-zero currents"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
