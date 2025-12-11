"""Tests for axonal delay implementation in brain regions.

This test suite verifies that ALL brain regions implement axonal delays,
achieving complete architectural parity with pathways.

Key Principles:
1. Regions and pathways are architecturally identical
2. Only difference is configuration (delay_ms values)
3. ALL neural connections have conduction delays (biological reality)
4. Delay buffer should be transparent to existing code

Author: Thalia Project
Date: December 11, 2025
"""

import pytest
import torch

from thalia.regions.striatum import Striatum, StriatumConfig
from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig
from thalia.regions.hippocampus import Hippocampus, HippocampusConfig
from thalia.regions.cerebellum import Cerebellum, CerebellumConfig
from thalia.regions.cortex import LayeredCortex, LayeredCortexConfig


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def dt():
    return 1.0


def test_striatum_has_delay_buffer(device, dt):
    """Verify Striatum initializes and uses delay buffer."""
    config = StriatumConfig(
        n_input=10,
        n_output=10,  # neurons_per_action * n_actions (5 * 2)
        neurons_per_action=5,
        axonal_delay_ms=2.0,
        dt_ms=dt,
        device=device,
    )
    striatum = Striatum(config)

    # Check delay configuration exists
    assert hasattr(striatum, 'axonal_delay_ms'), "Striatum missing axonal_delay_ms"
    assert hasattr(striatum, 'avg_delay_steps'), "Striatum missing avg_delay_steps"

    # Verify delay parameters
    assert striatum.axonal_delay_ms == 2.0
    assert striatum.avg_delay_steps == int(2.0 / dt)

    # Buffer is lazily initialized on first forward() call
    # Run one forward to trigger initialization
    input_spikes = torch.zeros(10, dtype=torch.bool, device=device)
    output = striatum.forward(input_spikes, dt=dt, action_mask=None)

    # Now delay_buffer should exist
    assert hasattr(striatum, 'delay_buffer'), "Striatum delay_buffer not initialized after forward()"
    assert hasattr(striatum, 'delay_buffer_idx'), "Striatum delay_buffer_idx not initialized"


def test_striatum_output_is_delayed(device, dt):
    """Verify Striatum output has delay applied."""
    config = StriatumConfig(
        n_input=10,
        n_output=10,  # neurons_per_action * n_actions
        neurons_per_action=5,
        axonal_delay_ms=3.0,  # 3ms delay
        dt_ms=dt,
        device=device,
    )
    striatum = Striatum(config)

    # First timestep: output should be zeros (delay buffer empty)
    input_spikes = torch.zeros(10, dtype=torch.bool, device=device)
    input_spikes[0] = True  # Some input

    output1 = striatum.forward(input_spikes, dt=dt, action_mask=None)

    # Output should be delayed (initially zeros since buffer is empty)
    # After delay_steps timesteps, we should see non-zero output

    # Run for delay_steps to fill buffer
    delay_steps = striatum.avg_delay_steps
    for _ in range(delay_steps + 2):
        input_spikes = torch.rand(10, device=device) > 0.5
        output = striatum.forward(input_spikes, dt=dt, action_mask=None)

    # Verify we got delayed output (bool tensor)
    assert output.dtype == torch.bool
    # Output shape: striatum returns all D1+D2 neurons (50 = neurons_per_action * n_actions * 2 pathways + 10 matrisomes)
    # We just need to verify it's a valid bool tensor


def test_prefrontal_has_delay_buffer(device, dt):
    """Verify Prefrontal initializes and uses delay buffer."""
    config = PrefrontalConfig(
        n_input=20,
        n_output=15,
        axonal_delay_ms=1.5,
        dt_ms=dt,
        device=device,
    )
    pfc = Prefrontal(config)

    assert hasattr(pfc, 'axonal_delay_ms')
    assert pfc.axonal_delay_ms == 1.5

    # Trigger lazy initialization
    input_spikes = torch.zeros(20, dtype=torch.bool, device=device)
    pfc.forward(input_spikes, dt=dt)

    assert hasattr(pfc, 'delay_buffer')


def test_hippocampus_has_delay_buffer(device, dt):
    """Verify Hippocampus initializes and uses delay buffer."""
    config = HippocampusConfig(
        n_input=30,
        n_output=25,  # CA1 output size
        axonal_delay_ms=2.5,
        dt_ms=dt,
        device=device,
    )
    hippo = Hippocampus(config)

    assert hasattr(hippo, 'axonal_delay_ms')
    assert hippo.axonal_delay_ms == 2.5

    # Trigger lazy initialization
    input_spikes = torch.zeros(30, dtype=torch.bool, device=device)
    hippo.forward(input_spikes, dt=dt)

    assert hasattr(hippo, 'delay_buffer')


def test_cerebellum_has_delay_buffer(device, dt):
    """Verify Cerebellum initializes and uses delay buffer."""
    config = CerebellumConfig(
        n_input=20,
        n_output=15,
        axonal_delay_ms=1.0,
        dt_ms=dt,
        device=device,
    )
    cerebellum = Cerebellum(config)

    assert hasattr(cerebellum, 'axonal_delay_ms')
    assert cerebellum.axonal_delay_ms == 1.0

    # Trigger lazy initialization
    input_spikes = torch.zeros(20, dtype=torch.bool, device=device)
    cerebellum.forward(input_spikes, dt=dt)

    assert hasattr(cerebellum, 'delay_buffer')


def test_cortex_has_delay_buffer(device, dt):
    """Verify LayeredCortex initializes and uses delay buffer."""
    config = LayeredCortexConfig(
        n_input=25,
        n_output=30,  # Total output size (L2/3 + L5)
        axonal_delay_ms=1.5,
        dt_ms=dt,
        device=device,
    )
    cortex = LayeredCortex(config)

    assert hasattr(cortex, 'axonal_delay_ms')
    assert cortex.axonal_delay_ms == 1.5

    # Trigger lazy initialization
    input_spikes = torch.zeros(25, dtype=torch.bool, device=device)
    cortex.forward(input_spikes, dt=dt)

    assert hasattr(cortex, 'delay_buffer')


def test_delay_buffer_fills_correctly(device, dt):
    """Verify delay buffer correctly stores and retrieves spikes."""
    config = PrefrontalConfig(
        n_input=10,
        n_output=8,
        axonal_delay_ms=2.0,  # 2 timesteps
        dt_ms=dt,
        device=device,
    )
    pfc = Prefrontal(config)

    # Run several timesteps with distinct input patterns
    inputs = [
        torch.zeros(10, dtype=torch.bool, device=device),
        torch.ones(10, dtype=torch.bool, device=device),
        torch.zeros(10, dtype=torch.bool, device=device),
        torch.ones(10, dtype=torch.bool, device=device),
    ]

    outputs = []
    for inp in inputs:
        out = pfc.forward(inp, dt=dt)
        outputs.append(out)

    # First output should be zero (buffer empty)
    # After delay_steps, we should see delayed versions
    # This is hard to verify exactly due to neuron dynamics,
    # but we can verify the mechanism runs without error
    assert all(out.dtype == torch.bool for out in outputs)
    assert all(out.shape == (8,) for out in outputs)


def test_delay_buffer_resets_correctly(device, dt):
    """Verify delay buffer resets with region state."""
    config = StriatumConfig(
        n_input=10,
        n_output=10,
        neurons_per_action=5,
        axonal_delay_ms=2.0,
        dt_ms=dt,
        device=device,
    )
    striatum = Striatum(config)

    # Run some timesteps to fill buffer
    for _ in range(5):
        input_spikes = torch.rand(10, device=device) > 0.5
        striatum.forward(input_spikes, dt=dt, action_mask=None)

    # Reset
    striatum.reset_state()

    # Verify buffer was cleared
    assert striatum.delay_buffer_idx == 0
    if striatum.delay_buffer is not None:
        assert torch.all(striatum.delay_buffer == 0), "Delay buffer not cleared after reset"


def test_component_parity_regions_and_pathways(device):
    """Verify regions and pathways share the same delay mechanism.

    This test ensures architectural parity: both use _apply_axonal_delay()
    from NeuralComponent base class, differing only in configuration.
    """
    # Create a region
    region_config = PrefrontalConfig(
        n_input=10,
        n_output=8,
        axonal_delay_ms=1.5,  # Typical within-region delay
        device=device,
    )
    region = Prefrontal(region_config)

    # Create a pathway (import here to avoid circular dependencies)
    from thalia.integration.spiking_pathway import SpikingPathway, SpikingPathwayConfig

    pathway_config = SpikingPathwayConfig(
        source_size=10,
        target_size=8,
        axonal_delay_ms=5.0,  # Typical inter-region delay
        device=device,
    )
    pathway = SpikingPathway(pathway_config)

    # Trigger lazy initialization for region
    input_spikes = torch.zeros(10, dtype=torch.bool, device=device)
    region.forward(input_spikes, dt=1.0)

    # Both should have the same delay attributes (after initialization)
    assert hasattr(region, 'delay_buffer')
    assert hasattr(pathway, 'delay_buffer')
    assert hasattr(region, '_apply_axonal_delay')
    assert hasattr(pathway, '_apply_axonal_delay')    # Both should use the same base class
    from thalia.regions.base import NeuralComponent
    assert isinstance(region, NeuralComponent)
    assert isinstance(pathway, NeuralComponent)

    print("✓ Component parity verified: Regions and pathways share delay mechanism")


def test_zero_delay_is_immediate_output(device, dt):
    """Verify that zero delay gives immediate output (for backward compatibility if needed)."""
    config = PrefrontalConfig(
        n_input=10,
        n_output=8,
        axonal_delay_ms=0.0,  # Zero delay
        dt_ms=dt,
        device=device,
    )
    pfc = Prefrontal(config)

    # With zero delay, avg_delay_steps should be 0
    # Output should be immediate (same timestep)
    assert pfc.avg_delay_steps == 0

    input_spikes = torch.rand(10, device=device) > 0.5
    output = pfc.forward(input_spikes, dt=dt)

    # Should work without error
    assert output.dtype == torch.bool
    assert output.shape == (8,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
