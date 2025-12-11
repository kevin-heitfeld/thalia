"""Test oscillator broadcasts to pathways."""

import pytest
import torch
import numpy as np

from thalia.integration.spiking_pathway import SpikingPathway, SpikingPathwayConfig


def test_pathway_receives_oscillator_phases():
    """Test that pathways can receive oscillator broadcasts."""
    config = SpikingPathwayConfig(
        source_size=64,
        target_size=32,
        device=torch.device("cpu"),
    )
    pathway = SpikingPathway(config)

    # Simulate oscillator broadcast from brain
    phases = {
        'delta': 1.2,
        'theta': 3.4,
        'alpha': 0.5,
        'beta': 2.1,
        'gamma': 4.8,
    }
    signals = {
        'delta': 0.8,
        'theta': -0.3,
        'alpha': 0.2,
        'beta': -0.6,
        'gamma': 0.9,
    }
    theta_slot = 3
    coupled_amplitudes = {
        'delta': 1.0,
        'theta': 0.73,
        'gamma': 0.48,
    }

    # Broadcast oscillators
    pathway.set_oscillator_phases(phases, signals, theta_slot, coupled_amplitudes)

    # Verify stored
    assert hasattr(pathway, '_oscillator_phases')
    assert hasattr(pathway, '_oscillator_signals')
    assert hasattr(pathway, '_oscillator_theta_slot')
    assert hasattr(pathway, '_coupled_amplitudes')

    assert pathway._oscillator_phases == phases
    assert pathway._oscillator_signals == signals
    assert pathway._oscillator_theta_slot == theta_slot
    assert pathway._coupled_amplitudes == coupled_amplitudes


def test_pathway_oscillators_dont_break_forward():
    """Test that oscillator broadcasts don't break forward pass."""
    config = SpikingPathwayConfig(
        source_size=64,
        target_size=32,
        device=torch.device("cpu"),
    )
    pathway = SpikingPathway(config)

    # Broadcast oscillators first
    phases = {'theta': 1.5, 'gamma': 3.2}
    pathway.set_oscillator_phases(phases)

    # Forward should still work
    input_spikes = torch.zeros(64)
    input_spikes[0:10] = 1.0

    output = pathway(input_spikes)
    assert output.shape == (32,)
    assert output.dtype == torch.bool


def test_pathway_oscillators_optional():
    """Test that pathways work without oscillator broadcasts (backward compat)."""
    config = SpikingPathwayConfig(
        source_size=64,
        target_size=32,
        device=torch.device("cpu"),
    )
    pathway = SpikingPathway(config)

    # Forward without oscillators should work
    input_spikes = torch.ones(64)
    output = pathway(input_spikes)

    assert output.shape == (32,)
    # Should not crash without oscillator info


def test_pathway_oscillators_persist_across_timesteps():
    """Test that oscillator info persists until next broadcast."""
    config = SpikingPathwayConfig(
        source_size=64,
        target_size=32,
        device=torch.device("cpu"),
    )
    pathway = SpikingPathway(config)

    # First broadcast
    phases1 = {'theta': 1.0}
    pathway.set_oscillator_phases(phases1)
    assert pathway._oscillator_phases['theta'] == 1.0

    # Run forward pass
    pathway(torch.zeros(64))

    # Oscillator info should persist
    assert pathway._oscillator_phases['theta'] == 1.0

    # Second broadcast updates
    phases2 = {'theta': 2.0}
    pathway.set_oscillator_phases(phases2)
    assert pathway._oscillator_phases['theta'] == 2.0


def test_derived_pathways_inherit_oscillators():
    """Test that derived pathways (replay, attention) inherit oscillator capability."""
    from thalia.integration.pathways.spiking_replay import (
        SpikingReplayPathway,
        SpikingReplayPathwayConfig,
    )
    from thalia.integration.pathways.spiking_attention import (
        SpikingAttentionPathway,
        SpikingAttentionPathwayConfig,
    )

    # Test replay pathway
    replay_config = SpikingReplayPathwayConfig(
        source_size=64,
        target_size=32,
        device=torch.device("cpu"),
    )
    replay = SpikingReplayPathway(replay_config)

    phases = {'theta': 1.5, 'gamma': 3.2}
    replay.set_oscillator_phases(phases)
    assert replay._oscillator_phases == phases

    # Test attention pathway
    attention_config = SpikingAttentionPathwayConfig(
        source_size=64,
        target_size=32,
        input_size=128,
        cortex_size=32,
        device=torch.device("cpu"),
    )
    attention = SpikingAttentionPathway(attention_config)

    attention.set_oscillator_phases(phases)
    assert attention._oscillator_phases == phases


def test_pathway_oscillators_in_get_diagnostics():
    """Test that oscillator info doesn't break diagnostics."""
    config = SpikingPathwayConfig(
        source_size=64,
        target_size=32,
        device=torch.device("cpu"),
    )
    pathway = SpikingPathway(config)

    # Broadcast oscillators
    phases = {'theta': 1.5, 'gamma': 3.2}
    pathway.set_oscillator_phases(phases)

    # Run forward
    pathway(torch.ones(64))

    # Get diagnostics should work
    diag = pathway.get_diagnostics()
    assert isinstance(diag, dict)
    assert 'weight_mean' in diag


def test_pathway_oscillators_in_checkpointing():
    """Test that oscillator state doesn't affect checkpointing."""
    config = SpikingPathwayConfig(
        source_size=64,
        target_size=32,
        device=torch.device("cpu"),
    )
    pathway1 = SpikingPathway(config)

    # Broadcast oscillators and run forward
    phases = {'theta': 1.5}
    pathway1.set_oscillator_phases(phases)
    pathway1(torch.ones(64))

    # Save state
    state = pathway1.get_state()

    # Load into new pathway
    pathway2 = SpikingPathway(config)
    pathway2.load_state(state)

    # Should work without oscillator broadcast
    output = pathway2(torch.ones(64))
    assert output.shape == (32,)


def test_attention_pathway_can_use_beta():
    """Test that attention pathway can access beta for gain modulation."""
    from thalia.integration.pathways.spiking_attention import (
        SpikingAttentionPathway,
        SpikingAttentionPathwayConfig,
    )

    config = SpikingAttentionPathwayConfig(
        source_size=64,
        target_size=32,
        input_size=128,
        cortex_size=32,
        device=torch.device("cpu"),
    )
    pathway = SpikingAttentionPathway(config)

    # Broadcast with beta modulation
    coupled_amplitudes = {
        'beta': 0.8,  # Reduced beta amplitude
        'alpha': 0.6,  # Reduced alpha (less suppression)
    }
    pathway.set_oscillator_phases(
        phases={'beta': 1.5, 'alpha': 2.3},
        coupled_amplitudes=coupled_amplitudes,
    )

    # Pathway can access coupled amplitudes for attention modulation
    assert pathway._coupled_amplitudes['beta'] == 0.8
    assert pathway._coupled_amplitudes['alpha'] == 0.6

    # Could implement: attention_gain *= coupled_amplitudes['beta']
    # (demonstration of potential use case)


def test_replay_pathway_can_use_theta():
    """Test that replay pathway can access theta for gating."""
    from thalia.integration.pathways.spiking_replay import (
        SpikingReplayPathway,
        SpikingReplayPathwayConfig,
    )

    config = SpikingReplayPathwayConfig(
        source_size=64,
        target_size=32,
        device=torch.device("cpu"),
    )
    pathway = SpikingReplayPathway(config)

    # Broadcast with theta phase info
    phases = {
        'theta': np.pi / 2,  # Theta trough (encoding)
        'gamma': 1.2,
    }
    pathway.set_oscillator_phases(phases=phases)

    # Pathway can access theta phase for replay gating
    assert pathway._oscillator_phases['theta'] == pytest.approx(np.pi / 2)

    # Could implement: replay_active = (theta_phase > np.pi)  # Retrieval phase
    # (demonstration of potential use case)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
