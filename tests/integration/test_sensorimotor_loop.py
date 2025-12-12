"""
Integration test for sensorimotor loop.

Tests the complete sensorimotor processing pipeline from spike input through
motor output, including learning and action selection.
"""

import pytest
import torch

from thalia.core.brain import EventDrivenBrain
from thalia.config.thalia_config import ThaliaConfig
from thalia.config.global_config import GlobalConfig
from thalia.config.brain_config import BrainConfig, RegionSizes


@pytest.fixture
def small_brain_config():
    """Create minimal brain configuration for testing."""
    config = ThaliaConfig(
        global_=GlobalConfig(
            device='cpu',
            dt_ms=1.0,
        ),
        brain=BrainConfig(
            sizes=RegionSizes(
                input_size=128,
                cortex_size=64,
                hippocampus_size=32,
                pfc_size=32,
                n_actions=4,
            ),
            encoding_timesteps=10,
        ),
    )
    return config


@pytest.fixture
def brain(small_brain_config):
    """Create EventDrivenBrain instance."""
    return EventDrivenBrain.from_thalia_config(small_brain_config)


def test_brain_initialization(brain):
    """Test that brain initializes without errors."""
    assert brain is not None
    assert hasattr(brain, 'cortex')
    assert hasattr(brain, 'hippocampus')
    assert hasattr(brain, 'pfc')
    assert hasattr(brain, 'striatum')
    assert hasattr(brain, 'cerebellum')
    assert hasattr(brain, 'pathway_manager')
    assert hasattr(brain, 'neuromodulator_manager')


def test_forward_pass_smoke_test(brain, small_brain_config):
    """Test that a single forward pass completes without errors."""
    input_size = small_brain_config.brain.sizes.input_size
    sensory_input = torch.zeros(input_size, dtype=torch.bool, device=brain.config.device)
    sensory_input[::4] = True  # 25% activity

    output = brain.forward(sensory_input)

    assert output is not None
    assert 'spike_counts' in output
    assert 'events_processed' in output
    assert 'final_time' in output
    assert isinstance(output['spike_counts'], dict)


def test_multi_step_simulation(brain, small_brain_config):
    """Test that brain can process multiple timesteps without crashes."""
    n_steps = 10
    results = []
    input_size = small_brain_config.brain.sizes.input_size

    for _ in range(n_steps):
        sensory_input = torch.rand(input_size, device=brain.config.device) < 0.2
        output = brain.forward(sensory_input)
        results.append(output)

    assert len(results) == n_steps


def test_learning_updates_weights(brain, small_brain_config):
    """Test that learning actually modifies weights over multiple trials."""
    input_size = small_brain_config.brain.sizes.input_size
    cortex_to_striatum = brain.pathway_manager.cortex_to_striatum
    initial_weights = cortex_to_striatum.weights.clone()

    n_trials = 20
    for i in range(n_trials):
        sensory_input = torch.rand(input_size, device=brain.config.device) < 0.2
        brain.forward(sensory_input)
        reward = 1.0 if i % 2 == 0 else -0.5
        brain.deliver_reward(external_reward=reward)

    final_weights = cortex_to_striatum.weights
    weight_change = (final_weights - initial_weights).abs().mean().item()
    assert weight_change > 0, "Weights should change after learning"


def test_pathway_health_checks(brain):
    """Test that pathway health checks work."""
    pathway_health = brain.pathway_manager.check_health()
    assert isinstance(pathway_health, dict)
    assert len(pathway_health) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
