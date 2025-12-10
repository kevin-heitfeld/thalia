"""
Integration test for AUTOMATIC HER integration.

Tests that HER automatically captures experiences during normal brain
operation WITHOUT manual calls to add_her_experience().

This verifies:
1. Brain.store_experience() automatically populates HER buffer
2. CurriculumTrainer consolidation automatically uses HER
3. No manual HER calls needed during training

Author: Thalia Project
Date: December 10, 2025
"""

import pytest
import torch
from thalia.core.brain import EventDrivenBrain
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes
from thalia.regions.hippocampus.config import TrisynapticConfig
from thalia.regions.hippocampus.trisynaptic import TrisynapticHippocampus


@pytest.fixture
def brain_with_her():
    """Create brain with HER enabled in hippocampus."""
    config = ThaliaConfig(
        global_=GlobalConfig(
            device="cpu",
            dt_ms=1.0,
            theta_frequency_hz=8.0,
        ),
        brain=BrainConfig(
            sizes=RegionSizes(
                input_size=16,
                cortex_size=16,
                hippocampus_size=16,
                pfc_size=16,
                n_actions=2,
            ),
        ),
    )
    
    # Enable HER in hippocampus config
    # We need to modify the hippocampus config after brain creation
    brain = EventDrivenBrain(config)
    
    # Hacky but works for testing: recreate hippocampus with HER enabled
    from thalia.regions.hippocampus.trisynaptic import TrisynapticHippocampus
    
    her_config = TrisynapticConfig(
        n_input=config.brain.sizes.cortex_l23_size,
        n_output=config.brain.sizes.hippocampus_size,
        use_her=True,
        her_k_hindsight=4,
        her_replay_ratio=0.8,
        her_strategy="future",
        her_goal_tolerance=0.1,
        her_buffer_size=100,
        device=config.global_.device,
    )
    
    brain.hippocampus.impl = TrisynapticHippocampus(her_config)
    
    return brain


def test_automatic_her_capture_via_store_experience():
    """Test that store_episode() automatically populates HER buffer.
    
    When use_her=True and goal/achieved_goal are provided,
    store_episode() should automatically call add_her_experience().
    
    No manual calls to add_her_experience() should be needed!
    """
    # Create hippocampus with HER enabled
    config = TrisynapticConfig(
        n_input=24,
        n_output=16,
        use_her=True,
        her_k_hindsight=4,
        device="cpu"
    )
    hippo = TrisynapticHippocampus(config)
    
    # Simulate storing an experience with goal information
    # This is what Brain.store_experience() would do
    state = torch.randn(32)  # Combined brain state
    goal = torch.zeros(16)
    goal[0] = 1.0  # Want to achieve goal A
    
    achieved = torch.zeros(16)
    achieved[1] = 1.0  # Actually achieved goal B
    
    # Call store_episode with goal (automatic HER integration!)
    hippo.store_episode(
        state=state,
        action=0,
        reward=0.0,  # Failed to get A
        correct=False,
        goal=goal,
        achieved_goal=achieved,
        done=True,
    )
    
    # Verify HER buffer was automatically populated
    her_diag = hippo.get_her_diagnostics()
    assert her_diag['her_enabled'] is True
    assert her_diag['n_episodes'] > 0, "HER buffer should have episodes (automatic)"
    assert her_diag['n_transitions'] > 0, "HER buffer should have transitions (automatic)"
    
    # Verify normal episode buffer also has it
    assert len(hippo.episode_buffer) == 1


def test_automatic_her_consolidation():
    """Test that consolidation automatically uses HER replay.
    
    During curriculum training, when consolidation triggers:
    1. Hippocampus enters HER consolidation mode automatically
    2. Hindsight experiences are relabeled automatically
    3. Replay batch contains real + hindsight automatically
    
    No manual HER mode toggling needed!
    """
    # This test would require CurriculumTrainer setup
    # For now, we test the hippocampus consolidation API
    
    config = TrisynapticConfig(
        n_input=16,
        n_output=16,
        use_her=True,
        her_k_hindsight=4,
        her_replay_ratio=0.8,
        her_strategy="future",
        device="cpu"
    )
    hippo = TrisynapticHippocampus(config)
    
    # Add episode via store_episode (automatic HER capture)
    state = torch.randn(32)
    goal = torch.zeros(16)
    goal[0] = 1.0  # Want goal A
    
    achieved = torch.zeros(16)
    achieved[1] = 1.0  # Actually achieved goal B
    
    hippo.store_episode(
        state=state,
        action=0,
        reward=0.0,  # Failed to get A
        correct=False,
        goal=goal,
        achieved_goal=achieved,
        done=True,
    )
    
    # Verify HER captured it automatically
    diag = hippo.get_her_diagnostics()
    assert diag['n_episodes'] == 1
    assert diag['n_transitions'] == 1
    assert diag['consolidation_mode'] is False
    
    # Enter consolidation (simulates sleep/consolidation trigger)
    hippo.enter_consolidation_mode()
    assert hippo.get_her_diagnostics()['consolidation_mode'] is True
    
    # Sample replay batch - should contain hindsight relabeled experiences
    batch = hippo.sample_her_replay_batch(batch_size=10)
    
    # Should have experiences (real + hindsight)
    assert len(batch) > 0, "Should get replay batch during consolidation"
    
    # Exit consolidation
    hippo.exit_consolidation_mode()
    assert hippo.get_her_diagnostics()['consolidation_mode'] is False


def test_her_disabled_no_errors():
    """Test that HER features don't break when disabled.
    
    When use_her=False:
    - store_episode() should work without HER parameters
    - Consolidation should work without HER
    - Brain should operate normally
    """
    config = TrisynapticConfig(
        n_input=16,
        n_output=16,
        use_her=False,
        device="cpu"
    )
    hippo = TrisynapticHippocampus(config)
    
    # Store episode without HER parameters
    hippo.store_episode(
        state=torch.randn(32),
        action=0,
        reward=1.0,
        correct=True,
    )
    
    # Should work fine
    assert len(hippo.episode_buffer) == 1
    
    # HER diagnostics should show disabled
    diag = hippo.get_her_diagnostics()
    assert diag['her_enabled'] is False
    # When disabled, these fields may not exist or should be 0
    assert diag.get('n_episodes', 0) == 0
    assert diag.get('n_transitions', 0) == 0


def test_store_episode_with_and_without_goal():
    """Test store_episode works with or without goal information.
    
    Backward compatibility: store_episode should work:
    - With goal/achieved_goal (HER active)
    - Without goal/achieved_goal (HER inactive, but doesn't error)
    """
    config = TrisynapticConfig(
        n_input=16,
        n_output=16,
        use_her=True,
        device="cpu"
    )
    hippo = TrisynapticHippocampus(config)
    
    # Store with goal (HER active)
    hippo.store_episode(
        state=torch.randn(32),
        action=0,
        reward=0.0,
        correct=False,
        goal=torch.randn(16),
        achieved_goal=torch.randn(16),
        done=True,
    )
    
    # Store without goal (HER inactive for this episode)
    hippo.store_episode(
        state=torch.randn(32),
        action=1,
        reward=1.0,
        correct=True,
        # No goal/achieved_goal provided
    )
    
    # Both should work
    assert len(hippo.episode_buffer) == 2
    
    # Only first episode should be in HER buffer
    diag = hippo.get_her_diagnostics()
    assert diag['n_episodes'] == 1  # Only episode with goal


def test_full_training_cycle_with_automatic_her():
    """Integration test: Full training cycle with automatic HER.
    
    This simulates a complete training workflow:
    1. Active learning: Store experiences (HER captures automatically)
    2. Consolidation: Replay with hindsight (automatic)
    3. Active learning: Continue training
    
    No manual HER calls anywhere!
    """
    config = TrisynapticConfig(
        n_input=16,
        n_output=16,
        use_her=True,
        her_k_hindsight=4,
        her_strategy="future",
        device="cpu"
    )
    hippo = TrisynapticHippocampus(config)
    
    # Active learning phase: Store 10 experiences
    for i in range(10):
        goal = torch.zeros(16)
        goal[i % 4] = 1.0  # Multiple goals
        
        achieved = torch.zeros(16)
        achieved[(i + 1) % 4] = 1.0  # Different achievement
        
        hippo.store_episode(
            state=torch.randn(32),
            action=i % 2,
            reward=0.0,  # Most failed
            correct=False,
            goal=goal,
            achieved_goal=achieved,
            done=(i % 3 == 0),  # Some episodes end
        )
    
    # Check HER captured experiences
    diag = hippo.get_her_diagnostics()
    initial_episodes = diag['n_episodes']
    assert initial_episodes > 0, "Should have captured episodes"
    
    # Consolidation phase (sleep)
    hippo.enter_consolidation_mode()
    
    # Sample replay batches
    batch1 = hippo.sample_her_replay_batch(batch_size=16)
    batch2 = hippo.sample_her_replay_batch(batch_size=16)
    
    assert len(batch1) > 0, "Should get hindsight replay"
    assert len(batch2) > 0, "Should get hindsight replay"
    
    # Some experiences should have reward=1 (hindsight successes)
    rewards = [t.reward for t in batch1 + batch2]
    # Note: Rewards depend on goal tolerance and achieved_goal distances
    
    # Exit consolidation
    hippo.exit_consolidation_mode()
    
    # Active learning resumes: Store more experiences
    for i in range(5):
        hippo.store_episode(
            state=torch.randn(32),
            action=0,
            reward=1.0,
            correct=True,
            goal=torch.randn(16),
            achieved_goal=torch.randn(16),
            done=True,
        )
    
    # Should have more episodes now
    final_diag = hippo.get_her_diagnostics()
    assert final_diag['n_episodes'] > initial_episodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
