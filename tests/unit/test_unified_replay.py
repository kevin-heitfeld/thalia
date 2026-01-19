"""
Tests for UnifiedReplayCoordinator integration (Phases 2 & 3).

This module tests that:
1. UnifiedReplayCoordinator is properly initialized in DynamicBrain
2. Sleep consolidation works through the unified API
3. Immediate replay is triggered after rewards
4. Forward planning works for action selection
5. Background planning updates values during idle time
6. Replay contexts are properly handled

Author: Thalia Project
Date: January 19, 2026 (Phase 3 - Unified replay system)
"""

import weakref

import pytest
import torch

from thalia.config import BrainConfig
from thalia.core.brain_builder import BrainBuilder
from thalia.replay import ReplayContext


@pytest.fixture
def device():
    """Device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def minimal_brain(device):
    """Create minimal brain with hippocampus, cortex, striatum, pfc for replay."""
    brain_config = BrainConfig(
        device=device,
        dt_ms=1.0,
    )

    # Create builder
    builder = BrainBuilder(brain_config)

    # Add regions required for replay (hippocampus, striatum, cortex, pfc)
    builder.add_component(
        "cortex",
        "cortex",
        input_size=64,
        n_output=128,
        n_input=64,
        l4_size=64,
        l23_size=64,
        l5_size=64,
        l6a_size=0,
        l6b_size=0,
    )
    builder.add_component("hippocampus", "hippocampus", n_output=64)
    builder.add_component("striatum", "striatum", n_output=32, n_actions=2)
    builder.add_component("pfc", "prefrontal", input_size=128, n_neurons=32)

    # Add connections
    builder.connect("cortex", "hippocampus", "axonal_projection")
    builder.connect("hippocampus", "cortex", "axonal_projection")
    builder.connect("hippocampus", "striatum", "axonal_projection")
    builder.connect("cortex", "striatum", "axonal_projection", source_port="l5")
    builder.connect("pfc", "striatum", "axonal_projection")

    # Build brain
    brain = builder.build()

    return brain


class TestUnifiedReplayCoordinator:
    """Test suite for UnifiedReplayCoordinator public API."""

    def test_coordinator_initialized(self, minimal_brain):
        """Test that UnifiedReplayCoordinator is initialized in DynamicBrain."""
        manager = minimal_brain.consolidation_manager
        assert manager is not None, "UnifiedReplayCoordinator should be initialized"

        # Check it has replay engine and sleep controller
        assert hasattr(manager, "replay_engine"), "Should have replay_engine"
        assert hasattr(manager, "sleep_controller"), "Should have sleep_controller"
        assert manager.replay_engine is not None, "Replay engine should be initialized"
        assert manager.sleep_controller is not None, "Sleep controller should be initialized"

    def test_brain_reference_storage(self, minimal_brain):
        """Test that coordinator stores weak reference to brain."""
        manager = minimal_brain.consolidation_manager

        # Check brain reference is set
        assert manager._brain_ref is not None, "Brain reference should be set"

        # Verify it's a weak reference
        assert isinstance(manager._brain_ref, weakref.ref), "Should be weakref"

        # Verify it points to brain
        brain_from_ref = manager._brain_ref()
        assert brain_from_ref is minimal_brain, "Weak reference should point to brain"

    def test_no_consolidation_weights_in_striatum(self, minimal_brain):
        """Test that striatum doesn't have special 'consolidation' weights."""
        striatum = minimal_brain.components["striatum"]

        # Check all weight keys
        weight_keys = list(striatum.synaptic_weights.keys())

        # No keys should contain 'consolidation'
        consolidation_keys = [key for key in weight_keys if "consolidation" in key.lower()]

        assert (
            len(consolidation_keys) == 0
        ), f"Striatum should NOT have consolidation weights, found: {consolidation_keys}"

        # Verify normal sources exist (e.g., hippocampus_d1, cortex:l5_d1)
        expected_sources = ["hippocampus", "cortex:l5"]
        for source in expected_sources:
            # Check D1 pathway
            d1_key = f"{source}_d1"
            assert d1_key in weight_keys, f"Missing normal source: {d1_key}"

            # Check D2 pathway
            d2_key = f"{source}_d2"
            assert d2_key in weight_keys, f"Missing normal source: {d2_key}"

    def test_sleep_consolidation_api(self, minimal_brain):
        """Test that sleep_consolidation() works through the public API."""
        manager = minimal_brain.consolidation_manager
        hippocampus = minimal_brain.components["hippocampus"]

        # Store some experiences through hippocampus (simulate training)
        for i in range(5):
            # Simulate experience storage via hippocampus
            state = torch.randn(256, device=minimal_brain.device)
            action = i % 2
            reward = 1.0 if i == 4 else 0.0

            # Store directly on hippocampus
            hippocampus.store_episode(
                state=state,
                action=action,
                reward=reward, correct=(reward > 0.5),
            )

        # Run sleep consolidation (should not crash)
        try:
            result = manager.sleep_consolidation(
                n_cycles=2,
                batch_size=2,
            )
            # Check result structure
            assert isinstance(result, dict), "Should return dict"
            assert "cycles_completed" in result, "Should have cycles_completed"
        except Exception as e:
            pytest.fail(f"sleep_consolidation() raised unexpected exception: {e}")

    def test_immediate_replay_api(self, minimal_brain):
        """Test that immediate_replay() works through the public API."""
        manager = minimal_brain.consolidation_manager
        hippocampus = minimal_brain.components["hippocampus"]

        # Store an experience
        state = torch.randn(256, device=minimal_brain.device)
        hippocampus.store_episode(
            state=state,
            action=1,
            reward=1.0, correct=True,
        )

        # Trigger immediate replay (should not crash)
        try:
            manager.immediate_replay(
                episode_index=0,  # Replay the first (and only) episode
                surprise_level=1.0,
            )
        except Exception as e:
            pytest.fail(f"immediate_replay() raised unexpected exception: {e}")

    def test_plan_action_api(self, minimal_brain):
        """Test that plan_action() works through the public API."""
        manager = minimal_brain.consolidation_manager
        hippocampus = minimal_brain.components["hippocampus"]

        # Store some experiences for planning
        for i in range(5):
            state = torch.randn(256, device=minimal_brain.device)
            hippocampus.store_episode(
                state=state,
                action=i % 2,
                reward=1.0 if i == 4 else 0.0,
                correct=(i == 4),  # Only last one is correct
            )

        # Test planning (should not crash)
        try:
            current_state = torch.randn(256, device=minimal_brain.device)
            action = manager.plan_action(
                current_state=current_state,
                available_actions=[0, 1],
                depth=5,
            )
            # Check result
            assert action in [0, 1], f"Action should be 0 or 1, got {action}"
        except Exception as e:
            pytest.fail(f"plan_action() raised unexpected exception: {e}")

    def test_background_planning_api(self, minimal_brain):
        """Test that background_planning() works through the public API."""
        manager = minimal_brain.consolidation_manager
        hippocampus = minimal_brain.components["hippocampus"]

        # Store some experiences
        for i in range(5):
            state = torch.randn(256, device=minimal_brain.device)
            hippocampus.store_episode(
                state=state,
                action=i % 2,
                reward=1.0 if i == 4 else 0.0,
                correct=(i == 4),  # Only last one is correct
            )

        # Test background planning (should not crash)
        try:
            manager.background_planning(n_simulations=3)
        except Exception as e:
            pytest.fail(f"background_planning() raised unexpected exception: {e}")

    def test_trigger_replay_with_different_contexts(self, minimal_brain):
        """Test that trigger_replay() works with different replay contexts."""
        manager = minimal_brain.consolidation_manager
        hippocampus = minimal_brain.components["hippocampus"]

        # Store experiences
        for i in range(5):
            state = torch.randn(256, device=minimal_brain.device)
            hippocampus.store_episode(
                state=state,
                action=i % 2,
                reward=1.0 if i == 4 else 0.0,
                correct=(i == 4),  # Only last one is correct
            )

        # Test each context (should not crash)
        contexts = [
            ReplayContext.SLEEP_CONSOLIDATION,
            ReplayContext.AWAKE_IMMEDIATE,
            ReplayContext.FORWARD_PLANNING,
            ReplayContext.BACKGROUND_PLANNING,
        ]

        for context in contexts:
            try:
                manager.trigger_replay(
                    context=context,
                    n_episodes=2,
                )
            except Exception as e:
                pytest.fail(f"trigger_replay({context}) raised unexpected exception: {e}")

    def test_replay_engine_integration(self, minimal_brain):
        """Test that ReplayEngine is properly integrated."""
        manager = minimal_brain.consolidation_manager
        replay_engine = manager.replay_engine

        # Check replay engine has required methods
        assert hasattr(replay_engine, "replay"), "Should have replay method"
        assert hasattr(replay_engine, "get_state"), "Should have get_state method"

        # Verify it has config
        assert hasattr(replay_engine, "config"), "Should have config"
        assert replay_engine.config is not None, "Config should be initialized"

    def test_sleep_controller_integration(self, minimal_brain):
        """Test that SleepStageController is properly integrated."""
        manager = minimal_brain.consolidation_manager
        sleep_controller = manager.sleep_controller

        # Check sleep controller has required methods
        assert hasattr(sleep_controller, "get_current_stage"), "Should have get_current_stage method"
        assert hasattr(sleep_controller, "is_cycle_complete"), "Should have is_cycle_complete method"

        # Verify initial state
        current_stage = sleep_controller.get_current_stage(0)
        assert current_stage is not None, "Should return current stage"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
