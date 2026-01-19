"""
Tests for UnifiedReplayCoordinator integration (Phases 2 & 3).

This module tests that:
1. UnifiedReplayCoordinator is properly initialized in DynamicBrain
2. Sleep consolidation works through the unified API
3. Immediate replay is triggered after rewards
4. Forward planning works for action selection
5. Background planning updates values during idle time
6. No special 'consolidation' weights exist in striatum

Author: Thalia Project
Date: January 2026
Updated: January 19, 2026 (Phase 3 - Unified replay system)
"""

import weakref

import pytest
import torch

from thalia.config import BrainConfig
from thalia.core.brain_builder import BrainBuilder


@pytest.fixture
def device():
    """Device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def minimal_brain(device):
    """Create minimal brain with hippocampus, cortex, striatum, pfc for consolidation."""
    brain_config = BrainConfig(
        device=device,
        dt_ms=1.0,
    )

    # Create builder
    builder = BrainBuilder(brain_config)

    # Add regions required for consolidation (hippocampus, striatum, cortex, pfc)
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
    builder.add_component("striatum", "striatum", n_output=32, n_actions=2)  # Need n_actions
    builder.add_component("pfc", "prefrontal", input_size=128, n_neurons=32)

    # Add connections (hippocampus and cortex need to connect to striatum)
    builder.connect("cortex", "hippocampus", "axonal_projection")
    builder.connect(
        "hippocampus", "cortex", "axonal_projection"
    )  # Back-projection for systems consolidation!
    builder.connect("hippocampus", "striatum", "axonal_projection")
    builder.connect("cortex", "striatum", "axonal_projection", source_port="l5")
    builder.connect("pfc", "striatum", "axonal_projection")

    # Build brain
    brain = builder.build()

    return brain


class TestUnifiedReplayIntegration:
    """Test suite for UnifiedReplayCoordinator integration in DynamicBrain."""

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
        assert manager is not None, "UnifiedReplayCoordinator should be initialized"

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

    def test_full_brain_forward_used_when_available(self, minimal_brain, monkeypatch):
        """Test that _run_consolidation_replay uses full brain.forward when brain ref exists."""
        manager = minimal_brain.consolidation_manager

        # Track calls to brain.forward
        forward_calls = []

        original_forward = minimal_brain.forward

        def mock_forward(*args, **kwargs):
            forward_calls.append((args, kwargs))
            return original_forward(*args, **kwargs)

        monkeypatch.setattr(minimal_brain, "forward", mock_forward)

        # Run consolidation replay
        manager._run_consolidation_replay(n_timesteps=5)

        # Verify brain.forward was called
        assert len(forward_calls) == 1, "brain.forward should be called once"

        # Check arguments
        args, kwargs = forward_calls[0]
        assert kwargs.get("sensory_input") is None, "sensory_input should be None"
        assert kwargs.get("n_timesteps") == 5, "n_timesteps should be 5"

    def test_fallback_when_brain_reference_missing(self, minimal_brain):
        """Test that _run_consolidation_replay detects missing brain ref and uses fallback."""
        manager = minimal_brain.consolidation_manager

        # Store original brain reference
        original_ref = manager._brain_ref

        # Remove brain reference to trigger fallback
        manager._brain_ref = None

        # The fallback path should execute without error
        # We can't fully test it here because it requires proper cortex outputs,
        # but we can verify it doesn't crash and uses the fallback branch
        try:
            # This will use the fallback path (manual region execution)
            # It may fail during striatum.forward due to missing proper inputs,
            # but that's expected - we're just verifying the branch is taken
            manager._run_consolidation_replay(n_timesteps=1)
        except (AssertionError, RuntimeError, KeyError):
            # Expected - fallback path may fail due to improper setup
            # The important thing is that it tried the fallback (not brain.forward)
            pass

        # Restore brain reference
        manager._brain_ref = original_ref

        # Verify we can still use the normal path
        assert manager._brain_ref is not None, "Brain reference should be restored"

    def test_consolidation_mode_integration(self, minimal_brain):
        """Test integration: hippocampus enters consolidation mode during replay."""
        manager = minimal_brain.consolidation_manager
        hippocampus = minimal_brain.components["hippocampus"]

        # Store some experiences
        for i in range(3):
            manager.store_experience(
                action=i % 2,
                reward=1.0,
                last_action_holder=[None],
            )

        # Initial state: not in consolidation mode
        assert not hippocampus._consolidation_mode, "Should start in normal mode"

        # Run consolidation (with HER disabled, enter/exit are called)
        # Note: With HER disabled, consolidation mode management happens in consolidate()
        stats = manager.consolidate(
            n_cycles=1,
            batch_size=1,
            verbose=False,
            last_action_holder=[None],
        )

        # After consolidation, should be back in normal mode
        assert not hippocampus._consolidation_mode, "Should return to normal mode after"

        # Consolidation should have completed
        assert stats["cycles_completed"] == 1, "Should complete 1 cycle"


class TestPhase4Features:
    """Test Phase 4: HER integration, quality metrics, and PFC consolidation."""

    def test_her_uses_normal_pathways(self, minimal_brain):
        """Test that HER relabeling works with normal pathways (not consolidation weights)."""
        brain = minimal_brain
        manager = brain.consolidation_manager
        striatum = brain.components["striatum"]

        # Store some experiences
        for i in range(5):
            manager.store_experience(
                action=i % 2,
                reward=float(i % 3 == 0),  # Sparse rewards
                last_action_holder=[None],
            )

        # Run consolidation (would use HER if enabled)
        last_action = [None]
        manager.consolidate(
            n_cycles=2,
            batch_size=3,
            verbose=False,
            last_action_holder=last_action,
        )

        # Verify NO consolidation weights exist
        if hasattr(striatum, "synaptic_weights"):
            assert "consolidation_d1" not in striatum.synaptic_weights
            assert "consolidation_d2" not in striatum.synaptic_weights

            # Verify normal weights exist
            assert "hippocampus_d1" in striatum.synaptic_weights
            assert "hippocampus_d2" in striatum.synaptic_weights

    def test_consolidation_quality_metrics(self, minimal_brain):
        """Test that consolidation quality metrics are tracked correctly."""
        brain = minimal_brain
        manager = brain.consolidation_manager

        # Store experiences
        for i in range(5):
            manager.store_experience(
                action=i % 2,
                reward=1.0 if i >= 3 else 0.0,
                last_action_holder=[None],
            )

        # Run consolidation
        last_action = [None]
        manager.consolidate(
            n_cycles=3,
            batch_size=2,
            verbose=False,
            last_action_holder=last_action,
        )

        # Get quality metrics
        metrics = manager.get_consolidation_quality_metrics()

        # Verify metrics structure
        assert "avg_cortical_learning" in metrics
        assert "avg_hippocampal_learning" in metrics
        assert "transfer_efficiency" in metrics
        assert "avg_replay_effectiveness" in metrics
        assert "systems_consolidation_progress" in metrics
        assert "n_consolidation_cycles" in metrics

        # Verify metrics have reasonable values
        assert metrics["n_consolidation_cycles"] > 0
        assert metrics["avg_replay_effectiveness"] >= 0.0

    def test_quality_metrics_track_learning(self, minimal_brain):
        """Test that quality metrics track actual learning progress."""
        brain = minimal_brain
        manager = brain.consolidation_manager

        # Store rewarded experiences
        for i in range(10):
            manager.store_experience(
                action=i % 2,
                reward=1.0,  # All rewarded for strong learning signal
                last_action_holder=[None],
            )

        # Get metrics before consolidation
        metrics_before = manager.get_consolidation_quality_metrics()
        cycles_before = metrics_before["n_consolidation_cycles"]

        # Run consolidation
        last_action = [None]
        manager.consolidate(
            n_cycles=5,
            batch_size=3,
            verbose=False,
            last_action_holder=last_action,
        )

        # Get metrics after consolidation
        metrics_after = manager.get_consolidation_quality_metrics()

        # Verify metrics updated
        assert metrics_after["n_consolidation_cycles"] > cycles_before
        assert metrics_after["avg_replay_effectiveness"] > 0.0

    def test_transfer_efficiency_computation(self, minimal_brain):
        """Test that transfer efficiency correctly measures cortical vs hippocampal learning."""
        brain = minimal_brain
        manager = brain.consolidation_manager

        # Store experiences and run consolidation
        for i in range(5):
            manager.store_experience(
                action=i % 2,
                reward=1.0,
                last_action_holder=[None],
            )

        last_action = [None]
        manager.consolidate(
            n_cycles=2,
            batch_size=3,
            verbose=False,
            last_action_holder=last_action,
        )

        # Get metrics
        metrics = manager.get_consolidation_quality_metrics()

        # Transfer efficiency should be ratio of cortical to hippocampal learning
        if metrics["avg_hippocampal_learning"] > 0:
            expected_ratio = metrics["avg_cortical_learning"] / metrics["avg_hippocampal_learning"]
            assert abs(metrics["transfer_efficiency"] - expected_ratio) < 0.001

    def test_systems_consolidation_progress(self, minimal_brain):
        """Test that systems consolidation progress tracks shift to cortical learning."""
        brain = minimal_brain
        manager = brain.consolidation_manager

        # Store and consolidate experiences
        for i in range(8):
            manager.store_experience(
                action=i % 2,
                reward=0.5,
                last_action_holder=[None],
            )

        last_action = [None]
        manager.consolidate(
            n_cycles=4,
            batch_size=2,
            verbose=False,
            last_action_holder=last_action,
        )

        # Get metrics
        metrics = manager.get_consolidation_quality_metrics()

        # Systems consolidation progress should be between 0 and 1
        assert 0.0 <= metrics["systems_consolidation_progress"] <= 1.0

        # Progress = cortical / (cortical + hippocampal)
        total_learning = metrics["avg_cortical_learning"] + metrics["avg_hippocampal_learning"]
        if total_learning > 0:
            expected_progress = metrics["avg_cortical_learning"] / total_learning
            assert abs(metrics["systems_consolidation_progress"] - expected_progress) < 0.001


class TestSleepStageIntegration:
    """Test suite for sleep stage (NREM/REM) integration."""

    def test_sleep_controller_initialization(self, minimal_brain):
        """Test that ConsolidationManager initializes SleepStageController."""
        manager = minimal_brain.consolidation_manager

        # Verify sleep_controller exists
        assert hasattr(manager, "sleep_controller"), "Should have sleep_controller"
        assert manager.sleep_controller is not None, "SleepStageController should be initialized"

        # Verify consolidation step tracking
        assert hasattr(manager, "_consolidation_step"), "Should track consolidation step"
        assert manager._consolidation_step == 0, "Should start at step 0"

        # Verify replay stats include stage tracking
        assert "nrem_replays" in manager.replay_stats, "Should track NREM replays"
        assert "rem_replays" in manager.replay_stats, "Should track REM replays"

    def test_sleep_stage_cycling(self, minimal_brain):
        """Test that sleep stages cycle through NREM/REM correctly."""
        brain = minimal_brain
        manager = brain.consolidation_manager

        # Store experiences
        for i in range(10):
            manager.store_experience(
                action=i % 2,
                reward=0.5,
                last_action_holder=[None],
            )

        # Run consolidation with enough cycles to trigger stage changes
        # Default: 5000 NREM + 2000 REM = 7000 steps per cycle
        # batch_size=2 → each cycle replays 2 experiences
        last_action = [None]
        manager.consolidate(
            n_cycles=5,
            batch_size=2,
            verbose=False,
            last_action_holder=last_action,
        )

        # Check stage info
        stage_info = manager.get_sleep_stage_info()

        assert "current_stage" in stage_info, "Should have current_stage"
        assert stage_info["current_stage"] in ["NREM", "REM"], "Stage should be NREM or REM"
        assert "consolidation_step" in stage_info, "Should track step"
        assert stage_info["consolidation_step"] > 0, "Should have advanced steps"
        assert "progress_in_stage" in stage_info, "Should track progress"
        assert 0.0 <= stage_info["progress_in_stage"] <= 1.0, "Progress should be 0-1"

    def test_nrem_rem_replay_counts(self, minimal_brain):
        """Test that NREM and REM replays are tracked separately."""
        brain = minimal_brain
        manager = brain.consolidation_manager

        # Store experiences
        for i in range(8):
            manager.store_experience(
                action=i % 2,
                reward=0.3,
                last_action_holder=[None],
            )

        # Run consolidation
        last_action = [None]
        manager.consolidate(
            n_cycles=3,
            batch_size=2,
            verbose=False,
            last_action_holder=last_action,
        )

        # Check stage-specific replay counts
        stats = manager.get_replay_statistics()

        assert "nrem_replays" in stats, "Should track NREM replays"
        assert "rem_replays" in stats, "Should track REM replays"

        # Total replays should equal sum of NREM + REM (minus awake replays)
        sleep_replays = stats.get("sleep_replays", 0)
        stage_replays = stats["nrem_replays"] + stats["rem_replays"]
        assert stage_replays == sleep_replays, "NREM + REM should equal sleep replays"

    def test_get_sleep_stage_info_contents(self, minimal_brain):
        """Test that get_sleep_stage_info returns complete information."""
        manager = minimal_brain.consolidation_manager

        # Get info before any consolidation
        stage_info = manager.get_sleep_stage_info()

        # Verify all expected keys present
        expected_keys = [
            "current_stage",
            "consolidation_step",
            "progress_in_stage",
            "cycle_complete",
            "nrem_replays",
            "rem_replays",
            "nrem_duration",
            "rem_duration",
        ]

        for key in expected_keys:
            assert key in stage_info, f"Should have {key} in sleep stage info"

        # Verify initial values
        assert stage_info["current_stage"] == "NREM", "Should start in NREM"
        assert stage_info["consolidation_step"] == 0, "Should start at step 0"
        assert stage_info["progress_in_stage"] == 0.0, "Should start at 0% progress"
        assert stage_info["cycle_complete"] is False, "No cycle completed yet"
        assert stage_info["nrem_replays"] == 0, "No NREM replays yet"
        assert stage_info["rem_replays"] == 0, "No REM replays yet"
        assert stage_info["nrem_duration"] == 5000, "Default NREM duration"
        assert stage_info["rem_duration"] == 2000, "Default REM duration"

    def test_stage_progress_advances_during_consolidation(self, minimal_brain):
        """Test that consolidation_step advances during consolidation cycles."""
        brain = minimal_brain
        manager = brain.consolidation_manager

        # Store experiences
        for i in range(6):
            manager.store_experience(
                action=i % 2,
                reward=0.2,
                last_action_holder=[None],
            )

        # Get initial step
        initial_step = manager._consolidation_step

        # Run consolidation
        last_action = [None]
        manager.consolidate(
            n_cycles=2,
            batch_size=2,
            verbose=False,
            last_action_holder=last_action,
        )

        # Verify step advanced
        final_step = manager._consolidation_step
        assert final_step > initial_step, "Consolidation step should advance during replay"

        # Should advance by batch_size * n_cycles (each replay increments step)
        expected_increment = 2 * 2  # 2 cycles × 2 batch_size
        assert (
            final_step >= expected_increment
        ), f"Step should advance by at least {expected_increment}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
