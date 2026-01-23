"""Integration test for new DynamicBrain features.

Verifies that the newly added methods work correctly in a realistic scenario.
"""

import pytest
import torch

from tests.utils import create_test_brain
from thalia.core.diagnostics import (
    BrainSystemDiagnostics,
    HippocampusDiagnostics,
    StriatumDiagnostics,
)


@pytest.fixture
def brain():
    """Create a DynamicBrain for testing."""
    return create_test_brain(
        device="cpu",
        dt_ms=1.0,
        input_size=128,
        thalamus_size=128,
        cortex_size=128,
        hippocampus_size=64,
        pfc_size=128,  # Match striatum's default pfc_size config (128)
        n_actions=4,
    )


class TestCounterfactualLearning:
    """Test counterfactual learning functionality."""

    def test_deliver_reward_with_counterfactual_basic(self, brain):
        """Test basic counterfactual reward delivery."""
        # Run forward pass and select action
        input_data = {"thalamus": torch.randn(128, device=brain.device)}
        brain.forward(input_data, n_timesteps=10)
        action, _ = brain.select_action(explore=True, use_planning=False)

        # Deliver counterfactual reward
        result = brain.deliver_reward_with_counterfactual(
            reward=1.0,
            is_match=True,
            selected_action=action,
            counterfactual_scale=0.5,
        )

        assert "real" in result
        assert "counterfactual" in result
        assert result["selected_action"] == action
        assert result["other_action"] == 1 - action
        assert "novelty_boost" in result

    def test_counterfactual_reward_computation(self, brain):
        """Test that counterfactual rewards are computed correctly."""
        input_data = {"thalamus": torch.randn(128, device=brain.device)}
        brain.forward(input_data, n_timesteps=10)
        brain.select_action(explore=True, use_planning=False)  # Need to select action first

        # Test MATCH trial with MATCH action (correct)
        result = brain.deliver_reward_with_counterfactual(
            reward=1.0,
            is_match=True,
            selected_action=0,  # MATCH
            counterfactual_scale=0.5,
        )
        # Other action (NOMATCH) would have been wrong, so reward = -1.0
        assert result["counterfactual_reward"] == -1.0

        # Test MATCH trial with NOMATCH action (incorrect)
        brain.forward(input_data, n_timesteps=10)
        brain.select_action(explore=True, use_planning=False)  # Need to select action first
        result = brain.deliver_reward_with_counterfactual(
            reward=-1.0,
            is_match=True,
            selected_action=1,  # NOMATCH (wrong)
            counterfactual_scale=0.5,
        )
        # Other action (MATCH) would have been correct, so reward = 1.0
        assert result["counterfactual_reward"] == 1.0

    def test_counterfactual_scaling(self, brain):
        """Test that counterfactual_scale parameter works."""
        input_data = {"thalamus": torch.randn(128, device=brain.device)}

        # Test with different scales
        for scale in [0.0, 0.5, 1.0]:
            brain.forward(input_data, n_timesteps=10)
            brain.select_action(explore=True, use_planning=False)  # Need to select action first
            result = brain.deliver_reward_with_counterfactual(
                reward=1.0,
                is_match=True,
                selected_action=0,
                counterfactual_scale=scale,
            )
            # Scaled reward should be base reward * novelty * scale
            scaled = result["counterfactual"]["scaled_reward"]
            expected = result["counterfactual_reward"] * result["novelty_boost"] * scale
            assert abs(scaled - expected) < 1e-6


class TestNoveltyBoost:
    """Test novelty boost functionality."""

    def test_novelty_boost_default(self, brain):
        """Test that novelty boost defaults to 1.0."""
        boost = brain._get_novelty_boost()
        assert boost == 1.0

    def test_novelty_boost_always_positive(self, brain):
        """Test that novelty boost is always >= 1.0."""
        # Set various novelty signals
        for value in [0.0, 0.5, 1.0, 1.5, 2.0]:
            brain._novelty_signal = value
            boost = brain._get_novelty_boost()
            assert boost >= 1.0, f"Novelty boost should be >= 1.0, got {boost}"


class TestStriatumDiagnostics:
    """Test striatum diagnostic collection."""

    def test_collect_striatum_diagnostics_structure(self, brain):
        """Test that striatum diagnostics have correct structure."""
        diag = brain._collect_striatum_diagnostics()

        assert isinstance(diag, StriatumDiagnostics)
        assert isinstance(diag.d1_per_action, list)
        assert isinstance(diag.d2_per_action, list)
        assert isinstance(diag.net_per_action, list)
        assert len(diag.d1_per_action) == 10  # n_actions (default for BrainBuilder.preset)
        assert len(diag.d2_per_action) == 10
        assert len(diag.net_per_action) == 10

    def test_striatum_diagnostics_after_action(self, brain):
        """Test diagnostics after action selection."""
        # Select an action
        input_data = {"thalamus": torch.randn(128, device=brain.device)}
        brain.forward(input_data, n_timesteps=10)
        action, _ = brain.select_action(explore=True, use_planning=False)

        # Get diagnostics
        diag = brain._collect_striatum_diagnostics()
        assert diag.last_action == action
        assert isinstance(diag.exploring, bool)
        assert isinstance(diag.exploration_prob, float)


class TestHippocampusDiagnostics:
    """Test hippocampus diagnostic collection."""

    def test_collect_hippocampus_diagnostics_structure(self, brain):
        """Test that hippocampus diagnostics have correct structure."""
        diag = brain._collect_hippocampus_diagnostics()

        assert isinstance(diag, HippocampusDiagnostics)
        assert isinstance(diag.ca1_total_spikes, float)
        assert isinstance(diag.ca1_normalized, float)
        assert isinstance(diag.dg_spikes, float)
        assert isinstance(diag.ca3_spikes, float)

    def test_hippocampus_diagnostics_after_activity(self, brain):
        """Test diagnostics after hippocampus activity."""
        # Run forward passes to generate activity
        input_data = {"thalamus": torch.randn(128, device=brain.device)}
        for _ in range(5):
            brain.forward(input_data, n_timesteps=10)

        diag = brain._collect_hippocampus_diagnostics()
        # Normalized should be between 0 and reasonable max
        assert 0.0 <= diag.ca1_normalized <= 10.0


class TestStructuredDiagnostics:
    """Test structured diagnostics collection."""

    def test_get_structured_diagnostics_returns_dataclass(self, brain):
        """Test that get_structured_diagnostics returns correct type."""
        diag = brain.get_structured_diagnostics()
        assert isinstance(diag, BrainSystemDiagnostics)

    def test_structured_diagnostics_includes_all_components(self, brain):
        """Test that all component diagnostics are included."""
        diag = brain.get_structured_diagnostics()

        assert isinstance(diag.striatum, StriatumDiagnostics)
        assert isinstance(diag.hippocampus, HippocampusDiagnostics)
        assert isinstance(diag.trial_num, int)
        assert isinstance(diag.is_match, bool)
        assert isinstance(diag.selected_action, int)
        assert isinstance(diag.correct, bool)

    def test_structured_diagnostics_updates_with_actions(self, brain):
        """Test that diagnostics update correctly after actions."""
        # Initial diagnostics
        _ = brain.get_structured_diagnostics()  # Verify diagnostics work

        # Take action
        input_data = {"thalamus": torch.randn(128, device=brain.device)}
        brain.forward(input_data, n_timesteps=10)
        action, _ = brain.select_action(explore=True, use_planning=False)

        # Updated diagnostics
        diag2 = brain.get_structured_diagnostics()
        assert diag2.selected_action == action


class TestFeatureParity:
    """Test that DynamicBrain now has full parity with EventDrivenBrain."""

    def test_all_critical_methods_present(self, brain):
        """Verify all critical methods are implemented."""
        # Core RL interface
        assert hasattr(brain, "forward")
        assert hasattr(brain, "select_action")
        assert hasattr(brain, "deliver_reward")

        # NEW: Counterfactual learning
        assert hasattr(brain, "deliver_reward_with_counterfactual")
        assert callable(brain.deliver_reward_with_counterfactual)

        # NEW: Structured diagnostics (tests novelty boost and component diagnostics behaviorally)
        assert hasattr(brain, "get_structured_diagnostics")
        assert callable(brain.get_structured_diagnostics)

        # Test behavioral contract: structured diagnostics should work
        diag = brain.get_structured_diagnostics()
        assert isinstance(diag, BrainSystemDiagnostics), "Should return structured diagnostics"
        assert hasattr(diag, "striatum"), "Should include striatum diagnostics"
        assert hasattr(diag, "hippocampus"), "Should include hippocampus diagnostics"

        # State management
        assert hasattr(brain, "get_full_state")
        assert hasattr(brain, "load_full_state")
        assert hasattr(brain, "reset_state")

        # Growth
        assert hasattr(brain, "check_growth_needs")
        assert hasattr(brain, "auto_grow")

        # Diagnostics
        assert hasattr(brain, "get_diagnostics")

    def test_full_rl_episode_with_all_features(self, brain):
        """Test a complete RL episode using all new features."""
        rewards = []

        for trial in range(3):
            # Forward pass
            input_data = {"thalamus": torch.randn(128, device=brain.device)}
            brain.forward(input_data, n_timesteps=10)

            # Select action
            action, _ = brain.select_action(explore=True, use_planning=False)

            # Deliver counterfactual reward
            is_match = trial % 2 == 0
            reward = 1.0 if (action == 0 and is_match) or (action == 1 and not is_match) else -1.0

            _ = brain.deliver_reward_with_counterfactual(
                reward=reward,
                is_match=is_match,
                selected_action=action,
                counterfactual_scale=0.5,
            )
            rewards.append(reward)

            # Get all diagnostic types
            dict_diag = brain.get_diagnostics()
            struct_diag = brain.get_structured_diagnostics()
            striatum_diag = brain._collect_striatum_diagnostics()
            hippo_diag = brain._collect_hippocampus_diagnostics()

            # Verify everything is working
            assert "components" in dict_diag
            assert isinstance(struct_diag, BrainSystemDiagnostics)
            assert isinstance(striatum_diag, StriatumDiagnostics)
            assert isinstance(hippo_diag, HippocampusDiagnostics)

        # Episode completed successfully
        assert len(rewards) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
