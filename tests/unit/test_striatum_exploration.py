"""
Integration tests for Striatum with ExplorationManager

Tests the integration of exploration functionality into Striatum,
including action selection, UCB exploration, and adaptive tonic DA.
"""

import pytest
import torch

from thalia.regions.striatum.striatum import Striatum
from thalia.regions.striatum.config import StriatumConfig


@pytest.fixture
def device():
    """Device for testing (CPU)."""
    return torch.device("cpu")


@pytest.fixture
def striatum_config(device):
    """Basic Striatum configuration."""
    return StriatumConfig(
        n_input=10,
        n_output=2,  # 2 actions
        device=device,
        population_coding=True,
        neurons_per_action=5,
        ucb_exploration=True,
        ucb_coefficient=1.0,
        adaptive_exploration=True,
        performance_window=10,
        min_tonic_dopamine=0.0,
        max_tonic_dopamine=0.3,
        tonic_dopamine=0.1,
    )


@pytest.fixture
def striatum(striatum_config):
    """Create Striatum for testing."""
    return Striatum(striatum_config)


class TestStriatumExplorationIntegration:
    """Test Striatum integration with ExplorationManager."""
    
    def test_exploration_manager_created(self, striatum):
        """Test that exploration manager is created during init."""
        assert hasattr(striatum, 'exploration_manager')
        assert striatum.exploration_manager is not None
        assert striatum.exploration_manager.n_actions == 2
    
    def test_property_delegation(self, striatum):
        """Test that properties delegate to exploration manager."""
        # Access properties
        action_counts = striatum._action_counts
        total_trials = striatum._total_trials
        tonic_da = striatum.tonic_dopamine
        
        # Verify delegation
        assert torch.all(action_counts == striatum.exploration_manager._action_counts)
        assert total_trials == striatum.exploration_manager._total_trials
        assert tonic_da == striatum.exploration_manager.tonic_dopamine
    
    def test_tonic_dopamine_setter(self, striatum):
        """Test that tonic_dopamine property setter works."""
        striatum.tonic_dopamine = 0.25
        
        assert striatum.tonic_dopamine == 0.25
        assert striatum.exploration_manager.tonic_dopamine == 0.25


class TestActionSelectionWithExploration:
    """Test action selection with exploration."""
    
    def test_finalize_action_ucb(self, striatum):
        """Test that finalize_action uses UCB bonus."""
        # Accumulate some votes
        striatum._d1_votes_accumulated[0] = 10.0
        striatum._d2_votes_accumulated[0] = 5.0
        striatum._d1_votes_accumulated[1] = 8.0
        striatum._d2_votes_accumulated[1] = 3.0
        
        # Take action 0 many times to bias UCB
        for _ in range(10):
            striatum.exploration_manager.update_action_counts(0)
        
        # Finalize action (should favor action 1 due to UCB bonus)
        result = striatum.finalize_action(explore=True)
        
        assert "ucb_bonus" in result
        assert "selected_action" in result
        assert result["ucb_bonus"][1] > result["ucb_bonus"][0]
    
    def test_update_action_counts_called(self, striatum):
        """Test that finalize_action updates action counts."""
        initial_trials = striatum._total_trials
        
        # Set up votes
        striatum._d1_votes_accumulated[0] = 10.0
        striatum._d2_votes_accumulated[0] = 2.0
        
        # Finalize action
        result = striatum.finalize_action(explore=False)
        selected = result["selected_action"]
        
        # Check counts updated
        assert striatum._total_trials == initial_trials + 1
        assert striatum._action_counts[selected] > 0


class TestAdaptiveExplorationIntegration:
    """Test adaptive exploration in Striatum."""
    
    def test_deliver_reward_adjusts_tonic_da(self, striatum):
        """Test that deliver_reward adjusts tonic DA via exploration manager."""
        initial_tonic = striatum.tonic_dopamine
        
        # Set up for learning
        input_spikes = torch.zeros(10, dtype=torch.bool, device=striatum.device)
        input_spikes[0] = True
        
        # Forward pass to build eligibility
        striatum.forward(input_spikes)
        
        # Set last action for learning
        striatum.last_action = 0
        
        # Set dopamine (mimics Brain's VTA)
        striatum.set_dopamine(0.5)
        
        # Deliver reward (should adjust tonic DA)
        for _ in range(5):
            striatum.deliver_reward(reward=1.0)
        
        # Tonic DA should have changed (decreased with success)
        assert striatum.tonic_dopamine != initial_tonic


class TestCheckpointing:
    """Test state saving and loading with exploration."""
    
    def test_get_full_state_includes_exploration(self, striatum):
        """Test that get_full_state includes exploration manager state."""
        # Build up some state
        striatum.exploration_manager.update_action_counts(0)
        striatum.exploration_manager.adjust_tonic_dopamine(reward=1.0)
        
        state = striatum.get_full_state()
        
        assert "exploration_state" in state
        assert "manager_state" in state["exploration_state"]
        
        manager_state = state["exploration_state"]["manager_state"]
        assert "action_counts" in manager_state
        assert "tonic_dopamine" in manager_state
    
    def test_load_full_state_restores_exploration(self, striatum_config):
        """Test that load_full_state restores exploration manager state."""
        # Create striatum and build state
        striatum1 = Striatum(striatum_config)
        striatum1.exploration_manager.update_action_counts(0)
        striatum1.exploration_manager.update_action_counts(0)
        striatum1.exploration_manager.update_action_counts(1)
        striatum1.exploration_manager.adjust_tonic_dopamine(reward=1.0)
        
        state1 = striatum1.get_full_state()
        
        # Create new striatum and load state
        striatum2 = Striatum(striatum_config)
        striatum2.load_full_state(state1)
        
        # Check exploration state matches
        assert torch.all(striatum2._action_counts == striatum1._action_counts)
        assert striatum2._total_trials == striatum1._total_trials
        assert striatum2.tonic_dopamine == striatum1.tonic_dopamine
    
    def test_backward_compatibility_old_checkpoint_format(self, striatum_config):
        """Test loading old checkpoint format (before ExplorationManager)."""
        # Create striatum
        striatum = Striatum(striatum_config)
        
        # Create old-format state (direct action_counts, not manager_state)
        old_state = striatum.get_full_state()
        
        # Modify to old format
        exploration_state = old_state["exploration_state"]
        exploration_state["action_counts"] = torch.tensor([3, 1], device=striatum.device)
        exploration_state["total_trials"] = 4
        exploration_state["recent_rewards"] = [1.0, 0.0, 1.0]
        exploration_state["recent_accuracy"] = 0.67
        # Remove new format
        if "manager_state" in exploration_state:
            del exploration_state["manager_state"]
        
        # Should load without error
        striatum.load_full_state(old_state)
        
        # Check state loaded correctly
        assert striatum._action_counts[0] == 3
        assert striatum._action_counts[1] == 1
        assert striatum._total_trials == 4


class TestGrowthWithExploration:
    """Test neuron growth with exploration state."""
    
    def test_add_neurons_grows_exploration(self, striatum):
        """Test that add_neurons grows exploration manager."""
        # Set up initial state
        striatum.exploration_manager.update_action_counts(0)
        striatum.exploration_manager.update_action_counts(1)
        
        initial_n_actions = striatum.n_actions
        old_counts = striatum._action_counts.clone()
        
        # Add 1 action (with population coding = neurons_per_action neurons)
        striatum.add_neurons(n_new=1)
        
        # Check action space expanded
        assert striatum.n_actions == initial_n_actions + 1
        assert striatum.exploration_manager.n_actions == initial_n_actions + 1
        
        # Check old counts preserved
        assert torch.all(striatum._action_counts[:initial_n_actions] == old_counts)
        
        # Check new action initialized to zero
        assert striatum._action_counts[initial_n_actions] == 0


class TestExplorationDisabled:
    """Test behavior when exploration is disabled."""
    
    @pytest.fixture
    def no_exploration_config(self, device):
        """Config with exploration disabled."""
        return StriatumConfig(
            n_input=10,
            n_output=2,
            device=device,
            ucb_exploration=False,
            adaptive_exploration=False,
        )
    
    def test_ucb_disabled(self, no_exploration_config):
        """Test UCB returns zeros when disabled."""
        striatum = Striatum(no_exploration_config)
        
        # Take some actions
        striatum.exploration_manager.update_action_counts(0)
        
        # UCB should be zero
        ucb_bonus = striatum.exploration_manager.compute_ucb_bonus()
        assert torch.all(ucb_bonus == 0)
    
    def test_adaptive_exploration_disabled(self, no_exploration_config):
        """Test tonic DA doesn't change when adaptive exploration disabled."""
        striatum = Striatum(no_exploration_config)
        
        initial_tonic = striatum.tonic_dopamine
        
        # Try to adjust
        striatum.exploration_manager.adjust_tonic_dopamine(reward=1.0)
        
        assert striatum.tonic_dopamine == initial_tonic


class TestExplorationDiagnostics:
    """Test exploration diagnostic information."""
    
    def test_get_diagnostics(self, striatum):
        """Test getting exploration diagnostics."""
        # Build up state
        striatum.exploration_manager.update_action_counts(0)
        striatum.exploration_manager.update_action_counts(0)
        striatum.exploration_manager.update_action_counts(1)
        
        diagnostics = striatum.exploration_manager.get_diagnostics()
        
        assert diagnostics["total_trials"] == 3
        assert diagnostics["action_counts"] == [2, 1]
        assert diagnostics["most_tried_action"] == 0
        assert diagnostics["least_tried_action"] == 1
