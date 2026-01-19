"""
Unit tests for EmergentGoalSystem.

Tests the core mechanisms of emergent goal representation:
1. Goal tagging from working memory activity
2. Tag decay over time
3. Hebbian transition learning
4. Goal selection by value and recency
5. Subgoal prediction from abstract patterns
6. Value consolidation with dopamine
7. State dict serialization/deserialization
8. Abstract/concrete population separation
"""

import pytest
import torch

from thalia.regions.prefrontal.goal_emergence import EmergentGoalSystem


@pytest.fixture
def device():
    """Device for tests."""
    return "cpu"


@pytest.fixture
def goal_system(device):
    """Create a test EmergentGoalSystem."""
    n_wm_neurons = 100
    n_abstract = 30  # 30% abstract (rostral PFC)
    n_concrete = 70  # 70% concrete (caudal PFC)

    return EmergentGoalSystem(
        n_wm_neurons=n_wm_neurons,
        n_abstract=n_abstract,
        n_concrete=n_concrete,
        device=device,
    )


def test_goal_tags_created_by_activity(goal_system, device):
    """Test that active WM patterns create synaptic tags."""
    # Create WM pattern with some active neurons
    wm_pattern = torch.zeros(100, device=device)
    wm_pattern[10:20] = 0.8  # Strong activation
    wm_pattern[30:35] = 0.2  # Weak activation (below threshold)

    # Initially no tags
    assert torch.allclose(goal_system.goal_tags, torch.zeros(100, device=device))

    # Update tags
    goal_system.update_goal_tags(wm_pattern)

    # Strong activations should create tags
    assert torch.all(goal_system.goal_tags[10:20] > 0.7)

    # Weak activations (below 0.3 threshold) should not create tags
    assert torch.all(goal_system.goal_tags[30:35] < 0.01)

    # Inactive neurons should have no tags
    assert torch.all(goal_system.goal_tags[50:60] == 0.0)


def test_goal_tags_decay_over_time(goal_system, device):
    """Test that synaptic tags decay exponentially."""
    # Create initial tags
    wm_pattern = torch.zeros(100, device=device)
    wm_pattern[10:20] = 1.0
    goal_system.update_goal_tags(wm_pattern)

    initial_tags = goal_system.goal_tags[10:20].clone()

    # Multiple timesteps without activity
    for _ in range(10):
        inactive_pattern = torch.zeros(100, device=device)
        goal_system.update_goal_tags(inactive_pattern)

    # Tags should decay (but not to zero with 10 steps)
    final_tags = goal_system.goal_tags[10:20]
    assert torch.all(final_tags < initial_tags)
    assert torch.all(final_tags > 0.0)

    # Decay should be exponential: tag_t = tag_0 * decay^t
    expected_tags = initial_tags * (goal_system.tag_decay**10)
    assert torch.allclose(final_tags, expected_tags, atol=1e-5)


def test_transition_learning_hebbian(goal_system, device):
    """Test that goal transitions are learned via Hebbian rule."""
    # Create abstract and concrete patterns
    abstract_pattern = torch.zeros(30, device=device)
    abstract_pattern[0:5] = 1.0  # Abstract goal A

    concrete_pattern = torch.zeros(70, device=device)
    concrete_pattern[10:20] = 1.0  # Concrete subgoal B

    # Initially no associations
    initial_weights = goal_system.transition_weights.clone()

    # Learn transition A→B
    goal_system.learn_transition(abstract_pattern, concrete_pattern, learning_rate=0.1)

    # Weights should increase where both patterns were active
    # transition[concrete, abstract] should increase at [10:20, 0:5]
    updated_weights = goal_system.transition_weights[10:20, 0:5]
    assert torch.all(updated_weights > initial_weights[10:20, 0:5])

    # Weights where patterns were inactive should not change much
    assert torch.allclose(
        goal_system.transition_weights[50:60, 20:25],
        initial_weights[50:60, 20:25],
        atol=1e-5,
    )


def test_subgoal_prediction_from_abstract(goal_system, device):
    """Test that concrete subgoals can be predicted from abstract goals."""
    # Train a transition: abstract[0:5] → concrete[10:20]
    abstract_pattern = torch.zeros(30, device=device)
    abstract_pattern[0:5] = 1.0

    concrete_pattern = torch.zeros(70, device=device)
    concrete_pattern[10:20] = 1.0

    # Train the transition multiple times
    for _ in range(20):
        goal_system.learn_transition(abstract_pattern, concrete_pattern, learning_rate=0.1)

    # Now predict subgoal from abstract pattern
    predicted_subgoal = goal_system.predict_subgoal(abstract_pattern)

    # Prediction should activate neurons in the trained range [10:20]
    # Winner-take-all selects top 10% = 7 neurons
    assert predicted_subgoal.sum() > 0  # Some neurons active
    assert torch.any(predicted_subgoal[10:20] > 0.5)  # Some overlap with trained pattern


def test_goal_selection_by_value(goal_system, device):
    """Test that high-value goals are preferentially selected."""
    # Create two competing goal patterns
    goal_a = torch.zeros(100, device=device)
    goal_a[0:10] = 1.0

    goal_b = torch.zeros(100, device=device)
    goal_b[20:30] = 1.0

    # Set high value for goal A
    goal_system.value_weights[0:10] = 1.0  # High value
    goal_system.value_weights[20:30] = -0.5  # Low value

    # Select from competing goals (with moderate dopamine)
    selected = goal_system.select_goal([goal_a, goal_b], dopamine=0.5)

    # Should select goal A (higher value) most of the time
    # Run multiple times to check stochastic selection
    selections_a = 0
    for _ in range(50):
        selected = goal_system.select_goal([goal_a, goal_b], dopamine=0.5)
        if torch.sum(selected[0:10]) > 0.5:
            selections_a += 1

    # At least 70% should select high-value goal
    assert selections_a > 35


def test_goal_selection_by_recency(goal_system, device):
    """Test that recently-active goals are preferentially selected."""
    # Create two competing goal patterns
    goal_a = torch.zeros(100, device=device)
    goal_a[0:10] = 1.0

    goal_b = torch.zeros(100, device=device)
    goal_b[20:30] = 1.0

    # Tag goal A as recently active
    goal_system.goal_tags[0:10] = 0.8
    goal_system.goal_tags[20:30] = 0.1

    # Select from competing goals (with moderate dopamine)
    selections_a = 0
    for _ in range(50):
        selected = goal_system.select_goal([goal_a, goal_b], dopamine=0.5)
        if torch.sum(selected[0:10]) > 0.5:
            selections_a += 1

    # At least 70% should select recently-tagged goal
    assert selections_a > 35


def test_value_consolidation_with_dopamine(goal_system, device):
    """Test that tagged goals strengthen with dopamine."""
    # Tag some goal patterns
    goal_system.goal_tags[10:20] = 0.8  # Recently active
    goal_system.goal_tags[30:40] = 0.2  # Weakly active

    initial_values = goal_system.value_weights.clone()

    # Consolidate with high dopamine
    goal_system.consolidate_valuable_goals(dopamine=1.0, learning_rate=0.1)

    # Strongly-tagged patterns should increase in value
    assert torch.all(goal_system.value_weights[10:20] > initial_values[10:20])

    # Weakly-tagged patterns should increase less
    value_increase_strong = (goal_system.value_weights[10:20] - initial_values[10:20]).mean()
    value_increase_weak = (goal_system.value_weights[30:40] - initial_values[30:40]).mean()
    assert value_increase_strong > value_increase_weak

    # Untagged patterns should not change
    assert torch.allclose(goal_system.value_weights[50:60], initial_values[50:60], atol=1e-5)


def test_value_consolidation_requires_dopamine(goal_system, device):
    """Test that consolidation does not occur without dopamine."""
    # Tag some goal patterns
    goal_system.goal_tags[10:20] = 0.8

    initial_values = goal_system.value_weights.clone()

    # Try to consolidate with no dopamine
    goal_system.consolidate_valuable_goals(dopamine=0.0, learning_rate=0.1)

    # Values should not change without dopamine
    assert torch.allclose(goal_system.value_weights, initial_values, atol=1e-6)


def test_abstract_concrete_split(goal_system, device):
    """Test that neurons are correctly split into abstract/concrete populations."""
    # Check population sizes
    assert len(goal_system.abstract_neurons) == 30
    assert len(goal_system.concrete_neurons) == 70

    # Check populations are non-overlapping
    abstract_set = set(goal_system.abstract_neurons.tolist())
    concrete_set = set(goal_system.concrete_neurons.tolist())
    assert len(abstract_set & concrete_set) == 0

    # Check populations cover all neurons
    assert len(abstract_set | concrete_set) == 100

    # Check abstract neurons come first
    assert torch.all(goal_system.abstract_neurons == torch.arange(30, device=device))
    assert torch.all(goal_system.concrete_neurons == torch.arange(30, 100, device=device))


def test_state_dict_serialization(goal_system, device):
    """Test that state can be saved and restored."""
    # Modify state
    goal_system.transition_weights[10:20, 0:5] = 0.7
    goal_system.value_weights[10:20] = 0.9
    goal_system.goal_tags[5:15] = 0.6

    # Save state
    state_dict = goal_system.get_state_dict()

    # Create new system and load state
    new_system = EmergentGoalSystem(
        n_wm_neurons=100,
        n_abstract=30,
        n_concrete=70,
        device=device,
    )
    new_system.load_state_dict(state_dict)

    # Verify state matches
    assert torch.allclose(new_system.transition_weights, goal_system.transition_weights)
    assert torch.allclose(new_system.value_weights, goal_system.value_weights)
    assert torch.allclose(new_system.goal_tags, goal_system.goal_tags)


def test_reset_tags(goal_system, device):
    """Test that tags can be reset to zero."""
    # Create some tags
    goal_system.goal_tags[10:20] = 0.8
    assert torch.any(goal_system.goal_tags > 0)

    # Reset tags
    goal_system.reset_tags()

    # All tags should be zero
    assert torch.allclose(goal_system.goal_tags, torch.zeros(100, device=device))


def test_transition_weights_clipping(goal_system, device):
    """Test that transition weights are clipped to [0, 1]."""
    # Create patterns
    abstract_pattern = torch.zeros(30, device=device)
    abstract_pattern[0:5] = 1.0

    concrete_pattern = torch.zeros(70, device=device)
    concrete_pattern[10:20] = 1.0

    # Learn many times to potentially exceed bounds
    for _ in range(100):
        goal_system.learn_transition(abstract_pattern, concrete_pattern, learning_rate=0.1)

    # Weights should be clipped to [0, 1]
    assert torch.all(goal_system.transition_weights >= 0.0)
    assert torch.all(goal_system.transition_weights <= 1.0)


def test_value_weights_clipping(goal_system, device):
    """Test that value weights are clipped to [-1, 1]."""
    # Create very strong tags
    goal_system.goal_tags[:] = 1.0

    # Consolidate many times with high dopamine
    for _ in range(100):
        goal_system.consolidate_valuable_goals(dopamine=1.0, learning_rate=0.1)

    # Values should be clipped to [-1, 1]
    assert torch.all(goal_system.value_weights >= -1.0)
    assert torch.all(goal_system.value_weights <= 1.0)
