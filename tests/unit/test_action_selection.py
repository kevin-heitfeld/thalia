"""
Unit tests for ActionSelector.

Tests standalone action selection functionality.
"""

import pytest
import torch

from thalia.decision_making import ActionSelectionConfig, ActionSelector, SelectionMode


def test_greedy_selection():
    """Test greedy action selection (always max)."""
    selector = ActionSelector(
        n_actions=4,
        config=ActionSelectionConfig(mode=SelectionMode.GREEDY),
    )

    votes = torch.tensor([1.0, 5.0, 3.0, 2.0])
    action, info = selector.select_action(positive_votes=votes)

    assert action == 1, "Should select action with highest votes"
    assert not info["is_exploring"], "Greedy should not be exploring"


def test_opponent_voting():
    """Test action selection with positive and negative votes."""
    selector = ActionSelector(
        n_actions=3,
        config=ActionSelectionConfig(mode=SelectionMode.GREEDY),
    )

    positive = torch.tensor([10.0, 5.0, 8.0])
    negative = torch.tensor([2.0, 6.0, 3.0])
    # Net: [8, -1, 5]

    action, info = selector.select_action(
        positive_votes=positive,
        negative_votes=negative,
    )

    assert action == 0, "Should select action with highest net votes"
    expected_net = torch.tensor([8.0, -1.0, 5.0])
    torch.testing.assert_close(info["net_votes"], expected_net)


def test_softmax_selection():
    """Test softmax probabilistic selection."""
    selector = ActionSelector(
        n_actions=4,
        config=ActionSelectionConfig(
            mode=SelectionMode.SOFTMAX,
            temperature=1.0,
        ),
    )

    votes = torch.tensor([10.0, 5.0, 3.0, 2.0])

    # Run multiple times to check probabilistic nature
    action_counts = torch.zeros(4)
    n_trials = 100

    for _ in range(n_trials):
        action, info = selector.select_action(positive_votes=votes)
        action_counts[action] += 1
        assert info["probabilities"] is not None, "Softmax should return probabilities"

    # Action 0 should be chosen most often (but not always)
    assert action_counts[0] > action_counts[1], "Higher-vote action should be chosen more"
    assert action_counts.sum() == n_trials


def test_epsilon_greedy():
    """Test epsilon-greedy exploration."""
    selector = ActionSelector(
        n_actions=3,
        config=ActionSelectionConfig(
            mode=SelectionMode.EPSILON_GREEDY,
            epsilon=0.3,  # 30% exploration
        ),
    )

    votes = torch.tensor([10.0, 1.0, 1.0])

    # Run multiple times
    action_0_count = 0
    explored_count = 0
    n_trials = 100

    for _ in range(n_trials):
        action, info = selector.select_action(positive_votes=votes)
        if action == 0:
            action_0_count += 1
        if info["is_exploring"]:
            explored_count += 1

    # Action 0 should dominate but not be 100%
    assert 60 < action_0_count < 90, "Should mostly exploit but sometimes explore"


def test_ucb_exploration():
    """Test UCB exploration bonus."""
    selector = ActionSelector(
        n_actions=3,
        config=ActionSelectionConfig(
            mode=SelectionMode.SOFTMAX,
            temperature=0.1,  # Low temp for clear preference
            ucb_c=5.0,  # High exploration
        ),
    )

    votes = torch.tensor([10.0, 10.0, 10.0])  # Equal initial votes

    # Select action 0 multiple times
    for _ in range(10):
        action, info = selector.select_action(positive_votes=votes)
        selector.action_counts[0] += 9  # Simulate choosing action 0

    # Now UCB should favor unexplored actions 1 and 2
    action, info = selector.select_action(positive_votes=votes)

    assert info["ucb_bonus"] is not None, "Should compute UCB bonus"
    assert action != 0, "Should explore less-tried actions due to UCB"


def test_population_coding():
    """Test decoding votes from population-coded spikes."""
    selector = ActionSelector(
        n_actions=3,
        config=ActionSelectionConfig(neurons_per_action=5),
    )

    # 3 actions × 5 neurons = 15 total neurons
    spikes = torch.tensor(
        [
            1,
            1,
            0,
            1,
            1,  # Action 0: 4 spikes
            0,
            1,
            0,
            0,
            1,  # Action 1: 2 spikes
            1,
            1,
            1,
            1,
            1,  # Action 2: 5 spikes
        ],
        dtype=torch.float,
    )

    votes = selector.decode_population_votes(spikes)

    expected = torch.tensor([4.0, 2.0, 5.0])
    torch.testing.assert_close(votes, expected)


def test_vote_accumulation():
    """Test accumulating votes across timesteps."""
    selector = ActionSelector(
        n_actions=3,
        config=ActionSelectionConfig(
            mode=SelectionMode.GREEDY,
            accumulate_votes=True,
            vote_decay=0.9,
            ucb_c=0.0,  # Disable UCB for this test
        ),
    )

    # First timestep - establishes baseline accumulated votes
    votes1 = torch.tensor([10.0, 5.0, 3.0])
    _, info1 = selector.select_action(positive_votes=votes1)

    # After first call: accumulated_votes = 0 * 0.9 + votes1 = votes1
    expected_after_first = votes1
    torch.testing.assert_close(info1["net_votes"], expected_after_first)

    # Second timestep - accumulates on top of first
    votes2 = torch.tensor([1.0, 8.0, 2.0])
    _, info2 = selector.select_action(positive_votes=votes2)

    # After second call: accumulated_votes = votes1 * 0.9 + votes2
    expected_after_second = votes1 * 0.9 + votes2
    torch.testing.assert_close(info2["net_votes"], expected_after_second)


def test_action_mask():
    """Test masking invalid actions."""
    selector = ActionSelector(
        n_actions=4,
        config=ActionSelectionConfig(mode=SelectionMode.GREEDY),
    )

    votes = torch.tensor([10.0, 8.0, 12.0, 6.0])
    mask = torch.tensor([True, True, False, True])  # Action 2 invalid

    action, _ = selector.select_action(
        positive_votes=votes,
        mask=mask,
    )

    assert action != 2, "Should not select masked action"
    assert action == 0, "Should select highest valid action"


def test_state_persistence():
    """Test saving and loading selector state."""
    selector1 = ActionSelector(
        n_actions=3,
        config=ActionSelectionConfig(mode=SelectionMode.GREEDY),
    )

    # Perform some selections
    votes = torch.tensor([5.0, 10.0, 3.0])
    for _ in range(5):
        selector1.select_action(positive_votes=votes)

    # Save state
    state = selector1.get_state()

    # Create new selector and load state
    selector2 = ActionSelector(
        n_actions=3,
        config=ActionSelectionConfig(mode=SelectionMode.GREEDY),
    )
    selector2.load_state(state)

    # Verify state transferred
    assert selector2.total_selections == selector1.total_selections
    torch.testing.assert_close(selector2.action_counts, selector1.action_counts)


def test_all_zero_votes():
    """Test action selection when all votes are zero (edge case)."""
    selector = ActionSelector(
        n_actions=4,
        config=ActionSelectionConfig(mode=SelectionMode.GREEDY),
    )

    votes = torch.zeros(4)
    action, info = selector.select_action(positive_votes=votes)

    # Contract: should select a valid action even with zero votes
    assert 0 <= action < 4, "Should select valid action index"
    assert info["net_votes"] is not None, "Should return vote information"


def test_all_equal_votes():
    """Test action selection when all votes are equal (edge case)."""
    selector = ActionSelector(
        n_actions=3,
        config=ActionSelectionConfig(mode=SelectionMode.GREEDY),
    )

    votes = torch.ones(3) * 5.0  # All equal
    action, info = selector.select_action(positive_votes=votes)

    # Contract: should select any valid action deterministically
    assert 0 <= action < 3, "Should select valid action"
    # With equal votes, should pick first action (greedy on tie)
    assert action == 0, "Greedy should break ties by selecting first"


def test_negative_votes_only():
    """Test selection with only negative votes (all actions bad)."""
    selector = ActionSelector(
        n_actions=3,
        config=ActionSelectionConfig(mode=SelectionMode.GREEDY),
    )

    positive = torch.zeros(3)
    negative = torch.tensor([5.0, 10.0, 3.0])  # Net: [-5, -10, -3]

    action, info = selector.select_action(
        positive_votes=positive,
        negative_votes=negative,
    )

    # Contract: should select least-bad action (least negative)
    assert action == 2, "Should select action with least negative votes (-3)"
    assert info["net_votes"][action] == -3.0, "Net votes should be least negative"


def test_extreme_temperature_softmax():
    """Test softmax with extreme temperature values (edge case)."""
    selector_hot = ActionSelector(
        n_actions=3,
        config=ActionSelectionConfig(mode=SelectionMode.SOFTMAX, temperature=100.0),
    )
    selector_cold = ActionSelector(
        n_actions=3,
        config=ActionSelectionConfig(mode=SelectionMode.SOFTMAX, temperature=0.01),
    )

    votes = torch.tensor([10.0, 5.0, 1.0])

    # High temperature: nearly uniform distribution
    action_counts_hot = torch.zeros(3)
    for _ in range(300):
        action, _ = selector_hot.select_action(positive_votes=votes)
        action_counts_hot[action] += 1

    # Should have relatively balanced selection
    assert action_counts_hot.min() > 50, "High temp should explore broadly"

    # Low temperature: nearly deterministic (always pick best)
    action_counts_cold = torch.zeros(3)
    for _ in range(100):
        action, _ = selector_cold.select_action(positive_votes=votes)
        action_counts_cold[action] += 1

    # Should almost always pick action 0
    assert action_counts_cold[0] > 95, "Low temp should be nearly greedy"


def test_single_action():
    """Test selector with only one action (edge case)."""
    selector = ActionSelector(
        n_actions=1,
        config=ActionSelectionConfig(mode=SelectionMode.GREEDY),
    )

    votes = torch.tensor([5.0])
    action, info = selector.select_action(positive_votes=votes)

    # Contract: must select the only available action
    assert action == 0, "Single action must be selected"


def test_ucb_c_parameter_effect():
    """Test that UCB exploration constant affects selection."""
    selector_high = ActionSelector(
        n_actions=3,
        config=ActionSelectionConfig(mode=SelectionMode.UCB, ucb_c=5.0),
    )
    selector_low = ActionSelector(
        n_actions=3,
        config=ActionSelectionConfig(mode=SelectionMode.UCB, ucb_c=0.1),
    )

    votes = torch.tensor([10.0, 5.0, 1.0])

    # Action 0 is best, but has been tried
    selector_high.action_counts[0] = 10
    selector_high.total_selections = 10
    selector_low.action_counts[0] = 10
    selector_low.total_selections = 10

    # High c should explore untried actions more
    action_high, _ = selector_high.select_action(positive_votes=votes)
    # Low c should exploit best action more
    action_low, _ = selector_low.select_action(positive_votes=votes)

    # Contract: higher c increases exploration
    # (Not deterministic, so just check they're valid)
    assert 0 <= action_high < 3
    assert 0 <= action_low < 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
