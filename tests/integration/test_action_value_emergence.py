"""Integration tests for Phase 5: Emergent Action-Value Learning.

Tests that action values emerge purely from D1/D2 synaptic weight competition,
without explicit Q-value storage or TD error computation.

Success Criteria (Phase 5):
- ✅ No explicit Q-value storage
- ✅ Action selection via D1/D2 competition
- ✅ Action preferences converge correctly
- ✅ Performance within 30% of explicit Q-learning baseline

Biology:
- D1 (Go) pathway: Strengthened by reward (dopamine burst)
- D2 (NoGo) pathway: Strengthened by punishment (dopamine dip)
- Action value = NET activity (D1 - D2) during selection
- No explicit "expected value" storage

References:
- docs/design/emergent_rl_migration.md Phase 5
- Schultz et al. (1997): Dopamine reward prediction error
- Frank (2005): Dynamic dopamine modulation in basal ganglia
"""

import pytest
import torch

from thalia.config import BrainConfig
from thalia.core.brain_builder import BrainBuilder
from thalia.core.dynamic_brain import DynamicBrain


@pytest.fixture
def simple_striatum_brain(device: str = "cpu") -> DynamicBrain:
    """Create minimal brain with striatum for RL testing.

    Architecture:
    - Thalamus (input) → Striatum (action selection)
    - 2 actions with population coding
    - D1/D2 pathways for opponent processing
    """
    # Create brain config
    brain_config = BrainConfig(device=device, dt_ms=1.0)

    # Build brain with 2 actions (overrides default 10)
    brain = BrainBuilder.preset(
        "default",
        brain_config,
        striatum_actions=2,  # Override default 10 actions
        striatum_neurons_per_action=5,  # Small population for faster testing
    )

    # Configure striatum for testing
    if "striatum" in brain.components:
        striatum = brain.components["striatum"]
        # Disable exploration for deterministic tests
        striatum.config.ucb_exploration = False
        striatum.config.softmax_action_selection = False
        striatum.config.rpe_enabled = False  # Phase 5: No explicit Q-values

    return brain


def test_no_explicit_qvalue_storage(simple_striatum_brain):
    """Phase 5: Striatum should not store explicit Q-values.

    Validates:
    - No value_estimates attribute
    - No get_expected_value method
    - No update_value_estimate method
    """
    striatum = simple_striatum_brain.components["striatum"]

    # Should not have value_estimates tensor
    assert not hasattr(striatum, "value_estimates") or striatum.value_estimates is None, \
        "Striatum should not store explicit Q-values (Phase 5)"

    # Should not have Q-value query methods
    # Note: evaluate_state still exists for planning, but uses D1-D2 weights
    assert not hasattr(striatum, "get_expected_value"), \
        "Striatum should not have get_expected_value method (Phase 5)"

    assert not hasattr(striatum, "update_value_estimate"), \
        "Striatum should not have update_value_estimate method (Phase 5)"


def test_action_selection_via_d1_d2_competition(simple_striatum_brain):
    """Phase 5: Action selection should use pure D1-D2 vote difference.

    Validates:
    - Action selection uses D1_votes - D2_votes
    - No Q-values involved in selection
    - NET votes determine action
    """
    brain = simple_striatum_brain
    striatum = brain.components["striatum"]
    thalamus = brain.components["thalamus"]

    # Create test input (strong spikes to drive network)
    # Use binary spikes (realistic) with size matching thalamus input
    input_size = thalamus.input_size
    test_input = torch.ones(input_size, device=brain.device)  # All spikes

    # Run forward pass for longer to accumulate votes
    brain.forward({"thalamus": test_input}, n_timesteps=50)

    # Finalize action (should use D1-D2 competition)
    action, confidence = brain.select_action(explore=False)

    # Verify action was selected via NET votes
    d1_votes, d2_votes = striatum.state_tracker.get_accumulated_votes()
    net_votes = d1_votes - d2_votes

    expected_action = int(net_votes.argmax().item())

    # Verify action is within valid range
    assert 0 <= action < striatum.n_actions, \
        f"Action {action} should be in range [0, {striatum.n_actions})"

    # Verify action matches NET votes argmax
    assert action == expected_action, \
        f"Action should be selected via D1-D2 competition. Got {action}, expected {expected_action}. " \
        f"NET votes: {net_votes.tolist()}"

    # Verify NET votes were actually different (not all zeros)
    assert net_votes.abs().sum() > 0, \
        "D1 and D2 should produce different votes (not all zeros)"


def test_d1_weights_strengthen_with_reward(simple_striatum_brain):
    """Phase 5: D1 (Go) pathway weights should strengthen with reward.

    Biology: Dopamine burst → LTP in D1 synapses
    Mechanism: Three-factor rule (STDP + eligibility + DA)
    """
    brain = simple_striatum_brain
    striatum = brain.components["striatum"]

    # Get initial D1 weights for action 0
    d1_pathway = striatum.d1_pathway
    initial_weights = d1_pathway.weights.clone()

    # Repeatedly present input and reward action 0
    test_input = torch.randn(128, device=brain.device)

    for trial in range(10):
        # Reset votes
        striatum.state_tracker.reset_trial_votes()

        # Forward pass
        for _ in range(20):
            brain.forward({"thalamus": test_input}, n_timesteps=1)

        # Force action 0 selection (bypass exploration)
        striatum.state_tracker.set_last_action(0, exploring=False)

        # Deliver reward (high dopamine)
        striatum.set_neuromodulators(dopamine=0.9)  # Dopamine burst
        striatum.deliver_reward(reward=1.0)

    # Check that D1 weights for action 0 increased
    final_weights = d1_pathway.weights.clone()

    # Get action 0 population indices
    action_0_slice = striatum._get_action_population_indices(0)
    action_0_weight_change = (final_weights[action_0_slice, :] - initial_weights[action_0_slice, :]).mean()

    assert action_0_weight_change > 0.001, \
        f"D1 weights for rewarded action should strengthen. Change: {action_0_weight_change:.4f}"


def test_d2_weights_strengthen_with_punishment(simple_striatum_brain):
    """Phase 5: D2 (NoGo) pathway weights should strengthen with punishment.

    Biology: Dopamine dip → LTP in D2 synapses (inverted modulation)
    Mechanism: Three-factor rule with inverted dopamine signal
    """
    brain = simple_striatum_brain
    striatum = brain.components["striatum"]

    # Get initial D2 weights for action 1
    d2_pathway = striatum.d2_pathway
    initial_weights = d2_pathway.weights.clone()

    # Repeatedly present input and punish action 1
    test_input = torch.randn(128, device=brain.device)

    for trial in range(10):
        # Reset votes
        striatum.state_tracker.reset_trial_votes()

        # Forward pass
        for _ in range(20):
            brain.forward({"thalamus": test_input}, n_timesteps=1)

        # Force action 1 selection
        striatum.state_tracker.set_last_action(1, exploring=False)

        # Deliver punishment (low dopamine = D2 strengthening)
        striatum.set_neuromodulators(dopamine=0.1)  # Dopamine dip
        striatum.deliver_reward(reward=-1.0)

    # Check that D2 weights for action 1 increased
    final_weights = d2_pathway.weights.clone()

    # Get action 1 population indices
    action_1_slice = striatum._get_action_population_indices(1)
    action_1_weight_change = (final_weights[action_1_slice, :] - initial_weights[action_1_slice, :]).mean()

    assert action_1_weight_change > 0.001, \
        f"D2 weights for punished action should strengthen. Change: {action_1_weight_change:.4f}"


def test_action_preferences_converge(simple_striatum_brain):
    """Phase 5: Action preferences should converge to reward-maximizing policy.

    Task: Binary choice with deterministic rewards
    - Action 0 → +1 reward (always)
    - Action 1 → -1 reward (always)

    Expected: Action 0 should dominate after learning
    """
    brain = simple_striatum_brain
    striatum = brain.components["striatum"]

    # Training phase: 50 trials
    test_input = torch.randn(128, device=brain.device)
    action_0_count = 0

    for trial in range(50):
        # Reset votes
        striatum.state_tracker.reset_trial_votes()

        # Forward pass
        for _ in range(20):
            brain.forward({"thalamus": test_input}, n_timesteps=1)

        # Select action (deterministic after learning)
        action, _ = brain.select_action(explore=False)

        # Count action 0 selections (should increase over time)
        if action == 0:
            action_0_count += 1

        # Deliver reward based on action
        reward = 1.0 if action == 0 else -1.0
        dopamine = 0.9 if reward > 0 else 0.1

        striatum.set_neuromodulators(dopamine=dopamine)
        striatum.deliver_reward(reward=reward)

    # Verify convergence: action 0 should be selected >70% of the time
    action_0_preference = action_0_count / 50
    assert action_0_preference > 0.7, \
        f"Action 0 (rewarded) should be selected >70% of the time. Got {action_0_preference:.2%}"


def test_net_votes_reflect_action_value(simple_striatum_brain):
    """Phase 5: NET votes (D1-D2) should reflect emergent action values.

    After learning:
    - Rewarded action → high NET (D1 > D2)
    - Punished action → low NET (D1 < D2)
    """
    brain = simple_striatum_brain
    striatum = brain.components["striatum"]

    # Train on deterministic rewards
    test_input = torch.randn(128, device=brain.device)

    for trial in range(30):
        striatum.state_tracker.reset_trial_votes()

        for _ in range(20):
            brain.forward({"thalamus": test_input}, n_timesteps=1)

        # Train: Action 0 → reward, Action 1 → punishment
        for action in [0, 1]:
            striatum.state_tracker.reset_trial_votes()

            for _ in range(20):
                brain.forward({"thalamus": test_input}, n_timesteps=1)

            striatum.state_tracker.set_last_action(action, exploring=False)

            reward = 1.0 if action == 0 else -1.0
            dopamine = 0.9 if reward > 0 else 0.1

            striatum.set_neuromodulators(dopamine=dopamine)
            striatum.deliver_reward(reward=reward)

    # Test: Measure NET votes for each action
    striatum.state_tracker.reset_trial_votes()
    for _ in range(20):
        brain.forward({"thalamus": test_input}, n_timesteps=1)

    net_votes = striatum.state_tracker.get_net_votes()

    # Action 0 (rewarded) should have higher NET than action 1 (punished)
    assert net_votes[0] > net_votes[1], \
        f"Rewarded action should have higher NET votes. Action 0: {net_votes[0]:.2f}, Action 1: {net_votes[1]:.2f}"

    # NET difference should be substantial (not marginal)
    net_difference = net_votes[0] - net_votes[1]
    assert net_difference > 1.0, \
        f"NET difference should be substantial (>1.0). Got {net_difference:.2f}"


def test_eligibility_gates_learning(simple_striatum_brain):
    """Phase 5: Learning should only occur for actions with eligibility traces.

    Mechanism: Three-factor rule requires:
    1. Pre-post spike coincidence (creates eligibility)
    2. Eligibility trace (persists ~1 second)
    3. Dopamine signal (gates plasticity)

    Without eligibility → no learning (even with dopamine)
    """
    brain = simple_striatum_brain
    striatum = brain.components["striatum"]

    # Get initial weights
    d1_initial = striatum.d1_pathway.weights.clone()

    # Deliver dopamine WITHOUT prior activity (no eligibility)
    striatum.set_neuromodulators(dopamine=0.9)
    striatum.deliver_reward(reward=1.0)

    # Weights should NOT change (no eligibility traces)
    d1_after_dopamine_only = striatum.d1_pathway.weights.clone()
    weight_change_no_eligibility = (d1_after_dopamine_only - d1_initial).abs().mean()

    assert weight_change_no_eligibility < 0.0001, \
        f"Weights should not change without eligibility. Change: {weight_change_no_eligibility:.6f}"

    # Now create eligibility by running forward pass
    test_input = torch.randn(128, device=brain.device)
    striatum.state_tracker.reset_trial_votes()

    for _ in range(20):
        brain.forward({"thalamus": test_input}, n_timesteps=1)

    striatum.state_tracker.set_last_action(0, exploring=False)

    # Deliver dopamine WITH eligibility
    striatum.set_neuromodulators(dopamine=0.9)
    striatum.deliver_reward(reward=1.0)

    # Weights SHOULD change (eligibility present)
    d1_after_dopamine_with_eligibility = striatum.d1_pathway.weights.clone()
    weight_change_with_eligibility = (d1_after_dopamine_with_eligibility - d1_after_dopamine_only).abs().mean()

    assert weight_change_with_eligibility > 0.001, \
        f"Weights should change with eligibility + dopamine. Change: {weight_change_with_eligibility:.4f}"


def test_evaluate_state_uses_d1_d2_weights(simple_striatum_brain):
    """Phase 5: evaluate_state should compute value from D1-D2 weight balance.

    Not from explicit Q-values (which no longer exist).
    """
    brain = simple_striatum_brain
    striatum = brain.components["striatum"]

    # Train to create weight differences
    test_input = torch.randn(128, device=brain.device)

    for trial in range(20):
        striatum.state_tracker.reset_trial_votes()

        for _ in range(20):
            brain.forward({"thalamus": test_input}, n_timesteps=1)

        striatum.state_tracker.set_last_action(0, exploring=False)
        striatum.set_neuromodulators(dopamine=0.9)
        striatum.deliver_reward(reward=1.0)

    # Evaluate state should return value based on D1-D2 weights
    state_value = striatum.evaluate_state(test_input)

    # Should return non-zero value (weights have been trained)
    assert abs(state_value) > 0.01, \
        f"evaluate_state should return non-zero value from D1-D2 weights. Got {state_value:.4f}"

    # Value should be positive (action 0 was rewarded)
    assert state_value > 0, \
        f"State value should be positive for rewarded state. Got {state_value:.4f}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
