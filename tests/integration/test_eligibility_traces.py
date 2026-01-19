"""
Integration tests for eligibility-based credit assignment (Phase 3 Emergent RL).

Validates that multi-step credit assignment works through eligibility trace
dynamics WITHOUT explicit TD error computation.

Biological Foundation:
- Eligibility traces persist ~1000ms in striatum (Yagishita et al., 2014)
- Dopamine gates learning when it arrives (seconds after action)
- Credit propagates backward through trace dynamics, not TD(λ) calculation

Author: Thalia Development Team
Date: January 19, 2026
"""

import pytest
import torch

from thalia.config import BrainConfig
from thalia.core.brain_builder import BrainBuilder


@pytest.fixture
def test_brain():
    """Create minimal brain for eligibility trace testing."""
    config = BrainConfig(
        device="cpu",
        dt_ms=1.0,
    )

    # Build brain with striatum (uses three-factor learning)
    brain = BrainBuilder.preset("default", config)

    return brain


def test_multi_step_credit_assignment(test_brain):
    """Credit should propagate backward through eligibility traces.

    Test 5-step chain: state_0 → state_1 → ... → state_4 → reward
    Without eligibility traces, only state_4 would learn.
    WITH eligibility traces, state_0 should also show weight changes.
    """
    brain = test_brain
    striatum = brain.components["striatum"]

    # Create distinct input patterns for 5-step chain
    input_size = striatum.input_size
    chain_states = [torch.randn(input_size) for _ in range(5)]

    # Get initial weights for state_0 → action 0
    initial_weights = {}
    for source in striatum.synaptic_weights.keys():
        if "d1" in source:
            initial_weights[source] = striatum.synaptic_weights[source][0, :].clone()

    # Experience chain with delayed reward
    for step in range(5):
        # Present state
        brain.forward({"thalamus": chain_states[step]})

        # Select action (creates eligibility trace)
        _ = brain.select_action()

        # No reward yet (except at end)
        if step < 4:
            brain.deliver_reward(0.0)

    # Deliver reward at end
    brain.deliver_reward(1.0)  # Dopamine spike

    # Check that weights for state_0 changed (despite 5 steps delay)
    weight_change_detected = False
    for source in striatum.synaptic_weights.keys():
        if "d1" in source:
            final_weights = striatum.synaptic_weights[source][0, :]
            weight_change = (final_weights - initial_weights[source]).abs().sum()

            if weight_change > 0.01:
                weight_change_detected = True
                break

    assert weight_change_detected, (
        "Credit should propagate back to state_0 via eligibility traces (5-step delay). "
        "This validates that multi-step credit assignment works WITHOUT TD(λ) computation."
    )


def test_eligibility_trace_decay(test_brain):
    """Traces should decay exponentially without reinforcement."""
    brain = test_brain
    striatum = brain.components["striatum"]

    # Create test state
    input_size = striatum.input_size
    test_state = torch.randn(input_size)

    # Create eligibility trace
    brain.forward({"thalamus": test_state})
    _ = brain.select_action()

    # Get initial eligibility from D1 pathway learning strategy
    d1_pathway = striatum.d1_pathway
    if hasattr(d1_pathway.learning_strategy, "eligibility"):
        initial_eligibility = d1_pathway.learning_strategy.eligibility.clone()
        initial_sum = initial_eligibility.sum().item()
    else:
        pytest.skip("D1 pathway strategy doesn't have eligibility attribute")

    # Let decay for 200 timesteps (no new activity, no dopamine)
    for _ in range(200):
        empty_state = torch.zeros(input_size)
        brain.forward({"thalamus": empty_state})

    # Check eligibility decayed
    if hasattr(d1_pathway.learning_strategy, "eligibility"):
        final_eligibility = d1_pathway.learning_strategy.eligibility
        final_sum = final_eligibility.sum().item()
        decay_ratio = final_sum / (initial_sum + 1e-10)

        # With tau=1000ms and dt=1ms, decay per step = 1 - 1/1000 = 0.999
        # After 200 steps: 0.999^200 ≈ 0.819
        assert 0.7 < decay_ratio < 0.9, (
            f"Eligibility should decay to ~82% after 200ms (tau=1000ms). "
            f"Got {decay_ratio:.2%} of original"
        )


def test_no_td_error_calculation(test_brain):
    """System should learn without computing explicit TD errors."""
    brain = test_brain
    striatum = brain.components["striatum"]

    # Verify that TD(λ) components don't exist
    assert not hasattr(
        striatum, "td_lambda_d1"
    ), "Should not have explicit TD(λ) calculator (Phase 3 removal)"

    assert not hasattr(
        striatum, "td_lambda_d2"
    ), "Should not have explicit TD(λ) calculator (Phase 3 removal)"

    # Verify learning still works via three-factor rule
    input_size = striatum.input_size
    test_state = torch.randn(input_size)

    initial_weights = {}
    for source in striatum.synaptic_weights.keys():
        if "d1" in source:
            initial_weights[source] = striatum.synaptic_weights[source].clone()

    # Run learning loop with random rewards
    for _ in range(100):
        brain.forward({"thalamus": test_state})
        brain.select_action()
        brain.deliver_reward(torch.rand(1).item())

    # Weights should change (learning happened)
    total_weight_change = 0.0
    for source in striatum.synaptic_weights.keys():
        if "d1" in source:
            final_weights = striatum.synaptic_weights[source]
            weight_change = (final_weights - initial_weights[source]).abs().sum().item()
            total_weight_change += weight_change

    assert total_weight_change > 1.0, (
        "Learning should occur via three-factor rule WITHOUT TD error computation. "
        f"Got total weight change: {total_weight_change:.3f}"
    )


def test_eligibility_accumulates_before_dopamine(test_brain):
    """Eligibility should accumulate over multiple timesteps before reward."""
    brain = test_brain
    striatum = brain.components["striatum"]
    d1_pathway = striatum.d1_pathway

    # Skip if strategy doesn't expose eligibility
    if not hasattr(d1_pathway.learning_strategy, "eligibility"):
        pytest.skip("D1 pathway strategy doesn't have eligibility attribute")

    input_size = striatum.input_size
    test_state = torch.randn(input_size)

    # Reset eligibility
    d1_pathway.learning_strategy.reset_state()

    # Present same state 10 times (build eligibility)
    for _ in range(10):
        brain.forward({"thalamus": test_state})
        brain.select_action()
        # No dopamine yet

    # Check eligibility accumulated
    eligibility = d1_pathway.learning_strategy.eligibility
    if eligibility is not None:
        elig_sum = eligibility.sum().item()
        assert elig_sum > 1.0, (
            "Eligibility should accumulate over multiple presentations. " f"Got sum: {elig_sum:.3f}"
        )

        # Now deliver dopamine
        initial_weights = striatum.synaptic_weights["default_d1"].clone()
        brain.deliver_reward(1.0)
        final_weights = striatum.synaptic_weights["default_d1"]

        # Weights should change significantly (accumulated eligibility × dopamine)
        weight_change = (final_weights - initial_weights).abs().sum().item()
        assert weight_change > 0.1, (
            "Accumulated eligibility should create large weight change when dopamine arrives. "
            f"Got change: {weight_change:.3f}"
        )


def test_delayed_reward_5_steps(test_brain):
    """Test 5-step credit assignment without TD(λ)."""
    brain = test_brain
    striatum = brain.components["striatum"]

    input_size = striatum.input_size

    # Create 5 distinct states
    states = [torch.randn(input_size) for _ in range(5)]

    # Record initial weights
    initial_weights = striatum.synaptic_weights["default_d1"].clone()

    # Execute 5-step sequence
    for i in range(5):
        brain.forward({"thalamus": states[i]})
        brain.select_action()

        # Only reward at end
        if i == 4:
            brain.deliver_reward(1.0)
        else:
            brain.deliver_reward(0.0)

    # Check that early states received credit
    final_weights = striatum.synaptic_weights["default_d1"]
    weight_change = (final_weights - initial_weights).abs().sum().item()

    assert weight_change > 0.01, (
        f"5-step credit assignment should work via eligibility traces. "
        f"Got weight change: {weight_change:.3f}"
    )


def test_dopamine_gates_learning(test_brain):
    """No learning should occur without dopamine signal."""
    brain = test_brain
    striatum = brain.components["striatum"]

    input_size = striatum.input_size
    test_state = torch.randn(input_size)

    # Record initial weights
    initial_weights = striatum.synaptic_weights["default_d1"].clone()

    # Create eligibility WITHOUT dopamine
    for _ in range(50):
        brain.forward({"thalamus": test_state})
        brain.select_action()
        # NO reward delivery → no dopamine → no learning

    # Weights should NOT change significantly
    final_weights = striatum.synaptic_weights["default_d1"]
    weight_change = (final_weights - initial_weights).abs().sum().item()

    assert weight_change < 0.01, (
        "Without dopamine, eligibility should NOT cause weight changes "
        "(three-factor rule: eligibility × dopamine × lr). "
        f"Got weight change: {weight_change:.3f}"
    )
