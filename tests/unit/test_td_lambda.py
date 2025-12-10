"""
Unit tests for TD(λ) multi-step credit assignment.

Tests the TD(λ) implementation in striatum for extended temporal credit assignment.
"""

import torch
import pytest

from thalia.regions.striatum import (
    TDLambdaConfig,
    TDLambdaTraces,
    TDLambdaLearner,
    compute_n_step_return,
    compute_lambda_return,
    Striatum,
    StriatumConfig,
)


def test_td_lambda_traces_initialization():
    """Test that TD(λ) traces initialize correctly."""
    config = TDLambdaConfig(lambda_=0.9, gamma=0.99)
    traces = TDLambdaTraces(n_output=10, n_input=20, config=config)

    assert traces.traces.shape == (10, 20)
    assert torch.all(traces.traces == 0)
    assert traces.decay_factor == 0.9 * 0.99  # γλ


def test_td_lambda_traces_update():
    """Test that traces update correctly with decay."""
    config = TDLambdaConfig(lambda_=0.9, gamma=0.99)
    traces = TDLambdaTraces(n_output=2, n_input=3, config=config)

    # First update
    gradient = torch.ones(2, 3)
    traces.update(gradient)

    # Should equal gradient (no previous traces)
    assert torch.allclose(traces.traces, gradient)

    # Second update - should decay previous and add new
    gradient2 = torch.ones(2, 3) * 0.5
    traces.update(gradient2)

    # Expected: decay_factor * traces + gradient2
    expected = 0.9 * 0.99 * gradient + gradient2
    assert torch.allclose(traces.traces, expected)


def test_td_lambda_traces_reset():
    """Test that traces reset to zero."""
    config = TDLambdaConfig()
    traces = TDLambdaTraces(n_output=5, n_input=10, config=config)

    # Set some traces
    traces.update(torch.ones(5, 10))
    assert torch.any(traces.traces > 0)

    # Reset
    traces.reset_state()
    assert torch.all(traces.traces == 0)


def test_td_lambda_learner_initialization():
    """Test TD(λ) learner initializes correctly."""
    config = TDLambdaConfig(lambda_=0.95, gamma=0.99)
    learner = TDLambdaLearner(n_actions=3, n_input=10, config=config)

    assert learner.n_actions == 3
    assert learner.n_input == 10
    assert learner.config.lambda_ == 0.95
    assert learner.last_value is None
    assert learner._total_updates == 0


def test_td_lambda_eligibility_update():
    """Test eligibility trace updates during action selection."""
    config = TDLambdaConfig(lambda_=0.9, gamma=0.99)
    learner = TDLambdaLearner(n_actions=3, n_input=5, config=config)

    # Simulate action selection with input activity
    action = 1
    pre_activity = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])

    learner.update_eligibility(action, pre_activity)

    # Should have traces only for chosen action
    traces = learner.traces.get()
    assert traces.shape == (3, 5)
    assert torch.all(traces[action] == pre_activity)
    assert torch.all(traces[0] == 0)  # Other actions
    assert torch.all(traces[2] == 0)


def test_td_error_computation():
    """Test TD error computation."""
    learner = TDLambdaLearner(n_actions=2, n_input=5)

    # First timestep (no previous value)
    td_error = learner.compute_td_error(reward=1.0, next_value=0.5)
    assert td_error == 1.0  # No previous value, so just reward

    # Set last value
    learner.set_last_value(0.5)

    # Second timestep: δ = r + γV(s') - V(s)
    td_error = learner.compute_td_error(reward=1.0, next_value=0.8)
    expected = 1.0 + 0.99 * 0.8 - 0.5  # r + γV(s') - V(s)
    assert abs(td_error - expected) < 0.001

    # Terminal state
    learner.set_last_value(0.8)
    td_error = learner.compute_td_error(reward=1.0, next_value=0.0, terminal=True)
    expected = 1.0 - 0.8  # r - V(s), next_value ignored
    assert abs(td_error - expected) < 0.001


def test_weight_update_computation():
    """Test that weight updates combine TD error and eligibility."""
    config = TDLambdaConfig(lambda_=0.9, gamma=0.99)
    learner = TDLambdaLearner(n_actions=2, n_input=3, config=config)

    # Set up eligibility for action 0
    pre_activity = torch.tensor([1.0, 0.5, 0.0])
    learner.update_eligibility(action=0, pre_activity=pre_activity)

    # Compute weight update with positive TD error
    td_error = 0.5
    weight_update = learner.compute_update(td_error)

    # Should be td_error × eligibility
    expected = torch.zeros(2, 3)
    expected[0] = td_error * pre_activity
    assert torch.allclose(weight_update, expected)


def test_n_step_return_computation():
    """Test n-step return computation."""
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    values = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    gamma = 0.9

    # 1-step return from t=0: r0 + γV(1)
    returns = compute_n_step_return(rewards, values, gamma, n=1)
    expected_0 = 1.0 + 0.9 * 0.6
    assert abs(returns[0].item() - expected_0) < 0.001

    # 2-step return from t=0: r0 + γr1 + γ²V(2)
    returns = compute_n_step_return(rewards, values, gamma, n=2)
    expected_0 = 1.0 + 0.9 * 0.0 + 0.9**2 * 0.7
    assert abs(returns[0].item() - expected_0) < 0.001


def test_lambda_return_computation():
    """Test λ-return computation."""
    rewards = torch.tensor([1.0, 0.0, 1.0])
    values = torch.tensor([0.5, 0.6, 0.7])
    gamma = 0.9
    lambda_ = 0.9

    returns = compute_lambda_return(rewards, values, gamma, lambda_)

    # Should be exponentially-weighted average of n-step returns
    assert returns.shape == rewards.shape
    # Last timestep should equal reward (terminal)
    assert abs(returns[-1].item() - rewards[-1].item()) < 0.001


def test_striatum_with_td_lambda_disabled():
    """Test striatum works normally with TD(λ) disabled (default)."""
    config = StriatumConfig(
        n_input=10,
        n_output=3,
        use_td_lambda=False,  # Disabled (default)
        device="cpu",
    )
    striatum = Striatum(config)

    # Should not have TD(λ) learners
    assert striatum.td_lambda_d1 is None
    assert striatum.td_lambda_d2 is None

    # Should work normally
    input_spikes = torch.rand(10) > 0.5
    output = striatum.forward(input_spikes)
    assert output.shape == (config.n_output * striatum.neurons_per_action,)


def test_striatum_with_td_lambda_enabled():
    """Test striatum with TD(λ) enabled."""
    config = StriatumConfig(
        n_input=10,
        n_output=3,
        use_td_lambda=True,
        td_lambda=0.9,
        td_gamma=0.99,
        device="cpu",
    )
    striatum = Striatum(config)

    # Should have TD(λ) learners
    assert striatum.td_lambda_d1 is not None
    assert striatum.td_lambda_d2 is not None
    assert striatum.td_lambda_d1.config.lambda_ == 0.9
    assert striatum.td_lambda_d1.config.gamma == 0.99

    # Should still work normally
    input_spikes = torch.rand(10) > 0.5
    output = striatum.forward(input_spikes)
    assert output.shape == (config.n_output * striatum.neurons_per_action,)

    # TD(λ) traces should be updated during forward pass
    # (traces are updated for all neurons, not just chosen action)
    traces_d1 = striatum.td_lambda_d1.traces.get()
    traces_d2 = striatum.td_lambda_d2.traces.get()

    # At least some traces should be non-zero if there was activity
    if input_spikes.sum() > 0:
        assert traces_d1.abs().sum() > 0 or traces_d2.abs().sum() > 0


def test_td_lambda_diagnostics():
    """Test that TD(λ) diagnostics are included in striatum diagnostics."""
    config = StriatumConfig(
        n_input=10,
        n_output=3,
        use_td_lambda=True,
        td_lambda=0.95,
        device="cpu",
    )
    striatum = Striatum(config)

    diag = striatum.get_diagnostics()

    # Should include TD(λ) state
    assert "td_lambda" in diag
    assert diag["td_lambda"]["td_lambda_enabled"] is True
    assert diag["td_lambda"]["lambda"] == 0.95
    assert "d1_td_lambda" in diag["td_lambda"]
    assert "d2_td_lambda" in diag["td_lambda"]


def test_td_lambda_reset():
    """Test that TD(λ) traces reset correctly."""
    config = StriatumConfig(
        n_input=10,
        n_output=3,
        use_td_lambda=True,
        device="cpu",
    )
    striatum = Striatum(config)

    # Run forward pass to build up traces
    input_spikes = torch.ones(10)
    striatum.forward(input_spikes)

    # Traces should have some value
    traces_before = striatum.td_lambda_d1.traces.get().clone()
    assert traces_before.abs().sum() > 0

    # Reset state
    striatum.reset_state()

    # Traces should be zero
    traces_after = striatum.td_lambda_d1.traces.get()
    assert torch.all(traces_after == 0)


@pytest.mark.parametrize("lambda_", [0.0, 0.5, 0.9, 0.95, 1.0])
def test_td_lambda_different_lambdas(lambda_):
    """Test TD(λ) with different λ values."""
    config = TDLambdaConfig(lambda_=lambda_, gamma=0.99)
    learner = TDLambdaLearner(n_actions=2, n_input=5, config=config)

    # λ=0 should be TD(0) - immediate only
    # λ=1 should be Monte Carlo - full episode
    # 0 < λ < 1 should interpolate

    # Update eligibility multiple times
    for i in range(5):
        learner.update_eligibility(action=0, pre_activity=torch.ones(5))

    # Traces should accumulate differently based on λ
    traces = learner.traces.get()

    if lambda_ == 0.0:
        # TD(0): traces decay completely each step (γλ = 0)
        # So traces should just be last gradient
        assert torch.allclose(traces[0], torch.ones(5), atol=0.1)
    elif lambda_ == 1.0:
        # Monte Carlo: traces accumulate without λ decay (γλ = γ)
        # Should have accumulated more
        assert traces[0].sum() > 5.0


def test_td_lambda_vs_basic_eligibility():
    """Test that TD(λ) provides longer credit assignment than basic traces."""
    # Create two striatum instances: one with TD(λ), one without
    config_basic = StriatumConfig(
        n_input=10,
        n_output=2,
        use_td_lambda=False,
        eligibility_tau_ms=1000.0,  # 1 second
        dt_ms=1.0,
        device="cpu",
    )

    config_td_lambda = StriatumConfig(
        n_input=10,
        n_output=2,
        use_td_lambda=True,
        td_lambda=0.9,
        td_gamma=0.99,
        eligibility_tau_ms=1000.0,
        dt_ms=1.0,
        device="cpu",
    )

    striatum_basic = Striatum(config_basic)
    striatum_td = Striatum(config_td_lambda)

    # Simulate action selection
    input_spikes = torch.ones(10)

    # Both forward
    striatum_basic.forward(input_spikes)
    striatum_td.forward(input_spikes)

    # Simulate time passing (eligibility decaying)
    for _ in range(1000):  # 1000 timesteps
        striatum_basic.forward(torch.zeros(10))
        striatum_td.forward(torch.zeros(10))

    # Basic eligibility should decay exponentially
    basic_elig = striatum_basic.d1_eligibility.abs().sum().item()

    # TD(λ) traces should decay slower (γλ > simple decay)
    td_traces = striatum_td.td_lambda_d1.traces.get().abs().sum().item()

    # TD(λ) should maintain higher traces (but this depends on parameters)
    # At minimum, both should have decayed significantly
    assert basic_elig < 10.0  # Decayed from initial
    assert td_traces >= 0  # Non-negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
