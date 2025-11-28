"""
World Model Demo - Predictive processing and internal simulation.

This demo shows how the network builds an internal model of the world:
- Predicting sensory inputs before they arrive
- Detecting surprise (prediction errors)
- Simulating possible futures
- Planning actions through imagination

Run: python examples/world_model_demo.py
"""

import torch
import sys

sys.path.insert(0, "src")

from thalia.world import (
    WorldModel,
    WorldModelConfig,
    PredictiveLayer,
    PredictiveLayerConfig,
    PredictionMode,
    ActionSimulator,
    PredictiveCodingNetwork,
)


def demo_predictive_layer():
    """Demo 1: Basic predictive layer."""
    print("\n" + "="*60)
    print("Demo 1: Predictive Layer")
    print("="*60)

    config = PredictiveLayerConfig(
        n_neurons=32,
        n_prediction_neurons=16,
        error_gain=0.5,
    )

    layer = PredictiveLayer(config, input_size=24, higher_size=48)
    layer.reset_state(batch_size=1)

    print(f"Layer: {config.n_neurons} neurons, {config.n_prediction_neurons} prediction neurons")
    print(f"Error gain: {config.error_gain}")

    # Process with consistent input (should reduce error over time)
    consistent_input = torch.rand(1, 24)
    higher_context = torch.rand(1, 48)

    print("\nProcessing consistent input over time:")
    surprises = []
    for t in range(20):
        activity, prediction, error = layer(
            input_activity=consistent_input,
            higher_activity=higher_context,
        )
        surprise = layer.get_surprise()
        surprises.append(surprise)

    print(f"  Initial surprise: {surprises[0]:.4f}")
    print(f"  Final surprise:   {surprises[-1]:.4f}")

    # Now change input (should increase surprise)
    new_input = torch.rand(1, 24)
    activity, prediction, error = layer(input_activity=new_input, higher_activity=higher_context)
    new_surprise = layer.get_surprise()

    print(f"\nAfter input change:")
    print(f"  Surprise: {new_surprise:.4f} (should be higher)")

    precision = layer.get_precision()
    print(f"  Precision: mean={precision.mean().item():.4f}")


def demo_world_model():
    """Demo 2: Full world model."""
    print("\n" + "="*60)
    print("Demo 2: World Model")
    print("="*60)

    config = WorldModelConfig(
        n_sensory=32,
        n_hidden=64,
        n_action=8,
        n_layers=3,
    )
    model = WorldModel(config)
    model.reset_state(batch_size=1)

    print(f"World Model:")
    print(f"  Sensory: {config.n_sensory} neurons")
    print(f"  Hidden:  {config.n_hidden} neurons")
    print(f"  Layers:  {config.n_layers}")

    # Process a sequence of observations
    print("\nProcessing sensory sequence:")

    # Create a simple pattern that repeats
    pattern = torch.rand(1, 32)

    for t in range(30):
        # Slightly noisy version of pattern
        sensory = pattern + torch.randn(1, 32) * 0.1
        state, errors = model(sensory)

        if t % 10 == 0:
            surprise = model.get_surprise()
            print(f"  t={t:2d}: surprise={surprise:.4f}, belief_norm={state.norm().item():.4f}")

    print(f"\nTotal surprise over sequence: {model.get_total_surprise():.4f}")

    # Get belief state
    belief_mean, belief_precision = model.get_belief()
    print(f"Final belief: mean_norm={belief_mean.norm().item():.4f}")


def demo_simulation():
    """Demo 3: Simulating future without input."""
    print("\n" + "="*60)
    print("Demo 3: Future Simulation (Imagination)")
    print("="*60)

    config = WorldModelConfig(n_sensory=32, n_hidden=64, n_action=8, n_layers=2)
    model = WorldModel(config)
    model.reset_state(batch_size=1)

    # Initialize with some observations
    print("Initializing with observations...")
    for _ in range(10):
        sensory = torch.rand(1, 32)
        model(sensory)

    initial_belief, _ = model.get_belief()
    print(f"Initial belief norm: {initial_belief.norm().item():.4f}")

    # Simulate future (no sensory input)
    print("\nSimulating 20 steps into the future...")
    future_states = model.simulate(steps=20)

    print(f"Generated {len(future_states)} future states")

    # Measure divergence from initial state
    divergences = [
        (state - initial_belief).norm().item()
        for state in future_states
    ]

    print(f"  Step 5 divergence:  {divergences[4]:.4f}")
    print(f"  Step 10 divergence: {divergences[9]:.4f}")
    print(f"  Step 20 divergence: {divergences[19]:.4f}")

    print("\n(Divergence increases = imagination drifts from reality)")


def demo_action_planning():
    """Demo 4: Planning actions through simulation."""
    print("\n" + "="*60)
    print("Demo 4: Action Planning")
    print("="*60)

    config = WorldModelConfig(n_sensory=32, n_hidden=64, n_action=8, n_layers=2)
    model = WorldModel(config)
    model.reset_state(batch_size=1)

    # Initialize
    for _ in range(10):
        sensory = torch.rand(1, 32)
        model(sensory)

    # Create action simulator
    simulator = ActionSimulator(model, simulation_steps=30)

    # Define candidate actions
    actions = [
        torch.zeros(1, 8),  # Do nothing
        torch.randn(1, 8) * 0.5,  # Small action
        torch.randn(1, 8) * 1.0,  # Medium action
        torch.randn(1, 8) * 2.0,  # Large action
    ]

    print("Evaluating 4 candidate actions...")

    # Evaluate each action
    for i, action in enumerate(actions):
        result = simulator.simulate_action(action)
        print(f"  Action {i}: expected_surprise={result.expected_surprise:.4f}, "
              f"final_state_norm={result.final_state.norm().item():.4f}")

    # Select best action with a goal
    print("\nSelecting best action for goal (minimize state variance)...")

    def goal_fn(state):
        return -state.var().item()  # Prefer low variance (stable) states

    best_action, best_result = simulator.select_best_action(
        actions, goal_fn, prefer_low_surprise=True
    )

    print(f"  Best action index: {[torch.allclose(a, best_action) for a in actions].index(True)}")
    print(f"  Expected reward: {best_result.expected_reward:.4f}")
    print(f"  Expected surprise: {best_result.expected_surprise:.4f}")


def demo_predictive_coding():
    """Demo 5: Predictive coding network."""
    print("\n" + "="*60)
    print("Demo 5: Predictive Coding Network")
    print("="*60)

    network = PredictiveCodingNetwork(
        layer_sizes=[32, 64, 32],
        tau_mem=20.0,
        tau_error=5.0,
    )
    network.reset_state(batch_size=1)

    print("Predictive Coding Network:")
    print(f"  Layer sizes: {network.layer_sizes}")
    print(f"  3 levels of representation")
    print(f"  2 levels of prediction error")

    # Test with structured input
    sensory = torch.rand(1, 32)

    print("\nIterative inference:")
    for n_iter in [1, 3, 5, 10]:
        network.reset_state(batch_size=1)
        representations, errors = network(sensory, n_iterations=n_iter)
        total_error = network.get_total_error(errors)
        print(f"  {n_iter:2d} iterations: total_error={total_error:.4f}")

    print("\n(More iterations should reduce prediction error)")


def demo_surprise_detection():
    """Demo 6: Detecting surprising events."""
    print("\n" + "="*60)
    print("Demo 6: Surprise Detection")
    print("="*60)

    config = WorldModelConfig(n_sensory=32, n_hidden=64, n_layers=2)
    model = WorldModel(config)
    model.reset_state(batch_size=1)

    # Create a predictable pattern
    base_pattern = torch.rand(1, 32)

    print("Processing predictable sequence (same pattern + noise)...")

    surprise_history = []
    for t in range(50):
        if t == 25:
            # Inject a surprise at t=25
            sensory = torch.rand(1, 32)  # Completely new pattern
        else:
            sensory = base_pattern + torch.randn(1, 32) * 0.1

        model(sensory)
        surprise_history.append(model.get_surprise())

    # Find the surprise spike
    max_surprise = max(surprise_history)
    max_surprise_time = surprise_history.index(max_surprise)

    avg_before = sum(surprise_history[:25]) / 25
    avg_after = sum(surprise_history[26:]) / 24

    print(f"\nAverage surprise before t=25: {avg_before:.4f}")
    print(f"Surprise at t=25 (change):    {surprise_history[25]:.4f}")
    print(f"Average surprise after t=25:  {avg_after:.4f}")
    print(f"\nMax surprise at t={max_surprise_time}: {max_surprise:.4f}")

    if max_surprise_time == 25:
        print("✓ Correctly detected the surprising event!")
    else:
        print(f"(Surprise peak detected at t={max_surprise_time})")


def demo_belief_update():
    """Demo 7: Belief updates from evidence."""
    print("\n" + "="*60)
    print("Demo 7: Belief Updates from Evidence")
    print("="*60)

    config = WorldModelConfig(n_sensory=32, n_hidden=64, n_layers=2, tau_belief=50.0)
    model = WorldModel(config)
    model.reset_state(batch_size=1)

    # Two different "worlds"
    world_A = torch.rand(1, 32)
    world_B = torch.rand(1, 32)

    print("Two possible worlds: A and B")
    print(f"  World A pattern norm: {world_A.norm().item():.4f}")
    print(f"  World B pattern norm: {world_B.norm().item():.4f}")

    # Start with evidence for world A
    print("\nPhase 1: Evidence for World A (20 steps)")
    for _ in range(20):
        sensory = world_A + torch.randn(1, 32) * 0.1
        model(sensory)

    belief_A, _ = model.get_belief()
    print(f"  Belief after A evidence: norm={belief_A.norm().item():.4f}")

    # Now switch to evidence for world B
    print("\nPhase 2: Evidence for World B (20 steps)")
    for _ in range(20):
        sensory = world_B + torch.randn(1, 32) * 0.1
        model(sensory)

    belief_B, _ = model.get_belief()
    print(f"  Belief after B evidence: norm={belief_B.norm().item():.4f}")

    # Measure belief shift
    belief_shift = (belief_B - belief_A).norm().item()
    print(f"\nBelief shift magnitude: {belief_shift:.4f}")
    print("(Larger shift = beliefs updated based on new evidence)")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("THALIA World Model Demo")
    print("Predictive Processing and Internal Simulation")
    print("="*60)

    demos = [
        ("Predictive Layer", demo_predictive_layer),
        ("World Model", demo_world_model),
        ("Future Simulation", demo_simulation),
        ("Action Planning", demo_action_planning),
        ("Predictive Coding", demo_predictive_coding),
        ("Surprise Detection", demo_surprise_detection),
        ("Belief Updates", demo_belief_update),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n[Error in {name}]: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nKey takeaways:")
    print("• World model predicts sensory input before it arrives")
    print("• Prediction errors (surprise) drive learning and attention")
    print("• Internal simulation enables planning without acting")
    print("• Beliefs update based on accumulated evidence")
    print("• Predictive coding reduces error through iteration")


if __name__ == "__main__":
    main()
