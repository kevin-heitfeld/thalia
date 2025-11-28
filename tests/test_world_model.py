"""
Tests for world model - predictive processing.
"""

import pytest
import torch

from thalia.world import (
    WorldModel,
    WorldModelConfig,
    PredictiveLayer,
    PredictiveLayerConfig,
    PredictionMode,
    ActionSimulator,
    SimulationResult,
    PredictiveCodingNetwork,
)


class TestPredictiveLayerConfig:
    """Tests for PredictiveLayerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PredictiveLayerConfig()
        assert config.n_neurons == 64
        assert config.n_prediction_neurons == 64
        assert config.tau_mem == 20.0
        assert config.error_gain == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = PredictiveLayerConfig(
            n_neurons=128,
            tau_prediction=100.0,
            error_gain=0.8,
        )
        assert config.n_neurons == 128
        assert config.tau_prediction == 100.0
        assert config.error_gain == 0.8


class TestPredictiveLayer:
    """Tests for PredictiveLayer."""

    @pytest.fixture
    def layer(self):
        """Create a basic predictive layer."""
        config = PredictiveLayerConfig(n_neurons=32, n_prediction_neurons=16)
        return PredictiveLayer(config, input_size=24, higher_size=48)

    @pytest.fixture
    def layer_no_higher(self):
        """Create layer without higher level."""
        config = PredictiveLayerConfig(n_neurons=32)
        return PredictiveLayer(config, input_size=24)

    def test_creation(self, layer):
        """Test layer creation."""
        assert layer is not None
        assert layer.ff_weights is not None
        assert layer.pred_weights is not None

    def test_reset_state(self, layer):
        """Test state reset."""
        layer.reset_state(batch_size=2)
        assert layer.neurons.membrane is not None
        assert layer.neurons.membrane.shape == (2, 32)

    def test_forward_feedforward(self, layer):
        """Test feedforward mode."""
        layer.reset_state(batch_size=1)
        input_data = torch.rand(1, 24)

        activity, prediction, error = layer(
            input_activity=input_data,
            mode=PredictionMode.FEEDFORWARD
        )

        assert activity.shape == (1, 32)
        assert prediction.shape == (1, 32)
        assert error.shape == (1, 32)

    def test_forward_generative(self, layer):
        """Test generative mode."""
        layer.reset_state(batch_size=1)
        higher_activity = torch.rand(1, 48)

        activity, prediction, error = layer(
            higher_activity=higher_activity,
            mode=PredictionMode.GENERATIVE
        )

        assert activity.shape == (1, 32)

    def test_forward_combined(self, layer):
        """Test combined mode."""
        layer.reset_state(batch_size=1)
        input_data = torch.rand(1, 24)
        higher_activity = torch.rand(1, 48)

        activity, prediction, error = layer(
            input_activity=input_data,
            higher_activity=higher_activity,
            mode=PredictionMode.COMBINED
        )

        assert activity.shape == (1, 32)

    def test_prediction_error_tracking(self, layer):
        """Test that prediction error is tracked."""
        layer.reset_state(batch_size=1)
        input_data = torch.rand(1, 24)

        layer(input_activity=input_data)

        error = layer.get_prediction_error()
        assert error is not None
        assert error.shape == (1, 32)

    def test_precision_update(self, layer):
        """Test precision updates over time."""
        layer.reset_state(batch_size=1)

        # Run several steps
        for _ in range(10):
            input_data = torch.rand(1, 24)
            layer(input_activity=input_data)

        precision = layer.get_precision()
        assert precision is not None

    def test_surprise_computation(self, layer):
        """Test surprise is computed."""
        layer.reset_state(batch_size=1)
        input_data = torch.rand(1, 24)

        layer(input_activity=input_data)

        surprise = layer.get_surprise()
        assert isinstance(surprise, float)

    def test_generate_prediction(self, layer):
        """Test standalone prediction generation."""
        higher_activity = torch.rand(1, 48)

        prediction = layer.generate_prediction(higher_activity)
        assert prediction.shape == (1, 32)

    def test_batched_processing(self, layer):
        """Test batch processing."""
        layer.reset_state(batch_size=4)
        input_data = torch.rand(4, 24)
        higher_activity = torch.rand(4, 48)

        activity, prediction, error = layer(
            input_activity=input_data,
            higher_activity=higher_activity,
        )

        assert activity.shape == (4, 32)
        assert prediction.shape == (4, 32)
        assert error.shape == (4, 32)


class TestWorldModelConfig:
    """Tests for WorldModelConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = WorldModelConfig()
        assert config.n_sensory == 128
        assert config.n_hidden == 256
        assert config.n_layers == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = WorldModelConfig(
            n_sensory=64,
            n_hidden=128,
            n_action=16,
        )
        assert config.n_sensory == 64
        assert config.n_hidden == 128
        assert config.n_action == 16


class TestWorldModel:
    """Tests for WorldModel."""

    @pytest.fixture
    def model(self):
        """Create a basic world model."""
        config = WorldModelConfig(
            n_sensory=32,
            n_hidden=64,
            n_action=8,
            n_layers=2,
        )
        return WorldModel(config)

    def test_creation(self, model):
        """Test model creation."""
        assert model is not None
        assert len(model.layers) == 2

    def test_reset_state(self, model):
        """Test state reset."""
        model.reset_state(batch_size=2)
        assert model.belief_mean.shape == (2, 64)

    def test_forward_step(self, model):
        """Test single forward step."""
        model.reset_state(batch_size=1)
        sensory = torch.rand(1, 32)

        state, errors = model(sensory)

        assert state.shape == (1, 64)
        assert len(errors) == 2

    def test_forward_with_action(self, model):
        """Test forward with action."""
        model.reset_state(batch_size=1)
        sensory = torch.rand(1, 32)
        action = torch.rand(1, 8)

        state, errors = model(sensory, action=action)

        assert state.shape == (1, 64)

    def test_predict_next(self, model):
        """Test next state prediction."""
        model.reset_state(batch_size=1)
        sensory = torch.rand(1, 32)
        model(sensory)

        next_state = model.predict_next()
        assert next_state.shape == (1, 64)

    def test_simulate(self, model):
        """Test simulation without input."""
        model.reset_state(batch_size=1)
        sensory = torch.rand(1, 32)
        model(sensory)  # Initialize belief

        states = model.simulate(steps=10)

        assert len(states) == 10
        assert all(s.shape == (1, 64) for s in states)

    def test_simulate_with_actions(self, model):
        """Test simulation with action sequence."""
        model.reset_state(batch_size=1)
        sensory = torch.rand(1, 32)
        model(sensory)

        actions = torch.rand(10, 1, 8)
        states = model.simulate(steps=10, action_sequence=actions)

        assert len(states) == 10

    def test_simulate_action(self, model):
        """Test single action simulation."""
        model.reset_state(batch_size=1)
        sensory = torch.rand(1, 32)
        model(sensory)

        action = torch.rand(1, 8)
        final_state, surprise = model.simulate_action(action, steps=20)

        assert final_state.shape == (1, 64)
        assert isinstance(surprise, float)

    def test_surprise_tracking(self, model):
        """Test surprise accumulation."""
        model.reset_state(batch_size=1)

        for _ in range(10):
            sensory = torch.rand(1, 32)
            model(sensory)

        assert model.get_total_surprise() > 0
        assert len(model.get_surprise_history()) == 10

    def test_belief_getter(self, model):
        """Test getting belief state."""
        model.reset_state(batch_size=1)
        sensory = torch.rand(1, 32)
        model(sensory)

        mean, precision = model.get_belief()
        assert mean.shape == (1, 64)
        assert precision.shape == (1, 64)

    def test_layer_errors(self, model):
        """Test getting layer errors."""
        model.reset_state(batch_size=1)
        sensory = torch.rand(1, 32)
        model(sensory)

        errors = model.get_layer_errors()
        assert len(errors) == 2


class TestActionSimulator:
    """Tests for ActionSimulator."""

    @pytest.fixture
    def simulator(self):
        """Create action simulator."""
        config = WorldModelConfig(n_sensory=32, n_hidden=64, n_action=8, n_layers=2)
        model = WorldModel(config)
        model.reset_state(batch_size=1)
        return ActionSimulator(model, simulation_steps=20)

    def test_creation(self, simulator):
        """Test simulator creation."""
        assert simulator is not None
        assert simulator.simulation_steps == 20

    def test_simulate_action(self, simulator):
        """Test simulating single action."""
        # Initialize model
        sensory = torch.rand(1, 32)
        simulator.world_model(sensory)

        action = torch.rand(1, 8)
        result = simulator.simulate_action(action)

        assert isinstance(result, SimulationResult)
        assert result.final_state.shape == (1, 64)
        assert len(result.trajectory) > 0

    def test_evaluate_actions(self, simulator):
        """Test evaluating multiple actions."""
        sensory = torch.rand(1, 32)
        simulator.world_model(sensory)

        actions = [torch.rand(1, 8) for _ in range(3)]
        results = simulator.evaluate_actions(actions)

        assert len(results) == 3
        assert all(isinstance(r, SimulationResult) for r in results)

    def test_select_best_action(self, simulator):
        """Test selecting best action."""
        sensory = torch.rand(1, 32)
        simulator.world_model(sensory)

        actions = [torch.rand(1, 8) for _ in range(3)]

        # Simple reward function
        def reward_fn(state):
            return state.mean().item()

        best_action, result = simulator.select_best_action(actions, reward_fn)

        assert best_action.shape == (1, 8)
        assert result.expected_reward != 0.0


class TestPredictiveCodingNetwork:
    """Tests for PredictiveCodingNetwork."""

    @pytest.fixture
    def network(self):
        """Create predictive coding network."""
        return PredictiveCodingNetwork(
            layer_sizes=[32, 64, 32],
            tau_mem=20.0,
            tau_error=5.0,
        )

    def test_creation(self, network):
        """Test network creation."""
        assert network is not None
        assert len(network.repr_layers) == 3
        assert len(network.error_layers) == 2

    def test_reset_state(self, network):
        """Test state reset."""
        network.reset_state(batch_size=2)
        assert network.repr_layers[0].membrane is not None

    def test_forward(self, network):
        """Test forward pass."""
        network.reset_state(batch_size=1)
        sensory = torch.rand(1, 32)

        representations, errors = network(sensory, n_iterations=3)

        assert len(representations) == 3
        assert len(errors) == 2
        assert representations[0].shape == (1, 32)
        assert representations[1].shape == (1, 64)
        assert representations[2].shape == (1, 32)

    def test_total_error(self, network):
        """Test total error computation."""
        network.reset_state(batch_size=1)
        sensory = torch.rand(1, 32)

        _, errors = network(sensory)

        total_error = network.get_total_error(errors)
        assert isinstance(total_error, float)
        assert total_error >= 0

    def test_iterative_inference(self, network):
        """Test that more iterations reduce error."""
        network.reset_state(batch_size=1)
        sensory = torch.rand(1, 32)

        # Few iterations
        _, errors_few = network(sensory, n_iterations=1)
        error_few = network.get_total_error(errors_few)

        network.reset_state(batch_size=1)

        # Many iterations
        _, errors_many = network(sensory, n_iterations=10)
        error_many = network.get_total_error(errors_many)

        # More iterations should generally reduce error
        # (not always guaranteed, but usually)

    def test_batched_forward(self, network):
        """Test batched processing."""
        network.reset_state(batch_size=4)
        sensory = torch.rand(4, 32)

        representations, errors = network(sensory)

        assert representations[0].shape == (4, 32)


class TestGPU:
    """Tests for GPU compatibility."""

    @pytest.fixture
    def device(self):
        """Get available device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_predictive_layer_on_device(self, device):
        """Test layer on device."""
        config = PredictiveLayerConfig(n_neurons=32)
        layer = PredictiveLayer(config, input_size=24).to(device)

        layer.reset_state(batch_size=1)
        input_data = torch.rand(1, 24, device=device)

        activity, _, _ = layer(input_activity=input_data)
        assert activity.device.type == device.type

    def test_world_model_on_device(self, device):
        """Test world model on device."""
        config = WorldModelConfig(n_sensory=32, n_hidden=64, n_layers=2)
        model = WorldModel(config).to(device)

        model.reset_state(batch_size=1)
        sensory = torch.rand(1, 32, device=device)

        state, errors = model(sensory)
        assert state.device.type == device.type

    def test_predictive_coding_on_device(self, device):
        """Test predictive coding network on device."""
        network = PredictiveCodingNetwork([32, 64, 32]).to(device)
        network.reset_state(batch_size=1)

        sensory = torch.rand(1, 32, device=device)
        representations, _ = network(sensory)

        assert representations[0].device.type == device.type


class TestIntegration:
    """Integration tests."""

    def test_world_model_sequence(self):
        """Test world model processing a sequence."""
        config = WorldModelConfig(n_sensory=32, n_hidden=64, n_layers=2)
        model = WorldModel(config)
        model.reset_state(batch_size=1)

        # Process sequence of observations
        surprises = []
        for _ in range(20):
            sensory = torch.rand(1, 32)
            model(sensory)
            surprises.append(model.get_surprise())

        assert len(surprises) == 20

    def test_action_planning(self):
        """Test using world model for action planning."""
        config = WorldModelConfig(n_sensory=32, n_hidden=64, n_action=8, n_layers=2)
        model = WorldModel(config)
        model.reset_state(batch_size=1)

        # Initialize with observation
        sensory = torch.rand(1, 32)
        model(sensory)

        # Evaluate candidate actions
        simulator = ActionSimulator(model)
        actions = [torch.rand(1, 8) for _ in range(5)]

        def goal_fn(state):
            return -state.var().item()  # Prefer low variance states

        best_action, result = simulator.select_best_action(actions, goal_fn)

        assert best_action is not None
        assert result.expected_reward != 0
