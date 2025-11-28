"""
World Model - Predictive processing and internal simulation.

This module implements the network's internal model of the world,
enabling it to:

- Predict future sensory inputs
- Simulate actions without executing them
- Detect prediction errors (surprise)
- Update beliefs based on evidence

The World Model is based on predictive processing theory:
1. The brain constantly predicts incoming sensory data
2. Prediction errors drive learning and attention
3. Internal simulation enables planning and imagination

Key components:
- PredictiveLayer: Generates predictions and computes errors
- WorldModel: Multi-layer predictive hierarchy
- ActionSimulator: Simulates outcomes of potential actions
- BeliefState: Represents probabilistic beliefs about world state

References:
- Rao & Ballard (1999) - Predictive Coding
- Friston (2010) - Free Energy Principle
- Hawkins & Blakeslee (2004) - On Intelligence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.core.neuron import LIFNeuron, LIFConfig


class PredictionMode(Enum):
    """How the world model generates predictions."""
    FEEDFORWARD = auto()     # Bottom-up sensory processing
    GENERATIVE = auto()      # Top-down prediction generation
    COMBINED = auto()        # Bidirectional predictive coding


@dataclass
class PredictiveLayerConfig:
    """Configuration for a predictive processing layer.
    
    Attributes:
        n_neurons: Number of neurons in this layer
        n_prediction_neurons: Neurons for generating predictions
        tau_mem: Membrane time constant
        tau_prediction: Time constant for prediction updates
        noise_std: Noise level
        learning_rate: Rate for updating prediction weights
        error_gain: How much prediction errors affect activity
        dt: Simulation timestep
    """
    n_neurons: int = 64
    n_prediction_neurons: int = 64
    tau_mem: float = 20.0
    tau_prediction: float = 50.0  # Slower than membrane
    noise_std: float = 0.02
    learning_rate: float = 0.01
    error_gain: float = 0.5
    dt: float = 1.0


@dataclass
class WorldModelConfig:
    """Configuration for the full world model.
    
    Attributes:
        n_sensory: Number of sensory input neurons
        n_hidden: Number of hidden state neurons
        n_action: Number of action neurons (for simulation)
        n_layers: Number of predictive layers
        tau_belief: Time constant for belief updates
        prediction_horizon: How far ahead to predict (timesteps)
        simulation_steps: Steps to run for action simulation
        precision_weighting: Whether to weight by precision
        dt: Simulation timestep
    """
    n_sensory: int = 128
    n_hidden: int = 256
    n_action: int = 32
    n_layers: int = 3
    tau_belief: float = 100.0
    prediction_horizon: int = 10
    simulation_steps: int = 50
    precision_weighting: bool = True
    dt: float = 1.0


class PredictiveLayer(nn.Module):
    """A layer that generates predictions and computes errors.
    
    This implements the core of predictive coding:
    - Representation neurons encode the current state
    - Prediction neurons generate expected input from higher levels
    - Error neurons compute difference between prediction and input
    
    The layer learns to predict its inputs, with prediction errors
    propagating up the hierarchy to update beliefs.
    
    Example:
        >>> config = PredictiveLayerConfig(n_neurons=64)
        >>> layer = PredictiveLayer(config, input_size=32)
        >>> 
        >>> # Forward pass with input
        >>> output, prediction, error = layer(input_tensor)
        >>> 
        >>> # Top-down prediction (generative)
        >>> pred = layer.generate_prediction(higher_layer_activity)
    """
    
    def __init__(
        self, 
        config: PredictiveLayerConfig,
        input_size: Optional[int] = None,
        higher_size: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        n = config.n_neurons
        
        # Representation neurons (encode state)
        neuron_config = LIFConfig(
            tau_mem=config.tau_mem,
            noise_std=config.noise_std,
            dt=config.dt,
        )
        self.neurons = LIFNeuron(n_neurons=n, config=neuron_config)
        
        # Prediction neurons (separate population)
        pred_config = LIFConfig(
            tau_mem=config.tau_prediction,
            noise_std=config.noise_std * 0.5,  # Less noisy predictions
            dt=config.dt,
        )
        self.pred_neurons = LIFNeuron(
            n_neurons=config.n_prediction_neurons,
            config=pred_config,
        )
        
        # Feedforward weights (input → representation)
        if input_size is not None:
            self.ff_weights = nn.Parameter(
                torch.randn(input_size, n) * 0.1 / (input_size ** 0.5)
            )
        else:
            self.ff_weights = None
            
        # Prediction weights (higher → prediction → representation)
        if higher_size is not None:
            self.pred_weights = nn.Parameter(
                torch.randn(higher_size, config.n_prediction_neurons) * 0.1 / (higher_size ** 0.5)
            )
            self.pred_to_repr = nn.Parameter(
                torch.randn(config.n_prediction_neurons, n) * 0.1 / (config.n_prediction_neurons ** 0.5)
            )
        else:
            self.pred_weights = None
            self.pred_to_repr = None
            
        # Error computation weights
        self.register_buffer(
            "error_weights",
            torch.eye(n) if input_size == n else torch.randn(n, n) * 0.1
        )
        
        # Recurrent weights for temporal prediction
        self.recurrent = nn.Parameter(torch.randn(n, n) * 0.01)
        self.register_buffer("recurrent_mask", 1 - torch.eye(n))
        
        # State
        self._last_activity: Optional[torch.Tensor] = None
        self._last_prediction: Optional[torch.Tensor] = None
        self._prediction_error: Optional[torch.Tensor] = None
        self._precision: Optional[torch.Tensor] = None
        
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset layer state."""
        self.neurons.reset_state(batch_size)
        self.pred_neurons.reset_state(batch_size)
        self._last_activity = None
        self._last_prediction = None
        self._prediction_error = None
        device = self._get_device()
        self._precision = torch.ones(batch_size, self.config.n_neurons, device=device)
        
    def _get_device(self) -> torch.device:
        """Get the device this layer is on."""
        if self.ff_weights is not None:
            return self.ff_weights.device
        return self.recurrent.device
    
    def forward(
        self,
        input_activity: Optional[torch.Tensor] = None,
        higher_activity: Optional[torch.Tensor] = None,
        mode: PredictionMode = PredictionMode.COMBINED,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process one timestep.
        
        Args:
            input_activity: Bottom-up input from lower layer
            higher_activity: Top-down input from higher layer
            mode: How to combine feedforward and predictive
            
        Returns:
            activity: Current layer activity (spikes)
            prediction: What was predicted for input
            error: Prediction error (input - prediction)
        """
        device = self._get_device()
        batch_size = 1
        
        if input_activity is not None:
            batch_size = input_activity.shape[0]
        elif higher_activity is not None:
            batch_size = higher_activity.shape[0]
        elif self.neurons.membrane is not None:
            batch_size = self.neurons.membrane.shape[0]
            
        if self.neurons.membrane is None:
            self.reset_state(batch_size)
            
        # === Generate prediction from higher layer ===
        prediction = torch.zeros(batch_size, self.config.n_neurons, device=device)
        
        if higher_activity is not None and self.pred_weights is not None:
            # Higher activity → prediction neurons → representation prediction
            pred_input = torch.matmul(higher_activity, self.pred_weights)
            pred_spikes, _ = self.pred_neurons(pred_input)
            prediction = torch.matmul(pred_spikes, self.pred_to_repr)
            
        # === Compute prediction error ===
        if input_activity is not None:
            # Project input to representation space if needed
            if self.ff_weights is not None:
                input_repr = torch.matmul(input_activity, self.ff_weights)
            else:
                input_repr = input_activity
                
            error = input_repr - prediction
        else:
            error = torch.zeros(batch_size, self.config.n_neurons, device=device)
            input_repr = prediction  # Use prediction as input when no external input
            
        # === Combine based on mode ===
        if mode == PredictionMode.FEEDFORWARD:
            total_input = input_repr if input_activity is not None else torch.zeros_like(prediction)
        elif mode == PredictionMode.GENERATIVE:
            total_input = prediction
        else:  # COMBINED
            # Precision-weighted combination
            if self._precision is not None:
                # High precision = trust the prediction error more
                total_input = prediction + error * self.config.error_gain * self._precision
            else:
                total_input = prediction + error * self.config.error_gain
                
        # === Add recurrent input ===
        if self._last_activity is not None:
            recurrent_input = torch.matmul(
                self._last_activity,
                self.recurrent * self.recurrent_mask
            )
            total_input = total_input + recurrent_input
            
        # === Update neurons ===
        spikes, membrane = self.neurons(total_input)
        
        # === Update state ===
        self._last_activity = spikes
        self._last_prediction = prediction
        self._prediction_error = error
        
        # === Update precision (inverse variance of error) ===
        # High error variance = low precision = less trust in predictions
        if self._precision is not None:
            error_var = error.pow(2).mean(dim=1, keepdim=True) + 1e-6
            new_precision = 1.0 / error_var
            # Slow update
            alpha = self.config.dt / self.config.tau_prediction
            self._precision = (1 - alpha) * self._precision + alpha * new_precision.expand_as(self._precision)
            
        return spikes, prediction, error
    
    def generate_prediction(self, higher_activity: torch.Tensor) -> torch.Tensor:
        """Generate top-down prediction without updating state."""
        if self.pred_weights is None:
            return torch.zeros(higher_activity.shape[0], self.config.n_neurons, device=higher_activity.device)
            
        pred_input = torch.matmul(higher_activity, self.pred_weights)
        # Use membrane potential for continuous prediction
        self.pred_neurons.reset_state(higher_activity.shape[0])
        _, membrane = self.pred_neurons(pred_input)
        prediction = torch.matmul(torch.sigmoid(membrane), self.pred_to_repr)
        return prediction
    
    def get_prediction_error(self) -> Optional[torch.Tensor]:
        """Get the current prediction error."""
        return self._prediction_error
    
    def get_precision(self) -> Optional[torch.Tensor]:
        """Get current precision estimate."""
        return self._precision
    
    def get_surprise(self) -> float:
        """Get surprise (magnitude of prediction error)."""
        if self._prediction_error is None:
            return 0.0
        return self._prediction_error.abs().mean().item()


class WorldModel(nn.Module):
    """Multi-layer predictive world model.
    
    Implements a hierarchical predictive processing system where:
    - Lower layers predict sensory input
    - Higher layers predict patterns in lower layers  
    - Prediction errors drive updates and learning
    - The model can simulate future states
    
    Example:
        >>> config = WorldModelConfig(n_sensory=64, n_hidden=128)
        >>> model = WorldModel(config)
        >>> 
        >>> # Process sensory input
        >>> state, errors = model(sensory_input)
        >>> 
        >>> # Simulate future (without input)
        >>> future_states = model.simulate(steps=50)
        >>> 
        >>> # Evaluate potential action
        >>> predicted_outcome = model.simulate_action(action)
    """
    
    def __init__(self, config: Optional[WorldModelConfig] = None):
        super().__init__()
        self.config = config or WorldModelConfig()
        
        # Build predictive hierarchy
        self.layers = nn.ModuleList()
        
        # Layer sizes (expand then contract)
        sizes = self._compute_layer_sizes()
        
        for i in range(self.config.n_layers):
            layer_config = PredictiveLayerConfig(
                n_neurons=sizes[i],
                n_prediction_neurons=sizes[i] // 2,
                tau_mem=20.0 * (1.5 ** i),  # Slower at higher levels
                tau_prediction=50.0 * (1.5 ** i),
            )
            
            input_size = self.config.n_sensory if i == 0 else sizes[i-1]
            higher_size = sizes[i+1] if i < self.config.n_layers - 1 else None
            
            layer = PredictiveLayer(layer_config, input_size, higher_size)
            self.layers.append(layer)
            
        # Belief state (probabilistic estimate of world state)
        self.register_buffer(
            "belief_mean",
            torch.zeros(1, self.config.n_hidden)
        )
        self.register_buffer(
            "belief_precision",
            torch.ones(1, self.config.n_hidden)
        )
        
        # Action model (for simulation)
        self.action_weights = nn.Parameter(
            torch.randn(self.config.n_action, self.config.n_hidden) * 0.1
        )
        
        # Temporal prediction (state → next state)
        self.transition_weights = nn.Parameter(
            torch.randn(self.config.n_hidden, self.config.n_hidden) * 0.01
        )
        
        # State
        self._timestep = 0
        self._total_surprise = 0.0
        self._surprise_history: List[float] = []
        
    def _compute_layer_sizes(self) -> List[int]:
        """Compute sizes for each layer."""
        sizes = []
        n = self.config.n_sensory
        
        for i in range(self.config.n_layers):
            if i < self.config.n_layers // 2:
                # Expand
                sizes.append(int(n * (1.5 ** (i + 1))))
            else:
                # Contract toward hidden size
                progress = (i - self.config.n_layers // 2) / (self.config.n_layers // 2 + 1)
                size = int(sizes[-1] * (1 - progress * 0.5))
                size = max(size, self.config.n_hidden)
                sizes.append(size)
                
        return sizes
    
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset world model state."""
        for layer in self.layers:
            layer.reset_state(batch_size)
            
        device = self.action_weights.device
        self.belief_mean = torch.zeros(batch_size, self.config.n_hidden, device=device)
        self.belief_precision = torch.ones(batch_size, self.config.n_hidden, device=device)
        
        self._timestep = 0
        self._total_surprise = 0.0
        self._surprise_history = []
        
    def forward(
        self,
        sensory_input: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        mode: PredictionMode = PredictionMode.COMBINED,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Process one timestep of world model.
        
        Args:
            sensory_input: Current sensory observation
            action: Current action being taken
            mode: Prediction mode
            
        Returns:
            state: Current hidden state
            errors: Prediction errors at each layer
        """
        self._timestep += 1
        
        # Initialize if needed
        if len(self.layers) > 0 and self.layers[0].neurons.membrane is None:
            batch_size = sensory_input.shape[0] if sensory_input is not None else 1
            self.reset_state(batch_size)
            
        # === Bottom-up pass ===
        activities = []
        errors = []
        
        current_input = sensory_input
        for i, layer in enumerate(self.layers):
            # Get top-down prediction from higher layer
            higher_activity = activities[-1] if i > 0 and len(activities) > 0 else None
            
            # But we need higher layer activity from previous timestep
            # Use stored prediction for now
            higher_pred = None
            if i < len(self.layers) - 1:
                higher_pred = self.layers[i + 1]._last_activity
                
            activity, prediction, error = layer(
                input_activity=current_input,
                higher_activity=higher_pred,
                mode=mode,
            )
            
            activities.append(activity)
            errors.append(error)
            current_input = activity  # Feed to next layer
            
        # === Update belief state ===
        if len(activities) > 0:
            top_activity = activities[-1]
            
            # Precision-weighted belief update
            alpha = self.config.dt / self.config.tau_belief
            
            # Project top activity to hidden size
            if top_activity.shape[1] != self.config.n_hidden:
                # Simple averaging if sizes don't match
                if top_activity.shape[1] > self.config.n_hidden:
                    top_hidden = top_activity[:, :self.config.n_hidden]
                else:
                    top_hidden = F.pad(top_activity, (0, self.config.n_hidden - top_activity.shape[1]))
            else:
                top_hidden = top_activity
                
            self.belief_mean = (1 - alpha) * self.belief_mean + alpha * top_hidden
            
            # Apply action effect
            if action is not None:
                action_effect = torch.matmul(action, self.action_weights)
                self.belief_mean = self.belief_mean + action_effect * 0.1
                
        # === Track surprise ===
        layer_surprise = sum(layer.get_surprise() for layer in self.layers) / len(self.layers)
        self._total_surprise += layer_surprise
        self._surprise_history.append(layer_surprise)
        
        return self.belief_mean.clone(), errors
    
    def predict_next(self) -> torch.Tensor:
        """Predict next state from current belief."""
        next_state = torch.matmul(self.belief_mean, self.transition_weights)
        # Add current state (residual connection)
        next_state = next_state + self.belief_mean
        return torch.tanh(next_state)  # Bounded state
    
    def simulate(
        self,
        steps: int,
        action_sequence: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Simulate future states without sensory input.
        
        This is "imagination" - running the world model forward
        without actual sensory data.
        
        Args:
            steps: Number of steps to simulate
            action_sequence: Optional actions to apply (steps, batch, n_action)
            
        Returns:
            List of predicted states
        """
        states = []
        
        for t in range(steps):
            # Predict next state
            next_state = self.predict_next()
            
            # Apply action if provided
            if action_sequence is not None and t < action_sequence.shape[0]:
                action = action_sequence[t]
                action_effect = torch.matmul(action, self.action_weights)
                next_state = next_state + action_effect * 0.1
                
            states.append(next_state)
            
            # Update belief for next iteration
            self.belief_mean = next_state
            
        return states
    
    def simulate_action(
        self,
        action: torch.Tensor,
        steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Simulate the outcome of taking an action.
        
        Runs the world model forward with the action to predict
        what will happen.
        
        Args:
            action: Action to simulate
            steps: Steps to simulate (default: config.simulation_steps)
            
        Returns:
            final_state: Predicted final state
            expected_surprise: Expected surprise along trajectory
        """
        if steps is None:
            steps = self.config.simulation_steps
            
        # Save current belief
        saved_belief = self.belief_mean.clone()
        
        # Simulate with action
        action_seq = action.unsqueeze(0).expand(steps, -1, -1)
        states = self.simulate(steps, action_seq)
        
        final_state = states[-1] if states else self.belief_mean
        
        # Estimate expected surprise (simple heuristic)
        # Large state changes = more surprise expected
        if len(states) > 1:
            deltas = [
                (states[i+1] - states[i]).abs().mean().item()
                for i in range(len(states)-1)
            ]
            expected_surprise = sum(deltas) / len(deltas)
        else:
            expected_surprise = 0.0
            
        # Restore belief
        self.belief_mean = saved_belief
        
        return final_state, expected_surprise
    
    def get_surprise(self) -> float:
        """Get current surprise level."""
        if len(self._surprise_history) == 0:
            return 0.0
        return self._surprise_history[-1]
    
    def get_total_surprise(self) -> float:
        """Get cumulative surprise."""
        return self._total_surprise
    
    def get_surprise_history(self) -> List[float]:
        """Get history of surprise values."""
        return list(self._surprise_history)
    
    def get_belief(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current belief state (mean, precision)."""
        return self.belief_mean.clone(), self.belief_precision.clone()
    
    def get_layer_errors(self) -> List[Optional[torch.Tensor]]:
        """Get prediction errors from all layers."""
        return [layer.get_prediction_error() for layer in self.layers]


@dataclass
class SimulationResult:
    """Result of action simulation.
    
    Attributes:
        action: The action that was simulated
        final_state: Predicted final state
        trajectory: Predicted state trajectory
        expected_surprise: Expected surprise
        expected_reward: Expected reward (if available)
    """
    action: torch.Tensor
    final_state: torch.Tensor
    trajectory: List[torch.Tensor]
    expected_surprise: float
    expected_reward: float = 0.0


class ActionSimulator:
    """Evaluates potential actions through simulation.
    
    Uses the world model to "imagine" outcomes of different
    actions before executing them.
    
    Example:
        >>> simulator = ActionSimulator(world_model)
        >>> 
        >>> # Evaluate multiple actions
        >>> actions = [action1, action2, action3]
        >>> results = simulator.evaluate_actions(actions)
        >>> 
        >>> # Pick best action
        >>> best = simulator.select_best_action(actions, reward_fn)
    """
    
    def __init__(
        self,
        world_model: WorldModel,
        simulation_steps: int = 50,
    ):
        self.world_model = world_model
        self.simulation_steps = simulation_steps
        
    def simulate_action(
        self,
        action: torch.Tensor,
        steps: Optional[int] = None,
    ) -> SimulationResult:
        """Simulate a single action and return detailed result."""
        if steps is None:
            steps = self.simulation_steps
            
        # Save state
        saved_belief = self.world_model.belief_mean.clone()
        
        # Simulate
        action_seq = action.unsqueeze(0).expand(steps, -1, -1)
        trajectory = self.world_model.simulate(steps, action_seq)
        
        final_state = trajectory[-1] if trajectory else self.world_model.belief_mean
        
        # Compute expected surprise
        if len(trajectory) > 1:
            deltas = [
                (trajectory[i+1] - trajectory[i]).abs().mean().item()
                for i in range(len(trajectory)-1)
            ]
            expected_surprise = sum(deltas) / len(deltas)
        else:
            expected_surprise = 0.0
            
        # Restore state
        self.world_model.belief_mean = saved_belief
        
        return SimulationResult(
            action=action,
            final_state=final_state,
            trajectory=trajectory,
            expected_surprise=expected_surprise,
        )
    
    def evaluate_actions(
        self,
        actions: List[torch.Tensor],
        steps: Optional[int] = None,
    ) -> List[SimulationResult]:
        """Evaluate multiple actions through simulation."""
        return [self.simulate_action(action, steps) for action in actions]
    
    def select_best_action(
        self,
        actions: List[torch.Tensor],
        reward_fn: Callable[[torch.Tensor], float],
        prefer_low_surprise: bool = True,
        surprise_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, SimulationResult]:
        """Select best action based on expected reward and surprise.
        
        Args:
            actions: List of candidate actions
            reward_fn: Function that scores states
            prefer_low_surprise: Whether to prefer predictable outcomes
            surprise_weight: How much to weight surprise in decision
            
        Returns:
            best_action: The selected action
            result: Simulation result for best action
        """
        results = self.evaluate_actions(actions)
        
        # Score each action
        scores = []
        for result in results:
            reward = reward_fn(result.final_state)
            result.expected_reward = reward
            
            if prefer_low_surprise:
                score = reward - surprise_weight * result.expected_surprise
            else:
                score = reward + surprise_weight * result.expected_surprise  # Curious agent
                
            scores.append(score)
            
        # Select best
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return actions[best_idx], results[best_idx]


class PredictiveCodingNetwork(nn.Module):
    """Full predictive coding network with bidirectional message passing.
    
    Implements the classic predictive coding architecture:
    - Error units compute prediction errors
    - Representation units encode state
    - Top-down connections carry predictions
    - Bottom-up connections carry prediction errors
    
    This creates a hierarchy where each level tries to predict
    the level below, with only unpredicted (surprising) information
    propagating upward.
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        tau_mem: float = 20.0,
        tau_error: float = 5.0,
        learning_rate: float = 0.01,
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        
        # Representation neurons at each level
        self.repr_layers = nn.ModuleList()
        for size in layer_sizes:
            config = LIFConfig(tau_mem=tau_mem, noise_std=0.02)
            self.repr_layers.append(LIFNeuron(n_neurons=size, config=config))
            
        # Error neurons at each level (except top)
        self.error_layers = nn.ModuleList()
        for size in layer_sizes[:-1]:
            config = LIFConfig(tau_mem=tau_error, noise_std=0.01)
            self.error_layers.append(LIFNeuron(n_neurons=size, config=config))
            
        # Bottom-up weights (error → representation)
        self.bu_weights = nn.ParameterList()
        for i in range(len(layer_sizes) - 1):
            w = nn.Parameter(
                torch.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1 / (layer_sizes[i] ** 0.5)
            )
            self.bu_weights.append(w)
            
        # Top-down weights (representation → prediction)
        self.td_weights = nn.ParameterList()
        for i in range(len(layer_sizes) - 1):
            w = nn.Parameter(
                torch.randn(layer_sizes[i+1], layer_sizes[i]) * 0.1 / (layer_sizes[i+1] ** 0.5)
            )
            self.td_weights.append(w)
            
        self.learning_rate = learning_rate
        
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset all layers."""
        for layer in self.repr_layers:
            layer.reset_state(batch_size)
        for layer in self.error_layers:
            layer.reset_state(batch_size)
            
    def forward(
        self,
        sensory_input: torch.Tensor,
        n_iterations: int = 5,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run predictive coding inference.
        
        Runs multiple iterations of message passing to settle
        into a coherent interpretation.
        
        Args:
            sensory_input: Bottom-level input
            n_iterations: Iterations of message passing
            
        Returns:
            representations: Activity at each level
            errors: Prediction error at each level
        """
        batch_size = sensory_input.shape[0]
        device = sensory_input.device
        
        # Initialize if needed
        if self.repr_layers[0].membrane is None:
            self.reset_state(batch_size)
            
        # Initialize representations with feedforward sweep
        representations = [sensory_input]
        for i, layer in enumerate(self.repr_layers):
            if i == 0:
                _, membrane = layer(sensory_input)
                representations[0] = torch.sigmoid(membrane)
            else:
                input_activity = torch.matmul(representations[-1], self.bu_weights[i-1])
                _, membrane = layer(input_activity)
                representations.append(torch.sigmoid(membrane))
                
        # Iterative inference
        for _ in range(n_iterations):
            errors = []
            
            # Compute errors (bottom-up)
            for i in range(len(self.layer_sizes) - 1):
                # Prediction from higher level
                prediction = torch.matmul(representations[i+1], self.td_weights[i])
                
                # Error = representation - prediction
                error = representations[i] - prediction
                
                # Error neurons
                _, membrane = self.error_layers[i](error)
                errors.append(torch.tanh(membrane))  # Bounded error
                
            # Update representations (combine bottom-up and top-down)
            for i in range(len(self.repr_layers)):
                # Bottom-up input
                if i == 0:
                    bu_input = sensory_input
                else:
                    bu_input = torch.matmul(errors[i-1], self.bu_weights[i-1])
                    
                # Top-down input (prediction)
                if i < len(self.layer_sizes) - 1:
                    td_input = torch.matmul(representations[i+1], self.td_weights[i])
                else:
                    td_input = torch.zeros(batch_size, self.layer_sizes[i], device=device)
                    
                # Combined input
                total_input = bu_input + td_input * 0.5
                
                _, membrane = self.repr_layers[i](total_input)
                representations[i] = torch.sigmoid(membrane)
                
        # Final error computation
        final_errors = []
        for i in range(len(self.layer_sizes) - 1):
            prediction = torch.matmul(representations[i+1], self.td_weights[i])
            error = representations[i] - prediction
            final_errors.append(error)
            
        return representations, final_errors
    
    def get_total_error(self, errors: List[torch.Tensor]) -> float:
        """Compute total prediction error (free energy proxy)."""
        return sum(e.pow(2).mean().item() for e in errors)
