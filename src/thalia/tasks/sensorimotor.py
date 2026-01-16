"""
Sensorimotor Tasks for Stage -0.5 (Curriculum Training).

Implements embodied learning tasks for establishing sensorimotor coordination:
- Basic motor control (movement commands)
- Visual-motor coordination (reaching)
- Object manipulation (push/pull/grasp)
- Sensorimotor prediction (forward models)

These tasks provide continuous reward feedback based on behavioral accuracy,
following the continuous learning architecture where:
- Learning happens automatically during forward passes (STDP, BCM, Hebbian)
- Rewards modulate dopamine for striatum/PFC three-factor learning
- Accuracy measured by behavioral outcomes (movement direction, reaching error)

Author: Thalia Project
Date: December 9, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import random

import torch
import numpy as np

from thalia.constants.task import (
    PROPRIOCEPTION_NOISE_SCALE,
)
from thalia.tasks.stimulus_utils import (
    create_zero_stimulus,
    create_random_position,
    add_proprioceptive_noise,
)


# ============================================================================
# Task Types
# ============================================================================

class SensorimotorTaskType(Enum):
    """Types of sensorimotor tasks."""
    MOTOR_CONTROL = "motor_control"
    REACHING = "reaching"
    MANIPULATION = "manipulation"
    PREDICTION = "prediction"


class MovementDirection(Enum):
    """Basic movement directions."""
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    FORWARD = 4
    BACK = 5
    STOP = 6


# ============================================================================
# Task Configurations
# ============================================================================

@dataclass
class MotorControlConfig:
    """Configuration for basic motor control tasks."""

    input_size: int = 128  # Sensory input dimension
    n_timesteps: int = 10  # Duration of trial
    difficulty: float = 0.5  # Task difficulty (0-1)
    noise_level: float = PROPRIOCEPTION_NOISE_SCALE  # Noise in proprioceptive feedback

    # Movement parameters
    movement_types: List[MovementDirection] = field(
        default_factory=lambda: list(MovementDirection)
    )

    # Reward parameters
    accuracy_threshold: float = 0.7  # Minimum cosine similarity for success
    reward_scale: float = 1.0  # Scale factor for rewards


@dataclass
class ReachingConfig:
    """Configuration for visual-motor reaching tasks."""

    input_size: int = 128
    n_timesteps: int = 15  # Longer for reaching movements
    difficulty: float = 0.6

    # Spatial parameters
    workspace_size: float = 10.0  # Size of reachable workspace
    target_size: float = 0.5  # Size of target

    # Visual feedback
    visual_input_size: int = 64  # Visual input dimension
    proprioceptive_size: int = 32  # Proprioception dimension

    # Reward parameters
    distance_threshold: float = 1.0  # Success distance
    reward_scale: float = 1.0


@dataclass
class ManipulationConfig:
    """Configuration for object manipulation tasks."""

    input_size: int = 128
    n_timesteps: int = 20  # Longer for manipulation sequences
    difficulty: float = 0.7

    # Object parameters
    n_objects: int = 5  # Number of objects in environment
    object_properties: List[str] = field(
        default_factory=lambda: ["pushable", "graspable", "pullable"]
    )

    # Reward parameters
    success_reward: float = 1.0
    failure_penalty: float = -0.3


# ============================================================================
# Motor Control Task
# ============================================================================

class MotorControlTask:
    """Basic motor control task for Stage -0.5.

    Generates simple movement commands (left, right, up, down, etc.) and
    provides reward based on execution accuracy. Measures behavioral outcome
    using cosine similarity between intended and actual movement.

    **Architecture Integration**:
    - Input: Proprioceptive state + movement command encoding
    - Brain processes via motor cortex + cerebellum
    - Output: Motor commands (striatum action selection)
    - Reward: Based on movement accuracy (dopamine modulation)

    **Example Usage**:
    ```python
    task = MotorControlTask(config)
    task_data = task.get_task("basic_movement")

    # Brain processes (learning happens inside)
    output = brain.process_sample(task_data['input'], n_timesteps=10)

    # Compute reward from movement accuracy
    reward = task.compute_reward(output, task_data['target'])
    brain.deliver_reward(external_reward=reward)
    ```
    """

    def __init__(self, config: MotorControlConfig, device: str = "cpu"):
        """Initialize motor control task.

        Args:
            config: Task configuration
            device: PyTorch device
        """
        self.config = config
        self.device = device

        # Movement encoding (one-hot for each direction)
        self.n_directions = len(MovementDirection)
        self.current_target = None
        self.current_input = None

    def get_task(self, task_name: str = "motor_control") -> Dict[str, Any]:
        """Generate one motor control trial.

        Args:
            task_name: Name of task (unused, for interface compatibility)

        Returns:
            Dictionary with:
                - input: Sensory input tensor [input_size]
                - target: Target movement direction [n_directions]
                - n_timesteps: Trial duration
                - task_type: 'motor_control' (triggers reward delivery)
                - reward: Placeholder (computed after brain output)
                - metadata: Additional trial information
        """
        # Sample random movement direction
        direction = random.choice(self.config.movement_types)

        # Create target encoding (one-hot)
        target = create_zero_stimulus(self.n_directions, self.device)
        target[direction.value] = 1.0

        # Create input: proprioceptive state + movement command
        # First half: current state (noisy)
        proprioception = add_proprioceptive_noise(
            torch.zeros(self.config.input_size // 2, device=self.device),
            noise_scale=self.config.noise_level
        )

        # Second half: movement command embedding
        command_encoding = create_zero_stimulus(
            self.config.input_size // 2,
            self.device
        )
        # Encode target direction in input
        command_start = direction.value * (len(command_encoding) // self.n_directions)
        command_end = command_start + (len(command_encoding) // self.n_directions)
        command_encoding[command_start:command_end] = 1.0 * (1.0 - self.config.difficulty)

        input_tensor = torch.cat([proprioception, command_encoding])

        # Store for reward computation
        self.current_target = target
        self.current_input = input_tensor

        return {
            'input': input_tensor,
            'target': target,
            'n_timesteps': self.config.n_timesteps,
            'task_type': 'motor_control',  # Triggers reward delivery
            'reward': 0.0,  # Placeholder, computed after output
            'metadata': {
                'direction': direction.name,
                'difficulty': self.config.difficulty,
            }
        }

    def compute_reward(
        self,
        output: Dict[str, torch.Tensor],
        target: Optional[torch.Tensor] = None
    ) -> float:
        """Compute reward based on movement accuracy.

        Uses cosine similarity between brain output and target direction.
        This measures behavioral accuracy without inspecting neural activity.

        Args:
            output: Brain output dictionary (must contain 'spikes' or 'output')
            target: Target movement (uses self.current_target if None)

        Returns:
            Reward in [-1, +1] based on accuracy
        """
        if target is None:
            target = self.current_target

        if target is None:
            return 0.0

        # Extract brain output (use final timestep of spikes or dedicated output)
        if 'output' in output:
            brain_output = output['output']
        elif 'spikes' in output:
            # Use spike counts as output (sum over time)
            spikes = output['spikes']
            if spikes.dim() == 2:  # [time, neurons]
                brain_output = spikes.sum(dim=0)
            else:
                brain_output = spikes
        else:
            return 0.0

        # Ensure same dimensionality
        if brain_output.shape[0] != target.shape[0]:
            # Resize via average pooling
            pool_size = brain_output.shape[0] // target.shape[0]
            brain_output = brain_output[:pool_size * target.shape[0]].reshape(
                target.shape[0], pool_size
            ).mean(dim=1)

        # Compute cosine similarity (behavioral accuracy measure)
        brain_output_norm = brain_output / (brain_output.norm() + 1e-8)
        target_norm = target / (target.norm() + 1e-8)
        accuracy = torch.dot(brain_output_norm, target_norm).item()

        # Convert accuracy ∈ [-1, 1] to reward ∈ [-1, 1]
        # accuracy > threshold → positive reward
        # accuracy < threshold → negative reward
        if accuracy >= self.config.accuracy_threshold:
            reward = accuracy * self.config.reward_scale
        else:
            # Penalize poor performance
            reward = (accuracy - self.config.accuracy_threshold) * self.config.reward_scale

        return float(np.clip(reward, -1.0, 1.0))


# ============================================================================
# Reaching Task
# ============================================================================

class ReachingTask:
    """Visual-motor reaching task for Stage -0.5.

    Presents visual targets at different positions and rewards accurate
    reaching movements. Combines visual processing with motor control.

    **Architecture Integration**:
    - Input: Visual target position + proprioceptive state
    - Brain: Visual cortex → Motor cortex → Cerebellum (forward model)
    - Output: Reaching trajectory
    - Reward: Based on distance to target (closer = higher reward)
    """

    def __init__(self, config: ReachingConfig, device: str = "cpu"):
        """Initialize reaching task.

        Args:
            config: Task configuration
            device: PyTorch device
        """
        self.config = config
        self.device = device

        # Current trial state
        self.current_target_pos = None
        self.current_start_pos = None

    def get_task(self, task_name: str = "reaching") -> Dict[str, Any]:
        """Generate one reaching trial.

        Returns:
            Dictionary with input, target, reward parameters
        """
        # Sample random target position in workspace
        target_pos = create_random_position(self.config.workspace_size, self.device)

        # Sample start position (current effector position)
        start_pos = create_random_position(self.config.workspace_size, self.device)

        # Create visual input: target position encoded spatially
        visual_input = self._encode_visual_target(target_pos)

        # Create proprioceptive input: current position
        proprioceptive_input = self._encode_proprioception(start_pos)

        # Combine inputs
        input_tensor = torch.cat([visual_input, proprioceptive_input])

        # Pad to input_size if needed
        if input_tensor.shape[0] < self.config.input_size:
            padding = torch.zeros(
                self.config.input_size - input_tensor.shape[0],
                device=self.device
            )
            input_tensor = torch.cat([input_tensor, padding])
        else:
            input_tensor = input_tensor[:self.config.input_size]

        # Store for reward computation
        self.current_target_pos = target_pos
        self.current_start_pos = start_pos

        return {
            'input': input_tensor,
            'target': target_pos,
            'start_pos': start_pos,
            'n_timesteps': self.config.n_timesteps,
            'task_type': 'reaching',
            'reward': 0.0,
            'metadata': {
                'target_pos': target_pos.tolist(),
                'start_pos': start_pos.tolist(),
                'difficulty': self.config.difficulty,
            }
        }

    def _encode_visual_target(self, position: torch.Tensor) -> torch.Tensor:
        """Encode target position as visual input."""
        # Simple spatial encoding: Gaussian bump at target position
        visual_input = create_zero_stimulus(self.config.visual_input_size, self.device)

        # Map 2D position to 1D visual array
        pos_idx = int((position[0] / self.config.workspace_size) * len(visual_input))
        pos_idx = min(pos_idx, len(visual_input) - 1)

        # Gaussian bump (sigma = 10% of visual field)
        sigma = len(visual_input) * PROPRIOCEPTION_NOISE_SCALE
        for i in range(len(visual_input)):
            dist = abs(i - pos_idx)
            visual_input[i] = torch.exp(torch.tensor(-dist**2 / (2 * sigma**2), device=self.device))

        return visual_input

    def _encode_proprioception(self, position: torch.Tensor) -> torch.Tensor:
        """Encode current effector position as proprioceptive input."""
        # Direct encoding of position + some noise
        proprio = torch.cat([
            position,
            torch.randn(self.config.proprioceptive_size - 2, device=self.device) * PROPRIOCEPTION_NOISE_SCALE
        ])
        return proprio

    def compute_reward(
        self,
        output: Dict[str, torch.Tensor],
        target_pos: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None
    ) -> float:
        """Compute reward based on reaching accuracy.

        Reward based on final distance to target:
        - Distance < threshold: Full reward
        - Distance > threshold: Decaying reward

        Args:
            output: Brain output
            target_pos: Target position (uses self.current_target_pos if None)
            start_pos: Start position (uses self.current_start_pos if None)

        Returns:
            Reward in [-1, +1]
        """
        if target_pos is None:
            target_pos = self.current_target_pos
        if start_pos is None:
            start_pos = self.current_start_pos

        if target_pos is None or start_pos is None:
            return 0.0

        # Extract final position from output
        # For now, decode from spike patterns (simplified)
        if 'output' in output:
            brain_output = output['output']
        elif 'spikes' in output:
            spikes = output['spikes']
            if spikes.dim() == 2:
                brain_output = spikes.sum(dim=0)
            else:
                brain_output = spikes
        else:
            return 0.0

        # Decode position from brain output (first 2 dimensions)
        if brain_output.shape[0] >= 2:
            final_pos = brain_output[:2] * (self.config.workspace_size / 10.0)
        else:
            # Not enough output dimensions, return small reward
            return -0.5

        # Compute distance to target
        distance = torch.norm(final_pos - target_pos).item()

        # Compute reward based on distance
        if distance <= self.config.distance_threshold:
            # Success! Full reward
            reward = 1.0
        else:
            # Partial reward based on improvement over start
            start_distance = torch.norm(start_pos - target_pos).item()
            improvement = (start_distance - distance) / start_distance
            reward = improvement * self.config.reward_scale

        return float(np.clip(reward, -1.0, 1.0))


# ============================================================================
# Manipulation Task
# ============================================================================

class ManipulationTask:
    """Object manipulation task for Stage -0.5.

    Learn to interact with objects: push, pull, grasp, release.
    Develops understanding of object affordances and cause-effect.
    """

    def __init__(self, config: ManipulationConfig, device: str = "cpu"):
        """Initialize manipulation task."""
        self.config = config
        self.device = device

        self.current_action = None
        self.current_object = None

    def get_task(self, task_name: str = "manipulation") -> Dict[str, Any]:
        """Generate one manipulation trial.

        Returns:
            Dictionary with input, action, reward parameters
        """
        # Sample object with affordance
        object_idx = random.randint(0, self.config.n_objects - 1)
        affordance = random.choice(self.config.object_properties)

        # Create visual input: object representation
        visual_input = create_zero_stimulus(
            self.config.input_size // 2,
            self.device
        )
        # Encode object identity
        obj_start = object_idx * (len(visual_input) // self.config.n_objects)
        obj_end = obj_start + (len(visual_input) // self.config.n_objects)
        visual_input[obj_start:obj_end] = 1.0

        # Create action command: what to do with object
        action_input = create_zero_stimulus(
            self.config.input_size // 2,
            self.device
        )
        affordance_idx = self.config.object_properties.index(affordance)
        action_start = affordance_idx * (len(action_input) // len(self.config.object_properties))
        action_end = action_start + (len(action_input) // len(self.config.object_properties))
        action_input[action_start:action_end] = 1.0

        input_tensor = torch.cat([visual_input, action_input])

        # Store for reward computation
        self.current_action = affordance
        self.current_object = object_idx

        return {
            'input': input_tensor,
            'target_action': affordance,
            'object_id': object_idx,
            'n_timesteps': self.config.n_timesteps,
            'task_type': 'manipulation',
            'reward': 0.0,
            'metadata': {
                'action': affordance,
                'object': object_idx,
                'difficulty': self.config.difficulty,
            }
        }

    def compute_reward(
        self,
        output: Dict[str, torch.Tensor],
        target_action: Optional[str] = None
    ) -> float:
        """Compute reward for manipulation success.

        Simplified: check if output indicates successful manipulation.
        In real scenario, would have physics simulation.

        Args:
            output: Brain output
            target_action: Expected action (uses self.current_action if None)

        Returns:
            Reward in [-1, +1]
        """
        if target_action is None:
            target_action = self.current_action

        if target_action is None:
            return 0.0

        # Extract output
        if 'output' in output:
            brain_output = output['output']
        elif 'spikes' in output:
            spikes = output['spikes']
            if spikes.dim() == 2:
                brain_output = spikes.sum(dim=0)
            else:
                brain_output = spikes
        else:
            return self.config.failure_penalty

        # Check if manipulation successful (simplified)
        # In reality: physics simulation checks if object moved correctly
        output_mean = brain_output.mean().item()

        # Success if output is sufficiently active
        if output_mean > 0.2:  # Threshold for "action executed"
            reward = self.config.success_reward
        else:
            reward = self.config.failure_penalty

        return float(np.clip(reward, -1.0, 1.0))


# ============================================================================
# Sensorimotor Task Loader (Manager)
# ============================================================================

class SensorimotorTaskLoader:
    """Unified task loader for Stage -0.5 sensorimotor training.

    Orchestrates all sensorimotor tasks with proper mixing ratios:
    - 40% basic motor control
    - 35% reaching
    - 20% manipulation
    - 5% prediction

    **Usage**:
    ```python
    loader = SensorimotorTaskLoader(device='cuda')

    # Get task with automatic mixing
    task_data = loader.get_task()  # Samples task by mixing ratio

    # Or specify task type
    task_data = loader.get_task('motor_control')

    # After brain processes, compute reward
    output = brain.process_sample(task_data['input'], n_timesteps=task_data['n_timesteps'])
    reward = loader.compute_reward(output, task_data)
    brain.deliver_reward(external_reward=reward)
    ```
    """

    def __init__(
        self,
        device: str = "cpu",
        motor_control_config: Optional[MotorControlConfig] = None,
        reaching_config: Optional[ReachingConfig] = None,
        manipulation_config: Optional[ManipulationConfig] = None,
    ):
        """Initialize sensorimotor task loader.

        Args:
            device: PyTorch device
            motor_control_config: Motor control configuration (uses defaults if None)
            reaching_config: Reaching configuration (uses defaults if None)
            manipulation_config: Manipulation configuration (uses defaults if None)
        """
        self.device = device

        # Initialize task generators
        self.motor_control = MotorControlTask(
            motor_control_config or MotorControlConfig(),
            device=device
        )
        self.reaching = ReachingTask(
            reaching_config or ReachingConfig(),
            device=device
        )
        self.manipulation = ManipulationTask(
            manipulation_config or ManipulationConfig(),
            device=device
        )

        # Task mixing ratios (from curriculum strategy)
        self.task_weights = {
            'motor_control': 0.40,
            'reaching': 0.35,
            'manipulation': 0.20,
            'prediction': 0.05,  # Not implemented yet
        }

        # Current task tracker
        self.current_task_type = None
        self.current_task_data = None

    def get_task(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Get next sensorimotor task.

        Args:
            task_name: Specific task to get, or None for automatic sampling

        Returns:
            Task data dictionary
        """
        # Sample task if not specified
        if task_name is None:
            task_name = random.choices(
                list(self.task_weights.keys()),
                weights=list(self.task_weights.values()),
                k=1
            )[0]

        # Generate task
        if task_name == 'motor_control':
            task_data = self.motor_control.get_task()
        elif task_name == 'reaching':
            task_data = self.reaching.get_task()
        elif task_name == 'manipulation':
            task_data = self.manipulation.get_task()
        elif task_name == 'prediction':
            # Placeholder: use motor control for now
            task_data = self.motor_control.get_task()
            task_data['task_type'] = 'prediction'
        else:
            raise ValueError(f"Unknown task: {task_name}")

        # Store for reward computation
        self.current_task_type = task_name
        self.current_task_data = task_data

        return task_data

    def compute_reward(
        self,
        output: Dict[str, torch.Tensor],
        task_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute reward for current task.

        Args:
            output: Brain output dictionary
            task_data: Task data (uses self.current_task_data if None)

        Returns:
            Reward in [-1, +1]
        """
        if task_data is None:
            task_data = self.current_task_data
            task_type = self.current_task_type
        else:
            task_type = task_data.get('task_type', 'motor_control')

        if task_data is None:
            return 0.0

        # Compute reward based on task type
        if task_type == 'motor_control':
            reward = self.motor_control.compute_reward(
                output,
                task_data.get('target')
            )
        elif task_type == 'reaching':
            reward = self.reaching.compute_reward(
                output,
                task_data.get('target'),
                task_data.get('start_pos')
            )
        elif task_type == 'manipulation':
            reward = self.manipulation.compute_reward(
                output,
                task_data.get('target_action')
            )
        elif task_type == 'prediction':
            # Placeholder
            reward = self.motor_control.compute_reward(output)
        else:
            reward = 0.0

        return reward
