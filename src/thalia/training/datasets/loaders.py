"""
Task Loaders for Curriculum Training Stages (with Sensory Pathways).

This module provides real task loaders for each curriculum stage that:
- Load actual datasets (MNIST, CIFAR-10, phonological, sensorimotor)
- Use SENSORY PATHWAYS to encode raw data to fixed-size spike patterns
- Provide reward signals for RL tasks
- Support environment integration

CRITICAL ARCHITECTURE:
======================
Task loaders use sensory pathways to handle variable input sizes biologically:

    Raw Input (variable size: 275, 784, 3072, etc.)
        ↓
    Sensory Pathway (modality-specific encoding)
        ↓
    Spike Pattern (STANDARDIZED output_size, e.g., 256 neurons)
        ↓
    Brain (fixed input_size=256)

This matches how real brains work: sensory organs (retina, cochlea) perform
dimensionality reduction before cortical processing. Once information is
converted to SPIKES, the brain processes all modalities uniformly.

Author: Thalia Project
Date: December 9, 2025
"""

from __future__ import annotations

from typing import Protocol, Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import numpy as np

from thalia.training.constants import (
    REWARD_MOVEMENT_THRESHOLD,
    REWARD_SMALL_SUCCESS,
    REWARD_REACHING_THRESHOLD,
    REWARD_HIGH_SUCCESS,
    REWARD_MANIPULATION_BASE,
)
from thalia.sensory import (
    VisualConfig,
    RetinalEncoder,
)
from thalia.tasks.stimulus_utils import create_motor_spikes
from thalia.training.datasets.constants import (
    SPIKE_PROBABILITY_LOW,
    SPIKE_PROBABILITY_MEDIUM,
    SPIKE_PROBABILITY_HIGH,
    SENSORIMOTOR_WEIGHT_MOTOR_CONTROL,
    SENSORIMOTOR_WEIGHT_REACHING,
    SENSORIMOTOR_WEIGHT_MANIPULATION,
    SENSORIMOTOR_WEIGHT_PREDICTION,
    DATASET_WEIGHT_MNIST,
    DATASET_WEIGHT_TEMPORAL,
    DATASET_WEIGHT_PHONOLOGY,
    DATASET_WEIGHT_GAZE,
    REWARD_SCALE_PREDICTION,
)


# ============================================================================
# Task Loader Protocol
# ============================================================================

class BaseTaskLoader(Protocol):
    """Protocol for task loaders used by CurriculumTrainer.

    All task loaders must implement this interface.
    """

    def get_task(self, task_name: str) -> Dict[str, Any]:
        """Get a single task sample.

        Args:
            task_name: Name of task type to sample

        Returns:
            Dictionary with task data:
            - 'input': Input spikes (torch.Tensor, bool) [output_size]
            - 'n_timesteps': Number of timesteps to run (int)
            - 'reward': Reward signal (float, optional for RL tasks)
            - 'target': Target output (torch.Tensor, optional for supervised)
            - 'label': Classification label (int, optional)
            - Additional task-specific fields
        """
        ...

    def get_task_types(self) -> List[str]:
        """Get list of available task types."""
        ...

    def reset(self) -> None:
        """Reset task loader state."""
        ...


# ============================================================================
# Stage -0.5: Sensorimotor Task Loader
# ============================================================================

class TaskType(Enum):
    """Task types for sensorimotor stage."""
    MOTOR_CONTROL = "motor_control"
    REACHING = "reaching"
    MANIPULATION = "manipulation"
    PREDICTION = "prediction"


@dataclass
class SensorimotorConfig:
    """Configuration for sensorimotor task loader."""
    # Output size (must match brain.input_size)
    output_size: int = 256

    # Task mixing
    task_probabilities: Dict[str, float] = None

    # Episode settings
    max_episode_steps: int = 1000

    device: str = 'cpu'

    def __post_init__(self):
        if self.task_probabilities is None:
            # Equal probability for all tasks
            self.task_probabilities = {
                'motor_control': SENSORIMOTOR_WEIGHT_MOTOR_CONTROL,
                'reaching': SENSORIMOTOR_WEIGHT_REACHING,
                'manipulation': SENSORIMOTOR_WEIGHT_MANIPULATION,
                'prediction': SENSORIMOTOR_WEIGHT_PREDICTION,
            }


class SensorimotorTaskLoader:
    """Task loader for Stage -0.5: Sensorimotor grounding.

    Provides motor control, reaching, manipulation, and prediction tasks
    using MuJoCo environments through SensorimotorWrapper.

    ARCHITECTURE:
    =============
    SensorimotorWrapper produces variable-size spike patterns (e.g., 275 neurons).
    This loader projects them to fixed output_size to match brain.input_size.

    Raw proprioception (275) → Linear projection → Fixed spikes (256)

    This is biologically plausible: proprioceptive/motor neurons project to
    fixed-size cortical columns via adjustable synaptic connectivity.

    Tasks:
    ------
    - motor_control: Random motor commands (exploration)
    - reaching: Reach toward visual targets
    - manipulation: Push/pull objects
    - prediction: Learn forward/inverse models

    Returns:
    --------
    Task data with:
    - 'input': Observation spikes [output_size] (projected to fixed size)
    - 'reward': Task-specific reward signal
    - 'action': Motor command executed
    - 'target': Next observation (for prediction)
    """

    def __init__(
        self,
        wrapper: Any,
        config: Optional[SensorimotorConfig] = None,
        output_size: Optional[int] = None,
    ):
        """Initialize sensorimotor task loader.

        Args:
            wrapper: SensorimotorWrapper instance
            config: Task loader configuration
            output_size: Override config output_size (must match brain.input_size)
        """
        self.wrapper = wrapper
        self.config = config or SensorimotorConfig()

        # Override output_size if provided
        if output_size is not None:
            self.config.output_size = output_size

        self.task_types = [t.value for t in TaskType]

        # Track statistics
        self.task_counts = {t: 0 for t in self.task_types}
        self.task_successes = {t: 0 for t in self.task_types}

        # Environment state
        self.current_episode_step = 0
        self.last_obs = None
        self.last_action = None

        # Initialize wrapper with first observation
        self.current_obs = self.wrapper.reset()

        # Sensory projection: wrapper output → fixed size
        # Biologically: Proprioceptive neurons project to cortical columns
        wrapper_output_size = wrapper.n_sensory_neurons
        self.sensory_projection = nn.Linear(
            wrapper_output_size,
            self.config.output_size,
            bias=False,
        )
        # Initialize with sparse random connectivity
        nn.init.sparse_(self.sensory_projection.weight, sparsity=0.7)
        self.sensory_projection.to(self.config.device)

        print(f"[OK] SensorimotorTaskLoader: {wrapper_output_size} -> {self.config.output_size} neurons")

    def _encode_sensory(self, raw_spikes: torch.Tensor) -> torch.Tensor:
        """Project raw sensorimotor spikes to fixed output size.

        Args:
            raw_spikes: Raw spikes from wrapper [wrapper_output_size]

        Returns:
            projected_spikes: Fixed-size spikes [output_size]
        """
        # Ensure correct device
        raw_spikes = raw_spikes.to(self.config.device)

        # Convert to float for linear projection
        raw_float = raw_spikes.float()

        # Project: [wrapper_size] → [output_size]
        projected = self.sensory_projection(raw_float)

        # Apply threshold to maintain sparsity
        # Use sigmoid to get probabilities, then stochastic spiking
        probabilities = torch.sigmoid(projected)
        projected_spikes = torch.rand_like(probabilities) < probabilities

        return projected_spikes

    def get_task(self, task_name: str) -> Dict[str, Any]:
        """Get a sensorimotor task sample.

        Args:
            task_name: One of ['motor_control', 'reaching', 'manipulation', 'prediction']

        Returns:
            Task data dictionary with spikes, reward, and action
        """
        self.task_counts[task_name] += 1
        self.current_episode_step += 1

        # Use current observation (updated by previous step)
        raw_obs_spikes = self.current_obs

        # Project to fixed size
        obs_spikes = self._encode_sensory(raw_obs_spikes)

        # Route to task-specific logic
        if task_name == TaskType.MOTOR_CONTROL.value:
            return self._motor_control_task(obs_spikes)
        elif task_name == TaskType.REACHING.value:
            return self._reaching_task(obs_spikes)
        elif task_name == TaskType.MANIPULATION.value:
            return self._manipulation_task(obs_spikes)
        elif task_name == TaskType.PREDICTION.value:
            return self._prediction_task(obs_spikes)
        else:
            raise ValueError(f"Unknown task type: {task_name}")

    def _motor_control_task(self, obs_spikes: torch.Tensor) -> Dict[str, Any]:
        """Motor control task: Execute basic motor commands.

        Task: Random motor babbling (exploration) or directed movement.
        """
        # Random motor babbling or directed movement
        if np.random.rand() < 0.5:
            # Random exploration
            motor_spikes = create_motor_spikes(self.wrapper.n_motor_neurons, SPIKE_PROBABILITY_LOW, self.config.device)
        else:
            # Directed command (simple policy)
            motor_spikes = torch.zeros(self.wrapper.n_motor_neurons, dtype=torch.bool, device=self.config.device)
            if self.wrapper.n_motor_neurons > 0:
                motor_spikes[0] = True

        # Execute action in environment
        next_obs, reward, terminated, truncated = self.wrapper.step(motor_spikes)

        # Update current observation for next call
        self.current_obs = next_obs

        # Store for next step
        self.last_obs = next_obs
        self.last_action = motor_spikes

        # Reward based on movement execution
        movement_reward = REWARD_SMALL_SUCCESS if reward > REWARD_MOVEMENT_THRESHOLD else 0.0

        return {
            'input': obs_spikes,
            'n_timesteps': 10,
            'reward': movement_reward,
            'target': self._encode_sensory(next_obs),
            'action': motor_spikes,
            'task_type': 'motor_control',
        }

    def _reaching_task(self, obs_spikes: torch.Tensor) -> Dict[str, Any]:
        """Reaching task: Move effector toward visual target."""
        # Simple heuristic policy
        motor_spikes = create_motor_spikes(self.wrapper.n_motor_neurons, SPIKE_PROBABILITY_MEDIUM, self.config.device)

        # Execute action
        next_obs, reward, terminated, truncated = self.wrapper.step(motor_spikes)

        # Update current observation for next call
        self.current_obs = next_obs
 # Store state
        self.last_obs = next_obs
        self.last_action = motor_spikes

        # Reward is based on reaching accuracy
        reaching_reward = max(0.0, reward + REWARD_SMALL_SUCCESS)

        # Track success
        if reaching_reward > REWARD_HIGH_SUCCESS:
            self.task_successes['reaching'] += 1

        return {
            'input': obs_spikes,
            'n_timesteps': 10,
            'reward': reaching_reward,
            'target': self._encode_sensory(next_obs),
            'action': motor_spikes,
            'task_type': 'reaching',
            'success': reaching_reward > REWARD_HIGH_SUCCESS,
        }

    def _manipulation_task(self, obs_spikes: torch.Tensor) -> Dict[str, Any]:
        """Manipulation task: Push/pull objects."""
        # Generate motor command
        motor_spikes = create_motor_spikes(self.wrapper.n_motor_neurons, SPIKE_PROBABILITY_HIGH, self.config.device)

        # Execute action
        next_obs, reward, _, _ = self.wrapper.step(motor_spikes)

        # Update current observation for next call
        self.current_obs = next_obs
 # Store state
        self.last_obs = next_obs
        self.last_action = motor_spikes

        # Reward for any interaction
        manipulation_reward = REWARD_MANIPULATION_BASE if reward > REWARD_REACHING_THRESHOLD else REWARD_SMALL_SUCCESS

        # Track success
        if manipulation_reward > 0.4:
            self.task_successes['manipulation'] += 1

        return {
            'input': obs_spikes,
            'n_timesteps': 10,
            'reward': manipulation_reward,
            'target': self._encode_sensory(next_obs),
            'action': motor_spikes,
            'task_type': 'manipulation',
            'success': manipulation_reward > 0.4,
        }

    def _prediction_task(self, obs_spikes: torch.Tensor) -> Dict[str, Any]:
        """Prediction task: Learn forward/inverse models."""
        # Generate motor command
        motor_spikes = create_motor_spikes(self.wrapper.n_motor_neurons, SPIKE_PROBABILITY_LOW, self.config.device)

        # Execute action
        next_obs, reward, terminated, truncated = self.wrapper.step(motor_spikes)

        # Compute prediction error reward
        if self.last_obs is not None and self.last_action is not None:
            prediction_error = torch.sum(torch.abs(next_obs.float() - self.last_obs.float()))
            prediction_error = prediction_error / next_obs.numel()
            prediction_reward = max(0.0, REWARD_SCALE_PREDICTION - prediction_error.item())
        else:
            prediction_reward = 0.0

        # Store state
        self.last_obs = next_obs
        self.last_action = motor_spikes

        return {
            'input': obs_spikes,
            'n_timesteps': 10,
            'reward': prediction_reward,
            'target': self._encode_sensory(next_obs),
            'action': motor_spikes,
            'task_type': 'prediction',
        }

    def get_task_types(self) -> List[str]:
        """Get available task types."""
        return self.task_types

    def reset(self) -> None:
        """Reset task loader state."""
        self.current_episode_step = 0
        self.last_obs = None
        self.last_action = None

    def get_statistics(self) -> Dict[str, Any]:
        """Get task statistics."""
        success_rates = {}
        for task_type in self.task_types:
            count = self.task_counts[task_type]
            if count > 0:
                success_rates[task_type] = self.task_successes[task_type] / count
            else:
                success_rates[task_type] = 0.0

        return {
            'task_counts': self.task_counts.copy(),
            'success_rates': success_rates,
        }


# ============================================================================
# Stage 0: Phonology Task Loader
# ============================================================================

class PhonologyTaskType(Enum):
    """Task types for phonology stage."""
    MNIST = "mnist"
    TEMPORAL = "temporal"
    PHONOLOGY = "phonology"
    GAZE_FOLLOWING = "gaze_following"


@dataclass
class PhonologyConfig:
    """Configuration for phonology task loader (Stage 0)."""
    # Output size (must match brain.input_size)
    output_size: int = 256

    # Task mixing
    task_probabilities: Dict[str, float] = None

    # Encoding parameters
    n_timesteps: int = 10
    mnist_spike_rate: float = 0.3
    temporal_sequence_length: int = 5

    device: str = 'cpu'

    def __post_init__(self):
        if self.task_probabilities is None:
            # Default task distribution (per curriculum strategy)
            self.task_probabilities = {
                'mnist': DATASET_WEIGHT_MNIST,          # Visual foundation
                'temporal': DATASET_WEIGHT_TEMPORAL,       # Sequence learning
                'phonology': DATASET_WEIGHT_PHONOLOGY,      # Phoneme discrimination
                'gaze_following': DATASET_WEIGHT_GAZE, # Social attention
            }


class PhonologyTaskLoader:
    """Task loader for Stage 0: Phonology and sensory foundations.

    Provides MNIST digit recognition, temporal sequences, phoneme
    discrimination, and gaze following tasks.

    ARCHITECTURE:
    =============
    Uses sensory pathways to encode raw inputs to fixed-size spike patterns:

    - MNIST (784 pixels) → RetinalEncoder → Fixed spikes (256)
    - Temporal sequences (variable) → Projection → Fixed spikes (256)
    - Phonemes (40 features) → Audio encoding → Fixed spikes (256)

    This matches biological sensory encoding: retina/cochlea perform
    dimensionality reduction before cortical processing.

    Tasks:
    ------
    - mnist: Digit recognition (visual cortex training)
    - temporal: A-B-C sequence prediction
    - phonology: Phoneme categorical perception
    - gaze_following: Social attention foundations

    Returns:
    --------
    Task data with:
    - 'input': Encoded spike patterns [output_size]
    - 'label' or 'target': Ground truth
    - 'task_type': Task identifier
    """

    def __init__(
        self,
        config: Optional[PhonologyConfig] = None,
        device: str = 'cpu',
        output_size: Optional[int] = None,
    ):
        """Initialize phonology task loader.

        Args:
            config: Task loader configuration
            device: Device for tensors ('cpu' or 'cuda')
            output_size: Override config output_size (must match brain.input_size)
        """
        self.config = config or PhonologyConfig(device=device)
        self.device = device

        # Override output_size if provided
        if output_size is not None:
            self.config.output_size = output_size

        self.task_types = [t.value for t in PhonologyTaskType]

        # Track statistics
        self.task_counts = {t: 0 for t in self.task_types}
        self.task_accuracies = {t: [] for t in self.task_types}

        # Initialize datasets (lazy loading)
        self._mnist_dataset = None
        self._temporal_dataset = None
        self._phonology_dataset = None

        # Iterator state
        self._mnist_iter = None
        self._temporal_iter = None
        self._phonology_iter = None

        # Sensory encoders (lazy initialization)
        self._visual_encoder = None
        self._temporal_projection = None
        self._phonology_projection = None

    @property
    def visual_encoder(self):
        """Lazy initialize visual encoder (RetinalEncoder)."""
        if self._visual_encoder is None:
            # Create visual config matching output size
            visual_config = VisualConfig(
                output_size=self.config.output_size,
                n_timesteps=self.config.n_timesteps,
                input_height=28,  # MNIST
                input_width=28,
                input_channels=1,
                device=self.device,
            )
            self._visual_encoder = RetinalEncoder(visual_config)
            self._visual_encoder.to(self.device)
            print(f"[OK] Initialized RetinalEncoder: 784 -> {self.config.output_size}")

        return self._visual_encoder

    @property
    def mnist_dataset(self):
        """Lazy load MNIST dataset."""
        if self._mnist_dataset is None:
            try:
                from torchvision import datasets, transforms

                # Load MNIST
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])

                self._mnist_dataset = datasets.MNIST(
                    root='./data',
                    train=True,
                    download=True,
                    transform=transform,
                )

                print(f"[OK] Loaded MNIST: {len(self._mnist_dataset)} samples")
            except Exception as e:
                print(f"[WARN] Failed to load MNIST: {e}")
                self._mnist_dataset = None

        return self._mnist_dataset

    @property
    def temporal_dataset(self):
        """Lazy load temporal sequence dataset."""
        if self._temporal_dataset is None:
            try:
                from thalia.datasets import create_stage0_temporal_dataset

                self._temporal_dataset = create_stage0_temporal_dataset(
                    device=self.device
                )

                print("[OK] Loaded Temporal Sequences dataset")
            except Exception as e:
                print(f"[WARN] Failed to load temporal dataset: {e}")
                self._temporal_dataset = None

        return self._temporal_dataset

    @property
    def phonology_dataset(self):
        """Lazy load phonological dataset."""
        if self._phonology_dataset is None:
            try:
                from thalia.datasets import PhonologicalDataset, PhonologicalConfig, Language

                phon_config = PhonologicalConfig()
                phon_config.device = self.config.device

                self._phonology_dataset = PhonologicalDataset(
                    config=phon_config,
                    language=Language.ENGLISH,
                )

                print("[OK] Loaded Phonological dataset (English)")
            except Exception as e:
                print(f"[WARN] Failed to load phonological dataset: {e}")
                self._phonology_dataset = None

        return self._phonology_dataset

    @property
    def temporal_projection(self):
        """Lazy initialize temporal projection layer."""
        if self._temporal_projection is None:
            # Get actual n_symbols from dataset if available
            if self.temporal_dataset is not None:
                input_size = self.temporal_dataset.config.n_symbols
            else:
                input_size = 20  # Fallback

            self._temporal_projection = nn.Linear(input_size, self.config.output_size, bias=False)
            nn.init.sparse_(self._temporal_projection.weight, sparsity=0.8)
            self._temporal_projection.to(self.device)
            print(f"[OK] Initialized temporal projection: {input_size} -> {self.config.output_size}")
        return self._temporal_projection

    @property
    def phonology_projection(self):
        """Lazy initialize phonology projection layer."""
        if self._phonology_projection is None:
            # Get actual spectrogram dimensions from dataset if available
            if self.phonology_dataset is not None:
                # Spectrogram is (n_freq_channels × n_time_steps)
                input_size = (self.phonology_dataset.config.n_freq_channels *
                             self.phonology_dataset.config.n_time_steps)
            else:
                input_size = 40  # Fallback for feature-based encoding

            self._phonology_projection = nn.Linear(input_size, self.config.output_size, bias=False)
            nn.init.sparse_(self._phonology_projection.weight, sparsity=0.7)
            self._phonology_projection.to(self.device)
            print(f"[OK] Initialized phonology projection: {input_size} -> {self.config.output_size}")
        return self._phonology_projection

    def get_task(self, task_name: str) -> Dict[str, Any]:
        """Get a phonology task sample.

        Args:
            task_name: One of ['mnist', 'temporal', 'phonology', 'gaze_following']

        Returns:
            Task data dictionary with spikes and labels
        """
        self.task_counts[task_name] += 1

        if task_name == PhonologyTaskType.MNIST.value:
            return self._mnist_task()
        elif task_name == PhonologyTaskType.TEMPORAL.value:
            return self._temporal_task()
        elif task_name == PhonologyTaskType.PHONOLOGY.value:
            return self._phonology_task()
        elif task_name == PhonologyTaskType.GAZE_FOLLOWING.value:
            return self._gaze_following_task()
        else:
            raise ValueError(f"Unknown task type: {task_name}")

    def _mnist_task(self) -> Dict[str, Any]:
        """MNIST digit recognition task.

        Returns digit image encoded through RetinalEncoder.
        Biologically: 784 pixels → Retinal processing → 256 ganglion cells
        """
        if self.mnist_dataset is None:
            # Fallback: Random retinal input
            random_image = torch.rand(28, 28, device=self.device)
            spikes, _ = self.visual_encoder(random_image)
            # Take first timestep for single-timestep input
            spikes = spikes[0, :]  # [n_timesteps, output_size] → [output_size]
            label = torch.randint(0, 10, (1,), device=self.device).item()
        else:
            # Get next sample from dataset
            if self._mnist_iter is None:
                self._mnist_iter = iter(torch.utils.data.DataLoader(
                    self.mnist_dataset,
                    batch_size=1,
                    shuffle=True,
                ))

            try:
                image, label = next(self._mnist_iter)
            except StopIteration:
                # Reset iterator
                self._mnist_iter = iter(torch.utils.data.DataLoader(
                    self.mnist_dataset,
                    batch_size=1,
                    shuffle=True,
                ))
                image, label = next(self._mnist_iter)

            # Encode through RetinalEncoder
            # image shape: [1, 1, 28, 28] → squeeze to [28, 28]
            image = image.squeeze().to(self.device)  # [28, 28]

            # RetinalEncoder expects [C, H, W] or [H, W]
            spikes, metadata = self.visual_encoder(image)  # Returns [n_timesteps, output_size]

            # Take first timestep (or could use all timesteps for temporal coding)
            spikes = spikes[0, :]  # [output_size]

            label = label.item()

        return {
            'input': spikes,
            'n_timesteps': self.config.n_timesteps,
            'label': label,
            'task_type': 'mnist',
        }

    def _temporal_task(self) -> Dict[str, Any]:
        """Temporal sequence prediction task.

        Returns A-B-C pattern for next-item prediction.
        """
        if self.temporal_dataset is None:
            # Fallback: Simple pattern
            sequence_length = self.config.temporal_sequence_length
            pattern = torch.arange(sequence_length, device=self.device)
            spikes_raw = torch.nn.functional.one_hot(pattern, num_classes=20).float()
            target = pattern[1].item()  # Predict next item
        else:
            # Generate sequence from dataset
            sequence, targets, pattern_type = self.temporal_dataset.generate_sequence()

            # sequence shape: (sequence_length, n_symbols)
            spikes_raw = sequence[0].to(self.device)  # Take first item [n_symbols]

            # Convert first target from one-hot to index
            target = torch.argmax(targets[0]).item()

        # Project to output size
        spikes = self.temporal_projection(spikes_raw)
        spikes = torch.sigmoid(spikes) > 0.5  # Threshold to binary spikes

        return {
            'input': spikes,
            'n_timesteps': self.config.n_timesteps,
            'target': target,
            'task_type': 'temporal',
        }

    def _phonology_task(self) -> Dict[str, Any]:
        """Phonological discrimination task.

        Returns phoneme pair for categorical perception.
        """
        if self.phonology_dataset is None:
            # Fallback: Random phoneme features
            n_features = 40  # Typical phoneme feature dimension
            spikes_raw = torch.rand(n_features, device=self.device)
            is_same = torch.randint(0, 2, (1,), device=self.device).item()
        else:
            # Generate sample from dataset
            from thalia.datasets import PhonemeCategory

            # Use simple voiced/voiceless contrast
            contrast = (PhonemeCategory.P, PhonemeCategory.B)
            same = torch.randint(0, 2, (1,), device=self.device).item() == 1
            phoneme1, phoneme2, is_same = self.phonology_dataset.generate_discrimination_pair(contrast, same)

            # Use first phoneme features (flatten spectrogram)
            spikes_raw = phoneme1.flatten()

        # Project to output size
        spikes = self.phonology_projection(spikes_raw)
        spikes = torch.sigmoid(spikes) > 0.5  # Threshold to binary spikes

        return {
            'input': spikes,
            'n_timesteps': self.config.n_timesteps,
            'target': is_same,
            'task_type': 'phonology',
        }

    def _gaze_following_task(self) -> Dict[str, Any]:
        """Gaze following task (social attention).

        Returns: Visual scene with gaze direction cue.
        """
        # Simple implementation: Random visual pattern with attention cue
        # In full implementation, would use actual gaze following dataset

        # Generate random "visual scene"
        scene = torch.rand(28, 28, device=self.device)

        # Encode through visual pathway
        spikes, _ = self.visual_encoder(scene)
        spikes = spikes[0, :]  # Take first timestep

        # Target is gaze-attended region (simplified: just a location)
        target_x = torch.randint(0, 28, (1,), device=self.device).item()
        target_y = torch.randint(0, 28, (1,), device=self.device).item()

        return {
            'input': spikes,
            'n_timesteps': self.config.n_timesteps,
            'target': (target_x, target_y),
            'task_type': 'gaze_following',
        }

    def get_task_types(self) -> List[str]:
        """Get available task types."""
        return self.task_types

    def reset(self) -> None:
        """Reset task loader state."""
        self._mnist_iter = None
        self._temporal_iter = None
        self._phonology_iter = None

    def get_statistics(self) -> Dict[str, Any]:
        """Get task statistics."""
        avg_accuracies = {}
        for task_type in self.task_types:
            if len(self.task_accuracies[task_type]) > 0:
                avg_accuracies[task_type] = np.mean(self.task_accuracies[task_type])
            else:
                avg_accuracies[task_type] = 0.0

        return {
            'task_counts': self.task_counts.copy(),
            'avg_accuracies': avg_accuracies,
        }


# ============================================================================
# Task Loader Registry
# ============================================================================

class TaskLoaderRegistry:
    """Registry for task loaders by curriculum stage."""

    _loaders = {
        'sensorimotor': SensorimotorTaskLoader,
        'phonology': PhonologyTaskLoader,
    }

    @classmethod
    def get_loader_class(cls, stage_name: str):
        """Get task loader class for stage."""
        stage_key = stage_name.lower().replace('-', '').replace('_', '')

        # Map stage names to loader keys
        mapping = {
            'stage05': 'sensorimotor',
            'sensorimotor': 'sensorimotor',
            'stage0': 'phonology',
            'phonology': 'phonology',
        }

        loader_key = mapping.get(stage_key)
        if loader_key is None:
            raise ValueError(f"No task loader registered for stage: {stage_name}")

        return cls._loaders[loader_key]

    @classmethod
    def create_loader(cls, stage_name: str, **kwargs):
        """Create task loader instance for stage."""
        loader_class = cls.get_loader_class(stage_name)
        return loader_class(**kwargs)

    @classmethod
    def register_loader(cls, stage_name: str, loader_class):
        """Register a new task loader."""
        cls._loaders[stage_name.lower()] = loader_class


# ============================================================================
# Convenience Functions
# ============================================================================

def create_sensorimotor_loader(
    wrapper: Any,
    config: Optional[SensorimotorConfig] = None,
    output_size: Optional[int] = None,
) -> SensorimotorTaskLoader:
    """Create sensorimotor task loader for Stage -0.5.

    Args:
        wrapper: SensorimotorWrapper instance
        config: Optional configuration
        output_size: Output size to match brain.input_size

    Returns:
        SensorimotorTaskLoader instance
    """
    return SensorimotorTaskLoader(wrapper, config, output_size)


def create_phonology_loader(
    config: Optional[PhonologyConfig] = None,
    device: str = 'cpu',
    output_size: Optional[int] = None,
) -> PhonologyTaskLoader:
    """Create phonology task loader for Stage 0.

    Args:
        config: Optional configuration
        device: Device for tensors
        output_size: Output size to match brain.input_size

    Returns:
        PhonologyTaskLoader instance
    """
    return PhonologyTaskLoader(config, device, output_size)
