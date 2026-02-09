"""
Bootstrap Tasks for Stage 0 (Developmental Initialization).

Implements the three phases of bootstrap training:
- Phase 0A: Spontaneous activity (noise-driven bursts)
- Phase 0B: Simple pattern exposure (single pixels, pure tones)
- Phase 0C: Parameter transition (gradually more complex)

These tasks establish functional connectivity BEFORE curriculum learning begins,
solving the bootstrap problem where weak random weights prevent neurons from firing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import numpy as np


# ============================================================================
# Task Configurations
# ============================================================================


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap tasks."""

    # Output size (must match brain input)
    output_size: int = 256

    # Task timing
    n_timesteps: int = 10  # Timesteps per trial

    # Phase 0A: Spontaneous Activity
    noise_sigma: float = 0.05  # OU noise strength
    noise_tau_ms: float = 20.0  # OU noise timescale
    burst_probability: float = 0.1  # Chance of population burst

    # Phase 0B: Simple Patterns
    pattern_spike_rate: float = 0.3  # Firing rate during pattern
    pattern_repetitions: int = 100  # Reps per pattern before switching

    # Phase 0C: Transition
    transition_complexity: int = 4  # Number of patterns (2 → 4)

    device: str = "cpu"


# ============================================================================
# Phase 0A: Spontaneous Activity
# ============================================================================


class SpontaneousActivityTask:
    """Generate spontaneous activity patterns (no external input).

    Simulates prenatal/eyes-closed period where the brain generates its own
    activity through noise and intrinsic dynamics. This refines genetically-
    specified connections through Hebbian learning before sensory input arrives.

    **Key Features**:
    - Ornstein-Uhlenbeck noise (temporally correlated, biological)
    - Occasional population bursts (wave-like activity)
    - No reward signal (pure Hebbian/STDP refinement)
    - Goal: Establish stable spontaneous firing rates (0.05-0.15)
    """

    def __init__(self, config: BootstrapConfig):
        """Initialize spontaneous activity generator.

        Args:
            config: Bootstrap configuration
        """
        self.config = config
        self.device = config.device

        # OU noise state
        self.noise_state = torch.zeros(config.output_size, device=self.device)

        # Burst generator
        self.burst_active = False
        self.burst_duration = 0

    def get_task(self) -> Dict[str, Any]:
        """Generate one spontaneous activity trial.

        Returns:
            Dictionary with:
                - input: OU noise pattern [output_size]
                - n_timesteps: Trial duration
                - task_type: 'spontaneous'
                - reward: None (no reward signal)
                - metadata: Trial information
        """
        # Generate OU noise
        dt = 1.0  # Assuming 1ms timesteps
        tau = self.config.noise_tau_ms
        sigma = self.config.noise_sigma

        # Ornstein-Uhlenbeck process:
        # dx = -x/tau * dt + sigma * sqrt(2*dt/tau) * dW
        noise_decay = np.exp(-dt / tau)
        noise_diffusion = sigma * np.sqrt(1 - noise_decay**2)

        # Update noise state
        self.noise_state = (
            noise_decay * self.noise_state +
            noise_diffusion * torch.randn(self.config.output_size, device=self.device)
        )

        # Occasionally trigger population burst
        if not self.burst_active and torch.rand(1).item() < self.config.burst_probability:
            self.burst_active = True
            self.burst_duration = torch.randint(3, 8, (1,)).item()  # 3-7 timesteps

        # Add burst component
        input_tensor = self.noise_state.clone()
        if self.burst_active:
            # Strong correlated input to simulate wave
            burst_strength = 0.3 * torch.ones_like(input_tensor)
            # Spatial gradient (wave-like)
            spatial_gradient = torch.linspace(0, 1, len(input_tensor), device=self.device)
            burst_pattern = burst_strength * (1 + 0.5 * torch.sin(spatial_gradient * 2 * np.pi))
            input_tensor = input_tensor + burst_pattern

            self.burst_duration -= 1
            if self.burst_duration <= 0:
                self.burst_active = False

        # Convert continuous values to binary spikes using Poisson spiking
        # Rates are clamped to [0, 1] and used as spike probabilities
        spike_probs = torch.clamp(input_tensor, 0, 1)
        spikes = (torch.rand_like(spike_probs) < spike_probs).float()

        return {
            "input": {"thalamus": spikes},  # Route to thalamus (sensory input)
            "n_timesteps": self.config.n_timesteps,
            "task_type": "spontaneous",
            "reward": None,  # No reward signal
            "metadata": {
                "phase": "0A",
                "burst_active": self.burst_active,
                "noise_mean": float(self.noise_state.mean()),
                "noise_std": float(self.noise_state.std()),
            },
        }


# ============================================================================
# Phase 0B: Simple Pattern Exposure
# ============================================================================


class SimplePatternTask:
    """Ultra-simple sensory patterns (single pixels, pure tones).

    Verifies that learning can strengthen connections with minimal input.
    Uses only 2 patterns initially, repeated 100x each to drive Hebbian
    strengthening of responding neurons.

    **Patterns**:
    - Single pixel ON (location A)
    - Single pixel ON (location B)
    - Pure tone (frequency 1)
    - Pure tone (frequency 2)

    **Goal**: Cortex fires reliably (>90% of trials) and discriminates patterns.
    """

    def __init__(self, config: BootstrapConfig):
        """Initialize simple pattern generator.

        Args:
            config: Bootstrap configuration
        """
        self.config = config
        self.device = config.device

        # Pattern library
        self.patterns = self._create_patterns()
        self.pattern_names = list(self.patterns.keys())

        # Repetition tracking
        self.current_pattern = 0
        self.repetition_count = 0

    def _create_patterns(self) -> Dict[str, torch.Tensor]:
        """Create simple pattern library.

        Returns:
            Dictionary of pattern name -> pattern tensor
        """
        patterns = {}
        size = self.config.output_size
        rate = self.config.pattern_spike_rate

        # Visual patterns (first half of input space)
        visual_size = size // 2

        # Single pixel at location 1/4
        pixel_a = torch.zeros(size, device=self.device)
        pixel_a[visual_size // 4] = rate

        # Single pixel at location 3/4
        pixel_b = torch.zeros(size, device=self.device)
        pixel_b[3 * visual_size // 4] = rate

        patterns["pixel_a"] = pixel_a
        patterns["pixel_b"] = pixel_b

        # Auditory patterns (second half of input space)
        # Pure tone 1 (low frequency representation)
        tone_1 = torch.zeros(size, device=self.device)
        tone_1[visual_size : visual_size + size // 8] = rate

        # Pure tone 2 (high frequency representation)
        tone_2 = torch.zeros(size, device=self.device)
        tone_2[size - size // 8 :] = rate

        patterns["tone_1"] = tone_1
        patterns["tone_2"] = tone_2

        return patterns

    def get_task(self) -> Dict[str, Any]:
        """Generate one simple pattern trial.

        Returns:
            Dictionary with task data
        """
        # Select current pattern (cycle through with repetitions)
        pattern_name = self.pattern_names[self.current_pattern]
        pattern = self.patterns[pattern_name]

        # Add small amount of noise
        noisy_pattern = pattern + 0.05 * torch.randn_like(pattern)
        noisy_pattern = torch.clamp(noisy_pattern, 0, 1)

        # Convert to binary spikes using Poisson spiking
        spikes = (torch.rand_like(noisy_pattern) < noisy_pattern).float()

        # Update repetition counter
        self.repetition_count += 1
        if self.repetition_count >= self.config.pattern_repetitions:
            self.repetition_count = 0
            self.current_pattern = (self.current_pattern + 1) % len(self.pattern_names)

        return {
            "input": {"thalamus": spikes},  # Route to thalamus (sensory input)
            "n_timesteps": self.config.n_timesteps,
            "task_type": "simple_pattern",
            "reward": None,  # Pure Hebbian, no reward
            "metadata": {
                "phase": "0B",
                "pattern_name": pattern_name,
                "pattern_id": self.current_pattern,
                "repetition": self.repetition_count,
            },
        }


# ============================================================================
# Phase 0C: Slightly More Complex Patterns
# ============================================================================


class TransitionPatternTask:
    """Slightly more complex patterns during parameter transition.

    Increases to 4 patterns (2x complexity) while plasticity parameters
    gradually transition from developmental to adult levels. Verifies that
    learning still works as critical period closes.

    **Patterns**:
    - Two pixels ON (locations A+B)
    - Two pixels ON (locations C+D)
    - Tone pair (frequencies 1+2)
    - Tone pair (frequencies 3+4)
    """

    def __init__(self, config: BootstrapConfig):
        """Initialize transition pattern generator.

        Args:
            config: Bootstrap configuration
        """
        self.config = config
        self.device = config.device

        # Pattern library (more complex)
        self.patterns = self._create_complex_patterns()
        self.pattern_names = list(self.patterns.keys())
        self.current_pattern = 0

    def _create_complex_patterns(self) -> Dict[str, torch.Tensor]:
        """Create slightly more complex patterns.

        Returns:
            Dictionary of pattern name -> pattern tensor
        """
        patterns = {}
        size = self.config.output_size
        rate = self.config.pattern_spike_rate
        visual_size = size // 2

        # Two-pixel patterns
        two_pixels_ab = torch.zeros(size, device=self.device)
        two_pixels_ab[visual_size // 4] = rate
        two_pixels_ab[visual_size // 2] = rate

        two_pixels_cd = torch.zeros(size, device=self.device)
        two_pixels_cd[3 * visual_size // 4] = rate
        two_pixels_cd[visual_size - visual_size // 8] = rate

        patterns["two_pixels_ab"] = two_pixels_ab
        patterns["two_pixels_cd"] = two_pixels_cd

        # Tone pair patterns
        tone_pair_low = torch.zeros(size, device=self.device)
        tone_pair_low[visual_size : visual_size + size // 8] = rate
        tone_pair_low[visual_size + size // 8 : visual_size + size // 4] = rate

        tone_pair_high = torch.zeros(size, device=self.device)
        tone_pair_high[size - size // 4 : size - size // 8] = rate
        tone_pair_high[size - size // 8 :] = rate

        patterns["tone_pair_low"] = tone_pair_low
        patterns["tone_pair_high"] = tone_pair_high

        return patterns

    def get_task(self) -> Dict[str, Any]:
        """Generate one transition pattern trial.

        Returns:
            Dictionary with task data
        """
        # Cycle through patterns
        pattern_name = self.pattern_names[self.current_pattern]
        pattern = self.patterns[pattern_name]

        # Slightly more noise (increasing difficulty)
        noisy_pattern = pattern + 0.08 * torch.randn_like(pattern)
        noisy_pattern = torch.clamp(noisy_pattern, 0, 1)

        # Convert to binary spikes using Poisson spiking
        spikes = (torch.rand_like(noisy_pattern) < noisy_pattern).float()

        self.current_pattern = (self.current_pattern + 1) % len(self.pattern_names)

        return {
            "input": {"thalamus": spikes},  # Route to thalamus (sensory input)
            "n_timesteps": self.config.n_timesteps,
            "task_type": "transition_pattern",
            "reward": None,
            "metadata": {
                "phase": "0C",
                "pattern_name": pattern_name,
                "pattern_id": self.current_pattern,
            },
        }


# ============================================================================
# Unified Bootstrap Task Loader
# ============================================================================


class BootstrapTaskLoader:
    """Unified task loader for Stage 0 bootstrap training.

    Provides tasks for all three bootstrap phases:
    - Phase 0A: Spontaneous activity (40k steps)
    - Phase 0B: Simple patterns (40k steps)
    - Phase 0C: Transition patterns (20k steps)

    Usage:
        loader = BootstrapTaskLoader(config, phase="0A")
        task_data = loader.get_task("spontaneous_activity")
    """

    def __init__(
        self,
        config: BootstrapConfig,
        phase: str = "0A",
        device: str = "cpu",
    ):
        """Initialize bootstrap task loader.

        Args:
            config: Bootstrap configuration
            phase: Which phase ("0A", "0B", "0C")
            device: PyTorch device
        """
        self.config = config
        self.phase = phase
        self.device = device

        # Initialize phase-specific task generators
        self.spontaneous = SpontaneousActivityTask(config)
        self.simple_patterns = SimplePatternTask(config)
        self.transition_patterns = TransitionPatternTask(config)

        # Map task names to generators
        self.task_map = {
            "spontaneous_activity": self.spontaneous,
            "single_pixel": self.simple_patterns,
            "pure_tone": self.simple_patterns,
            "two_pixels": self.transition_patterns,
            "tone_pair": self.transition_patterns,
        }

    def get_task(self, task_name: str) -> Dict[str, Any]:
        """Get task data for specified task.

        Args:
            task_name: Name of task to generate

        Returns:
            Task data dictionary with input, metadata, etc.
        """
        if task_name not in self.task_map:
            raise ValueError(
                f"Unknown bootstrap task: {task_name}. "
                f"Available: {list(self.task_map.keys())}"
            )

        return self.task_map[task_name].get_task()

    def set_phase(self, phase: str) -> None:
        """Switch to different bootstrap phase.

        Args:
            phase: Phase to switch to ("0A", "0B", "0C")
        """
        if phase not in ("0A", "0B", "0C"):
            raise ValueError(f"Invalid phase: {phase}. Use '0A', '0B', or '0C'")
        self.phase = phase
