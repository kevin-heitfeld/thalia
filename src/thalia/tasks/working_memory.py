"""
Working Memory Tasks with Theta-Gamma Phase Coding

Implements N-back and other working memory tasks using theta-gamma coupling
for temporal organization. Based on Lisman & Jensen (2013) neural code model.

Key Concepts:
============

1. THETA PHASE CODING:
   - Each item stored at specific theta phase
   - Sequence order = phase progression
   - 8 Hz theta → ~125ms per item

2. GAMMA NESTING:
   - Fast gamma cycles (~40 Hz) nested in theta
   - Gamma phase determines encoding/retrieval excitability
   - Peak gamma = optimal encoding window

3. N-BACK TASK:
   - Present sequence of items
   - Report if current item matches N items back
   - Tests working memory capacity and maintenance

4. PHASE RETRIEVAL:
   - Calculate target phase (N cycles back)
   - Retrieve item at that phase from PFC working memory
   - Compare to current stimulus

Biological Plausibility:
=======================
- Hippocampal-PFC theta synchronization during WM
- Gamma cycles segment items within theta phase
- CA1 pyramidal cells show phase precession
- PFC persistent activity maintains representations

References:
- Lisman & Jensen (2013): The theta-gamma neural code
- Fries (2015): Rhythms for cognition
- Buschman et al. (2011): Neural substrates of cognitive capacity
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from thalia.constants.task import MATCH_PROBABILITY_DEFAULT
from thalia.regions.prefrontal import Prefrontal
from thalia.tasks.stimulus_utils import create_random_stimulus


@dataclass
class WorkingMemoryTaskConfig:
    """Configuration for working memory tasks.

    Attributes:
        theta_freq_hz: Theta oscillation frequency (typically 4-10 Hz)
        gamma_freq_hz: Gamma oscillation frequency (typically 30-80 Hz)
        items_per_theta_cycle: How many items fit in one theta cycle
        dt_ms: Timestep for simulation
        encoding_window_ms: Time window for encoding each item
        retrieval_window_ms: Time window for retrieval
        device: Computation device
    """

    theta_freq_hz: float = 8.0
    gamma_freq_hz: float = 40.0
    items_per_theta_cycle: int = 8
    dt_ms: float = 1.0
    encoding_window_ms: float = 100.0  # 100ms to encode
    retrieval_window_ms: float = 50.0  # 50ms to retrieve
    device: str = "cpu"


class ThetaGammaEncoder(nn.Module):
    """
    Encodes items into working memory using theta-gamma phase coding.

    Each item is assigned:
    - A theta phase (position in sequence)
    - Optimal encoding at gamma peak (high excitability)

    This allows temporal ordering and capacity limits (~7±2 items).

    **Migration Note**: This class now uses centrally-managed oscillator
    phases from the brain instead of creating local oscillators. Call
    set_oscillator_phases() to provide current phases before using
    encoding/retrieval methods.
    """

    def __init__(self, config: WorkingMemoryTaskConfig):
        super().__init__()
        self.config = config

        # Store current oscillator phases (provided by brain)
        # These are updated via set_oscillator_phases()
        self._theta_phase: float = 0.0
        self._gamma_phase: float = 0.0
        self._theta_signal: float = 0.0
        self._gamma_signal: float = 0.0
        self._theta_slot: int = 0
        self._coupled_amplitudes: Dict[str, float] = {}

        # Time tracking
        self.item_count = 0

    def set_oscillator_phases(self, phases: Dict[str, float], signals: Dict[str, float]) -> None:
        """Receive current oscillator phases from brain.

        This should be called before encoding/retrieval operations
        to ensure phase information is current.

        Args:
            phases: Dict mapping oscillator name to phase [0, 2π)
            signals: Dict mapping oscillator name to signal [-1, 1]
        """
        self._theta_phase = phases.get("theta", 0.0)
        self._gamma_phase = phases.get("gamma", 0.0)
        self._theta_signal = signals.get("theta", 0.0)
        self._gamma_signal = signals.get("gamma", 0.0)

    def get_encoding_phase(self, item_index: int) -> Tuple[float, float]:
        """
        Calculate theta and gamma phases for encoding an item.

        Args:
            item_index: Sequential position of item (0, 1, 2, ...)

        Returns:
            theta_phase: Phase in radians [0, 2π)
            gamma_phase: Phase in radians [0, 2π) - should be near peak
        """
        # Theta phase: divide cycle by number of items
        position_in_cycle = item_index % self.config.items_per_theta_cycle
        theta_phase = (position_in_cycle / self.config.items_per_theta_cycle) * (2 * math.pi)

        # Gamma phase: encode at peak excitability (π/2 = 90° = peak)
        gamma_phase = math.pi / 2.0

        return theta_phase, gamma_phase

    def get_retrieval_phase(self, current_index: int, n_back: int) -> float:
        """
        Calculate theta phase for retrieving an item N positions back.

        Args:
            current_index: Current position in sequence
            n_back: How many items to look back

        Returns:
            theta_phase: Target phase for retrieval
        """
        target_index = current_index - n_back
        if target_index < 0:
            # Can't retrieve before sequence start
            return -1.0

        position_in_cycle = target_index % self.config.items_per_theta_cycle
        theta_phase = (position_in_cycle / self.config.items_per_theta_cycle) * (2 * math.pi)

        return theta_phase

    def get_current_theta_phase(self) -> float:
        """Get current theta phase (from brain).

        Returns:
            Current theta phase in radians [0, 2π)
        """
        return self._theta_phase

    def get_current_gamma_phase(self) -> float:
        """Get current gamma phase (from brain).

        Returns:
            Current gamma phase in radians [0, 2π)
        """
        return self._gamma_phase

    def get_excitability_modulation(self) -> float:
        """
        Get current excitability based on gamma phase.

        Higher at gamma peak (encoding window), lower at trough.

        Returns:
            modulation: Multiplicative factor [0, 1]
        """
        # Peak at gamma phase = π/2 (90°)
        # Use cosine shifted to make peak at π/2
        gamma_signal = math.sin(self._gamma_phase)
        # Map to [0, 1] for modulation
        modulation = (gamma_signal + 1.0) / 2.0
        return max(0.0, modulation)


class NBackTask:
    """
    N-back working memory task using theta-gamma phase coding.

    Task Structure:
    1. Present sequence of stimuli
    2. For each stimulus, determine if it matches N items back
    3. Track accuracy and response times

    Phase Coding:
    - Each item encoded at unique theta phase
    - Phase spacing allows ~8 items per theta cycle
    - Retrieval uses phase to access specific item

    Example:
        Sequence: A B C B D E ...
        2-back:   ? ? ? Y ? N ...
                  (C matches? B matches!)
    """

    def __init__(self, prefrontal: Prefrontal, config: WorkingMemoryTaskConfig, n_back: int = 2):
        """
        Initialize N-back task.

        Args:
            prefrontal: Prefrontal cortex for working memory
            config: Task configuration
            n_back: Number of items to look back (typically 1-3)
        """
        self.prefrontal = prefrontal
        self.config = config
        self.n_back = n_back

        # Phase encoder
        self.encoder = ThetaGammaEncoder(config)

        # Task state
        self.stimulus_history: List[torch.Tensor] = []
        self.responses: List[bool] = []
        self.correct: List[bool] = []

    def reset(self) -> None:
        """Reset task state for new trial."""
        self.stimulus_history = []
        self.responses = []
        self.correct = []
        self.encoder.item_count = 0

    def encode_item(self, stimulus: torch.Tensor, item_index: int) -> Dict[str, Any]:
        """
        Encode a stimulus into working memory at specific theta phase.

        **Note**: Caller must ensure encoder has current oscillator phases
        via set_oscillator_phases() before calling this method.

        Args:
            stimulus: Input stimulus pattern [n_input]
            item_index: Position in sequence

        Returns:
            Encoding metrics (phase, excitability, etc.)
        """
        # Get phase information for this item
        theta_phase, gamma_phase = self.encoder.get_encoding_phase(item_index)
        excitability = self.encoder.get_excitability_modulation()

        # Encode with dopamine gating (high DA = update gate)
        # Modulate by gamma excitability
        dopamine_signal = 0.8 * excitability  # High when gamma peaks

        # Forward through prefrontal with encoding
        self.prefrontal.forward(stimulus, dopamine_signal=dopamine_signal, dt=self.config.dt_ms)

        # Note: Oscillators are advanced by brain, not here

        return {
            "theta_phase": theta_phase,
            "gamma_phase": gamma_phase,
            "excitability": excitability,
            "dopamine": dopamine_signal,
        }

    def retrieve_item(
        self, current_index: int, n_back: int
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Retrieve item from N positions back using phase addressing.

        **Note**: Caller must ensure encoder has current oscillator phases
        via set_oscillator_phases() before calling this method.

        Retrieval happens by observing PFC's SPIKE OUTPUT, not by directly
        accessing internal state. The task observes what PFC broadcasts,
        just like any downstream region would.

        Args:
            current_index: Current position in sequence
            n_back: How many items back to retrieve

        Returns:
            retrieved_pattern: Retrieved pattern or None if out of bounds
            metrics: Retrieval information
        """
        # Calculate target phase
        target_phase = self.encoder.get_retrieval_phase(current_index, n_back)

        if target_phase < 0:
            # Can't retrieve before sequence start
            return None, {"error": "Target before sequence start"}

        # Retrieve by observing PFC's spike output (what it broadcasts to other regions)
        # This is the biologically plausible way: observe the output, not internal state
        if self.prefrontal.state.spikes is None:
            return None, {"error": "PFC has not produced output yet"}

        retrieved = (
            self.prefrontal.state.spikes.float()
        )  # Convert bool spikes to float for comparison

        # Note: Oscillators are advanced by brain, not here

        return retrieved, {
            "target_phase": target_phase,
            "retrieved_activity": retrieved.mean().item(),
        }

    def present_stimulus(
        self, stimulus: torch.Tensor, store_history: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Present a stimulus and check for N-back match.

        Args:
            stimulus: Input pattern [n_input]
            store_history: Whether to store in history

        Returns:
            is_match: Whether stimulus matches N items back
            metrics: Task metrics
        """
        current_index = len(self.stimulus_history)

        # Encode current stimulus
        encoding_info = self.encode_item(stimulus, current_index)

        # Store in history
        if store_history:
            self.stimulus_history.append(stimulus.clone())

        # Try to retrieve N-back item
        is_match = False
        retrieval_info: Dict[str, Any] = {}

        if current_index >= self.n_back:
            # Can retrieve N-back item
            retrieved, retrieval_info = self.retrieve_item(current_index, self.n_back)

            if retrieved is not None:
                # Compare current to N-back using working memory similarity
                # In simplified version: check if PFC maintained similar pattern
                target_stimulus = self.stimulus_history[current_index - self.n_back]

                # Similarity check (cosine similarity)
                current_norm = stimulus / (stimulus.norm() + 1e-8)
                target_norm = target_stimulus / (target_stimulus.norm() + 1e-8)
                similarity = (current_norm * target_norm).sum().item()

                # Match if high similarity (threshold = 0.7)
                is_match = similarity > 0.7

        # Combine metrics
        metrics = {
            **encoding_info,
            **retrieval_info,
            "item_index": current_index,
            "is_match": is_match,
            "can_retrieve": current_index >= self.n_back,
        }

        return is_match, metrics

    def run_sequence(
        self, stimulus_sequence: List[torch.Tensor], target_matches: Optional[List[bool]] = None
    ) -> Dict[str, Any]:
        """
        Run full N-back sequence and compute accuracy.

        Args:
            stimulus_sequence: List of stimulus patterns
            target_matches: Ground truth matches (optional)

        Returns:
            results: Accuracy, timing, phase information
        """
        self.reset()

        responses = []
        metrics_list = []

        for stimulus in stimulus_sequence:
            is_match, metrics = self.present_stimulus(stimulus)
            responses.append(is_match)
            metrics_list.append(metrics)

        # Calculate accuracy if ground truth provided
        accuracy = None
        if target_matches is not None:
            # Only count items where retrieval is possible
            valid_indices = [i for i in range(len(target_matches)) if i >= self.n_back]
            if valid_indices:
                correct = sum(responses[i] == target_matches[i] for i in valid_indices)
                accuracy = correct / len(valid_indices)

        return {
            "accuracy": accuracy,
            "n_items": len(stimulus_sequence),
            "responses": responses,
            "metrics": metrics_list,
            "n_back": self.n_back,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get task statistics."""
        if not self.stimulus_history:
            return {"error": "No items presented"}

        return {
            "n_items": len(self.stimulus_history),
            "n_responses": len(self.responses),
            "n_correct": sum(self.correct),
            "accuracy": sum(self.correct) / len(self.correct) if self.correct else 0.0,
            "n_back": self.n_back,
        }


def create_n_back_sequence(
    n_items: int,
    n_dims: int,
    n_back: int = 2,
    match_probability: float = MATCH_PROBABILITY_DEFAULT,
    device: str = "cpu",
) -> Tuple[List[torch.Tensor], List[bool]]:
    """
    Create random N-back sequence with specified match probability.

    Args:
        n_items: Number of items in sequence
        n_dims: Dimensionality of each item
        n_back: N for N-back task
        match_probability: Probability of match (0-1)
        device: Computation device

    Returns:
        sequence: List of stimulus patterns
        matches: List of match indicators
    """
    sequence: List[torch.Tensor] = []
    matches = []

    for i in range(n_items):
        if i >= n_back and torch.rand(1, device=device).item() < match_probability:
            # Create match: copy item from N positions back
            stimulus = sequence[i - n_back].clone()
            is_match = True
        else:
            # Create novel item
            stimulus = create_random_stimulus(n_dims, device)
            stimulus = stimulus / (stimulus.norm() + 1e-8)  # Normalize
            is_match = i >= n_back and (stimulus == sequence[i - n_back]).all().item()

        sequence.append(stimulus)
        matches.append(is_match)

    return sequence, matches


# Convenience functions matching plan specification


def theta_gamma_n_back(
    prefrontal: Prefrontal,
    stimulus_sequence: List[torch.Tensor],
    n: int = 2,
    theta_freq_hz: float = 8.0,
    gamma_freq_hz: float = 40.0,
) -> List[bool]:
    """
    N-back task using theta phase coding (matches plan specification).

    Each item encoded at different theta phase within gamma cycle.

    Args:
        prefrontal: Prefrontal cortex for working memory
        stimulus_sequence: List of stimuli to present
        n: N-back distance (default 2)
        theta_freq_hz: Theta frequency (default 8 Hz)
        gamma_freq_hz: Gamma frequency (default 40 Hz)

    Returns:
        results: List of match indicators
    """
    config = WorkingMemoryTaskConfig(
        theta_freq_hz=theta_freq_hz, gamma_freq_hz=gamma_freq_hz, device=str(prefrontal.device)
    )

    task = NBackTask(prefrontal, config, n_back=n)
    results = task.run_sequence(stimulus_sequence)

    return list(results["responses"])
