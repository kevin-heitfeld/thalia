"""Two-pattern thalamic discrimination task.

Pattern A: relay neurons 0–124 fire at ~10 Hz Poisson, 125–249 silent.
Pattern B: opposite.

Trial timing (all in 1 ms steps):
    [0, 50)     baseline — no external input
    [50, 450)   stimulus presentation (pattern A or B)
    [450, 500)  settling — no external input
    [300, 500)  response window for spike counting

Readout: cortex_association L5 pyramidal cells split into two halves.
Reward: +1.0 correct, -0.5 wrong, 0.0 ambiguous.
Convergence: >80 % accuracy over 100 consecutive trials.
"""

from __future__ import annotations

import random

import torch

from thalia.typing import SynapticInput

from thalia.training.encoding.spike_encoder import encode_population_rate
from thalia.training.encoding.spike_decoder import ReadoutGroup
from thalia.training.tasks.base import Trial, TrialResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THALAMUS_REGION = "thalamus_sensory"
RELAY_SIZE = 250

# Neuron index boundaries for the two patterns
_HALF = RELAY_SIZE // 2  # 125
_PATTERN_A_INDICES = torch.arange(0, _HALF)
_PATTERN_B_INDICES = torch.arange(_HALF, RELAY_SIZE)

# Firing probability per timestep for ~10 Hz at 1 ms dt
_FIRING_PROB = 0.01  # P(spike per neuron per ms) = 10 Hz × 1 ms

# Trial structure (ms / steps)
BASELINE_STEPS = 50
STIMULUS_STEPS = 400
SETTLING_STEPS = 50
STIMULUS_START = BASELINE_STEPS  # 50
STIMULUS_END = STIMULUS_START + STIMULUS_STEPS  # 450
TRIAL_DURATION = BASELINE_STEPS + STIMULUS_STEPS + SETTLING_STEPS  # 500
RESPONSE_WINDOW = (300, TRIAL_DURATION)  # last 200 ms

# Reward and inter-trial timing
REWARD_FORWARD_STEPS = 10
ITI_STEPS = 100

# Readout configuration
READOUT_REGION = "cortex_association"
READOUT_POPULATION = "l5_pyr"
READOUT_SIZE = 375
_READOUT_HALF = READOUT_SIZE // 2  # 187

# Convergence
CONVERGENCE_WINDOW = 100
CONVERGENCE_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Readout groups (fixed, task-defined)
# ---------------------------------------------------------------------------


def make_readout_groups() -> list[ReadoutGroup]:
    """Return the two readout groups for this task (group_a, group_b)."""
    return [
        ReadoutGroup(
            name="group_a",
            region=READOUT_REGION,
            population=READOUT_POPULATION,
            start_neuron=0,
            end_neuron=_READOUT_HALF,
        ),
        ReadoutGroup(
            name="group_b",
            region=READOUT_REGION,
            population=READOUT_POPULATION,
            start_neuron=_READOUT_HALF,
            end_neuron=READOUT_SIZE,
        ),
    ]


# ---------------------------------------------------------------------------
# Task implementation
# ---------------------------------------------------------------------------


class PatternAssociationTask:
    """Two-pattern thalamic discrimination task."""

    name: str = "pattern_association"

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = device

    # -- Task protocol methods ------------------------------------------------

    def generate_trial(self, trial_idx: int) -> Trial:
        """Create a random A/B trial."""
        pattern_id = random.choice(["A", "B"])
        return Trial(
            pattern_id=pattern_id,
            duration_steps=TRIAL_DURATION,
            response_window=RESPONSE_WINDOW,
        )

    def make_input(self, trial: Trial, timestep: int) -> SynapticInput | None:
        """Return Poisson spikes during the stimulus window, else ``None``."""
        if timestep < STIMULUS_START or timestep >= STIMULUS_END:
            return None

        if trial.pattern_id == "A":
            indices = _PATTERN_A_INDICES
        else:
            indices = _PATTERN_B_INDICES

        return encode_population_rate(
            relay_size=RELAY_SIZE,
            neuron_indices=indices,
            firing_prob=_FIRING_PROB,
            thalamus_region=THALAMUS_REGION,
            device=self.device,
        )

    def evaluate(
        self,
        trial: Trial,
        spike_counts: dict[str, int],
    ) -> TrialResult:
        """Score the brain's response.

        Correct if the expected readout group has strictly more spikes.
        Ambiguous (draw) → reward 0.
        """
        count_a = spike_counts.get("group_a", 0)
        count_b = spike_counts.get("group_b", 0)

        if trial.pattern_id == "A":
            correct = count_a > count_b
            wrong = count_b > count_a
        else:
            correct = count_b > count_a
            wrong = count_a > count_b

        if correct:
            reward = 1.0
        elif wrong:
            reward = -0.5
        else:
            reward = 0.0  # ambiguous (tie)

        return TrialResult(
            reward=reward,
            correct=correct,
            metrics={
                "count_a": count_a,
                "count_b": count_b,
                "margin": abs(count_a - count_b),
                "pattern": trial.pattern_id,
            },
        )

    def is_learned(self, recent_results: list[TrialResult]) -> bool:
        """True when accuracy exceeds threshold over the convergence window."""
        if len(recent_results) < CONVERGENCE_WINDOW:
            return False
        window = recent_results[-CONVERGENCE_WINDOW:]
        accuracy = sum(1 for r in window if r.correct) / len(window)
        return accuracy >= CONVERGENCE_THRESHOLD
