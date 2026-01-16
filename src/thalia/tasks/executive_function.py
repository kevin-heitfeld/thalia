"""
Executive Function Tasks for developmental curriculum.

Implements inhibitory control, cognitive flexibility, and working memory tasks
across developmental stages (Stage 1-4).

Stage 1 (Toddler): Go/No-Go, Delayed Gratification
Stage 2 (Preschool): DCCS, Task Switching
Stage 3-4 (School Age): Tower of Hanoi, Planning, Raven's Matrices

References:
- Diamond (2013): Executive functions
- Zelazo et al. (2003): DCCS and cognitive flexibility
- Mischel et al. (1989): Marshmallow test
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import torch

from thalia.constants.task import (
    FEATURE_INCREMENT_BASE,
    FEATURE_INCREMENT_COLUMN,
    FEATURE_INCREMENT_INTERACTION,
    FEATURE_NOISE_MATCH,
    STIMULUS_STRENGTH_HIGH,
    WEIGHT_INIT_SCALE_SMALL,
)
from thalia.tasks.stimulus_utils import (
    create_random_stimulus,
    create_zero_stimulus,
)


class TaskType(Enum):
    """Types of executive function tasks."""
    # Stage 1 (Toddler)
    GO_NO_GO = "go_no_go"
    DELAYED_GRATIFICATION = "delayed_gratification"

    # Stage 2 (Preschool)
    DCCS = "dccs"  # Dimensional Change Card Sort
    TASK_SWITCHING = "task_switching"

    # Stage 3-4 (School Age)
    TOWER_OF_HANOI = "tower_of_hanoi"
    RAVENS_MATRICES = "ravens_matrices"
    ANALOGICAL_REASONING = "analogical_reasoning"


class StimulusType(Enum):
    """Stimulus categories for Go/No-Go."""
    TARGET = "target"        # Go signal
    DISTRACTOR = "distractor"  # No-go signal
    NEUTRAL = "neutral"


@dataclass
class GoNoGoConfig:
    """Configuration for Go/No-Go task."""
    n_stimuli: int = 100  # Total trials
    target_probability: float = 0.7  # 70% go, 30% no-go
    stimulus_dim: int = 64  # Input dimension
    response_threshold: float = 0.5  # Decision threshold
    device: str = "cpu"


@dataclass
class DelayedGratificationConfig:
    """Configuration for delayed gratification task."""
    immediate_reward: float = 1.0  # Small reward now
    delayed_reward: float = 3.0  # Large reward later
    delay_steps: int = 50  # Delay period (timesteps)
    discount_rate: float = 0.95  # Temporal discounting
    device: str = "cpu"


@dataclass
class DCCSConfig:
    """Configuration for Dimensional Change Card Sort."""
    n_trials: int = 24  # 12 pre-switch, 12 post-switch
    n_dimensions: int = 2  # Color and shape
    n_features_per_dim: int = 2  # 2 colors, 2 shapes
    switch_trial: int = 12  # When to switch rule
    device: str = "cpu"


@dataclass
class TaskSwitchingConfig:
    """Configuration for task switching."""
    n_trials: int = 40  # Total trials
    n_tasks: int = 2  # Number of different tasks
    switch_probability: float = 0.3  # Probability of switching task
    stimulus_dim: int = 64  # Input dimension
    n_responses: int = 2  # Response options per task
    device: str = "cpu"


@dataclass
class TowerOfHanoiConfig:
    """Configuration for Tower of Hanoi task."""
    n_disks: int = 3  # Number of disks (2-5 typical)
    encode_dim: int = 64  # Encoding dimension per disk/peg
    max_moves: int = 100  # Maximum allowed moves
    optimal_moves: int = 7  # Optimal solution (2^n - 1)
    device: str = "cpu"


@dataclass
class RavensMatricesConfig:
    """Configuration for Raven's Progressive Matrices."""
    grid_size: int = 3  # 3x3 matrix (standard)
    pattern_complexity: str = "simple"  # simple, medium, hard
    n_answer_choices: int = 8  # Number of answer options
    stimulus_dim: int = 64  # Feature dimension per cell
    rule_types: List[str] = None  # progression, constant, distribution
    device: str = "cpu"

    def __post_init__(self):
        if self.rule_types is None:
            self.rule_types = ["progression", "constant", "distribution"]


@dataclass
class TaskResult:
    """Result of an executive function task."""
    correct: bool
    response_time: float
    task_type: TaskType
    trial_num: int
    additional_info: Dict[str, Any]


class ExecutiveFunctionTasks:
    """
    Executive function tasks across developmental stages.

    Implements inhibitory control, cognitive flexibility, and planning tasks
    that develop from toddlerhood through school age.
    """

    def __init__(self):
        self.statistics = {
            "n_trials": 0,
            "n_correct": 0,
            "by_task": {},
        }

    def reset_statistics(self):
        """Reset performance statistics."""
        self.statistics = {
            "n_trials": 0,
            "n_correct": 0,
            "by_task": {},
        }

    # ========================================================================
    # Stage 1: Toddler Tasks (Inhibitory Control Foundation)
    # ========================================================================

    def go_no_go(
        self,
        config: Optional[GoNoGoConfig] = None,
    ) -> Tuple[List[torch.Tensor], List[StimulusType], List[bool]]:
        """
        Generate Go/No-Go task trials.

        Tests basic inhibitory control:
        - Go trials: Respond to target stimulus
        - No-go trials: Inhibit response to distractor

        Args:
            config: Task configuration

        Returns:
            stimuli: List of stimulus tensors
            stimulus_types: Target or distractor
            correct_responses: True = respond, False = inhibit
        """
        config = config or GoNoGoConfig()
        device = torch.device(config.device)

        stimuli = []
        stimulus_types = []
        correct_responses = []

        for _ in range(config.n_stimuli):
            # Decide trial type
            is_target = np.random.rand() < config.target_probability

            if is_target:
                # Go trial: Target stimulus
                stimulus = self._generate_target_stimulus(config.stimulus_dim, device)
                stimulus_type = StimulusType.TARGET
                correct_response = True  # Should respond
            else:
                # No-go trial: Distractor stimulus
                stimulus = self._generate_distractor_stimulus(config.stimulus_dim, device)
                stimulus_type = StimulusType.DISTRACTOR
                correct_response = False  # Should inhibit

            stimuli.append(stimulus)
            stimulus_types.append(stimulus_type)
            correct_responses.append(correct_response)

        return stimuli, stimulus_types, correct_responses

    def _generate_target_stimulus(self, dim: int, device: torch.device) -> torch.Tensor:
        """Generate target stimulus (e.g., green circle)."""
        # Pattern: High values in first half of dimensions
        stimulus = create_zero_stimulus(dim, device)
        stimulus[:dim//2] = STIMULUS_STRENGTH_HIGH + torch.randn(dim//2, device=device) * WEIGHT_INIT_SCALE_SMALL
        return torch.clamp(stimulus, 0, 1)

    def _generate_distractor_stimulus(self, dim: int, device: torch.device) -> torch.Tensor:
        """Generate distractor stimulus (e.g., red square)."""
        # Pattern: High values in second half of dimensions
        stimulus = create_zero_stimulus(dim, device)
        stimulus[dim//2:] = STIMULUS_STRENGTH_HIGH + torch.randn(dim//2, device=device) * WEIGHT_INIT_SCALE_SMALL
        return torch.clamp(stimulus, 0, 1)

    def evaluate_go_no_go(
        self,
        responses: List[bool],
        correct_responses: List[bool],
        stimulus_types: List[StimulusType],
    ) -> Dict[str, float]:
        """
        Evaluate Go/No-Go performance.

        Metrics:
        - Overall accuracy
        - Go accuracy (hits)
        - No-go accuracy (correct rejections)
        - False alarm rate (responding on no-go)
        - d-prime (signal detection)

        Args:
            responses: Model responses (True = responded)
            correct_responses: Ground truth
            stimulus_types: Trial types

        Returns:
            metrics: Performance metrics
        """
        responses = np.array(responses)
        correct_responses = np.array(correct_responses)
        stimulus_types = np.array(stimulus_types)

        # Overall accuracy
        correct = (responses == correct_responses)
        overall_accuracy = correct.mean()

        # Go trials (target)
        go_trials = np.array([st == StimulusType.TARGET for st in stimulus_types])
        go_accuracy = (responses[go_trials] == correct_responses[go_trials]).mean()

        # No-go trials (distractor)
        no_go_trials = np.array([st == StimulusType.DISTRACTOR for st in stimulus_types])
        no_go_accuracy = (responses[no_go_trials] == correct_responses[no_go_trials]).mean()

        # False alarm rate (incorrectly responding on no-go)
        false_alarms = responses[no_go_trials].sum()
        fa_rate = false_alarms / no_go_trials.sum()

        # Hit rate (correctly responding on go)
        hits = responses[go_trials].sum()
        hit_rate = hits / go_trials.sum()

        # d-prime (signal detection sensitivity)
        from scipy.stats import norm
        hit_rate_adj = np.clip(hit_rate, 0.01, 0.99)
        fa_rate_adj = np.clip(fa_rate, 0.01, 0.99)
        d_prime = norm.ppf(hit_rate_adj) - norm.ppf(fa_rate_adj)

        # Update statistics
        self.statistics["n_trials"] += len(responses)
        self.statistics["n_correct"] += correct.sum()

        if "go_no_go" not in self.statistics["by_task"]:
            self.statistics["by_task"]["go_no_go"] = {"correct": 0, "total": 0}
        self.statistics["by_task"]["go_no_go"]["correct"] += correct.sum()
        self.statistics["by_task"]["go_no_go"]["total"] += len(responses)

        return {
            "overall_accuracy": overall_accuracy,
            "go_accuracy": go_accuracy,
            "no_go_accuracy": no_go_accuracy,
            "hit_rate": hit_rate,
            "false_alarm_rate": fa_rate,
            "d_prime": d_prime,
        }

    def delayed_gratification(
        self,
        config: Optional[DelayedGratificationConfig] = None,
    ) -> Dict[str, Any]:
        """
        Generate delayed gratification task (marshmallow test).

        Choice:
        - Immediate: Small reward now
        - Delayed: Large reward after waiting

        Tests ability to delay gratification for larger reward.

        Args:
            config: Task configuration

        Returns:
            task_info: Task parameters and optimal choice
        """
        config = config or DelayedGratificationConfig()

        # Calculate present value of delayed reward (with discounting)
        # PV = future_value * discount^delay
        present_value_delayed = config.delayed_reward * (config.discount_rate ** config.delay_steps)

        # Optimal choice
        optimal_choice = "delayed" if present_value_delayed > config.immediate_reward else "immediate"

        return {
            "immediate_reward": config.immediate_reward,
            "delayed_reward": config.delayed_reward,
            "delay_steps": config.delay_steps,
            "discount_rate": config.discount_rate,
            "present_value_delayed": present_value_delayed,
            "optimal_choice": optimal_choice,
        }

    def evaluate_delayed_gratification(
        self,
        choices: List[str],
        optimal_choices: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate delayed gratification performance.

        Metrics:
        - Proportion choosing delayed reward
        - Accuracy (optimal choice)
        - Average patience (delayed choices / total)

        Args:
            choices: "immediate" or "delayed"
            optimal_choices: Ground truth optimal

        Returns:
            metrics: Performance metrics
        """
        choices = np.array(choices)
        optimal_choices = np.array(optimal_choices)

        # Proportion delayed
        delayed_proportion = (choices == "delayed").mean()

        # Accuracy
        accuracy = (choices == optimal_choices).mean()

        # Update statistics
        correct = (choices == optimal_choices).sum()
        self.statistics["n_trials"] += len(choices)
        self.statistics["n_correct"] += correct

        if "delayed_gratification" not in self.statistics["by_task"]:
            self.statistics["by_task"]["delayed_gratification"] = {"correct": 0, "total": 0}
        self.statistics["by_task"]["delayed_gratification"]["correct"] += correct
        self.statistics["by_task"]["delayed_gratification"]["total"] += len(choices)

        return {
            "accuracy": accuracy,
            "delayed_proportion": delayed_proportion,
            "immediate_proportion": 1.0 - delayed_proportion,
        }

    # ========================================================================
    # Stage 2: Preschool Tasks (Cognitive Flexibility)
    # ========================================================================

    def dccs(
        self,
        config: Optional[DCCSConfig] = None,
    ) -> Tuple[List[torch.Tensor], List[str], List[int]]:
        """
        Generate Dimensional Change Card Sort (DCCS) task.

        Pre-switch: Sort by color (e.g., 12 trials)
        Post-switch: Sort by shape (requires inhibiting old rule)

        Tests cognitive flexibility and rule switching.

        Args:
            config: Task configuration

        Returns:
            cards: Card stimuli (encoded as tensors)
            rules: "color" or "shape" for each trial
            correct_bins: Target bin for each card
        """
        config = config or DCCSConfig()
        device = torch.device(config.device)

        cards = []
        rules = []
        correct_bins = []

        for trial in range(config.n_trials):
            # Determine current rule
            if trial < config.switch_trial:
                rule = "color"
            else:
                rule = "shape"

            # Generate card (color × shape)
            color = np.random.randint(0, config.n_features_per_dim)
            shape = np.random.randint(0, config.n_features_per_dim)

            # Encode as tensor (one-hot for color and shape)
            card = create_zero_stimulus(
                config.n_dimensions * config.n_features_per_dim,
                device=device
            )
            card[color] = 1.0  # Color dimension
            card[config.n_features_per_dim + shape] = 1.0  # Shape dimension

            # Correct bin depends on rule
            if rule == "color":
                correct_bin = color
            else:
                correct_bin = shape

            cards.append(card)
            rules.append(rule)
            correct_bins.append(correct_bin)

        return cards, rules, correct_bins

    def evaluate_dccs(
        self,
        responses: List[int],
        correct_bins: List[int],
        rules: List[str],
        switch_trial: int = 12,
    ) -> Dict[str, float]:
        """
        Evaluate DCCS performance.

        Metrics:
        - Pre-switch accuracy
        - Post-switch accuracy
        - Switch cost (accuracy drop)
        - Perseveration errors (using old rule)

        Args:
            responses: Model responses (bin choices)
            correct_bins: Ground truth
            rules: Rule for each trial
            switch_trial: Trial where rule switches

        Returns:
            metrics: Performance metrics
        """
        responses = np.array(responses)
        correct_bins = np.array(correct_bins)

        # Pre-switch accuracy
        pre_switch = responses[:switch_trial] == correct_bins[:switch_trial]
        pre_switch_accuracy = pre_switch.mean()

        # Post-switch accuracy
        post_switch = responses[switch_trial:] == correct_bins[switch_trial:]
        post_switch_accuracy = post_switch.mean()

        # Switch cost
        switch_cost = pre_switch_accuracy - post_switch_accuracy

        # Overall accuracy
        overall_accuracy = (responses == correct_bins).mean()

        # Update statistics
        correct = (responses == correct_bins).sum()
        self.statistics["n_trials"] += len(responses)
        self.statistics["n_correct"] += correct

        if "dccs" not in self.statistics["by_task"]:
            self.statistics["by_task"]["dccs"] = {"correct": 0, "total": 0}
        self.statistics["by_task"]["dccs"]["correct"] += correct
        self.statistics["by_task"]["dccs"]["total"] += len(responses)

        return {
            "overall_accuracy": overall_accuracy,
            "pre_switch_accuracy": pre_switch_accuracy,
            "post_switch_accuracy": post_switch_accuracy,
            "switch_cost": switch_cost,
        }

    # ========================================================================
    # Stage 2: Task Switching
    # ========================================================================

    def task_switching(
        self,
        config: Optional[TaskSwitchingConfig] = None,
    ) -> Tuple[List[torch.Tensor], List[int], List[int], List[bool]]:
        """
        Generate task switching paradigm.

        Two tasks alternate based on cue:
        - Task A: Respond to one stimulus dimension (e.g., magnitude)
        - Task B: Respond to different dimension (e.g., parity)

        Switch cost: Performance drop when task changes.

        Biology: Tests cognitive flexibility and task set maintenance.
        Requires PFC to maintain current task rule and switch when cued.

        Args:
            config: Task configuration

        Returns:
            stimuli: Task stimuli
            task_cues: Which task to perform (0 or 1)
            correct_responses: Correct response for each trial
            is_switch: Whether this trial is a switch
        """
        config = config or TaskSwitchingConfig()
        device = torch.device(config.device)

        stimuli = []
        task_cues = []
        correct_responses = []
        is_switch_list = []

        current_task = 0

        for trial in range(config.n_trials):
            # Determine if switch occurs
            if trial > 0 and np.random.rand() < config.switch_probability:
                current_task = 1 - current_task  # Toggle
                is_switch = True
            else:
                is_switch = False

            # Generate stimulus (random pattern)
            stimulus = create_random_stimulus(config.stimulus_dim, device)

            # Task-specific correct response
            # Task 0: Respond to first half of stimulus
            # Task 1: Respond to second half of stimulus
            if current_task == 0:
                response = int(stimulus[:config.stimulus_dim // 2].sum() > 0)
            else:
                response = int(stimulus[config.stimulus_dim // 2:].sum() > 0)

            stimuli.append(stimulus)
            task_cues.append(current_task)
            correct_responses.append(response)
            is_switch_list.append(is_switch)

        return stimuli, task_cues, correct_responses, is_switch_list

    def evaluate_task_switching(
        self,
        responses: List[int],
        correct_responses: List[int],
        is_switch: List[bool],
    ) -> Dict[str, float]:
        """
        Evaluate task switching performance.

        Metrics:
        - Overall accuracy
        - Switch trial accuracy (trials immediately after switch)
        - Non-switch (repeat) trial accuracy
        - Switch cost (repeat - switch accuracy)
        - Response time cost (if available)

        Args:
            responses: Model responses
            correct_responses: Ground truth
            is_switch: Whether each trial was a switch

        Returns:
            metrics: Performance metrics
        """
        responses = np.array(responses)
        correct_responses = np.array(correct_responses)
        is_switch = np.array(is_switch)

        # Overall accuracy
        correct = (responses == correct_responses)
        overall_accuracy = correct.mean()

        # Switch trial accuracy
        switch_trials = is_switch
        if switch_trials.sum() > 0:
            switch_accuracy = correct[switch_trials].mean()
        else:
            switch_accuracy = np.nan

        # Repeat trial accuracy
        repeat_trials = ~is_switch
        if repeat_trials.sum() > 0:
            repeat_accuracy = correct[repeat_trials].mean()
        else:
            repeat_accuracy = np.nan

        # Switch cost (drop in accuracy on switch trials)
        if not np.isnan(switch_accuracy) and not np.isnan(repeat_accuracy):
            switch_cost = repeat_accuracy - switch_accuracy
        else:
            switch_cost = np.nan

        # Update statistics
        correct_count = correct.sum()
        self.statistics["n_trials"] += len(responses)
        self.statistics["n_correct"] += correct_count

        if "task_switching" not in self.statistics["by_task"]:
            self.statistics["by_task"]["task_switching"] = {"correct": 0, "total": 0}
        self.statistics["by_task"]["task_switching"]["correct"] += correct_count
        self.statistics["by_task"]["task_switching"]["total"] += len(responses)

        return {
            "overall_accuracy": overall_accuracy,
            "switch_accuracy": switch_accuracy,
            "repeat_accuracy": repeat_accuracy,
            "switch_cost": switch_cost,
            "n_switch_trials": switch_trials.sum(),
            "n_repeat_trials": repeat_trials.sum(),
        }

    # ========================================================================
    # Stage 3-4: Planning and Abstract Reasoning (School Age)
    # ========================================================================

    def tower_of_hanoi(
        self,
        config: Optional[TowerOfHanoiConfig] = None,
    ) -> Tuple[List[torch.Tensor], List[int], List[Tuple[int, int]], int]:
        """
        Tower of Hanoi task: Move disks from source to target peg.

        Planning task requiring subgoaling and hierarchical decomposition.
        Tests: goal management, planning depth, subgoal creation.

        Biology:
        - Requires PFC for goal hierarchies
        - Dorsolateral PFC tracks subgoals
        - Optimal solution requires recursive thinking

        Args:
            config: Tower of Hanoi configuration

        Returns:
            states: List of board states (encoded tensors)
            disk_moved: List of which disk was moved (0 = smallest)
            moves: List of (from_peg, to_peg) tuples
            optimal_steps: Minimum moves needed (2^n - 1)
        """
        if config is None:
            config = TowerOfHanoiConfig()

        n_disks = config.n_disks
        encode_dim = config.encode_dim
        optimal_steps = config.optimal_moves or (2 ** n_disks - 1)

        # Initialize pegs: all disks on peg 0 (source)
        # Disks numbered 0 (smallest) to n-1 (largest)
        pegs = [list(range(n_disks)), [], []]  # [source, auxiliary, target]

        states = []
        disk_moved = []
        moves = []

        # Encode initial state
        states.append(self._encode_hanoi_state(pegs, encode_dim, n_disks, config.device))

        # Generate optimal solution via recursive algorithm
        solution_moves = self._solve_hanoi(n_disks, 0, 2, 1)  # source=0, target=2, aux=1

        # Execute moves and record states
        for from_peg, to_peg in solution_moves:
            if len(moves) >= config.max_moves:
                break

            # Move top disk
            if len(pegs[from_peg]) == 0:
                # Invalid move (shouldn't happen with correct solution)
                continue

            disk = pegs[from_peg].pop(0)  # Remove from top (index 0 = smallest)
            pegs[to_peg].insert(0, disk)  # Add to top

            # Record move
            moves.append((from_peg, to_peg))
            disk_moved.append(disk)

            # Encode new state
            states.append(self._encode_hanoi_state(pegs, encode_dim, n_disks, config.device))

        return states, disk_moved, moves, optimal_steps

    def _solve_hanoi(
        self,
        n: int,
        source: int,
        target: int,
        auxiliary: int,
    ) -> List[Tuple[int, int]]:
        """
        Recursive solution to Tower of Hanoi.

        Algorithm:
        1. Move n-1 disks from source to auxiliary (using target)
        2. Move largest disk from source to target
        3. Move n-1 disks from auxiliary to target (using source)

        Args:
            n: Number of disks to move
            source: Source peg
            target: Target peg
            auxiliary: Auxiliary peg

        Returns:
            List of (from_peg, to_peg) moves
        """
        if n == 0:
            return []

        if n == 1:
            return [(source, target)]

        # Move n-1 to auxiliary
        moves = self._solve_hanoi(n - 1, source, auxiliary, target)

        # Move largest to target
        moves.append((source, target))

        # Move n-1 from auxiliary to target
        moves.extend(self._solve_hanoi(n - 1, auxiliary, target, source))

        return moves

    def _encode_hanoi_state(
        self,
        pegs: List[List[int]],
        encode_dim: int,
        n_disks: int,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Encode Tower of Hanoi state as tensor.

        Encoding: For each peg, encode which disks are on it
        - Position encoding: disk size + position on peg
        - Total dim: encode_dim * 3 (three pegs)

        Args:
            pegs: List of 3 lists (disks on each peg)
            encode_dim: Dimension per peg
            n_disks: Total number of disks
            device: Computation device

        Returns:
            state: Encoded state [encode_dim * 3]
        """
        state = create_zero_stimulus(encode_dim * 3, device=torch.device(device))

        for peg_id, peg_disks in enumerate(pegs):
            offset = peg_id * encode_dim

            for position, disk in enumerate(peg_disks):
                # Encode disk identity and position
                # disk: 0 (smallest) to n_disks-1 (largest)
                # position: 0 (top) to len(peg)-1 (bottom)

                # Use position within encode_dim to encode disk
                idx = offset + (disk * encode_dim // n_disks)
                if idx < offset + encode_dim:
                    state[idx] = 1.0 - (position * 0.1)  # Top disk = 1.0, lower = 0.9, 0.8...

        return state

    def evaluate_tower_of_hanoi(
        self,
        moves: List[Tuple[int, int]],
        optimal_steps: int,
        final_state: torch.Tensor,
        target_state: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate Tower of Hanoi performance.

        Metrics:
        - Success: Whether puzzle was solved
        - Efficiency: Moves taken vs optimal (1.0 = optimal)
        - Planning quality: How close to optimal solution

        Args:
            moves: List of moves taken
            optimal_steps: Optimal number of moves
            final_state: Final board state
            target_state: Goal state (all disks on target peg)

        Returns:
            Dictionary with evaluation metrics
        """
        n_moves = len(moves)

        # Check success (final state matches target)
        success = torch.allclose(final_state, target_state, atol=0.1)

        # Efficiency: 1.0 if optimal, decreases with extra moves
        efficiency = optimal_steps / max(n_moves, optimal_steps) if n_moves > 0 else 0.0

        # Planning quality: inverse of extra moves ratio
        extra_moves = max(0, n_moves - optimal_steps)
        planning_quality = 1.0 / (1.0 + extra_moves / optimal_steps) if optimal_steps > 0 else 0.0

        # Update statistics
        self.statistics["n_trials"] += 1
        if success:
            self.statistics["n_correct"] += 1

        if "tower_of_hanoi" not in self.statistics["by_task"]:
            self.statistics["by_task"]["tower_of_hanoi"] = {
                "correct": 0,
                "total": 0,
                "avg_efficiency": 0.0,
                "n_optimal": 0,
            }

        task_stats = self.statistics["by_task"]["tower_of_hanoi"]
        task_stats["total"] += 1
        if success:
            task_stats["correct"] += 1

        # Update average efficiency
        prev_avg = task_stats["avg_efficiency"]
        n = task_stats["total"]
        task_stats["avg_efficiency"] = (prev_avg * (n - 1) + efficiency) / n

        if n_moves == optimal_steps:
            task_stats["n_optimal"] += 1

        return {
            "success": float(success),
            "n_moves": n_moves,
            "optimal_moves": optimal_steps,
            "efficiency": efficiency,
            "planning_quality": planning_quality,
            "extra_moves": extra_moves,
        }

    def ravens_matrices(
        self,
        config: Optional[RavensMatricesConfig] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """
        Raven's Progressive Matrices: Abstract reasoning task.

        Present 3x3 matrix with bottom-right cell missing.
        Find pattern rule and select correct completion from choices.

        Tests: Pattern induction, rule abstraction, analogical reasoning.

        Biology:
        - Requires PFC for rule maintenance
        - Parietal cortex for spatial transformations
        - Working memory to hold pattern elements

        Args:
            config: Raven's matrices configuration

        Returns:
            matrix: 3x3 matrix with last cell as zeros [9, stimulus_dim]
            answer_choices: N answer options [N, stimulus_dim]
            correct_answer: Index of correct answer
            rule_type: Type of rule ("progression", "constant", etc.)
        """
        if config is None:
            config = RavensMatricesConfig()

        grid_size = config.grid_size
        stimulus_dim = config.stimulus_dim
        n_choices = config.n_answer_choices
        device = torch.device(config.device)

        # Select rule type
        rule_type = np.random.choice(config.rule_types)

        # Generate matrix based on rule
        if rule_type == "progression":
            matrix, correct_cell = self._generate_progression_matrix(
                grid_size, stimulus_dim, config.pattern_complexity, device
            )
        elif rule_type == "constant":
            matrix, correct_cell = self._generate_constant_matrix(
                grid_size, stimulus_dim, config.pattern_complexity, device
            )
        elif rule_type == "distribution":
            matrix, correct_cell = self._generate_distribution_matrix(
                grid_size, stimulus_dim, config.pattern_complexity, device
            )
        else:
            # Default to progression
            matrix, correct_cell = self._generate_progression_matrix(
                grid_size, stimulus_dim, config.pattern_complexity, device
            )

        # Generate answer choices (correct + distractors)
        answer_choices = []
        correct_idx = np.random.randint(0, n_choices)

        for i in range(n_choices):
            if i == correct_idx:
                # Correct answer
                answer_choices.append(correct_cell)
            else:
                # Distractor (random or systematic perturbation)
                distractor = self._generate_distractor(
                    correct_cell, matrix, rule_type, device
                )
                answer_choices.append(distractor)

        answer_choices = torch.stack(answer_choices, dim=0)

        # Replace last cell with zeros (missing cell)
        matrix[-1] = torch.zeros_like(matrix[-1])

        return matrix, answer_choices, correct_idx, rule_type

    def _generate_progression_matrix(
        self,
        grid_size: int,
        stimulus_dim: int,
        complexity: str,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate matrix with progression rule.

        Rule: Each row/column has a progression (e.g., increasing size, rotation).
        """
        n_cells = grid_size ** 2
        matrix = create_zero_stimulus(n_cells * stimulus_dim, device=device).reshape(n_cells, stimulus_dim)

        # Base feature
        base_features = create_random_stimulus(stimulus_dim, device)

        # Progression increment
        increment = create_random_stimulus(stimulus_dim, device) * FEATURE_INCREMENT_BASE

        if complexity == "simple":
            # Row-wise progression
            for i in range(grid_size):
                for j in range(grid_size):
                    cell_idx = i * grid_size + j
                    matrix[cell_idx] = base_features + increment * j

        elif complexity == "medium":
            # Both row and column progression
            row_increment = increment
            col_increment = create_random_stimulus(stimulus_dim, device) * FEATURE_INCREMENT_COLUMN

            for i in range(grid_size):
                for j in range(grid_size):
                    cell_idx = i * grid_size + j
                    matrix[cell_idx] = base_features + row_increment * j + col_increment * i

        else:  # hard
            # Diagonal progression with interaction
            for i in range(grid_size):
                for j in range(grid_size):
                    cell_idx = i * grid_size + j
                    matrix[cell_idx] = (
                        base_features + increment * (i + j) +
                        create_random_stimulus(stimulus_dim, device) * FEATURE_INCREMENT_INTERACTION * i * j
                    )

        # Clip to [0, 1]
        matrix = torch.clamp(matrix, 0, 1)

        # Correct answer for bottom-right cell
        correct_cell = matrix[-1].clone()

        return matrix, correct_cell

    def _generate_constant_matrix(
        self,
        grid_size: int,
        stimulus_dim: int,
        complexity: str,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate matrix with constant rule.

        Rule: Certain features remain constant across row/column.
        """
        n_cells = grid_size ** 2
        matrix = create_zero_stimulus(n_cells * stimulus_dim, device=device).reshape(n_cells, stimulus_dim)

        # Constant features (same in all cells)
        constant_features = create_random_stimulus(stimulus_dim // 2, device)

        # Variable features (different per cell)
        for i in range(n_cells):
            variable = torch.rand(stimulus_dim // 2, device=device)
            matrix[i] = torch.cat([constant_features, variable])

        correct_cell = matrix[-1].clone()

        return matrix, correct_cell

    def _generate_distribution_matrix(
        self,
        grid_size: int,
        stimulus_dim: int,
        complexity: str,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate matrix with distribution rule.

        Rule: Each row/column contains each feature value exactly once.
        """
        n_cells = grid_size ** 2
        matrix = create_zero_stimulus(n_cells * stimulus_dim, device=device).reshape(n_cells, stimulus_dim)

        # Generate feature values for distribution
        feature_values = [
            torch.rand(stimulus_dim, device=device) for _ in range(grid_size)
        ]

        # Assign to cells following Latin square pattern
        for i in range(grid_size):
            for j in range(grid_size):
                cell_idx = i * grid_size + j
                value_idx = (i + j) % grid_size
                matrix[cell_idx] = feature_values[value_idx]

        correct_cell = matrix[-1].clone()

        return matrix, correct_cell

    def _generate_distractor(
        self,
        correct_cell: torch.Tensor,
        matrix: torch.Tensor,
        rule_type: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate distractor answer choice."""
        # Random perturbation of correct answer
        noise = torch.randn_like(correct_cell) * FEATURE_NOISE_MATCH
        distractor = correct_cell + noise
        distractor = torch.clamp(distractor, 0, 1)

        return distractor

    def evaluate_ravens_matrices(
        self,
        selected_answer: int,
        correct_answer: int,
        rule_type: str,
        response_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate Raven's matrices performance.

        Args:
            selected_answer: Index of selected answer
            correct_answer: Index of correct answer
            rule_type: Type of rule used
            response_time: Time taken (optional)

        Returns:
            Dictionary with evaluation metrics
        """
        correct = (selected_answer == correct_answer)

        # Update statistics
        self.statistics["n_trials"] += 1
        if correct:
            self.statistics["n_correct"] += 1

        if "ravens_matrices" not in self.statistics["by_task"]:
            self.statistics["by_task"]["ravens_matrices"] = {
                "correct": 0,
                "total": 0,
                "by_rule": {},
            }

        task_stats = self.statistics["by_task"]["ravens_matrices"]
        task_stats["total"] += 1
        if correct:
            task_stats["correct"] += 1

        # Track by rule type
        if rule_type not in task_stats["by_rule"]:
            task_stats["by_rule"][rule_type] = {"correct": 0, "total": 0}

        task_stats["by_rule"][rule_type]["total"] += 1
        if correct:
            task_stats["by_rule"][rule_type]["correct"] += 1

        result = {
            "correct": correct,
            "selected_answer": selected_answer,
            "correct_answer": correct_answer,
            "rule_type": rule_type,
        }

        if response_time is not None:
            result["response_time"] = response_time

        return result

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get cumulative performance statistics."""
        overall_accuracy = (
            self.statistics["n_correct"] / self.statistics["n_trials"]
            if self.statistics["n_trials"] > 0
            else 0.0
        )

        by_task_accuracy = {
            task: stats["correct"] / stats["total"]
            for task, stats in self.statistics["by_task"].items()
        }

        return {
            "overall_accuracy": overall_accuracy,
            "by_task": by_task_accuracy,
            "n_trials": self.statistics["n_trials"],
        }

    def generate_batch(
        self,
        task_type: TaskType,
        batch_size: int = 32,
        **task_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate batch for a specific task type.

        Args:
            task_type: Type of executive function task
            batch_size: Number of trials
            **task_kwargs: Task-specific parameters

        Returns:
            stimuli: Batch of task stimuli
            labels: Correct responses
        """
        if task_type == TaskType.GO_NO_GO:
            config = GoNoGoConfig(n_stimuli=batch_size, **task_kwargs)
            stimuli, stimulus_types, correct_responses = self.go_no_go(config)

            # Stack stimuli
            stimuli_tensor = torch.stack(stimuli, dim=0)
            labels_tensor = torch.tensor(correct_responses, dtype=torch.long)

            return stimuli_tensor, labels_tensor

        elif task_type == TaskType.DCCS:
            config = DCCSConfig(n_trials=batch_size, **task_kwargs)
            cards, rules, correct_bins = self.dccs(config)

            # Stack cards
            cards_tensor = torch.stack(cards, dim=0)
            labels_tensor = torch.tensor(correct_bins, dtype=torch.long)

            return cards_tensor, labels_tensor

        elif task_type == TaskType.TASK_SWITCHING:
            config = TaskSwitchingConfig(n_trials=batch_size, **task_kwargs)
            stimuli, task_cues, correct_responses, is_switch = self.task_switching(config)

            # Stack stimuli and concatenate with task cues
            stimuli_tensor = torch.stack(stimuli, dim=0)
            task_cues_tensor = torch.tensor(task_cues, dtype=torch.float32).unsqueeze(1)

            # Concatenate stimulus with task cue
            full_input = torch.cat([stimuli_tensor, task_cues_tensor], dim=1)
            labels_tensor = torch.tensor(correct_responses, dtype=torch.long)

            return full_input, labels_tensor

        else:
            raise ValueError(f"Batch generation not implemented for {task_type}")
