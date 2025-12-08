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

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Any
from enum import Enum
import numpy as np
import torch


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
        stimulus = torch.zeros(dim, device=device)
        stimulus[:dim//2] = 0.8 + torch.randn(dim//2, device=device) * 0.1
        return torch.clamp(stimulus, 0, 1)
    
    def _generate_distractor_stimulus(self, dim: int, device: torch.device) -> torch.Tensor:
        """Generate distractor stimulus (e.g., red square)."""
        # Pattern: High values in second half of dimensions
        stimulus = torch.zeros(dim, device=device)
        stimulus[dim//2:] = 0.8 + torch.randn(dim//2, device=device) * 0.1
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
            card = torch.zeros(config.n_dimensions * config.n_features_per_dim, device=device)
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
        
        else:
            raise ValueError(f"Batch generation not implemented for {task_type}")
