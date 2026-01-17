"""
Hierarchical Goal Management for Prefrontal Cortex.

Implements goal hierarchies, options framework, and hyperbolic temporal discounting
for Phase 3 of delayed gratification enhancements.

Uses existing PFC working memory for goal stack.
All learning via local rules (Hebbian, STDP) - NO backpropagation.

Biological Inspiration:
    - PFC represents abstract goals (Miller & Cohen, 2001)
    - Hierarchical organization of PFC (Badre & D'Esposito, 2009)
    - Rostral-caudal axis: Abstract → Concrete goals
    - Hyperbolic discounting models realistic temporal preferences

Psychological Framework:
    - Options framework (Sutton, Precup, Singh, 1999)
    - Hierarchical reinforcement learning
    - Goal decomposition and planning
    - Context-dependent impulsivity

Author: Thalia Project
Date: December 10, 2025
Phase: 3 - Hierarchical Goals & Temporal Abstraction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import torch

from thalia.constants.exploration import DEFAULT_EPSILON_EXPLORATION
from thalia.constants.regions import PREFRONTAL_PATIENCE_MIN


class GoalStatus(Enum):
    """Status of a goal in the hierarchy."""

    PENDING = "pending"  # Not started
    ACTIVE = "active"  # Currently pursuing
    COMPLETED = "completed"  # Successfully achieved
    FAILED = "failed"  # Could not achieve
    PAUSED = "paused"  # Temporarily suspended


@dataclass
class Goal:
    """
    Represents a goal at any level of the hierarchy.

    Goals can have subgoals (hierarchical decomposition).
    Goals can have policies (how to achieve them - options).
    Goals have value (how much they're worth).

    This is a DATA STRUCTURE for coordination, not a learned model.
    No backpropagation involved.
    """

    # Identity
    goal_id: int
    name: str
    description: str = ""

    # Hierarchy
    parent_goal: Optional[Goal] = None
    subgoals: List[Goal] = field(default_factory=list)
    level: int = 0  # 0=primitive actions, higher=more abstract

    # Status
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0  # 0-1, how much completed

    # Value
    intrinsic_value: float = 0.0  # Value of goal itself
    instrumental_value: float = 0.0  # Value for achieving parent goal
    total_value: float = 0.0  # Cached total

    # Policy (if this is an option)
    policy: Optional[Callable] = None  # state → action
    initiation_set: Optional[Callable] = None  # state → bool (can start?)
    termination_condition: Optional[Callable] = None  # state → bool (done?)

    # Temporal
    started_at: Optional[int] = None  # Timestep when started
    estimated_duration: Optional[int] = None  # Expected timesteps
    deadline: Optional[int] = None  # Must finish by

    # Context
    context: Dict[str, Any] = field(default_factory=dict)  # Additional info

    def add_subgoal(self, subgoal: Goal):
        """Add a subgoal to this goal."""
        subgoal.parent_goal = self
        subgoal.level = self.level - 1  # Subgoals are one level lower
        self.subgoals.append(subgoal)

    def compute_total_value(self) -> float:
        """
        Compute total value including subgoals.

        Value propagates up: Parent value = intrinsic + sum(subgoal values)

        This is pure computation - no learning involved.
        """
        if len(self.subgoals) == 0:
            # Leaf goal: only intrinsic value
            self.total_value = self.intrinsic_value
        else:
            # Internal goal: intrinsic + subgoal values
            subgoal_value = sum(g.compute_total_value() for g in self.subgoals)
            self.total_value = self.intrinsic_value + subgoal_value

        return self.total_value

    def is_achievable(self, state: torch.Tensor) -> bool:
        """Check if goal can be pursued from current state."""
        if self.initiation_set is not None:
            return bool(self.initiation_set(state))
        return True  # Always achievable if no constraint

    def is_achieved(self, state: torch.Tensor) -> bool:
        """Check if goal is achieved in current state."""
        if self.termination_condition is not None:
            return bool(self.termination_condition(state))

        # Default: Check if all subgoals completed
        if len(self.subgoals) > 0:
            return all(g.status == GoalStatus.COMPLETED for g in self.subgoals)

        return False  # Primitive goals need explicit termination


@dataclass
class GoalHierarchyConfig:
    """Configuration for goal hierarchy manager."""

    max_depth: int = 4  # Maximum hierarchy depth (0=actions, 4=high-level)
    max_active_goals: int = 5  # Limit on parallel goals (working memory constraint)

    # Goal selection
    use_value_based_selection: bool = True  # Select high-value goals
    epsilon_exploration: float = DEFAULT_EPSILON_EXPLORATION  # Explore low-value goals sometimes

    # Goal dynamics
    goal_persistence: float = 0.9  # Resist goal switching (stability)
    deadline_pressure_scale: float = 0.1  # Boost value as deadline approaches

    # Options
    enable_option_learning: bool = True  # Learn reusable policies
    option_discovery_threshold: float = 0.8  # Success rate to cache option

    # Device
    device: str = "cpu"


class GoalHierarchyManager:
    """
    Manages hierarchical goal structure and selection.

    Responsibilities:
    1. Maintain goal stack (active goals at each level)
    2. Select which goal to pursue (goal selection policy)
    3. Decompose goals into subgoals (goal elaboration)
    4. Track progress and completion
    5. Learn and reuse options (temporal abstraction)

    This is a COORDINATION system - no backprop, no global learning.
    Uses existing region mechanisms (PFC working memory, striatum values).
    """

    def __init__(self, config: GoalHierarchyConfig):
        self.config = config

        # Goal hierarchy
        self.root_goal: Optional[Goal] = None  # Top-level goal
        self.active_goals: List[Goal] = []  # Currently active goal stack
        self.goal_registry: Dict[int, Goal] = {}  # All goals by ID

        # Options library (reusable policies)
        self.options: Dict[str, Goal] = {}  # Learned options
        self.option_success_counts: Dict[str, int] = {}  # Success tracking
        self.option_attempt_counts: Dict[str, int] = {}  # Attempt tracking

        # State
        self.current_timestep = 0
        self.next_goal_id = 0

    def set_root_goal(self, goal: Goal):
        """Set the top-level goal."""
        self.root_goal = goal
        self.goal_registry[goal.goal_id] = goal

    def select_active_goal(self, state: torch.Tensor, available_goals: List[Goal]) -> Goal:
        """
        Select which goal to actively pursue.

        Uses value-based selection with deadline pressure.
        This is pure selection logic - no learning involved.

        Args:
            state: Current state
            available_goals: Goals that can be pursued from current state

        Returns:
            selected_goal: Goal to pursue
        """
        if len(available_goals) == 0:
            raise ValueError("No available goals")

        if not self.config.use_value_based_selection:
            # Random selection
            import random

            return random.choice(available_goals)

        # Compute selection scores
        scores = []
        for goal in available_goals:
            score = goal.total_value

            # Add deadline pressure
            if goal.deadline is not None:
                time_remaining = goal.deadline - self.current_timestep
                if time_remaining < (goal.estimated_duration or 10) * 1.5:
                    # Deadline approaching!
                    urgency = 1.0 / max(1, time_remaining)
                    score += self.config.deadline_pressure_scale * urgency

            # Add persistence bonus (if already active)
            if goal.status == GoalStatus.ACTIVE:
                score += self.config.goal_persistence

            scores.append(score)

        # Epsilon-greedy selection
        if torch.rand(1).item() < self.config.epsilon_exploration:
            import random

            return random.choice(available_goals)
        else:
            best_idx = int(torch.tensor(scores).argmax().item())
            return available_goals[best_idx]

    def decompose_goal(self, goal: Goal, state: torch.Tensor) -> List[Goal]:
        """
        Decompose a goal into subgoals.

        This is the key function for hierarchical planning:
        "How do I achieve X?" → Break into Y and Z

        Args:
            goal: Goal to decompose
            state: Current state (context for decomposition)

        Returns:
            subgoals: List of subgoals to achieve

        Implementation:
            Uses a hybrid approach:
            1. If goal has predefined subgoals (manual decomposition) → use them
            2. Check for learned options that match this goal type
            3. Consider state-dependent decomposition (which subgoals are achievable?)

        This enables both programmed and learned hierarchical planning.
        """
        # Start with existing subgoals (manual decomposition)
        subgoals = goal.subgoals.copy()

        # Check for learned options that could achieve this goal
        if self.config.enable_option_learning:
            for option_name, option in self.options.items():
                # If option can help achieve this goal and is initiatable
                if (
                    option.initiation_set is not None
                    and option.initiation_set(state)
                    and option.level == goal.level - 1
                ):  # Right hierarchical level
                    # Check if not already in subgoals
                    if not any(sg.name == option_name for sg in subgoals):
                        subgoals.append(option)

        # Filter to achievable subgoals based on current state
        achievable = [sg for sg in subgoals if sg.is_achievable(state)]

        return achievable if len(achievable) > 0 else subgoals

    def update_progress(self, goal: Goal, state: torch.Tensor):
        """
        Update goal progress based on current state.

        Checks termination condition and updates status.
        Pure state checking - no learning.
        """
        if goal.is_achieved(state):
            goal.status = GoalStatus.COMPLETED
            goal.progress = 1.0

            # Propagate completion up hierarchy
            if goal.parent_goal is not None:
                self.update_progress(goal.parent_goal, state)
        else:
            # Estimate progress from subgoals
            if len(goal.subgoals) > 0:
                completed = sum(1 for g in goal.subgoals if g.status == GoalStatus.COMPLETED)
                goal.progress = completed / len(goal.subgoals)

    def push_goal(self, goal: Goal):
        """Push goal onto active stack (start pursuing it)."""
        if len(self.active_goals) >= self.config.max_active_goals:
            # Working memory full! Must pause lowest-priority goal
            # Find lowest value active goal
            min_idx = 0
            min_value = self.active_goals[0].total_value
            for i, g in enumerate(self.active_goals):
                if g.total_value < min_value:
                    min_value = g.total_value
                    min_idx = i

            # Pause it
            paused = self.active_goals.pop(min_idx)
            paused.status = GoalStatus.PAUSED

        goal.status = GoalStatus.ACTIVE
        goal.started_at = self.current_timestep
        self.active_goals.append(goal)

    def pop_goal(self) -> Goal:
        """Remove goal from active stack."""
        if len(self.active_goals) == 0:
            raise ValueError("No active goals")

        goal = self.active_goals.pop()
        goal.status = GoalStatus.PAUSED if goal.progress < 1.0 else GoalStatus.COMPLETED
        return goal

    def get_current_goal(self) -> Optional[Goal]:
        """Get the goal currently being pursued (top of stack)."""
        return self.active_goals[-1] if len(self.active_goals) > 0 else None

    def cache_option(self, goal: Goal, policy: Callable, success_rate: float):
        """
        Cache successful policy as reusable option.

        If we've achieved a goal many times with same policy,
        cache it for fast reuse (Hebbian-like caching).

        No backprop - just storing successful patterns.

        Args:
            goal: Goal that was achieved
            policy: Policy that achieved it
            success_rate: Success rate of this policy
        """
        if not self.config.enable_option_learning:
            return

        if success_rate > self.config.option_discovery_threshold:
            option = Goal(
                goal_id=self.next_goal_id,
                name=f"option_{goal.name}",
                description=f"Reusable policy for {goal.name}",
                policy=policy,
                initiation_set=goal.initiation_set,
                termination_condition=goal.termination_condition,
                level=goal.level,
                intrinsic_value=goal.intrinsic_value,
            )
            self.next_goal_id += 1
            self.options[option.name] = option

    def record_option_attempt(self, option_name: str, success: bool):
        """Record success/failure of option execution."""
        if option_name not in self.option_attempt_counts:
            self.option_attempt_counts[option_name] = 0
            self.option_success_counts[option_name] = 0

        self.option_attempt_counts[option_name] += 1
        if success:
            self.option_success_counts[option_name] += 1

    def get_option_success_rate(self, option_name: str) -> float:
        """Get success rate for an option."""
        if option_name not in self.option_attempt_counts:
            return 0.0
        attempts = self.option_attempt_counts[option_name]
        if attempts == 0:
            return 0.0
        successes = self.option_success_counts[option_name]
        return successes / attempts

    def advance_time(self):
        """Advance internal timestep counter."""
        self.current_timestep += 1


@dataclass
class HyperbolicDiscountingConfig:
    """Configuration for hyperbolic discounting."""

    # Base discounting
    base_k: float = 0.01  # Base hyperbolic discount rate
    k_min: float = PREFRONTAL_PATIENCE_MIN  # Minimum k (most patient)
    k_max: float = 0.20  # Maximum k (most impulsive)

    # Context modulation
    cognitive_load_scale: float = 0.5  # How much load affects k
    stress_scale: float = 0.3  # How much stress affects k
    fatigue_scale: float = 0.2  # How much fatigue affects k

    # Learning
    learn_k: bool = True  # Adapt k based on outcomes
    k_learning_rate: float = 0.01  # Step size for k adaptation

    # Device
    device: str = "cpu"


class HyperbolicDiscounter:
    """
    Hyperbolic temporal discounting with context-dependent parameters.

    Key insight: Humans are impulsive under pressure, patient when relaxed.
    Models this with adaptive k parameter.

    Formula: V(R, t) = R / (1 + k * t)

    Where:
        R = reward magnitude
        t = delay (timesteps)
        k = discount rate (higher = more impulsive)

    Psychological Inspiration:
        - Marshmallow test failures under stress
        - PFC damage increases temporal discounting
        - Cognitive load reduces self-control

    Biological Note:
        - No backprop - k adaptation uses simple delta rule (local learning)
        - Context modulation is pure computation (no learning)
    """

    def __init__(self, config: HyperbolicDiscountingConfig):
        self.config = config
        self.k = config.base_k  # Current discount rate

        # Context state
        self.cognitive_load = 0.0  # 0-1
        self.stress_level = 0.0  # 0-1
        self.fatigue_level = 0.0  # 0-1

    def discount(self, reward: float, delay: int) -> float:
        """
        Compute discounted value of delayed reward.

        V(R, t) = R / (1 + k * t)

        Args:
            reward: Reward magnitude
            delay: Delay in timesteps

        Returns:
            discounted_value: Present value of delayed reward
        """
        # Hyperbolic discounting
        k_effective = self.get_effective_k()
        discounted_value = reward / (1.0 + k_effective * delay)
        return discounted_value

    def get_effective_k(self) -> float:
        """
        Compute effective k based on context.

        Higher cognitive load/stress → higher k → more impulsive
        Pure computation - no learning.
        """
        k = self.config.base_k

        # Increase k (more impulsive) under pressure
        k += self.config.cognitive_load_scale * self.cognitive_load
        k += self.config.stress_scale * self.stress_level
        k += self.config.fatigue_scale * self.fatigue_level

        # Clamp to valid range
        k = max(self.config.k_min, min(self.config.k_max, k))
        return k

    def update_context(
        self,
        cognitive_load: Optional[float] = None,
        stress: Optional[float] = None,
        fatigue: Optional[float] = None,
    ):
        """Update context variables that affect discounting."""
        if cognitive_load is not None:
            self.cognitive_load = max(0.0, min(1.0, cognitive_load))
        if stress is not None:
            self.stress_level = max(0.0, min(1.0, stress))
        if fatigue is not None:
            self.fatigue_level = max(0.0, min(1.0, fatigue))

    def learn_from_choice(
        self,
        chose_delayed: bool,
        immediate_value: float,
        delayed_value: float,
        delay: int,
        outcome_quality: float,
    ):
        """
        Adapt k based on choice outcomes.

        Uses simple delta rule (local learning, no backprop).

        If chose delayed reward and it was good → decrease k (more patient)
        If chose immediate and regretted → decrease k
        If chose delayed and regretted → increase k (more impulsive)

        Args:
            chose_delayed: Whether delayed option was chosen
            immediate_value: Value of immediate option
            delayed_value: Value of delayed option
            delay: Delay duration
            outcome_quality: How good the outcome was (0-1)
        """
        if not self.config.learn_k:
            return

        # Compute what k would have made this choice indifferent
        # V_immediate = V_delayed → immediate_value = delayed_value / (1 + k * delay)
        # Solve for k: k = (delayed_value / immediate_value - 1) / delay
        implied_k = (delayed_value / max(immediate_value, 0.01) - 1.0) / max(delay, 1)
        implied_k = max(self.config.k_min, min(self.config.k_max, implied_k))

        if chose_delayed and outcome_quality > 0.7:
            # Good choice! Move k toward implied k (if lower = more patient)
            if implied_k < self.k:
                self.k += self.config.k_learning_rate * (implied_k - self.k)
        elif not chose_delayed and outcome_quality < 0.3:
            # Regret! Should have waited. Decrease k (more patient).
            self.k -= self.config.k_learning_rate * 0.1

        # Clamp
        self.k = max(self.config.k_min, min(self.config.k_max, self.k))
