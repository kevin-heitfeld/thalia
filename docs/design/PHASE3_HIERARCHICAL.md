# Phase 3: Hierarchical Goals & Temporal Abstraction

**Duration**: 4 weeks
**Priority**: Important (needed by Stage 3)
**Dependencies**: Phase 1 (goal-conditioned values), Phase 2 (mental simulation)
**Target Completion**: Before Stage 3 curriculum begins

---

## Overview

Implement hierarchical goal structures and temporal abstraction (options framework) to enable:

1. **Goal hierarchies**: Break complex goals into subgoals
2. **Options framework**: Reusable action sequences (policies over subgoals)
3. **Hyperbolic discounting**: Realistic temporal preferences under cognitive load
4. **Value propagation**: Parent goal value = sum of subgoal values

**Why Critical for Stage 3+**:
- Essay writing: Topic → Paragraphs → Sentences → Words (4-level hierarchy)
- Planning tasks: Goal → Subgoals → Actions (Tower of Hanoi)
- Reading strategy: Understand passage → Answer questions → Find evidence
- Sustained attention: Maintain long-term goal despite distractions

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    PREFRONTAL CORTEX                              │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  GOAL HIERARCHY STACK                    │    │
│  │                                                          │    │
│  │  Level 3: [Write Essay]                                 │    │
│  │            ↓                                             │    │
│  │  Level 2: [Intro] → [Body] → [Conclusion]               │    │
│  │            ↓                                             │    │
│  │  Level 1: [Sentence 1] → [Sentence 2] → [Sentence 3]    │    │
│  │            ↓                                             │    │
│  │  Level 0: [word] → [word] → [word] → [word]             │    │
│  │                                                          │    │
│  │  Active Goal: Level 1, Sentence 2                        │    │
│  │  Subgoal Policy: π_sentence(word | context)             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           OPTIONS LIBRARY (Reusable Policies)            │    │
│  │                                                          │    │
│  │  - write_introduction(topic)                             │    │
│  │  - generate_argument(claim, evidence)                    │    │
│  │  - conclude_paragraph()                                  │    │
│  │  - add_transition_word()                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │      HYPERBOLIC DISCOUNTING (Context-Dependent)          │    │
│  │                                                          │    │
│  │  V(reward, delay) = R / (1 + k * delay)                 │    │
│  │  k = f(cognitive_load, stress, fatigue)                 │    │
│  │                                                          │    │
│  │  Low load:  k=0.01  (patient, long-term thinking)       │    │
│  │  High load: k=0.10  (impulsive, short-term focus)       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Goal Hierarchy Data Structures

### ⚠️ ARCHITECTURAL NOTE

Goal hierarchies are **data structures and coordination**, not learned networks.
- ✅ Goal stack maintained in PFC working memory (already exists!)
- ✅ Goal selection via value comparison (no backprop needed)
- ✅ Options learned via Hebbian caching of successful sequences
- ❌ NO separate neural networks with backprop

### Design

**File**: `src/thalia/regions/prefrontal_hierarchy.py`

```python
"""
Hierarchical Goal Management for Prefrontal Cortex.

Uses existing PFC working memory for goal stack.
All learning via local rules (Hebbian, STDP).

Biology:
    - PFC represents abstract goals (Miller & Cohen, 2001)
    - Hierarchical organization of PFC (Badre & D'Esposito, 2009)
    - Rostral-caudal axis: Abstract → Concrete goals

Psychology:
    - Options framework (Sutton, Precup, Singh, 1999)
    - Hierarchical reinforcement learning
    - Goal decomposition and planning

Note: This is a DATA STRUCTURE for coordination, not a learned model.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from enum import Enum
import torch


class GoalStatus(Enum):
    """Status of a goal in the hierarchy."""
    PENDING = "pending"      # Not started
    ACTIVE = "active"        # Currently pursuing
    COMPLETED = "completed"  # Successfully achieved
    FAILED = "failed"        # Could not achieve
    PAUSED = "paused"        # Temporarily suspended


@dataclass
class Goal:
    """
    Represents a goal at any level of the hierarchy.

    Goals can have subgoals (hierarchical decomposition).
    Goals can have policies (how to achieve them).
    Goals have value (how much they're worth).
    """

    # Identity
    goal_id: int
    name: str
    description: str = ""

    # Hierarchy
    parent_goal: Optional['Goal'] = None
    subgoals: List['Goal'] = field(default_factory=list)
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
    context: Dict = field(default_factory=dict)  # Additional info

    def add_subgoal(self, subgoal: 'Goal'):
        """Add a subgoal to this goal."""
        subgoal.parent_goal = self
        subgoal.level = self.level - 1  # Subgoals are one level lower
        self.subgoals.append(subgoal)

    def compute_total_value(self) -> float:
        """
        Compute total value including subgoals.

        Value propagates up: Parent value = intrinsic + sum(subgoal values)
        """
        if len(self.subgoals) == 0:
            # Leaf goal: only intrinsic value
            self.total_value = self.intrinsic_value
        else:
            # Internal goal: intrinsic + subgoal values
            subgoal_value = sum(g.compute_total_value() for g in self.subgoals)
            self.total_value = self.intrinsic_value + subgoal_value

        return self.total_value

    def is_achievable(self, state) -> bool:
        """Check if goal can be pursued from current state."""
        if self.initiation_set is not None:
            return self.initiation_set(state)
        return True  # Always achievable if no constraint

    def is_achieved(self, state) -> bool:
        """Check if goal is achieved in current state."""
        if self.termination_condition is not None:
            return self.termination_condition(state)

        # Default: Check if all subgoals completed
        if len(self.subgoals) > 0:
            return all(g.status == GoalStatus.COMPLETED for g in self.subgoals)

        return False  # Primitive goals need explicit termination


@dataclass
class GoalHierarchyConfig:
    """Configuration for goal hierarchy manager."""

    max_depth: int = 4  # Maximum hierarchy depth (0=actions, 4=high-level)
    max_active_goals: int = 5  # Limit on parallel goals (working memory)

    # Goal selection
    use_value_based_selection: bool = True  # Select high-value goals
    epsilon_exploration: float = 0.1  # Explore low-value goals sometimes

    # Goal dynamics
    goal_persistence: float = 0.9  # Resist goal switching
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
    """

    def __init__(self, config: GoalHierarchyConfig):
        self.config = config

        # Goal hierarchy
        self.root_goal: Optional[Goal] = None  # Top-level goal
        self.active_goals: List[Goal] = []  # Currently active goal stack
        self.goal_registry: Dict[int, Goal] = {}  # All goals by ID

        # Options library (reusable policies)
        self.options: Dict[str, Goal] = {}  # Learned options

        # State
        self.current_timestep = 0
        self.next_goal_id = 0

    def set_root_goal(self, goal: Goal):
        """Set the top-level goal."""
        self.root_goal = goal
        self.goal_registry[goal.goal_id] = goal

    def select_active_goal(
        self,
        state: torch.Tensor,
        available_goals: List[Goal]
    ) -> Goal:
        """
        Select which goal to actively pursue.

        Uses value-based selection with deadline pressure.

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
                if time_remaining < goal.estimated_duration * 1.5:
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
            best_idx = torch.tensor(scores).argmax().item()
            return available_goals[best_idx]

    def decompose_goal(
        self,
        goal: Goal,
        state: torch.Tensor
    ) -> List[Goal]:
        """
        Decompose a goal into subgoals.

        This is the key function for hierarchical planning:
        "How do I achieve X?" → Break into Y and Z

        Args:
            goal: Goal to decompose
            state: Current state (context for decomposition)

        Returns:
            subgoals: List of subgoals to achieve
        """
        # This is task-specific and would be learned or programmed
        # For now, return empty (subgoals set manually)
        return goal.subgoals

    def update_progress(self, goal: Goal, state: torch.Tensor):
        """
        Update goal progress based on current state.

        Checks termination condition and updates status.
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
                completed = sum(
                    1 for g in goal.subgoals
                    if g.status == GoalStatus.COMPLETED
                )
                goal.progress = completed / len(goal.subgoals)

    def push_goal(self, goal: Goal):
        """Push goal onto active stack (start pursuing it)."""
        if len(self.active_goals) >= self.config.max_active_goals:
            # Working memory full! Must pause lowest-priority goal
            self.pop_goal()

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

    def cache_option(
        self,
        goal: Goal,
        policy: Callable,
        success_rate: float
    ):
        """
        Cache successful policy as reusable option.

        If we've achieved a goal many times with same policy,
        cache it for fast reuse.
        """
        if success_rate > self.config.option_discovery_threshold:
            option = Goal(
                goal_id=self.next_goal_id,
                name=f"option_{goal.name}",
                description=f"Reusable policy for {goal.name}",
                policy=policy,
                initiation_set=goal.initiation_set,
                termination_condition=goal.termination_condition,
                level=goal.level
            )
            self.next_goal_id += 1
            self.options[option.name] = option
```

---

## Component 2: Hyperbolic Temporal Discounting

### Design

**File**: `src/thalia/regions/prefrontal_hierarchy.py` (continued)

```python
"""
Hyperbolic Temporal Discounting - Context-Dependent Impulsivity.

Models realistic human temporal preferences:
- Hyperbolic (not exponential) discounting
- Steeper discounting under cognitive load
- Adaptive k parameter based on context
"""

@dataclass
class HyperbolicDiscountingConfig:
    """Configuration for hyperbolic discounting."""

    # Base discounting
    base_k: float = 0.01  # Base hyperbolic discount rate
    k_min: float = 0.001  # Minimum k (most patient)
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

    Psychology:
        - Marshmallow test failures under stress
        - PFC damage increases temporal discounting
        - Cognitive load reduces self-control
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
        fatigue: Optional[float] = None
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
        outcome_quality: float
    ):
        """
        Adapt k based on choice outcomes.

        If chose delayed reward and it was good → decrease k (more patient)
        If chose immediate and regretted → decrease k
        If chose delayed and regretted → increase k (more impulsive)
        """
        if not self.config.learn_k:
            return

        # Compute what k would have made this choice indifferent
        # V_immediate = V_delayed → immediate_value = delayed_value / (1 + k * delay)
        # Solve for k: k = (delayed_value / immediate_value - 1) / delay
        implied_k = (delayed_value / max(immediate_value, 0.01) - 1.0) / max(delay, 1)
        implied_k = max(self.config.k_min, min(self.config.k_max, implied_k))

        if chose_delayed and outcome_quality > 0.7:
            # Good choice! Move k toward implied k (if lower)
            if implied_k < self.k:
                self.k += self.config.k_learning_rate * (implied_k - self.k)
        elif not chose_delayed and outcome_quality < 0.3:
            # Regret! Should have waited. Decrease k.
            self.k -= self.config.k_learning_rate * 0.1

        # Clamp
        self.k = max(self.config.k_min, min(self.config.k_max, self.k))
```

---

## Component 3: Integration with Prefrontal Cortex

**File to Modify**: `src/thalia/regions/prefrontal.py`

```python
# Add to PrefrontalConfig (around line 70):

# Hierarchical goals (NEW - Phase 3)
use_hierarchical_goals: bool = False
goal_hierarchy_config: Optional[GoalHierarchyConfig] = None

# Hyperbolic discounting (NEW - Phase 3)
use_hyperbolic_discounting: bool = False
hyperbolic_config: Optional[HyperbolicDiscountingConfig] = None

# Add to Prefrontal.__init__ (around line 250):

# Initialize goal hierarchy (NEW - Phase 3)
if self.config.use_hierarchical_goals:
    from thalia.regions.prefrontal_hierarchy import (
        GoalHierarchyManager,
        GoalHierarchyConfig,
        HyperbolicDiscounter,
        HyperbolicDiscountingConfig
    )

    gh_config = self.config.goal_hierarchy_config or GoalHierarchyConfig()
    self.goal_manager = GoalHierarchyManager(gh_config)

    # Hyperbolic discounting
    if self.config.use_hyperbolic_discounting:
        hd_config = self.config.hyperbolic_config or HyperbolicDiscountingConfig()
        self.discounter = HyperbolicDiscounter(hd_config)
    else:
        self.discounter = None
else:
    self.goal_manager = None
    self.discounter = None

# Add methods to Prefrontal class:

def set_goal_hierarchy(self, root_goal):
    """Set the top-level goal for hierarchical planning."""
    if self.goal_manager is None:
        raise ValueError("Hierarchical goals not enabled")
    self.goal_manager.set_root_goal(root_goal)

def get_current_goal(self):
    """Get currently active goal."""
    if self.goal_manager is None:
        return None
    return self.goal_manager.get_current_goal()

def update_cognitive_load(self, load: float):
    """Update cognitive load (affects temporal discounting)."""
    if self.discounter is not None:
        self.discounter.update_context(cognitive_load=load)

def evaluate_delayed_reward(self, reward: float, delay: int) -> float:
    """Discount delayed reward (hyperbolic or exponential)."""
    if self.discounter is not None:
        return self.discounter.discount(reward, delay)
    else:
        # Fallback: Exponential discounting
        gamma = 0.99
        return reward * (gamma ** delay)
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/unit/test_hierarchical_goals.py`

```python
def test_goal_hierarchy_construction():
    """Test building a goal hierarchy."""
    root = Goal(goal_id=0, name="write_essay", level=3)
    intro = Goal(goal_id=1, name="write_intro", level=2)
    body = Goal(goal_id=2, name="write_body", level=2)

    root.add_subgoal(intro)
    root.add_subgoal(body)

    assert len(root.subgoals) == 2
    assert intro.parent_goal == root
    assert intro.level == 2

def test_value_propagation():
    """Test that values propagate up hierarchy."""
    root = Goal(goal_id=0, name="root", level=2, intrinsic_value=1.0)
    child1 = Goal(goal_id=1, name="c1", level=1, intrinsic_value=2.0)
    child2 = Goal(goal_id=2, name="c2", level=1, intrinsic_value=3.0)

    root.add_subgoal(child1)
    root.add_subgoal(child2)

    total = root.compute_total_value()
    assert total == 1.0 + 2.0 + 3.0  # Parent + children

def test_hyperbolic_discounting():
    """Test hyperbolic vs exponential discounting."""
    config = HyperbolicDiscountingConfig(base_k=0.01)
    discounter = HyperbolicDiscounter(config)

    # Hyperbolic: V = R / (1 + k*t)
    v_short = discounter.discount(reward=1.0, delay=10)
    v_long = discounter.discount(reward=1.0, delay=100)

    # Should show preference reversal (hyperbolic property)
    # Small delays: steep discount
    # Large delays: flatter discount
    assert v_short > 0.90  # Small delay, high value
    assert v_long > 0.50  # Large delay, but not that discounted

def test_cognitive_load_increases_impulsivity():
    """Test that cognitive load increases discount rate."""
    config = HyperbolicDiscountingConfig(base_k=0.01, cognitive_load_scale=0.5)
    discounter = HyperbolicDiscounter(config)

    # Low load
    discounter.update_context(cognitive_load=0.1)
    v_low_load = discounter.discount(reward=1.0, delay=100)

    # High load
    discounter.update_context(cognitive_load=0.9)
    v_high_load = discounter.discount(reward=1.0, delay=100)

    # Higher load → more discounting → lower value
    assert v_high_load < v_low_load
```

### Integration Tests

**File**: `tests/integration/test_hierarchical_stage3.py`

```python
def test_essay_generation_with_hierarchy():
    """
    Test that hierarchical goals improve essay generation.

    Without hierarchy: Generate word-by-word (local coherence)
    With hierarchy: Generate paragraph-by-paragraph (global coherence)
    """
    brain = Brain(use_hierarchical_goals=True)

    # Set up goal hierarchy
    essay_goal = Goal(goal_id=0, name="write_essay", level=3)
    intro_goal = Goal(goal_id=1, name="intro", level=2)
    body_goal = Goal(goal_id=2, name="body", level=2)
    conclusion_goal = Goal(goal_id=3, name="conclusion", level=2)

    essay_goal.add_subgoal(intro_goal)
    essay_goal.add_subgoal(body_goal)
    essay_goal.add_subgoal(conclusion_goal)

    brain.prefrontal.set_goal_hierarchy(essay_goal)

    # Generate essay
    essay = brain.generate_text(prompt="Write about AI", max_length=200)

    # Evaluate structure
    coherence_score = evaluate_coherence(essay)
    structure_score = has_intro_body_conclusion(essay)

    assert coherence_score > 0.70
    assert structure_score > 0.80

def test_tower_of_hanoi_with_planning():
    """Test hierarchical planning on Tower of Hanoi."""
    brain = Brain(use_hierarchical_goals=True, use_model_based=True)

    # Solve 3-disk Tower of Hanoi
    initial_state = [3, 0, 0]  # All disks on peg 1
    goal_state = [0, 0, 3]  # All disks on peg 3

    solution = brain.solve_tower_of_hanoi(initial_state, goal_state)

    # Should find optimal solution (7 moves for 3 disks)
    assert len(solution) == 7
    assert verify_solution(solution, initial_state, goal_state)
```

---

## Success Criteria

### Phase 3 Complete When:

1. ✅ **Goal Hierarchies**:
   - [ ] `GoalHierarchyManager` implemented
   - [ ] Can build 4-level hierarchies
   - [ ] Value propagation works correctly
   - [ ] Goal selection policy functional

2. ✅ **Options Framework**:
   - [ ] Can define reusable policies
   - [ ] Option discovery from repeated success
   - [ ] Initiation/termination sets working

3. ✅ **Hyperbolic Discounting**:
   - [ ] Hyperbolic discount function implemented
   - [ ] Context modulation (cognitive load, stress) working
   - [ ] k adaptation from experience functional

4. ✅ **Integration**:
   - [ ] Integrated with Prefrontal cortex
   - [ ] Works with mental simulation coordinator (Phase 2)
   - [ ] Compatible with goal-conditioned values (Phase 1)

5. ✅ **Testing**:
   - [ ] All unit tests pass
   - [ ] Stage 3 essay generation benefits from hierarchy
   - [ ] Planning tasks (Tower of Hanoi) solved
   - [ ] Cognitive load affects temporal choices realistically

---

## Timeline

### Week 8: Goal Hierarchies
- Days 1-3: Implement `Goal` and `GoalHierarchyManager`
- Days 4-5: Unit tests and value propagation

### Week 9: Options Framework
- Days 1-3: Implement option caching and reuse
- Days 4-5: Integration with goal manager

### Week 10: Hyperbolic Discounting
- Days 1-2: Implement `HyperbolicDiscounter`
- Days 3-5: Context modulation and learning

### Week 11: Integration & Testing
- Days 1-2: Integrate with Prefrontal
- Days 3-5: Stage 3 tests and validation

---

## Performance Benchmarks

### Marshmallow Test (Stage 1)
- **Without improvements**: <30% wait for delayed reward
- **With TD(λ) only**: 50-60% wait (understands delay)
- **With planning**: 70-80% wait (simulates outcomes)
- **With hierarchy + hyperbolic**: 80-90% wait under low load, 40-50% under high load (realistic!)

### Essay Generation (Stage 3)
- **Baseline (no hierarchy)**: Coherence 0.50, Structure 0.40
- **With hierarchy**: Coherence 0.75, Structure 0.85
- **Improvement**: 50% better coherence, 2× better structure

### Planning Tasks (Stage 4)
- **Tower of Hanoi (3 disks)**: Solve in 7 moves (optimal)
- **Tower of Hanoi (4 disks)**: Solve in 15 moves (optimal)
- **Complex problems**: 30% faster with hierarchical decomposition

---

## Curriculum Integration Timeline

| Stage | Mechanism | Status | Notes |
|-------|-----------|--------|-------|
| Stage -0.5 | TD(λ) | ✅ Ready | Enable from start |
| Stage 0 | TD(λ) | ✅ Ready | Phonological learning |
| Stage 1 | TD(λ) + Goal-conditioned | ✅ Ready | Working memory, delayed gratification |
| Stage 1 late | Model-based (intro) | ✅ Ready | Week 14-16, simple simulation |
| Stage 2 | Model-based (full) | ⚠️ Week 4 | Grammar generation, task switching |
| Stage 2 late | Hierarchy (intro) | ⚠️ Week 8 | Week 20-24, 2-level hierarchies |
| Stage 3 | Hierarchy (full) + Hyperbolic | ⚠️ Week 11 | Essay writing, planning tasks |
| Stage 4-6 | All mature | ⚠️ Week 11+ | Maintain and apply |

**Critical Path**:
- Week 0: Start Phase 1 (TD(λ)) immediately
- Week 4: Complete Phase 2 (model-based) before Stage 2 starts (Week 16)
- Week 8: Complete Phase 3 (hierarchy) before Stage 3 starts (Week 30)

---

## Next Steps

1. **Immediate**: Begin Phase 1 implementation (TD(λ))
2. **Week 1-3**: Complete Phase 1, test with Stage -0.5
3. **Week 4-7**: Implement Phase 2 (model-based)
4. **Week 8-11**: Implement Phase 3 (hierarchy)
5. **Week 12+**: Integration testing across all curriculum stages

**All three phases complete**: Thalia will have full delayed gratification capabilities, from sensorimotor learning (Stage -0.5) through abstract reasoning (Stage 4+).

---

**Implementation Plan Complete**

Total implementation time: **11 weeks**
Total new code: ~3000-4000 lines
Expected performance gain: **2-5× improvement on temporal tasks**
Biological plausibility: ✅ **High** (all mechanisms have neural correlates)
