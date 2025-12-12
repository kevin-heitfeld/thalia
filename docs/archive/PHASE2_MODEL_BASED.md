# Phase 2: Model-Based Planning via Region Coordination

**Status**: ⏳ **ARCHITECTURE COMPLETE - IMPLEMENTATION PENDING**
**Duration**: 3-4 weeks (estimated)
**Priority**: Medium (Stage 2 can begin with Phase 1, enhanced by Phase 2)
**Dependencies**: Phase 1 (goal-conditioned values) ✅ COMPLETE
**Started**: Not yet started
**Target Completion**: TBD (after initial Stage 2 validation)

---

## ⚠️ CRITICAL ARCHITECTURE PRINCIPLE ⚠️

**Model-based planning is NOT a separate module with backpropagation!**

It **emerges** from coordination between existing brain regions using their native,
biologically-plausible learning mechanisms. There is no "WorldModel" brain area in
biology, and we don't implement one in Thalia either.

### What Already Exists (No Backprop Required!)

1. ✅ **PFC**: Working memory + predictive coding → naturally predicts future states
2. ✅ **Hippocampus**: Episodic memory + pattern completion → predicts outcomes from similar experiences
3. ✅ **Cortex**: Predictive coding layers → generate state representations and predictions
4. ✅ **Cerebellum**: Forward models → sensorimotor predictions (action→outcome)
5. ✅ **Striatum**: TD(λ) value learning + goal-conditioning → evaluates state quality

### What Phase 2 Implements

**NOT**: ❌ Separate `WorldModel` class with `nn.Module` and backprop (biologically implausible)

**INSTEAD**: ✅ Coordination utilities that orchestrate existing regions

```
MENTAL SIMULATION = Orchestrated Region Coordination

1. PFC maintains "simulated state" in working memory
2. Hippocampus retrieves similar past experiences (pattern completion)
3. PFC/Cortex predictive coding generates "next state" prediction
4. Striatum evaluates simulated state using learned goal-conditioned values
5. Repeat for N steps ahead (depth-limited tree search)
```

**All learning remains LOCAL** (STDP, Hebbian, error-corrective, three-factor). **No backprop anywhere!**

**See**: `docs/design/ARCHITECTURE_REVIEW_BIOLOGICAL_PLAUSIBILITY.md` for detailed analysis
of why the original WorldModel design was replaced with this emergent architecture.

---

## Why Model-Based Planning Matters for Stage 2+

Currently (Phase 1 complete), Thalia uses **model-free RL**: Actions are chosen based on
cached Q-values learned from experience. This works well for familiar situations but struggles with:

1. **Novel situations**: No cached values for new state-action pairs
2. **Grammar tasks**: "If I use SVO word order, what comes next?" requires simulation
3. **Task switching**: Mental rehearsal helps prepare for new task demands
4. **Text generation**: Planning ahead multiple words requires simulating sequences
5. **Credit assignment**: Delayed gratification requires reasoning about future consequences

**Model-based planning** adds the ability to **mentally simulate** action sequences before executing them:
- "What if I choose action A? What state would I reach?"
- "Which sequence of actions leads to the highest cumulative reward?"
- "How do I reach my goal from this unfamiliar state?"

**Critical Insight**: We don't need to build a separate "world model" - our existing regions
already provide the necessary mechanisms! We just need to coordinate them properly.

---

## Architecture Overview

### Biological Inspiration

**Vicarious Trial and Error (VTE)** in rodents (Tolman 1948, Johnson & Redish 2007):
- Rats pause at decision points
- Hippocampal "theta sequences" sweep through possible futures
- PFC maintains simulated states
- Striatum evaluates options
- Choose action based on mental simulation

**Human Planning** (Daw et al. 2005, 2011):
- Model-based vs model-free system arbitration
- PFC-hippocampus interaction during planning
- Simulation of hypothetical futures
- Value-based decision making

### How Existing Regions Provide Planning Components

| Capability | Region | Mechanism | Already Exists? |
|------------|--------|-----------|-----------------|
| **State representation** | Cortex | Spike patterns from sensory processing | ✅ Yes |
| **Working memory** | PFC | Persistent activity via recurrence | ✅ Yes |
| **Next-state prediction** | PFC/Cortex | Predictive coding layers | ✅ Yes |
| **Outcome prediction** | Hippocampus | Pattern completion from similar episodes | ✅ Yes |
| **State evaluation** | Striatum | TD(λ) + goal-conditioned values | ✅ Yes |
| **Error signals** | Cerebellum | Forward model prediction errors | ✅ Yes |

**All we need**: Coordination utilities to orchestrate these existing capabilities!

---

## Component 1: Mental Simulation Coordinator

### What It Does

Orchestrates existing brain regions to perform mental simulation ("what if" planning):

1. **Initialize**: PFC loads current state into working memory
2. **Loop** for N steps:
   a. Hippocampus retrieves similar past experiences (k-nearest neighbors)
   b. PFC/Cortex predictive coding predicts next state given action
   c. Striatum evaluates predicted state quality
   d. Update simulated state in PFC working memory
3. **Choose**: Select action sequence with highest cumulative value

**Location**: `src/thalia/planning/coordinator.py` (NEW FILE)

### Design

```python
"""
Mental Simulation Coordinator for Model-Based Planning.

Orchestrates existing brain regions (PFC, hippocampus, striatum, cortex)
to perform mental simulation of action sequences.

NO BACKPROPAGATION. All learning remains local to each region.

Biological Inspiration:
    - Vicarious Trial and Error (Tolman, 1948)
    - Hippocampal theta sequences (Johnson & Redish, 2007)
    - PFC working memory (Miller & Cohen, 2001)
    - Model-based planning (Daw et al., 2005)

Author: Thalia Project
Date: December 10, 2025
Phase: 2 - Model-Based Planning
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import torch

@dataclass
class SimulationConfig:
    """Configuration for mental simulation."""
    
    # Search parameters
    depth: int = 3  # How many steps to look ahead
    branching_factor: int = 3  # How many actions to consider per state
    
    # Hippocampal retrieval
    n_similar_experiences: int = 5  # Top-K similar episodes
    similarity_threshold: float = 0.5  # Minimum similarity to use
    
    # PFC simulation
    simulation_noise: float = 0.1  # Add noise to simulations
    
    # Value evaluation
    discount_gamma: float = 0.95  # Discount future rewards
    
    # Computational budget
    max_simulations: int = 100  # Limit planning time


@dataclass
class Rollout:
    """Result of a simulated action sequence."""
    
    states: List[torch.Tensor]  # Sequence of simulated states
    actions: List[int]  # Sequence of actions
    rewards: List[float]  # Sequence of predicted rewards
    cumulative_value: float  # Total discounted return
    uncertainty: float  # Confidence in simulation


class MentalSimulationCoordinator:
    """
    Coordinates existing brain regions to perform mental simulation.
    
    Uses:
        - PFC: Maintains simulated state in working memory
        - Hippocampus: Retrieves similar past experiences
        - PFC/Cortex: Predictive coding for next-state prediction
        - Striatum: Evaluates simulated states
    
    NO separate world model. NO backpropagation. Just coordination!
    """
    
    def __init__(
        self,
        pfc,  # PrefrontalCortex instance
        hippocampus,  # Hippocampus (trisynaptic) instance
        striatum,  # Striatum instance
        cortex,  # Cortex instance
        config: SimulationConfig
    ):
        self.pfc = pfc
        self.hippocampus = hippocampus
        self.striatum = striatum
        self.cortex = cortex
        self.config = config
    
    def simulate_rollout(
        self,
        current_state: torch.Tensor,
        action_sequence: List[int],
        goal_context: Optional[torch.Tensor] = None
    ) -> Rollout:
        """
        Simulate a specific action sequence.
        
        Args:
            current_state: Starting state (spike pattern)
            action_sequence: Sequence of actions to simulate
            goal_context: Optional goal for goal-conditioned values
        
        Returns:
            rollout: Simulated trajectory with predicted outcomes
        """
        states = [current_state.clone()]
        actions = []
        rewards = []
        cumulative_value = 0.0
        discount = 1.0
        
        # Load current state into PFC working memory
        simulated_state = current_state.clone()
        
        for step, action in enumerate(action_sequence):
            # 1. Retrieve similar past experiences from hippocampus
            similar_episodes = self.hippocampus.retrieve_similar(
                query_state=simulated_state,
                query_action=action,
                k=self.config.n_similar_experiences
            )
            
            # 2. Predict next state using PFC/Cortex predictive coding
            #    Informed by similar past experiences
            next_state_pred = self._predict_next_state(
                current=simulated_state,
                action=action,
                similar_experiences=similar_episodes
            )
            
            # 3. Predict reward from similar experiences
            reward_pred = self._predict_reward(similar_episodes)
            
            # 4. Evaluate predicted state using striatum
            if goal_context is not None:
                state_value = self.striatum.evaluate_state(
                    next_state_pred,
                    goal_context=goal_context
                )
            else:
                state_value = self.striatum.evaluate_state(next_state_pred)
            
            # Store trajectory
            states.append(next_state_pred)
            actions.append(action)
            rewards.append(reward_pred)
            cumulative_value += discount * (reward_pred + self.config.discount_gamma * state_value)
            discount *= self.config.discount_gamma
            
            # Update simulated state
            simulated_state = next_state_pred
        
        # Estimate uncertainty from hippocampal retrieval
        uncertainty = self._estimate_uncertainty(similar_episodes)
        
        return Rollout(
            states=states,
            actions=actions,
            rewards=rewards,
            cumulative_value=cumulative_value,
            uncertainty=uncertainty
        )
    
    def plan_best_action(
        self,
        current_state: torch.Tensor,
        available_actions: List[int],
        goal_context: Optional[torch.Tensor] = None
    ) -> Tuple[int, Rollout]:
        """
        Search for best action using tree search.
        
        Implements breadth-first search with striatum value guidance.
        
        Args:
            current_state: Starting state
            available_actions: List of possible actions
            goal_context: Optional goal context from PFC
        
        Returns:
            best_action: First action of best sequence
            best_rollout: Full best trajectory
        """
        best_action = None
        best_value = float('-inf')
        best_rollout = None
        
        # Limit actions to consider (branching factor)
        if len(available_actions) > self.config.branching_factor:
            # Use striatum to prioritize which actions to explore
            action_priorities = self._get_action_priorities(
                current_state,
                available_actions,
                goal_context
            )
            top_actions = sorted(
                available_actions,
                key=lambda a: action_priorities[a],
                reverse=True
            )[:self.config.branching_factor]
        else:
            top_actions = available_actions
        
        # Simulate each action
        for action in top_actions:
            # Simulate depth steps ahead
            action_sequence = [action] + self._generate_greedy_sequence(
                current_state,
                action,
                self.config.depth - 1,
                goal_context
            )
            
            rollout = self.simulate_rollout(
                current_state,
                action_sequence,
                goal_context
            )
            
            # Keep best
            if rollout.cumulative_value > best_value:
                best_value = rollout.cumulative_value
                best_action = action
                best_rollout = rollout
        
        return best_action, best_rollout
    
    def _predict_next_state(
        self,
        current: torch.Tensor,
        action: int,
        similar_experiences: List[Dict]
    ) -> torch.Tensor:
        """
        Predict next state using PFC/Cortex predictive coding.
        
        Informed by similar past experiences from hippocampus.
        """
        if len(similar_experiences) > 0:
            # Weight predictions by similarity
            weighted_prediction = torch.zeros_like(current)
            total_weight = 0.0
            
            for exp in similar_experiences:
                similarity = exp['similarity']
                if similarity > self.config.similarity_threshold:
                    # Use actual outcome from memory
                    weighted_prediction += similarity * exp['next_state']
                    total_weight += similarity
            
            if total_weight > 0:
                next_state = weighted_prediction / total_weight
            else:
                # No good matches - use PFC predictive coding alone
                next_state = self.pfc.predict_next_state(current, action)
        else:
            # No similar experiences - use PFC predictive coding
            next_state = self.pfc.predict_next_state(current, action)
        
        # Add simulation noise
        noise = torch.randn_like(next_state) * self.config.simulation_noise
        next_state = next_state + noise
        
        return next_state
    
    def _predict_reward(self, similar_experiences: List[Dict]) -> float:
        """Predict reward from similar past experiences."""
        if len(similar_experiences) == 0:
            return 0.0
        
        # Weighted average of similar experiences
        weighted_reward = 0.0
        total_weight = 0.0
        
        for exp in similar_experiences:
            similarity = exp['similarity']
            if similarity > self.config.similarity_threshold:
                weighted_reward += similarity * exp['reward']
                total_weight += similarity
        
        if total_weight > 0:
            return weighted_reward / total_weight
        else:
            return 0.0
    
    def _estimate_uncertainty(self, similar_experiences: List[Dict]) -> float:
        """Estimate uncertainty based on hippocampal retrieval."""
        if len(similar_experiences) == 0:
            return 1.0  # Max uncertainty
        
        # Uncertainty = 1 - average similarity
        avg_similarity = sum(exp['similarity'] for exp in similar_experiences) / len(similar_experiences)
        return 1.0 - avg_similarity
    
    def _get_action_priorities(
        self,
        state: torch.Tensor,
        actions: List[int],
        goal_context: Optional[torch.Tensor]
    ) -> Dict[int, float]:
        """Get action priorities from striatum (for pruning search tree)."""
        priorities = {}
        for action in actions:
            if goal_context is not None:
                value = self.striatum.get_goal_conditioned_value(
                    state,
                    action,
                    goal_context
                )
            else:
                value = self.striatum.get_value(state, action)
            priorities[action] = value
        return priorities
    
    def _generate_greedy_sequence(
        self,
        start_state: torch.Tensor,
        first_action: int,
        remaining_depth: int,
        goal_context: Optional[torch.Tensor]
    ) -> List[int]:
        """Generate greedy action sequence for remainder of rollout."""
        sequence = []
        state = start_state
        
        for _ in range(remaining_depth):
            # Get similar experiences
            similar = self.hippocampus.retrieve_similar(
                query_state=state,
                k=self.config.n_similar_experiences
            )
            
            # Predict next state
            next_state = self._predict_next_state(state, first_action, similar)
            
            # Choose greedy action using striatum
            if goal_context is not None:
                action = self.striatum.select_goal_conditioned_action(
                    next_state,
                    goal_context,
                    epsilon=0.0  # Greedy
                )
            else:
                action = self.striatum.select_action(
                    next_state,
                    epsilon=0.0  # Greedy
                )
            
            sequence.append(action)
            state = next_state
        
        return sequence
```

### Key Methods to Add to Existing Regions

**PFC** (`src/thalia/regions/prefrontal.py`):
```python
def predict_next_state(
    self,
    current_state: torch.Tensor,
    action: int
) -> torch.Tensor:
    """
    Predict next state using predictive coding.
    
    Uses existing PredictiveCodingLayer - just needs action conditioning.
    """
    # Action-condition the prediction (simple approach: concatenate)
    action_one_hot = torch.zeros(self.n_actions, device=self.device)
    action_one_hot[action] = 1.0
    
    conditioned_state = torch.cat([current_state, action_one_hot])
    
    # Use existing predictive coding layer
    prediction = self.predictive_coding_layer.predict(conditioned_state)
    
    return prediction[:len(current_state)]  # Remove action part
```

**Hippocampus** (`src/thalia/regions/hippocampus/trisynaptic.py`):
```python
def retrieve_similar(
    self,
    query_state: torch.Tensor,
    query_action: Optional[int] = None,
    k: int = 5
) -> List[Dict]:
    """
    Retrieve K most similar past experiences.
    
    Uses existing episodic memory - just needs K-nearest neighbor retrieval.
    
    Returns:
        similar_episodes: List of dicts with keys:
            - 'state', 'action', 'next_state', 'reward', 'similarity'
    """
    # Compute similarity to all stored episodes
    similarities = []
    for episode in self.episodic_memory:
        # Cosine similarity
        sim = F.cosine_similarity(
            query_state.unsqueeze(0),
            episode['state'].unsqueeze(0)
        ).item()
        
        # If action provided, boost similarity for matching actions
        if query_action is not None and episode['action'] == query_action:
            sim *= 1.2
        
        similarities.append((sim, episode))
    
    # Sort and return top-K
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    similar = []
    for sim, episode in similarities[:k]:
        similar.append({
            'state': episode['state'],
            'action': episode['action'],
            'next_state': episode['next_state'],
            'reward': episode['reward'],
            'similarity': sim
        })
    
    return similar
```

**Striatum** (`src/thalia/regions/striatum/striatum.py`):
```python
def evaluate_state(
    self,
    state: torch.Tensor,
    goal_context: Optional[torch.Tensor] = None
) -> float:
    """
    Evaluate state quality (max Q-value over actions).
    
    Uses existing value function - just needs state evaluation.
    """
    if goal_context is not None:
        values = self.get_goal_conditioned_values(state, goal_context)
    else:
        values = self.get_values(state)
    
    return values.max().item()
```

---

## Component 2: Dyna-Style Background Planning

### What It Does

**Dyna algorithm** (Sutton 1990): Interleave real and simulated experience

1. **Real experience**: Update striatum values (model-free)
2. **Simulated experience**: Use mental simulation for additional value updates
3. **Background planning**: During "idle" time, simulate and learn

**Key insight**: We don't need a separate world model! Hippocampus + PFC already provide
the prediction capability needed.

**Location**: `src/thalia/planning/dyna.py` (NEW FILE)

### Design

```python
"""
Dyna-Style Background Planning.

Combines model-free learning (striatum) with model-based planning
(mental simulation coordinator).

NO separate world model. Uses existing region coordination.

Reference:
    Sutton (1990): Integrated architectures for learning, planning, and reacting

Author: Thalia Project
Date: December 10, 2025
Phase: 2 - Model-Based Planning
"""

from dataclasses import dataclass
from typing import Optional
import torch
import random

@dataclass
class DynaConfig:
    """Configuration for Dyna planning."""
    
    # Planning budget
    n_planning_steps: int = 5  # Simulations per real experience
    
    # Learning from simulations
    simulation_lr_scale: float = 0.5  # Discount simulated updates
    
    # Prioritization
    use_prioritized_sweeping: bool = True  # Focus on important states
    priority_threshold: float = 0.1  # Minimum priority


class DynaPlanner:
    """
    Dyna algorithm: Combine real experience with simulated planning.
    
    Process:
        1. Real experience → update striatum values
        2. Sample previous states from hippocampus
        3. Simulate outcomes using mental simulation coordinator
        4. Update striatum values from simulated experience
    
    NO separate world model - uses existing region coordination!
    """
    
    def __init__(
        self,
        coordinator: 'MentalSimulationCoordinator',
        striatum,
        hippocampus,
        config: DynaConfig
    ):
        self.coordinator = coordinator
        self.striatum = striatum
        self.hippocampus = hippocampus
        self.config = config
        
        # Priority queue for states to simulate (for prioritized sweeping)
        self.state_priorities = {}
    
    def process_real_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        goal_context: Optional[torch.Tensor] = None
    ):
        """
        Process real experience and trigger background planning.
        
        Args:
            state, action, reward, next_state, done: Real transition
            goal_context: Optional goal context
        """
        # 1. Update striatum from real experience (model-free)
        td_error = self.striatum.update_td_lambda(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            goal_context=goal_context
        )
        
        # 2. Hippocampus stores experience (already happens automatically)
        
        # 3. Update priority for this state (for prioritized sweeping)
        if self.config.use_prioritized_sweeping:
            self.state_priorities[self._state_hash(state)] = abs(td_error)
        
        # 4. Do background planning
        self.do_planning(goal_context)
    
    def do_planning(self, goal_context: Optional[torch.Tensor] = None):
        """
        Background planning: Simulate additional experience and learn.
        
        This is the "thinking" phase - using existing regions to imagine
        what would happen and learn from simulations.
        """
        for _ in range(self.config.n_planning_steps):
            # Sample a previous state to start simulation from
            if self.config.use_prioritized_sweeping and self.state_priorities:
                # Sample proportional to priority (TD error)
                sampled_state = self._sample_by_priority()
            else:
                # Random sample from hippocampus
                sampled_state = self._sample_random_state()
            
            if sampled_state is None:
                continue  # No experiences yet
            
            # Simulate action from this state
            available_actions = list(range(self.striatum.n_actions))
            action = random.choice(available_actions)
            
            # Use coordinator to simulate outcome
            rollout = self.coordinator.simulate_rollout(
                current_state=sampled_state,
                action_sequence=[action],
                goal_context=goal_context
            )
            
            # Update striatum from simulated experience
            # Scale learning rate (less confident in simulation)
            original_lr = self.striatum.config.learning_rate
            self.striatum.config.learning_rate *= self.config.simulation_lr_scale
            
            sim_next_state = rollout.states[1]  # State after action
            sim_reward = rollout.rewards[0]
            
            self.striatum.update_td_lambda(
                state=sampled_state,
                action=action,
                reward=sim_reward,
                next_state=sim_next_state,
                done=False,  # Simulations don't end episodes
                goal_context=goal_context
            )
            
            # Restore learning rate
            self.striatum.config.learning_rate = original_lr
    
    def _sample_by_priority(self) -> Optional[torch.Tensor]:
        """Sample state proportional to TD error magnitude."""
        if not self.state_priorities:
            return None
        
        # Convert priorities to probabilities
        states = list(self.state_priorities.keys())
        priorities = torch.tensor([self.state_priorities[s] for s in states])
        probs = priorities / priorities.sum()
        
        # Sample
        idx = torch.multinomial(probs, 1).item()
        state_hash = states[idx]
        
        # Retrieve actual state from hippocampus
        # (In practice, we'd need to store state->hash mapping)
        return self._retrieve_state_by_hash(state_hash)
    
    def _sample_random_state(self) -> Optional[torch.Tensor]:
        """Sample random state from hippocampal memory."""
        if len(self.hippocampus.episodic_memory) == 0:
            return None
        
        episode = random.choice(self.hippocampus.episodic_memory)
        return episode['state']
    
    def _state_hash(self, state: torch.Tensor) -> int:
        """Simple hash for state (for priority dict)."""
        return hash(state.cpu().numpy().tobytes())
    
    def _retrieve_state_by_hash(self, state_hash: int) -> Optional[torch.Tensor]:
        """Retrieve state from hippocampus by hash."""
        # Simplified - in practice would need proper state->hash mapping
        for episode in self.hippocampus.episodic_memory:
            if self._state_hash(episode['state']) == state_hash:
                return episode['state']
        return None
```

---

## Integration with Brain

### Brain-Level Interface

**Location**: `src/thalia/core/brain.py` (EXTEND EXISTING)

Add planning capabilities to the `EventDrivenBrain` class:

```python
class EventDrivenBrain:
    """Main brain class with model-based planning."""
    
    def __init__(self, config: BrainConfig):
        # Existing initialization...
        
        # NEW: Planning system (if enabled)
        if config.use_model_based_planning:
            from thalia.planning.coordinator import (
                MentalSimulationCoordinator,
                SimulationConfig
            )
            from thalia.planning.dyna import DynaPlanner, DynaConfig
            
            self.simulation_coordinator = MentalSimulationCoordinator(
                pfc=self.pfc,
                hippocampus=self.hippocampus,
                striatum=self.striatum,
                cortex=self.cortex,
                config=SimulationConfig()
            )
            
            self.dyna_planner = DynaPlanner(
                coordinator=self.simulation_coordinator,
                striatum=self.striatum,
                hippocampus=self.hippocampus,
                config=DynaConfig()
            )
        else:
            self.simulation_coordinator = None
            self.dyna_planner = None
    
    def select_action(
        self,
        state: torch.Tensor,
        available_actions: List[int],
        use_planning: bool = True,
        epsilon: float = 0.1
    ) -> int:
        """
        Select action, optionally using model-based planning.
        
        Args:
            state: Current state
            available_actions: List of possible actions
            use_planning: Whether to use mental simulation
            epsilon: Exploration rate
        
        Returns:
            action: Selected action index
        """
        # Exploration
        if random.random() < epsilon:
            return random.choice(available_actions)
        
        # Model-based planning (if enabled and requested)
        if use_planning and self.simulation_coordinator is not None:
            goal_context = self.pfc.get_goal_context()
            best_action, rollout = self.simulation_coordinator.plan_best_action(
                current_state=state,
                available_actions=available_actions,
                goal_context=goal_context
            )
            return best_action
        else:
            # Model-free (existing striatum selection)
            goal_context = self.pfc.get_goal_context()
            return self.striatum.select_action(
                state=state,
                goal_context=goal_context,
                epsilon=0.0  # Already handled exploration above
            )
    
    def store_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ):
        """
        Store experience and trigger learning (including Dyna planning).
        
        This is called automatically by curriculum trainer.
        """
        # Existing: Store in hippocampus
        goal_context = self.pfc.get_goal_context()
        achieved_goal = self.hippocampus.get_achieved_goal()
        
        self.hippocampus.store_episode(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            goal=goal_context,
            achieved_goal=achieved_goal
        )
        
        # NEW: Dyna planning (if enabled)
        if self.dyna_planner is not None:
            self.dyna_planner.process_real_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                goal_context=goal_context
            )
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/unit/test_mental_simulation.py`

```python
def test_coordinator_simulates_rollout():
    """Test that coordinator can simulate action sequences."""
    brain = create_test_brain()
    coordinator = brain.simulation_coordinator
    
    state = torch.randn(256)  # Example state
    actions = [0, 1, 2]  # 3-step sequence
    
    rollout = coordinator.simulate_rollout(state, actions)
    
    assert len(rollout.states) == 4  # Initial + 3 steps
    assert len(rollout.actions) == 3
    assert len(rollout.rewards) == 3
    assert isinstance(rollout.cumulative_value, float)
    assert 0.0 <= rollout.uncertainty <= 1.0

def test_coordinator_uses_hippocampal_memory():
    """Test that planning incorporates past experiences."""
    brain = create_test_brain()
    coordinator = brain.simulation_coordinator
    
    # Store some experiences in hippocampus
    for _ in range(10):
        brain.hippocampus.store_episode(...)
    
    # Simulate - should use retrieved experiences
    state = torch.randn(256)
    rollout = coordinator.simulate_rollout(state, [0])
    
    # Uncertainty should be lower when similar experiences exist
    assert rollout.uncertainty < 0.8

def test_dyna_improves_learning_speed():
    """Test that Dyna planning speeds up learning."""
    brain_without_dyna = create_test_brain(use_dyna=False)
    brain_with_dyna = create_test_brain(use_dyna=True)
    
    # Train both on same task
    for episode in range(100):
        # ... training loop ...
        pass
    
    # Dyna should learn faster (fewer real experiences needed)
    assert brain_with_dyna.performance > brain_without_dyna.performance * 1.3
```

### Integration Tests

**File**: `tests/integration/test_model_based_planning.py`

```python
def test_planning_helps_novel_situations():
    """Test that planning improves performance in novel situations."""
    brain = create_test_brain(use_planning=True)
    
    # Train on environment A
    train_in_environment_a(brain)
    
    # Test in novel environment B
    # Planning should help by simulating ahead
    performance_with_planning = evaluate_in_environment_b(brain, use_planning=True)
    performance_without_planning = evaluate_in_environment_b(brain, use_planning=False)
    
    assert performance_with_planning > performance_without_planning * 1.2

def test_planning_for_grammar_generation():
    """Test that planning improves grammar generation (Stage 2 task)."""
    brain = create_test_brain(use_planning=True)
    
    # Generate sentence with planning
    sentence_with_plan = brain.generate_text(
        prompt="The cat",
        use_planning=True,
        max_length=10
    )
    
    # Generate without planning
    sentence_without_plan = brain.generate_text(
        prompt="The cat",
        use_planning=False,
        max_length=10
    )
    
    # Planning should reduce grammar errors
    errors_with_plan = count_grammar_errors(sentence_with_plan)
    errors_without_plan = count_grammar_errors(sentence_without_plan)
    
    assert errors_with_plan < errors_without_plan
```

---

## Success Criteria

### Phase 2 Complete When:

1. ✅ **Mental Simulation Coordinator**:
   - [ ] `MentalSimulationCoordinator` implemented
   - [ ] Can simulate action sequences using existing regions
   - [ ] Retrieves similar experiences from hippocampus
   - [ ] Uses PFC/Cortex for state prediction
   - [ ] Uses striatum for state evaluation
   - [ ] Tree search finds better actions than greedy (>20% improvement)

2. ✅ **Dyna Planning**:
   - [ ] `DynaPlanner` implemented
   - [ ] Interleaves real and simulated experience
   - [ ] Background planning improves learning speed (>30% fewer episodes)
   - [ ] Prioritized sweeping focuses on important states

3. ✅ **Region Extensions**:
   - [ ] PFC: `predict_next_state()` method added
   - [ ] Hippocampus: `retrieve_similar()` K-NN method added
   - [ ] Striatum: `evaluate_state()` method added
   - [ ] All use existing mechanisms (no new learning)

4. ✅ **Brain Integration**:
   - [ ] `Brain.select_action()` supports `use_planning` flag
   - [ ] `Brain.store_experience()` triggers Dyna planning
   - [ ] Config flag: `use_model_based_planning`

5. ✅ **Testing**:
   - [ ] All unit tests pass (>90% coverage)
   - [ ] Planning improves novel situation performance
   - [ ] Stage 2 grammar task benefits from planning
   - [ ] No regression in existing tests

6. ✅ **Biological Plausibility**:
   - [ ] NO backpropagation anywhere
   - [ ] All learning remains local to regions
   - [ ] Mental simulation uses only region coordination
   - [ ] Reviewed and approved by architecture team

---

## Timeline

### Week 1: Region Extensions (5 days)
- Days 1-2: Add `PFC.predict_next_state()`
- Day 3: Add `Hippocampus.retrieve_similar()`
- Day 4: Add `Striatum.evaluate_state()`
- Day 5: Unit tests for region extensions

### Week 2: Mental Simulation Coordinator (5 days)
- Days 1-3: Implement `MentalSimulationCoordinator`
- Days 4-5: Unit tests and tree search validation

### Week 3: Dyna Integration (5 days)
- Days 1-2: Implement `DynaPlanner`
- Days 3-4: Brain integration
- Day 5: Integration tests

### Week 4: Validation & Documentation (5 days)
- Days 1-2: Stage 2 grammar tests
- Day 3: Novel situation tests
- Day 4: Performance profiling
- Day 5: Documentation and examples

**Total**: 4 weeks (20 days)

---

## What's Next After Phase 2?

**Phase 3: Hierarchical Goals** (see `PHASE3_HIERARCHICAL.md`)
- Goal hierarchies (subgoals)
- Options learning (reusable policies)
- Temporal abstraction

**Stage 2 Curriculum** (can start in parallel with Phase 2!)
- Grammar learning (SVO, case marking)
- Task switching
- Text generation
- Planning will enhance these capabilities when added

---

## References

**Biological Inspiration**:
- Tolman (1948): Cognitive maps in rats and men
- Johnson & Redish (2007): Neural ensembles in CA3 transiently encode paths forward
- Foster & Wilson (2006): Reverse replay of behavioural sequences
- Daw et al. (2005): Uncertainty-based competition between prefrontal and dorsolateral striatal systems

**Computational Models**:
- Sutton (1990): Integrated architectures for learning, planning, and reacting (Dyna)
- Mattar & Daw (2018): Prioritized memory access explains planning and hippocampal replay
- Doll et al. (2015): Model-based choices involve prospective neural activity

**Architecture Review**:
- `docs/design/ARCHITECTURE_REVIEW_BIOLOGICAL_PLAUSIBILITY.md` - Why no WorldModel with backprop

---

**Last Updated**: December 10, 2025
**Status**: Architecture complete, implementation pending
**Next Step**: Implement region extensions (Week 1)
