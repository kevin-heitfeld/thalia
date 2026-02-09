"""
Emergent Goal System - Goals emerge from PFC working memory dynamics.

This module implements emergent goal representations that arise from PFC working
memory attractor dynamics, rather than being explicitly programmed. Goals are
represented as sustained neural activity patterns, with hierarchical structure
emerging from learned transitions between abstract and concrete representations.

**Core Principles**:
===================
1. **Goals = WM Patterns**: Goals are sustained activity patterns in working memory,
   not symbolic objects with metadata.

2. **Hierarchy = Learned Transitions**: Goal decomposition emerges from Hebbian
   associations between abstract (rostral PFC) and concrete (caudal PFC) patterns.

3. **Selection = WM Competition**: Goal selection arises from winner-take-all
   competition between competing WM representations.

4. **Value = Learned Associations**: Goal value is learned through dopamine-gated
   consolidation, not explicitly computed.

**Biological Basis**:
====================
- Miller & Cohen (2001): PFC maintains goal representations as sustained activity
- Badre & D'Esposito (2009): Rostral-caudal hierarchy in PFC organization
- Koechlin & Summerfield (2007): Goals as maintained information, not symbols
- Frey & Morris (1997): Synaptic tagging for consolidation (same as hippocampus)

**Key Mechanisms**:
==================
1. **Synaptic Tagging**: Recently-activated goal patterns get tagged for consolidation
2. **Transition Learning**: Hebbian learning of abstract→concrete associations
3. **Value Learning**: Dopamine-gated strengthening of valuable goal patterns
4. **Winner-Take-All**: Competition between WM patterns for goal selection

This replaces the explicit Goal/GoalHierarchyManager system with emergent dynamics.
"""

from __future__ import annotations

from typing import List

import torch

from thalia.components.synapses import WeightInitializer


class EmergentGoalSystem:
    """Goals emerge from PFC working memory attractor dynamics.

    Biological mechanisms:
    1. Working memory patterns = goals (no symbolic representation)
    2. Recurrent attractors maintain goal representations
    3. Learned transitions = hierarchical decomposition
    4. Dopamine-tagged patterns = high-value goals
    5. Winner-take-all = goal selection

    No explicit Goal objects, no symbolic manipulation.
    Pure neural dynamics.

    Attributes:
        n_wm_neurons: Total number of working memory neurons
        n_abstract: Number of abstract (rostral PFC) neurons
        n_concrete: Number of concrete (caudal PFC) neurons
        device: Device for tensors ('cpu' or 'cuda')
        abstract_neurons: Indices of abstract neuron population
        concrete_neurons: Indices of concrete neuron population
        transition_weights: Learned associations [n_concrete, n_abstract]
        value_weights: Value associations [n_wm_neurons]
        goal_tags: Synaptic tags for consolidation [n_wm_neurons]
        tag_decay: Decay rate for tags (default 0.95)
    """

    def __init__(
        self,
        n_wm_neurons: int,
        n_abstract: int,  # Rostral PFC (slow, abstract)
        n_concrete: int,  # Caudal PFC (fast, concrete)
        device: str,
    ):
        """Initialize emergent goal system.

        Args:
            n_wm_neurons: Total number of working memory neurons
            n_abstract: Number of abstract (rostral) neurons with long time constants
            n_concrete: Number of concrete (caudal) neurons with short time constants
            device: Device for tensors ('cpu' or 'cuda')
        """
        self.n_wm_neurons = n_wm_neurons
        self.n_abstract = n_abstract
        self.n_concrete = n_concrete
        self.device = device

        # Split PFC neurons into abstract/concrete populations
        # Abstract: long time constants (tau ~500ms), slow update
        # Concrete: short time constants (tau ~100ms), fast update
        self.abstract_neurons = torch.arange(n_abstract, device=device)
        self.concrete_neurons = torch.arange(n_abstract, n_abstract + n_concrete, device=device)

        # Goal transition matrix: learned associations between WM patterns
        # "When abstract pattern A is active, which concrete pattern B follows?"
        # This replaces explicit goal decomposition
        self.transition_weights = torch.zeros(
            n_concrete, n_abstract, device=device
        )  # [concrete, abstract]

        # Goal value associations: OFC-like value mapping
        # Maps WM patterns to expected value (learned from experience)
        self.value_weights = WeightInitializer.gaussian(
            n_output=n_wm_neurons, n_input=1, mean=0.0, std=0.1, device=device
        ).squeeze()

        # Synaptic tags for goal patterns (Frey-Morris)
        # Recently-activated goals get tagged for consolidation
        self.goal_tags = torch.zeros(n_wm_neurons, device=device)
        self.tag_decay = 0.95  # Same as hippocampal tags

    def update_goal_tags(
        self,
        wm_pattern: torch.Tensor,  # [n_wm_neurons] - current WM activity
    ) -> None:
        """Tag currently active goal patterns.

        Similar to hippocampal synaptic tagging:
        - Active patterns get tagged
        - Tags decay over time
        - Dopamine consolidates tagged patterns

        Args:
            wm_pattern: Current working memory activity [n_wm_neurons]
        """
        # Decay existing tags
        self.goal_tags *= self.tag_decay

        # Tag current pattern (threshold to avoid noise)
        active_mask = wm_pattern > 0.3
        self.goal_tags[active_mask] = torch.maximum(
            self.goal_tags[active_mask], wm_pattern[active_mask]
        )

    def predict_subgoal(
        self,
        abstract_pattern: torch.Tensor,  # [n_abstract] - current abstract goal
    ) -> torch.Tensor:
        """Predict concrete subgoal from abstract goal.

        This is emergent goal decomposition:
        - No explicit Goal.subgoals list
        - Learned from experience (Hebbian)
        - Pattern completion via associative memory

        Replaces: GoalHierarchyManager.decompose_goal()

        Args:
            abstract_pattern: Current abstract goal pattern [n_abstract]

        Returns:
            Predicted concrete subgoal pattern [n_concrete]
        """
        # Project abstract pattern to concrete space via learned transitions
        # transition_weights[n_concrete, n_abstract] @ abstract[n_abstract]
        predicted_concrete = self.transition_weights @ abstract_pattern

        # Winner-take-all: Select most strongly predicted subgoal pattern
        # In biology: lateral inhibition in PFC
        k = max(1, int(self.n_concrete * 0.1))  # Top 10% of concrete neurons
        _, top_indices = torch.topk(predicted_concrete, k)

        subgoal_pattern = torch.zeros(self.n_concrete, device=self.device)
        subgoal_pattern[top_indices] = 1.0

        return subgoal_pattern

    def select_goal(
        self,
        competing_patterns: List[torch.Tensor],  # Candidate WM patterns
        dopamine: float,
    ) -> torch.Tensor:
        """Select goal from competing options (emergent selection).

        Selection based on:
        1. Learned value (value_weights)
        2. Recency (goal_tags)
        3. Dopamine modulation (urgency/motivation)

        Replaces: GoalHierarchyManager.select_active_goal()

        Args:
            competing_patterns: List of candidate WM patterns [n_wm_neurons]
            dopamine: Current dopamine level (modulates selection)

        Returns:
            Selected goal pattern [n_wm_neurons]
        """
        if len(competing_patterns) == 0:
            return torch.zeros(self.n_wm_neurons, device=self.device)

        # Compute selection scores for each pattern
        scores = []
        for pattern in competing_patterns:
            # Value component: learned from experience
            value_score = torch.sum(pattern * self.value_weights)

            # Recency component: recently-activated goals
            recency_score = torch.sum(pattern * self.goal_tags)

            # Dopamine modulation: high DA → boost urgent/rewarded goals
            da_modulation = 1.0 + dopamine * 0.5

            # Combined score
            total_score = (value_score + recency_score) * da_modulation
            scores.append(total_score.item())

        # Softmax selection (soft winner-take-all)
        scores_tensor = torch.tensor(scores, device=self.device)
        probs = torch.softmax(scores_tensor / 0.1, dim=0)  # Temperature 0.1

        # Sample from distribution
        selected_idx = torch.multinomial(probs, 1).item()
        return competing_patterns[selected_idx]

    def learn_transition(
        self,
        abstract_pattern: torch.Tensor,  # [n_abstract] - parent goal
        concrete_pattern: torch.Tensor,  # [n_concrete] - subgoal that followed
        learning_rate: float = 0.01,
    ) -> None:
        """Learn goal hierarchies via Hebbian association.

        When abstract goal A leads to concrete subgoal B:
        - Strengthen transition A→B
        - This is how "decomposition" is learned, not programmed

        Replaces: Explicit Goal.add_subgoal()

        Args:
            abstract_pattern: Parent goal pattern [n_abstract]
            concrete_pattern: Subgoal pattern that followed [n_concrete]
            learning_rate: Learning rate for Hebbian update (default 0.01)
        """
        # Outer product: strengthen associations between co-active patterns
        # transition[concrete, abstract] += lr * concrete × abstract
        dW = learning_rate * torch.outer(concrete_pattern, abstract_pattern)
        self.transition_weights += dW

        # Normalize to prevent runaway weights
        self.transition_weights.clamp_(0.0, 1.0)

    def consolidate_valuable_goals(
        self,
        dopamine: float,
        learning_rate: float = 0.01,
    ) -> None:
        """Strengthen value associations for tagged goals.

        Similar to hippocampal consolidation:
        - Tagged patterns (recent goals) get consolidated with dopamine
        - High dopamine → high value goals become more prominent

        Args:
            dopamine: Current dopamine level (gates consolidation)
            learning_rate: Learning rate for value update (default 0.01)
        """
        if dopamine < 0.01:
            return  # No consolidation without dopamine

        # Strengthen value weights for tagged goals
        # High tag + high DA → increase value association
        value_update = learning_rate * dopamine * self.goal_tags
        self.value_weights += value_update

        # Keep values bounded
        self.value_weights.clamp_(-1.0, 1.0)
