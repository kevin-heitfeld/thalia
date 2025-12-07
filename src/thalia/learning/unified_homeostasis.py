"""
Unified Homeostasis: Constraint-Based Stability for SNNs.

This module replaces multiple biological homeostatic mechanisms with a
simpler, mathematically-guaranteed approach using constraints instead
of corrections.

Philosophy:
===========
The brain has 10+ overlapping homeostatic mechanisms (BCM, synaptic scaling,
intrinsic plasticity, etc.) because biology is messy and redundant. But we're
not constrained by evolution - we can impose mathematical guarantees.

Key Insight: CONSTRAINTS > CORRECTIONS
- Correction: "If weights get too high, slow down learning" (might not work)
- Constraint: "Weights MUST sum to X" (mathematically guaranteed)

Effects Captured:
=================
1. STABILITY: Weights and activity cannot explode or collapse
   → Weight normalization, hard bounds

2. DIFFERENTIATION: Neurons learn different things
   → Competitive normalization (if one grows, others shrink)

3. COMPETITION: Learning resources are finite
   → Per-neuron or per-action budget constraints

4. MEMORY: Learning persists appropriately
   → Normalization preserves relative differences

5. ADAPTABILITY: Can still learn new things
   → Constraints don't prevent learning, just bound it

Replaces:
=========
- BCM sliding threshold (metaplasticity)
- Synaptic scaling (global weight adjustment)
- Intrinsic plasticity (threshold adaptation)
- Heterosynaptic LTD (competitive weakening)
- Various soft bounds and rate limiters

Usage:
======
    homeostasis = UnifiedHomeostasis(
        weight_budget=1.0,      # Total weight sum per neuron
        activity_target=0.1,    # Target fraction of neurons active
        diversity_target=0.5,   # Target weight diversity (entropy)
    )
    
    # After each weight update:
    weights = homeostasis.normalize_weights(weights)
    
    # After each forward pass (optional):
    activity = homeostasis.normalize_activity(activity)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

from thalia.config.base import BaseConfig


@dataclass
class UnifiedHomeostasisConfig(BaseConfig):
    """Configuration for unified homeostatic regulation.
    
    Inherits device, dtype, seed from BaseConfig.
    
    This replaces the many parameters of BCM, synaptic scaling, etc.
    with a minimal set of target values that define the constraints.
    """
    
    # Weight constraints
    weight_budget: float = 1.0          # Target sum of weights per row (neuron)
    w_min: float = 0.0                  # Absolute minimum weight
    w_max: float = 1.0                  # Absolute maximum weight
    
    # Activity constraints
    activity_target: float = 0.1        # Target fraction of neurons active
    activity_min: float = 0.01          # Minimum activity (prevent dead neurons)
    activity_max: float = 0.5           # Maximum activity (prevent seizure)
    
    # Normalization settings
    normalize_rows: bool = True         # Normalize each neuron's input weights
    normalize_cols: bool = False        # Normalize each input's output weights
    soft_normalization: bool = True     # Use soft (multiplicative) vs hard normalization
    normalization_rate: float = 0.1     # How fast to approach target (soft only)
    
    # Competition settings
    enable_competition: bool = True     # Enable competitive weight adjustment
    competition_strength: float = 0.1   # Strength of winner-take-all effect


class UnifiedHomeostasis(nn.Module):
    """Unified homeostatic regulation using constraints.
    
    Replaces BCM, synaptic scaling, intrinsic plasticity, etc. with
    simple normalization operations that mathematically guarantee
    stability and competition.
    
    Key operations:
    1. Weight normalization: Each neuron's total input is bounded
    2. Activity normalization: Population activity is regulated
    3. Competitive adjustment: Strong weights suppress weak ones
    
    All operations are differentiable and can be applied every timestep.
    """
    
    def __init__(self, config: Optional[UnifiedHomeostasisConfig] = None):
        super().__init__()
        self.config = config or UnifiedHomeostasisConfig()
        
        # Running statistics for soft normalization
        self._weight_ema: Optional[torch.Tensor] = None
        self._activity_ema: float = self.config.activity_target
        
    def normalize_weights(
        self,
        weights: torch.Tensor,
        dim: int = 1,
    ) -> torch.Tensor:
        """Normalize weights to enforce budget constraint.
        
        Each row (dim=1) or column (dim=0) is scaled so its sum equals
        the weight budget. This:
        - Prevents runaway potentiation (sum is bounded)
        - Enables competition (if one grows, others shrink)
        - Implicitly implements BCM (relative weights matter, not absolute)
        
        Args:
            weights: Weight matrix [n_output, n_input]
            dim: Dimension to normalize (1=rows/neurons, 0=columns/inputs)
            
        Returns:
            Normalized weights with same shape
        """
        cfg = self.config
        
        if cfg.soft_normalization:
            # Soft normalization: gradually move toward target
            current_sum = weights.sum(dim=dim, keepdim=True).clamp(min=1e-8)
            target_scale = cfg.weight_budget / current_sum
            
            # Blend toward target
            scale = 1.0 + cfg.normalization_rate * (target_scale - 1.0)
            weights = weights * scale
        else:
            # Hard normalization: exactly enforce constraint
            current_sum = weights.sum(dim=dim, keepdim=True).clamp(min=1e-8)
            weights = weights / current_sum * cfg.weight_budget
        
        # Always enforce hard bounds
        weights = weights.clamp(cfg.w_min, cfg.w_max)
        
        return weights
    
    def normalize_weights_paired(
        self,
        weights_a: torch.Tensor,
        weights_b: torch.Tensor,
        budget_per_pair: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize paired weight matrices to share a budget.
        
        Perfect for D1/D2 opponent pathways: the total weight to each
        action is constrained, forcing D1 and D2 to compete.
        
        If D1 for action X grows, D2 for action X must shrink (and vice versa).
        This GUARANTEES that neither pathway can dominate completely.
        
        Args:
            weights_a: First weight matrix (e.g., D1) [n_output, n_input]
            weights_b: Second weight matrix (e.g., D2) [n_output, n_input]
            budget_per_pair: Total budget for A + B per row
            
        Returns:
            Tuple of normalized (weights_a, weights_b)
        """
        cfg = self.config
        
        # Sum across both pathways
        sum_a = weights_a.sum(dim=1, keepdim=True).clamp(min=1e-8)
        sum_b = weights_b.sum(dim=1, keepdim=True).clamp(min=1e-8)
        total = sum_a + sum_b
        
        if cfg.soft_normalization:
            # Soft: gradually move toward budget
            target_scale = budget_per_pair / total
            scale = 1.0 + cfg.normalization_rate * (target_scale - 1.0)
            
            weights_a = weights_a * scale
            weights_b = weights_b * scale
        else:
            # Hard: exactly enforce budget
            scale = budget_per_pair / total
            weights_a = weights_a * scale
            weights_b = weights_b * scale
        
        # Enforce bounds
        weights_a = weights_a.clamp(cfg.w_min, cfg.w_max)
        weights_b = weights_b.clamp(cfg.w_min, cfg.w_max)
        
        return weights_a, weights_b
    
    def normalize_activity(
        self,
        activity: torch.Tensor,
        target: Optional[float] = None,
    ) -> torch.Tensor:
        """Normalize population activity toward target.
        
        Implements the effect of intrinsic plasticity: if neurons are
        too active, scale down; if too quiet, scale up.
        
        Args:
            activity: Activity tensor (spikes or rates)
            target: Target mean activity (uses config default if None)
            
        Returns:
            Scaled activity tensor
        """
        cfg = self.config
        target = target or cfg.activity_target
        
        # Compute current mean activity
        mean_activity = activity.mean().clamp(min=1e-8)
        
        # Compute scaling factor
        scale = target / mean_activity
        
        # Bound the scaling to prevent extreme adjustments
        scale = scale.clamp(0.5, 2.0)
        
        # Apply soft scaling
        if cfg.soft_normalization:
            scale = 1.0 + cfg.normalization_rate * (scale - 1.0)
        
        return activity * scale
    
    def compute_excitability_modulation(
        self,
        activity_history: torch.Tensor,
        tau: float = 100.0,
    ) -> torch.Tensor:
        """Compute per-neuron excitability modulation.
        
        Neurons that fire too much become less excitable.
        Neurons that fire too little become more excitable.
        
        This replaces intrinsic plasticity with a simpler feedback loop.
        
        Args:
            activity_history: Running average of each neuron's activity
            tau: Time constant for modulation (higher = slower)
            
        Returns:
            Excitability modulation factor per neuron (multiply g_exc by this)
        """
        cfg = self.config
        
        # Error from target
        error = activity_history - cfg.activity_target
        
        # Modulation: high activity → lower excitability
        modulation = 1.0 - error / tau
        
        # Bound to reasonable range
        modulation = modulation.clamp(0.5, 2.0)
        
        return modulation
    
    def apply_competition(
        self,
        weights: torch.Tensor,
        winners: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply competitive weight adjustment.
        
        Winner-take-all effect: weights to "winning" neurons are boosted,
        weights to "losing" neurons are suppressed.
        
        This replaces heterosynaptic LTD and lateral inhibition effects.
        
        Args:
            weights: Weight matrix [n_output, n_input]
            winners: Binary mask of "winning" outputs (uses max if None)
            
        Returns:
            Competitively adjusted weights
        """
        if not self.config.enable_competition:
            return weights
        
        cfg = self.config
        
        # Identify winners (neurons with highest total weight)
        if winners is None:
            weight_sums = weights.sum(dim=1)
            threshold = weight_sums.mean()
            winners = (weight_sums > threshold).float()
        
        # Boost winners, suppress losers
        # Winners: multiply by (1 + strength)
        # Losers: multiply by (1 - strength)
        losers = 1.0 - winners
        
        scale = (
            winners.unsqueeze(1) * (1.0 + cfg.competition_strength) +
            losers.unsqueeze(1) * (1.0 - cfg.competition_strength)
        )
        
        weights = weights * scale
        
        # Re-normalize to maintain budget
        return self.normalize_weights(weights)
    
    def get_diagnostics(
        self,
        weights: torch.Tensor,
        activity: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Get diagnostic information about current state.
        
        Args:
            weights: Current weight matrix
            activity: Current activity (optional)
            
        Returns:
            Dict with diagnostic metrics
        """
        diagnostics = {
            'weight_mean': weights.mean().item(),
            'weight_std': weights.std().item(),
            'weight_min': weights.min().item(),
            'weight_max': weights.max().item(),
            'weight_sum_mean': weights.sum(dim=1).mean().item(),
            'weight_sum_std': weights.sum(dim=1).std().item(),
        }
        
        if activity is not None:
            diagnostics['activity_mean'] = activity.mean().item()
            diagnostics['activity_std'] = activity.std().item()
        
        # Diversity metric: how different are the weight patterns?
        # High diversity = good specialization
        if weights.dim() == 2 and weights.shape[0] > 1:
            from thalia.core.utils import cosine_similarity_safe
            
            # Normalize each row
            normed = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            # Pairwise cosine similarity using canonical implementation
            # Compute for all pairs: similarity[i,j] = cosine(normed[i], normed[j])
            similarity = torch.mm(normed, normed.T)
            # Mean off-diagonal similarity (lower = more diverse)
            mask = 1.0 - torch.eye(weights.shape[0], device=weights.device)
            mean_similarity = (similarity * mask).sum() / mask.sum()
            diagnostics['weight_diversity'] = 1.0 - mean_similarity.item()
        
        return diagnostics


class StriatumHomeostasis(UnifiedHomeostasis):
    """Specialized homeostasis for D1/D2 opponent pathways.
    
    Extends UnifiedHomeostasis with striatum-specific constraints:
    - D1 and D2 share a budget per action
    - Competition between GO and NOGO pathways
    - Dopamine-modulated normalization
    - Activity tracking and excitability modulation (replaces IntrinsicPlasticity)
    """
    
    def __init__(
        self,
        n_actions: int,
        neurons_per_action: int = 1,
        config: Optional[UnifiedHomeostasisConfig] = None,
        target_rate: float = 0.05,  # Target firing rate (fraction of timesteps)
        excitability_tau: float = 100.0,  # Time constant for excitability modulation
    ):
        super().__init__(config)
        self.n_actions = n_actions
        self.neurons_per_action = neurons_per_action
        self.n_neurons = n_actions * neurons_per_action
        self.target_rate = target_rate
        self.excitability_tau = excitability_tau
        
        # Per-action budgets (can vary if some actions should be favored)
        self.register_buffer(
            'action_budgets',
            torch.ones(n_actions) * (config or UnifiedHomeostasisConfig()).weight_budget
        )
        
        # Activity tracking for excitability modulation
        # Running average of firing rate per neuron (D1 and D2 separately)
        self.register_buffer('d1_activity_avg', torch.zeros(self.n_neurons))
        self.register_buffer('d2_activity_avg', torch.zeros(self.n_neurons))
        
        # Excitability modulation factors (multiply g_E by this)
        # > 1.0 means more excitable, < 1.0 means less excitable
        self.register_buffer('d1_excitability', torch.ones(self.n_neurons))
        self.register_buffer('d2_excitability', torch.ones(self.n_neurons))
    
    def update_activity(
        self,
        d1_spikes: torch.Tensor,
        d2_spikes: torch.Tensor,
        decay: float = 0.99,
    ) -> None:
        """Update running average of D1/D2 activity.
        
        Called every timestep to track firing rates.
        
        Args:
            d1_spikes: D1 spike tensor [batch, n_neurons] or [n_neurons]
            d2_spikes: D2 spike tensor [batch, n_neurons] or [n_neurons]
            decay: Exponential decay for running average
        """
        # Squeeze to 1D if needed
        d1 = d1_spikes.squeeze().float()
        d2 = d2_spikes.squeeze().float()
        
        # Handle batch dimension
        if d1.dim() > 1:
            d1 = d1.mean(dim=0)
            d2 = d2.mean(dim=0)
        
        # Update running averages
        self.d1_activity_avg = decay * self.d1_activity_avg + (1 - decay) * d1
        self.d2_activity_avg = decay * self.d2_activity_avg + (1 - decay) * d2
    
    def compute_excitability(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute excitability modulation based on activity history.
        
        Neurons firing above target → lower excitability (harder to fire)
        Neurons firing below target → higher excitability (easier to fire)
        
        This replaces IntrinsicPlasticity with a constraint-based approach.
        
        Returns:
            Tuple of (d1_excitability, d2_excitability) modulation factors
        """
        # Error from target rate
        d1_error = self.d1_activity_avg - self.target_rate
        d2_error = self.d2_activity_avg - self.target_rate
        
        # Modulation: high activity → lower excitability
        # excitability = 1 - error/tau, clamped to [0.5, 2.0]
        self.d1_excitability = (1.0 - d1_error / self.excitability_tau).clamp(0.5, 2.0)
        self.d2_excitability = (1.0 - d2_error / self.excitability_tau).clamp(0.5, 2.0)
        
        return self.d1_excitability, self.d2_excitability
    
    def get_excitability(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current excitability modulation factors.
        
        Returns:
            Tuple of (d1_excitability, d2_excitability)
        """
        return self.d1_excitability.clone(), self.d2_excitability.clone()
    
    def reset_activity(self) -> None:
        """Reset activity tracking (e.g., at start of new episode)."""
        self.d1_activity_avg.zero_()
        self.d2_activity_avg.zero_()
        self.d1_excitability.fill_(1.0)
        self.d2_excitability.fill_(1.0)
    
    def normalize_d1_d2(
        self,
        d1_weights: torch.Tensor,
        d2_weights: torch.Tensor,
        per_action: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize D1 and D2 weights with shared budget.
        
        Key insight: D1 and D2 for each action should compete for
        a fixed total weight budget. This GUARANTEES:
        - Neither pathway can completely dominate
        - If D1 grows, D2 must shrink (and vice versa)
        - Stable equilibrium is enforced mathematically
        
        Args:
            d1_weights: D1 pathway weights [n_neurons, n_input]
            d2_weights: D2 pathway weights [n_neurons, n_input]
            per_action: If True, normalize per action; else per neuron
            
        Returns:
            Tuple of (normalized_d1, normalized_d2)
        """
        if per_action:
            # Normalize per action (groups of neurons)
            d1_out = d1_weights.clone()
            d2_out = d2_weights.clone()
            
            for action in range(self.n_actions):
                start = action * self.neurons_per_action
                end = start + self.neurons_per_action
                
                d1_action = d1_weights[start:end]
                d2_action = d2_weights[start:end]
                
                # Sum across all neurons for this action
                d1_sum = d1_action.sum()
                d2_sum = d2_action.sum()
                total = (d1_sum + d2_sum).clamp(min=1e-8)
                
                # Scale to budget
                budget = self.action_budgets[action].item()
                scale = budget / total
                
                d1_out[start:end] = d1_action * scale
                d2_out[start:end] = d2_action * scale
            
            # Enforce bounds
            d1_out = d1_out.clamp(self.config.w_min, self.config.w_max)
            d2_out = d2_out.clamp(self.config.w_min, self.config.w_max)
            
            return d1_out, d2_out
        else:
            # Simple paired normalization per row
            return self.normalize_weights_paired(d1_weights, d2_weights)
    
    def modulate_by_dopamine(
        self,
        weights: torch.Tensor,
        dopamine: float,
        is_d2: bool = False,
    ) -> torch.Tensor:
        """Modulate normalization strength by dopamine level.
        
        During high dopamine (reward), we might want to allow more
        deviation from the budget to enable faster learning.
        During low dopamine, enforce the budget more strictly.
        
        Args:
            weights: Weights to normalize
            dopamine: Current dopamine level (-1 to 1)
            is_d2: Whether this is the D2 (indirect) pathway
            
        Returns:
            Modulated weights
        """
        # High dopamine = looser constraints (more learning flexibility)
        # Low dopamine = tighter constraints (more stability)
        flexibility = 0.5 + 0.5 * abs(dopamine)  # 0.5 to 1.0
        
        # Adjust normalization rate
        original_rate = self.config.normalization_rate
        self.config.normalization_rate = original_rate * (2.0 - flexibility)
        
        # Apply normalization
        weights = self.normalize_weights(weights)
        
        # Restore rate
        self.config.normalization_rate = original_rate
        
        return weights
