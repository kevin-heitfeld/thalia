"""
Learning Manager for Striatum

Encapsulates D1/D2 opponent learning logic including:
1. Three-factor learning (eligibility × dopamine)
2. Counterfactual learning (What if...?)
3. Goal-modulated learning (PFC → Striatum)
4. Eligibility trace management

This module extracts learning logic from the Striatum god object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any

import torch
import torch.nn as nn

from thalia.core.base_manager import BaseManager, ManagerContext
from thalia.core.neuron_constants import WEIGHT_INIT_SCALE_SMALL

if TYPE_CHECKING:
    from thalia.regions.striatum.d1_pathway import D1Pathway
    from thalia.regions.striatum.d2_pathway import D2Pathway
    from thalia.regions.striatum.config import StriatumConfig


class LearningManager(BaseManager["StriatumConfig"]):
    """Manages three-factor dopamine-modulated learning for D1/D2 pathways.
    
    Implements the biological three-factor learning rule:
    Δw = eligibility_trace × dopamine_signal
    
    Key features:
    - D1 pathway: DA+ → LTP (strengthen GO), DA- → LTD (weaken GO)
    - D2 pathway: DA+ → LTD (weaken NOGO), DA- → LTP (strengthen NOGO)
    - Goal modulation: PFC context gates learning per action
    - Counterfactual learning: Learn from what could have happened
    """
    
    def __init__(
        self,
        config: StriatumConfig,
        context: ManagerContext,
        d1_pathway: D1Pathway,
        d2_pathway: D2Pathway,
    ):
        """Initialize learning manager.
        
        Args:
            config: Striatum configuration
            context: Manager context (device, dimensions, etc.)
            d1_pathway: D1-MSN pathway (direct/Go)
            d2_pathway: D2-MSN pathway (indirect/NoGo)
        """
        super().__init__(config, context)
        self.d1_pathway = d1_pathway
        self.d2_pathway = d2_pathway
        
        # Goal modulation weights (if enabled)
        if config.use_goal_conditioning:
            # PFC→D1/D2 modulation matrices [n_output, pfc_size]
            # These learn which striatal neurons participate in which goals
            self.pfc_modulation_d1 = nn.Parameter(
                torch.randn(config.n_output, config.pfc_size, device=self.context.device) * WEIGHT_INIT_SCALE_SMALL
            )
            self.pfc_modulation_d2 = nn.Parameter(
                torch.randn(config.n_output, config.pfc_size, device=self.context.device) * WEIGHT_INIT_SCALE_SMALL
            )
        else:
            self.pfc_modulation_d1 = None
            self.pfc_modulation_d2 = None
        
        # Track last spikes for goal modulation learning
        self._last_d1_spikes: Optional[torch.Tensor] = None
        self._last_d2_spikes: Optional[torch.Tensor] = None
    
    def apply_dopamine_learning(
        self,
        dopamine: float,
        goal_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Apply dopamine-modulated learning to D1/D2 pathways.
        
        Uses three-factor rule: Δw = eligibility × dopamine
        
        D1 and D2 have OPPOSITE responses:
        - D1: DA+ → LTP (strengthen), DA- → LTD (weaken)  
        - D2: DA+ → LTD (weaken), DA- → LTP (strengthen)
        
        Args:
            dopamine: Dopamine level (RPE, typically -1 to +1)
            goal_context: PFC goal representation [n_pfc_goal_neurons] (optional)
            
        Returns:
            Dict with learning metrics
        """
        # Apply dopamine modulation to pathways
        d1_metrics = self.d1_pathway.apply_dopamine_modulation(
            dopamine=dopamine,
            heterosynaptic_ratio=self.config.heterosynaptic_ratio,
        )
        
        d2_metrics = self.d2_pathway.apply_dopamine_modulation(
            dopamine=dopamine,
            heterosynaptic_ratio=self.config.heterosynaptic_ratio,
        )
        
        # Apply goal modulation if enabled
        if self.config.use_goal_conditioning and goal_context is not None:
            self._apply_goal_modulation(dopamine, goal_context)
        
        return {
            "d1_ltp": d1_metrics.get("ltp", 0.0),
            "d1_ltd": d1_metrics.get("ltd", 0.0),
            "d2_ltp": d2_metrics.get("ltp", 0.0),
            "d2_ltd": d2_metrics.get("ltd", 0.0),
            "net_change": (
                d1_metrics.get("ltp", 0.0) +
                d1_metrics.get("ltd", 0.0) +
                d2_metrics.get("ltp", 0.0) +
                d2_metrics.get("ltd", 0.0)
            ),
        }
    
    def _apply_goal_modulation(
        self,
        dopamine: float,
        goal_context: torch.Tensor,
    ) -> None:
        """Modulate learning based on current goal from PFC.
        
        Only neurons relevant to the current goal learn strongly.
        This implements hierarchical RL: goals gate which actions learn.
        
        Args:
            dopamine: Current dopamine level
            goal_context: PFC goal representation [n_pfc_goal_neurons]
        """
        # Compute goal-relevance weights for D1/D2
        # High values mean this neuron participates in current goal
        goal_weight_d1 = torch.sigmoid(
            torch.matmul(self.pfc_modulation_d1, goal_context)
        )
        goal_weight_d2 = torch.sigmoid(
            torch.matmul(self.pfc_modulation_d2, goal_context)
        )
        
        # Modulate recently applied weight updates
        # (This should be called AFTER apply_dopamine_modulation)
        # The pathway already updated weights, we retroactively scale them
        # NOTE: This is a simplified version - in practice we'd want to
        # integrate this into the pathway's apply_dopamine_modulation
        
        # Update PFC modulation weights via Hebbian learning
        if abs(dopamine) > 0.01:
            pfc_lr = self.config.goal_modulation_lr
            
            # D1 modulation learning
            if self._last_d1_spikes is not None:
                d1_hebbian = torch.outer(
                    self._last_d1_spikes.float(),
                    goal_context
                ) * dopamine * pfc_lr
                self.pfc_modulation_d1.data += d1_hebbian
                self.pfc_modulation_d1.data.clamp_(0.0, 1.0)
            
            # D2 modulation learning (inverted DA response)
            if self._last_d2_spikes is not None:
                d2_hebbian = torch.outer(
                    self._last_d2_spikes.float(),
                    goal_context
                ) * (-dopamine) * pfc_lr
                self.pfc_modulation_d2.data += d2_hebbian
                self.pfc_modulation_d2.data.clamp_(0.0, 1.0)
    
    def apply_counterfactual_learning(
        self,
        counterfactual_dopamine: float,
        chosen_action: int,
        neurons_per_action: int,
    ) -> Dict[str, Any]:
        """Apply counterfactual learning: "What if I had chosen differently?"
        
        Uses the SAME eligibility traces (from actual action) but with
        DIFFERENT dopamine (from counterfactual outcome).
        
        This allows learning from hypothetical scenarios without
        having to execute them.
        
        Args:
            counterfactual_dopamine: DA signal for counterfactual outcome
            chosen_action: The action that was actually taken
            neurons_per_action: Neurons per action (for masking)
            
        Returns:
            Dict with counterfactual learning metrics
        """
        # Create mask for chosen action neurons
        start = chosen_action * neurons_per_action
        end = start + neurons_per_action
        
        # Save original eligibility
        d1_elig_orig = self.d1_pathway.eligibility.clone()
        d2_elig_orig = self.d2_pathway.eligibility.clone()
        
        # Zero out eligibility for chosen action (already learned from actual outcome)
        self.d1_pathway.eligibility[start:end, :] = 0.0
        self.d2_pathway.eligibility[start:end, :] = 0.0
        
        # Apply counterfactual dopamine to remaining actions
        cf_metrics = self.apply_dopamine_learning(
            dopamine=counterfactual_dopamine,
            goal_context=None,  # No goal modulation for counterfactual
        )
        
        # Restore original eligibility
        self.d1_pathway.eligibility = d1_elig_orig
        self.d2_pathway.eligibility = d2_elig_orig
        
        return {
            "cf_d1_ltp": cf_metrics["d1_ltp"],
            "cf_d1_ltd": cf_metrics["d1_ltd"],
            "cf_d2_ltp": cf_metrics["d2_ltp"],
            "cf_d2_ltd": cf_metrics["d2_ltd"],
            "cf_net_change": cf_metrics["net_change"],
        }
    
    def reset_eligibility(self, action_only: bool = True, action: Optional[int] = None, neurons_per_action: int = 1) -> None:
        """Reset eligibility traces after learning.
        
        Args:
            action_only: If True, only reset the chosen action's traces
            action: Which action to reset (required if action_only=True)
            neurons_per_action: Neurons per action for indexing
        """
        if action_only and action is not None:
            start = action * neurons_per_action
            end = start + neurons_per_action
            self.d1_pathway.eligibility[start:end, :] = 0.0
            self.d2_pathway.eligibility[start:end, :] = 0.0
        else:
            self.d1_pathway.eligibility.zero_()
            self.d2_pathway.eligibility.zero_()
    
    def store_spikes(self, d1_spikes: torch.Tensor, d2_spikes: torch.Tensor) -> None:
        """Store recent spikes for goal modulation learning.
        
        Args:
            d1_spikes: D1 pathway spikes
            d2_spikes: D2 pathway spikes
        """
        self._last_d1_spikes = d1_spikes
        self._last_d2_spikes = d2_spikes
    
    def reset_state(self) -> None:
        """Reset learning manager state (trial boundaries)."""
        self._last_d1_spikes = None
        self._last_d2_spikes = None
        # Reset pathway eligibility traces
        self.reset_eligibility(action_only=False)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic metrics for monitoring.
        
        Returns:
            Dict with learning-related metrics
        """
        diagnostics = {
            "d1_eligibility_mean": self.d1_pathway.eligibility.mean().item(),
            "d1_eligibility_max": self.d1_pathway.eligibility.max().item(),
            "d2_eligibility_mean": self.d2_pathway.eligibility.mean().item(),
            "d2_eligibility_max": self.d2_pathway.eligibility.max().item(),
        }
        
        if self.pfc_modulation_d1 is not None:
            diagnostics["pfc_modulation_d1_mean"] = self.pfc_modulation_d1.mean().item()
            diagnostics["pfc_modulation_d2_mean"] = self.pfc_modulation_d2.mean().item()
        
        return diagnostics
    
    def to(self, device: torch.device) -> "LearningManager":
        """Move all tensors to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self.context.device = device
        
        if self.pfc_modulation_d1 is not None:
            self.pfc_modulation_d1.data = self.pfc_modulation_d1.data.to(device)
            self.pfc_modulation_d2.data = self.pfc_modulation_d2.data.to(device)
        
        if self._last_d1_spikes is not None:
            self._last_d1_spikes = self._last_d1_spikes.to(device)
        if self._last_d2_spikes is not None:
            self._last_d2_spikes = self._last_d2_spikes.to(device)
        
        return self
    
    def get_state(self) -> Dict[str, Any]:
        """Get learning manager state for checkpointing.
        
        Returns:
            Dict with current state
        """
        state = {}
        
        if self.pfc_modulation_d1 is not None:
            state["pfc_modulation_d1"] = self.pfc_modulation_d1.detach().clone()
            state["pfc_modulation_d2"] = self.pfc_modulation_d2.detach().clone()
        
        return state
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore learning manager state from checkpoint.
        
        Args:
            state: Dict from get_state()
        """
        if "pfc_modulation_d1" in state and self.pfc_modulation_d1 is not None:
            self.pfc_modulation_d1.data = state["pfc_modulation_d1"].to(self.context.device)
            self.pfc_modulation_d2.data = state["pfc_modulation_d2"].to(self.context.device)
