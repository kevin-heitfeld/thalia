"""
Checkpoint Manager for Striatum

Handles state serialization and deserialization for checkpointing.
Separates the complexity of state management from core Striatum logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, List

import torch

if TYPE_CHECKING:
    from thalia.regions.striatum.striatum import Striatum


class CheckpointManager:
    """Manages state checkpointing for Striatum.
    
    Handles:
    - Full state serialization (weights, eligibility, exploration, etc.)
    - State restoration with backward compatibility
    - Version migration for old checkpoint formats
    """
    
    def __init__(self, striatum: Striatum):
        """Initialize checkpoint manager.
        
        Args:
            striatum: The Striatum instance to manage
        """
        self.striatum = striatum
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get complete striatum state for checkpointing.
        
        Returns:
            Dict containing all state needed to restore striatum
        """
        s = self.striatum
        
        # 1. NEURON STATE
        neuron_state = {
            "membrane_potential": (
                s.d1_neurons.membrane.detach().clone()
                if s.d1_neurons is not None and s.d1_neurons.membrane is not None
                else None
            ),
            "n_output": s.config.n_output,
            "n_input": s.config.n_input,
        }
        
        # 2. PATHWAY STATE (D1/D2 weights, eligibility, etc.)
        pathway_state = {
            "d1_state": s.d1_pathway.get_state(),
            "d2_state": s.d2_pathway.get_state(),
        }
        
        # 3. LEARNING STATE
        learning_state = {
            # Trial accumulators
            "d1_votes_accumulated": s._d1_votes_accumulated.detach().clone(),
            "d2_votes_accumulated": s._d2_votes_accumulated.detach().clone(),
            
            # Homeostatic state
            "activity_ema": s._activity_ema,
            "trial_spike_count": s._trial_spike_count,
            "trial_timesteps": s._trial_timesteps,
            "homeostatic_scaling_applied": s._homeostatic_scaling_applied,
            
            # Homeostasis manager state (if enabled)
            "homeostasis_manager_state": s.homeostasis_manager.get_state() if s.homeostasis_manager is not None else None,
            # Backward compatibility: also save as unified_homeostasis_state
            "unified_homeostasis_state": s.homeostasis_manager.get_state() if s.homeostasis_manager is not None else None,
        }
        
        # 4. EXPLORATION STATE (delegate to ExplorationManager)
        exploration_state = {
            "exploring": s.exploring,
            "last_uncertainty": s._last_uncertainty,
            "last_exploration_prob": s._last_exploration_prob,
            # Get exploration manager state (includes action_counts, recent_rewards, etc.)
            "manager_state": s.exploration_manager.get_state(),
        }
        
        # 5. VALUE ESTIMATION STATE (if RPE enabled)
        rpe_state = {}
        if s.value_estimates is not None:
            rpe_state = {
                "value_estimates": s.value_estimates.detach().clone(),
                "last_rpe": s._last_rpe,
                "last_expected": s._last_expected,
            }
        
        # 6. GOAL MODULATION STATE (if enabled)
        goal_state = {}
        if hasattr(s, 'pfc_modulation_d1') and s.pfc_modulation_d1 is not None:
            goal_state = {
                "pfc_modulation_d1": s.pfc_modulation_d1.detach().clone(),
                "pfc_modulation_d2": s.pfc_modulation_d2.detach().clone(),
            }
        
        # 7. ACTION SELECTION STATE
        action_state = {
            "last_action": s.last_action,
            "recent_spikes": s.recent_spikes.detach().clone(),
        }
        
        return {
            "neuron_state": neuron_state,
            "pathway_state": pathway_state,
            "learning_state": learning_state,
            "exploration_state": exploration_state,
            "rpe_state": rpe_state,
            "goal_state": goal_state,
            "action_state": action_state,
        }
    
    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Restore complete striatum state from checkpoint.
        
        Args:
            state: Dict from get_full_state()
        """
        s = self.striatum
        
        # 1. RESTORE NEURON STATE
        neuron_state = state["neuron_state"]
        if s.d1_neurons is not None and neuron_state["membrane_potential"] is not None:
            s.d1_neurons.membrane = neuron_state["membrane_potential"].to(s.device)
        
        # 2. RESTORE PATHWAY STATE
        pathway_state = state["pathway_state"]
        s.d1_pathway.load_state(pathway_state["d1_state"])
        s.d2_pathway.load_state(pathway_state["d2_state"])
        
        # 3. RESTORE LEARNING STATE
        learning_state = state["learning_state"]
        
        # Trial accumulators
        s._d1_votes_accumulated = learning_state["d1_votes_accumulated"].to(s.device)
        s._d2_votes_accumulated = learning_state["d2_votes_accumulated"].to(s.device)
        
        # Homeostatic state
        s._activity_ema = learning_state["activity_ema"]
        s._trial_spike_count = learning_state["trial_spike_count"]
        s._trial_timesteps = learning_state["trial_timesteps"]
        s._homeostatic_scaling_applied = learning_state["homeostatic_scaling_applied"]
        
        # Homeostasis manager (with backward compatibility)
        if s.homeostasis_manager is not None:
            # Try new format first, fall back to old format
            if "homeostasis_manager_state" in learning_state and learning_state["homeostasis_manager_state"] is not None:
                s.homeostasis_manager.load_state(learning_state["homeostasis_manager_state"])
            elif "unified_homeostasis_state" in learning_state and learning_state["unified_homeostasis_state"] is not None:
                # Backward compatibility: load old unified_homeostasis state
                s.homeostasis_manager.load_state(learning_state["unified_homeostasis_state"])
        
        # 4. RESTORE EXPLORATION STATE (delegate to ExplorationManager)
        exploration_state = state["exploration_state"]
        s.exploring = exploration_state["exploring"]
        s._last_uncertainty = exploration_state["last_uncertainty"]
        s._last_exploration_prob = exploration_state["last_exploration_prob"]
        
        # Load exploration manager state if present (new format)
        if "manager_state" in exploration_state:
            s.exploration_manager.load_state(exploration_state["manager_state"])
        else:
            # Backward compatibility: load old format directly
            # Old format had action_counts, recent_rewards, etc. at top level
            old_state = {
                "action_counts": exploration_state.get("action_counts", torch.zeros(s.n_actions, device=s.device)),
                "total_trials": exploration_state.get("total_trials", 0),
                "recent_rewards": exploration_state.get("recent_rewards", []),
                "recent_accuracy": exploration_state.get("recent_accuracy", []),
                "tonic_dopamine": exploration_state.get("tonic_dopamine", 0.3),
            }
            s.exploration_manager.load_state(old_state)
        
        # 5. RESTORE RPE STATE (if present)
        if "rpe_state" in state and state["rpe_state"]:
            rpe_state = state["rpe_state"]
            s.value_estimates = rpe_state["value_estimates"].to(s.device)
            s._last_rpe = rpe_state["last_rpe"]
            s._last_expected = rpe_state["last_expected"]
        
        # 6. RESTORE GOAL MODULATION STATE (if present)
        if "goal_state" in state and state["goal_state"]:
            goal_state = state["goal_state"]
            if hasattr(s, 'pfc_modulation_d1'):
                s.pfc_modulation_d1.data = goal_state["pfc_modulation_d1"].to(s.device)
                s.pfc_modulation_d2.data = goal_state["pfc_modulation_d2"].to(s.device)
        
        # 7. RESTORE ACTION SELECTION STATE
        action_state = state["action_state"]
        s.last_action = action_state["last_action"]
        s.recent_spikes = action_state["recent_spikes"].to(s.device)
