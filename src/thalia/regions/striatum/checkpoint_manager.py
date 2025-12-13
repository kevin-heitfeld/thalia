"""
Striatum Checkpoint Manager - State Serialization and Restoration

This component manages checkpoint serialization and deserialization for the
Striatum region, extracted from the main Striatum class to improve separation
of concerns and maintainability.

**Responsibilities:**
- Serialize complete striatum state (weights, eligibility, exploration, etc.)
- Deserialize and restore state from checkpoints
- Handle backward compatibility with older checkpoint formats
- Manage version migration for checkpoint schema changes
- Provide get_full_state() and restore_from_state() interface

**Used By:**
- `Striatum` (main region class)
- Training scripts that save/load checkpoints
- Curriculum learning system for stage transitions

**Coordinates With:**
- `D1Pathway` and `D2Pathway`: Serializes pathway weights and eligibility traces
- `StriatumStateTracker`: Serializes vote accumulators and trial state
- `ExplorationComponent`: Serializes exploration parameters
- `LearningComponent`: Serializes learning history and statistics
- `StriatumHomeostasisComponent`: Serializes homeostatic state

**Why Extracted:**
- Orthogonal concern: State management is separate from forward/learning logic
- Complexity reduction: Checkpoint code is ~200 lines, cluttered main class
- Testability: Can test serialization/deserialization independently
- Maintainability: Checkpoint format changes isolated to single module
- Backward compatibility: Version migration logic centralized

**Checkpoint Format:**
The full state dict contains:
- `neuron_state`: Membrane potentials, dimensions
- `pathway_state`: D1/D2 weights, eligibility traces
- `learning_state`: Vote accumulators, trial statistics, homeostasis
- `exploration_state`: Exploration parameters and history
- `last_action`: Most recent action for credit assignment
- `td_lambda_state`: TD(λ) eligibility traces (if enabled)

Author: Thalia Project
Date: December 9, 2025 (extracted during striatum refactoring)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any

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
            # Elastic tensor capacity tracking (Phase 1)
            "n_neurons_active": s.n_neurons_active,
            "n_neurons_capacity": s.n_neurons_capacity,
        }

        # 2. PATHWAY STATE (D1/D2 weights, eligibility, etc.)
        pathway_state = {
            "d1_state": s.d1_pathway.get_state(),
            "d2_state": s.d2_pathway.get_state(),
        }

        # 3. LEARNING STATE
        learning_state = {
            # Trial accumulators (now managed by state_tracker)
            "d1_votes_accumulated": s.state_tracker._d1_votes_accumulated.detach().clone(),
            "d2_votes_accumulated": s.state_tracker._d2_votes_accumulated.detach().clone(),

            # Homeostatic state
            "activity_ema": s._activity_ema,
            "trial_spike_count": s._trial_spike_count,
            "trial_timesteps": s._trial_timesteps,
            "homeostatic_scaling_applied": s._homeostatic_scaling_applied,

            # Homeostasis manager state (if enabled)
            "homeostasis_manager_state": s.homeostasis_manager.unified_homeostasis.get_state() if (s.homeostasis_manager is not None and s.homeostasis_manager.unified_homeostasis is not None) else None,
        }

        # 4. EXPLORATION STATE (delegate to ExplorationManager)
        exploration_state = {
            "exploring": s.state_tracker.exploring,
            "last_uncertainty": s.state_tracker._last_uncertainty,
            "last_exploration_prob": s.state_tracker._last_exploration_prob,
            # Get exploration manager state (includes action_counts, recent_rewards, etc.)
            "manager_state": s.exploration_manager.get_state(),
        }

        # 5. VALUE ESTIMATION STATE (if RPE enabled)
        rpe_state = {}
        if s.value_estimates is not None:
            rpe_state = {
                "value_estimates": s.value_estimates.detach().clone(),
                "last_rpe": s.state_tracker._last_rpe,
                "last_expected": s.state_tracker._last_expected,
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
            "last_action": s.state_tracker.last_action,
            "recent_spikes": s.state_tracker.recent_spikes.detach().clone(),
        }

        # 8. D1/D2 PATHWAY DELAY BUFFERS (Temporal Competition)
        delay_state = {
            "d1_delay_buffer": s._d1_delay_buffer.detach().clone() if s._d1_delay_buffer is not None else None,
            "d2_delay_buffer": s._d2_delay_buffer.detach().clone() if s._d2_delay_buffer is not None else None,
            "d1_delay_ptr": s._d1_delay_ptr,
            "d2_delay_ptr": s._d2_delay_ptr,
        }

        return {
            "format_version": "1.0.0",  # Checkpoint format version
            "neuron_state": neuron_state,
            "pathway_state": pathway_state,
            "learning_state": learning_state,
            "exploration_state": exploration_state,
            "rpe_state": rpe_state,
            "goal_state": goal_state,
            "action_state": action_state,
            "delay_state": delay_state,
        }

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Restore complete striatum state from checkpoint.

        Supports elastic tensor format (Phase 1):
        - Auto-grows if checkpoint has more neurons than current brain
        - Partial restore if checkpoint has fewer neurons
        - Validates capacity metadata

        Args:
            state: Dict from get_full_state()
        """
        s = self.striatum

        # PHASE 1: ELASTIC TENSOR CAPACITY HANDLING
        # Check if checkpoint has capacity metadata (new format)
        neuron_state = state["neuron_state"]

        if "n_neurons_active" in neuron_state and "n_neurons_capacity" in neuron_state:
            # New elastic tensor format
            checkpoint_active = neuron_state["n_neurons_active"]
            checkpoint_capacity = neuron_state["n_neurons_capacity"]

            # Validate capacity
            if checkpoint_capacity < checkpoint_active:
                raise ValueError(
                    f"Checkpoint capacity ({checkpoint_capacity}) less than active "
                    f"neurons ({checkpoint_active}). Corrupted checkpoint?"
                )

            # Auto-grow if checkpoint has more neurons than current brain
            if checkpoint_active > s.n_neurons_active:
                n_grow_neurons = checkpoint_active - s.n_neurons_active

                # Convert neuron count to action count (for population coding)
                n_grow_actions = n_grow_neurons // s.neurons_per_action
                if n_grow_neurons % s.neurons_per_action != 0:
                    raise ValueError(
                        f"Checkpoint neuron count ({checkpoint_active}) is not aligned with "
                        f"population coding ({s.neurons_per_action} neurons/action). "
                        f"Cannot auto-grow brain."
                    )

                import warnings
                warnings.warn(
                    f"Checkpoint has {checkpoint_active} neurons but brain has "
                    f"{s.n_neurons_active}. Auto-growing by {n_grow_actions} actions "
                    f"({n_grow_neurons} neurons).",
                    UserWarning
                )
                s.add_neurons(n_new=n_grow_actions)

            # Warn if checkpoint has fewer neurons
            elif checkpoint_active < s.n_neurons_active:
                import warnings
                warnings.warn(
                    f"Checkpoint has {checkpoint_active} neurons but brain has "
                    f"{s.n_neurons_active}. Will restore {checkpoint_active} neurons, "
                    f"remaining {s.n_neurons_active - checkpoint_active} keep current state.",
                    UserWarning
                )
        else:
            # Old format without capacity metadata - assume full restore
            import warnings
            warnings.warn(
                "Loading checkpoint from old format without capacity metadata. "
                "Assuming checkpoint size matches brain size.",
                UserWarning
            )

        # 1. RESTORE NEURON STATE
        if s.d1_neurons is not None and neuron_state.get("membrane_potential") is not None:
            membrane = neuron_state["membrane_potential"].to(s.device)
            # Partial restore: only copy up to checkpoint size
            n_restore = min(membrane.shape[0], s.d1_neurons.membrane.shape[0])
            s.d1_neurons.membrane[:n_restore] = membrane[:n_restore]

        # 2. RESTORE PATHWAY STATE
        pathway_state = state["pathway_state"]
        s.d1_pathway.load_state(pathway_state["d1_state"])
        s.d2_pathway.load_state(pathway_state["d2_state"])

        # 3. RESTORE LEARNING STATE
        learning_state = state["learning_state"]

        # Trial accumulators (now managed by state_tracker)
        s.state_tracker._d1_votes_accumulated = learning_state["d1_votes_accumulated"].to(s.device)
        s.state_tracker._d2_votes_accumulated = learning_state["d2_votes_accumulated"].to(s.device)

        # Homeostatic state
        s._activity_ema = learning_state["activity_ema"]
        s._trial_spike_count = learning_state["trial_spike_count"]
        s._trial_timesteps = learning_state["trial_timesteps"]
        s._homeostatic_scaling_applied = learning_state["homeostatic_scaling_applied"]

        # Homeostasis manager (with backward compatibility)
        if s.homeostasis_manager is not None:
            # Try new format first, fall back to old format
            if "homeostasis_manager_state" in learning_state and learning_state["homeostasis_manager_state"] is not None:
                s.homeostasis_manager.unified_homeostasis.load_state(learning_state["homeostasis_manager_state"])

        # 4. RESTORE EXPLORATION STATE (delegate to ExplorationManager)
        exploration_state = state["exploration_state"]
        s.state_tracker.exploring = exploration_state["exploring"]
        s.state_tracker._last_uncertainty = exploration_state["last_uncertainty"]
        s.state_tracker._last_exploration_prob = exploration_state["last_exploration_prob"]

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

        # 6. RESTORE RPE STATE (if present)
        if "rpe_state" in state and state["rpe_state"]:
            rpe_state = state["rpe_state"]
            s.value_estimates = rpe_state["value_estimates"].to(s.device)
            s.state_tracker._last_rpe = rpe_state["last_rpe"]
            s.state_tracker._last_expected = rpe_state["last_expected"]

        # 6. RESTORE GOAL MODULATION STATE (if present)
        if "goal_state" in state and state["goal_state"]:
            goal_state = state["goal_state"]
            if hasattr(s, 'pfc_modulation_d1'):
                s.pfc_modulation_d1.data = goal_state["pfc_modulation_d1"].to(s.device)
                s.pfc_modulation_d2.data = goal_state["pfc_modulation_d2"].to(s.device)

        # 8. RESTORE ACTION SELECTION STATE
        action_state = state["action_state"]
        s.state_tracker.last_action = action_state["last_action"]
        s.state_tracker.recent_spikes = action_state["recent_spikes"].to(s.device)

        # 9. RESTORE D1/D2 PATHWAY DELAY BUFFERS (if present)
        if "delay_state" in state and state["delay_state"]:
            delay_state = state["delay_state"]
            if delay_state["d1_delay_buffer"] is not None:
                s._d1_delay_buffer = delay_state["d1_delay_buffer"].to(s.device)
                s._d1_delay_ptr = delay_state["d1_delay_ptr"]
            if delay_state["d2_delay_buffer"] is not None:
                s._d2_delay_buffer = delay_state["d2_delay_buffer"].to(s.device)
                s._d2_delay_ptr = delay_state["d2_delay_ptr"]
