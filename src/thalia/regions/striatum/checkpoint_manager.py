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

    # =========================================================================
    # NEUROMORPHIC FORMAT (Phase 2) - Neuron-Centric Checkpoints
    # =========================================================================

    def _extract_synapses_for_neuron(self, neuron_idx: int, weights: torch.Tensor, eligibility: torch.Tensor, source_prefix: str) -> list[Dict[str, Any]]:
        """Extract synapses for a single neuron.

        Args:
            neuron_idx: Index of target neuron
            weights: Weight matrix [n_output, n_input]
            eligibility: Eligibility traces [n_output, n_input]
            source_prefix: Prefix for source neuron IDs (e.g., "cortex_l4_neuron")

        Returns:
            List of synapse dicts with {from, weight, eligibility}
        """
        synapses = []
        neuron_weights = weights[neuron_idx]
        neuron_eligibility = eligibility[neuron_idx] if eligibility is not None else torch.zeros_like(neuron_weights)

        # Only store non-zero synapses (sparse format)
        nonzero_mask = neuron_weights.abs() > 1e-8
        nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=True)[0]

        for input_idx in nonzero_indices:
            synapse = {
                "from": f"{source_prefix}_{input_idx.item()}",
                "weight": neuron_weights[input_idx].item(),
                "eligibility": neuron_eligibility[input_idx].item(),
            }
            synapses.append(synapse)

        return synapses

    def get_neuromorphic_state(self, source_prefix: str = "input") -> Dict[str, Any]:
        """Get striatum state in neuromorphic (neuron-centric) format.

        Instead of weight matrices, stores per-neuron data with explicit synapses.
        This format is ID-based, supporting growth/pruning gracefully.

        Args:
            source_prefix: Prefix for input neuron IDs (defaults to "input")

        Returns:
            Dict with format: {
                "format": "neuromorphic",
                "neurons": [
                    {
                        "id": "striatum_d1_neuron_0_step0",
                        "type": "D1-MSN",
                        "region": "striatum",
                        "created_step": 0,
                        "membrane": 0.5,
                        "incoming_synapses": [
                            {"from": "input_0", "weight": 0.3, "eligibility": 0.1}
                        ]
                    }
                ]
            }
        """
        s = self.striatum

        neurons = []

        # Extract D1 pathway neurons
        d1_weights = s.d1_pathway.weights
        d1_eligibility = s.d1_pathway.eligibility if s.d1_pathway.eligibility is not None else torch.zeros_like(d1_weights)

        n_d1 = d1_weights.shape[0] // 2  # Half are D1

        for i in range(n_d1):
            neuron_id = s.neuron_ids[i] if i < len(s.neuron_ids) else f"striatum_d1_neuron_{i}_step0"

            # Extract creation step from ID (format: "..._step{N}")
            created_step = 0
            if "_step" in neuron_id:
                created_step = int(neuron_id.split("_step")[1])

            neuron_data = {
                "id": neuron_id,
                "type": "D1-MSN",
                "region": "striatum",
                "created_step": created_step,
                "membrane": s.d1_pathway.neurons.membrane[i].item() if s.d1_pathway.neurons.membrane is not None else 0.0,
                "incoming_synapses": self._extract_synapses_for_neuron(i, d1_weights, d1_eligibility, source_prefix),
            }
            neurons.append(neuron_data)

        # Extract D2 pathway neurons
        d2_weights = s.d2_pathway.weights
        d2_eligibility = s.d2_pathway.eligibility if s.d2_pathway.eligibility is not None else torch.zeros_like(d2_weights)

        n_d2 = d2_weights.shape[0] - n_d1  # Remaining are D2

        for i in range(n_d2):
            neuron_idx = n_d1 + i
            neuron_id = s.neuron_ids[neuron_idx] if neuron_idx < len(s.neuron_ids) else f"striatum_d2_neuron_{i}_step0"

            # Extract creation step from ID
            created_step = 0
            if "_step" in neuron_id:
                created_step = int(neuron_id.split("_step")[1])

            neuron_data = {
                "id": neuron_id,
                "type": "D2-MSN",
                "region": "striatum",
                "created_step": created_step,
                "membrane": s.d2_pathway.neurons.membrane[i].item() if s.d2_pathway.neurons.membrane is not None else 0.0,
                "incoming_synapses": self._extract_synapses_for_neuron(i, d2_weights, d2_eligibility, source_prefix),
            }
            neurons.append(neuron_data)

        return {
            "format": "neuromorphic",
            "format_version": "2.0.0",
            "neurons": neurons,
            # Also store global state (exploration, action selection, etc.)
            "exploration_state": {
                "manager_state": s.exploration_manager.get_state(),
            },
            "action_state": {
                "last_action": s.state_tracker.last_action,
            },
        }

    def load_neuromorphic_state(self, state: Dict[str, Any]) -> None:
        """Load striatum state from neuromorphic format.

        Matches neurons by ID and restores state. Handles:
        - Missing neurons (checkpoint has neurons brain doesn't) → skip with warning
        - Extra neurons (brain has neurons checkpoint doesn't) → keep current state
        - Orphaned synapses (source neuron deleted) → skip

        Args:
            state: Dict from get_neuromorphic_state()
        """
        s = self.striatum

        if state.get("format") != "neuromorphic":
            raise ValueError(f"Expected neuromorphic format, got {state.get('format')}")

        # Build ID → neuron data mapping
        neurons_by_id = {n["id"]: n for n in state["neurons"]}

        # Counters for logging
        restored_count = 0
        missing_count = 0

        # Build ID → index mapping for current brain
        current_neuron_indices = {neuron_id: idx for idx, neuron_id in enumerate(s.neuron_ids)}

        # Restore each neuron from checkpoint
        for neuron_data in state["neurons"]:
            neuron_id = neuron_data["id"]

            # Check if this neuron exists in current brain
            if neuron_id not in current_neuron_indices:
                missing_count += 1
                continue  # Skip neurons not in current brain

            idx = current_neuron_indices[neuron_id]
            neuron_type = neuron_data["type"]

            # Restore membrane potential
            if neuron_type == "D1-MSN":
                if idx < s.d1_pathway.neurons.membrane.shape[0]:
                    s.d1_pathway.neurons.membrane[idx] = neuron_data["membrane"]
            elif neuron_type == "D2-MSN":
                d2_idx = idx - (s.d1_pathway.neurons.membrane.shape[0] // 2)
                if d2_idx < s.d2_pathway.neurons.membrane.shape[0]:
                    s.d2_pathway.neurons.membrane[d2_idx] = neuron_data["membrane"]

            # Restore synapses (weights and eligibility)
            for synapse in neuron_data["incoming_synapses"]:
                # Parse source neuron ID to get input index
                source_id = synapse["from"]
                if "_" in source_id:
                    try:
                        input_idx = int(source_id.split("_")[-1])

                        # Restore weight (use .data[] to avoid in-place operation errors)
                        if neuron_type == "D1-MSN" and idx < s.d1_pathway.weights.shape[0]:
                            if input_idx < s.d1_pathway.weights.shape[1]:
                                s.d1_pathway.weights.data[idx, input_idx] = synapse["weight"]
                                if s.d1_pathway.eligibility is not None:
                                    s.d1_pathway.eligibility.data[idx, input_idx] = synapse["eligibility"]
                        elif neuron_type == "D2-MSN":
                            d2_idx = idx - (s.d1_pathway.weights.shape[0] // 2)
                            if d2_idx < s.d2_pathway.weights.shape[0] and input_idx < s.d2_pathway.weights.shape[1]:
                                s.d2_pathway.weights.data[d2_idx, input_idx] = synapse["weight"]
                                if s.d2_pathway.eligibility is not None:
                                    s.d2_pathway.eligibility.data[d2_idx, input_idx] = synapse["eligibility"]
                    except (ValueError, IndexError):
                        # Invalid source ID format, skip
                        pass

            restored_count += 1

        # Restore global state
        if "exploration_state" in state and "manager_state" in state["exploration_state"]:
            s.exploration_manager.load_state(state["exploration_state"]["manager_state"])

        if "action_state" in state:
            s.state_tracker.last_action = state["action_state"]["last_action"]

        # Log warnings
        if missing_count > 0:
            import warnings
            warnings.warn(
                f"Checkpoint has {missing_count} neurons not in current brain. "
                f"Skipped those neurons. Restored {restored_count} neurons.",
                UserWarning
            )

        # Check for extra neurons in brain
        checkpoint_ids = set(neurons_by_id.keys())
        current_ids = set(s.neuron_ids)
        extra_ids = current_ids - checkpoint_ids

        if len(extra_ids) > 0:
            import warnings
            warnings.warn(
                f"Brain has {len(extra_ids)} neurons not in checkpoint. "
                f"Keeping their current state.",
                UserWarning
            )

    # =========================================================================
    # Phase 3: Hybrid Format (Auto-Selection)
    # =========================================================================

    def _should_use_neuromorphic(self) -> bool:
        """Determine if neuromorphic format should be used.

        Decision criteria:
        - Small regions (<100 neurons): Use neuromorphic (more inspectable)
        - High growth frequency (>0.1): Use neuromorphic (ID-based matching)
        - Growth enabled + small: Use neuromorphic
        - Otherwise: Use elastic tensor (more efficient)

        Returns:
            bool: True if neuromorphic format should be used
        """
        s = self.striatum

        # Count total neurons (D1 + D2)
        n_neurons = s.config.n_output

        # Threshold: small regions benefit from neuromorphic format
        SIZE_THRESHOLD = 100

        # If region is small, use neuromorphic for better inspectability
        if n_neurons < SIZE_THRESHOLD:
            return True

        # If growth enabled and region not too large, use neuromorphic
        if s.config.growth_enabled and n_neurons < SIZE_THRESHOLD * 2:
            return True

        # For large stable regions, elastic tensor is more efficient
        return False

    def save(self, path: str | Path) -> Dict[str, Any]:
        """Save checkpoint with automatic format selection.

        Automatically chooses between elastic tensor and neuromorphic formats
        based on region size and properties.

        Args:
            path: Path where checkpoint will be saved

        Returns:
            Dict with checkpoint metadata
        """
        import torch
        from pathlib import Path

        path = Path(path)

        # Auto-select format
        use_neuromorphic = self._should_use_neuromorphic()

        if use_neuromorphic:
            state = self.get_neuromorphic_state()
            format_name = "neuromorphic"
        else:
            state = self.get_full_state()
            format_name = "elastic_tensor"

        # Add hybrid format metadata
        state["hybrid_metadata"] = {
            "auto_selected": True,
            "selected_format": format_name,
            "selection_criteria": {
                "n_neurons": self.striatum.config.n_output,
                "growth_enabled": self.striatum.config.growth_enabled,
            }
        }

        # Save to disk
        torch.save(state, path)

        return {
            "path": str(path),
            "format": format_name,
            "n_neurons": self.striatum.config.n_output,
            "file_size": path.stat().st_size if path.exists() else 0,
        }

    def load(self, path: str | Path) -> None:
        """Load checkpoint with automatic format detection.

        Automatically detects the checkpoint format and calls the appropriate
        load method (load_full_state for elastic, load_neuromorphic_state for neuromorphic).

        Args:
            path: Path to checkpoint file
        """
        from pathlib import Path
        import torch

        path = Path(path)

        # Load checkpoint
        state = torch.load(path, weights_only=False)

        # Detect format
        if "format" in state and state["format"] == "neuromorphic":
            # Neuromorphic format
            self.load_neuromorphic_state(state)
        elif "format_version" in state and state.get("format_version", "").startswith("1."):
            # Elastic tensor format (Phase 1)
            self.load_full_state(state)
        elif "hybrid_metadata" in state:
            # Hybrid format - check selected format
            selected_format = state["hybrid_metadata"]["selected_format"]
            if selected_format == "neuromorphic":
                self.load_neuromorphic_state(state)
            else:
                self.load_full_state(state)
        else:
            # Legacy format or unknown - try elastic tensor
            self.load_full_state(state)
