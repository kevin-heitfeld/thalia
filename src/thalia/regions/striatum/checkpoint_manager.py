"""
Striatum Checkpoint Manager - State Serialization and Restoration

This component manages checkpoint serialization and deserialization for the
Striatum region, extracted from the main Striatum class to improve separation
of concerns and maintainability.

**Responsibilities:**
- Serialize complete striatum state (weights, eligibility, exploration, etc.)
- Deserialize and restore state from checkpoints
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

Author: Thalia Project
Date: December 9, 2025 (extracted during striatum refactoring)
"""

from __future__ import annotations

import warnings
import weakref
from typing import TYPE_CHECKING, Any, Dict

import torch

from thalia.managers import BaseCheckpointManager

if TYPE_CHECKING:
    from .striatum import Striatum


class StriatumCheckpointManager(BaseCheckpointManager):
    """Manages state checkpointing for Striatum.

    Handles:
    - Full state serialization (weights, eligibility, exploration, etc.)
    - State restoration with elastic tensor support
    - Neuromorphic format for growth-enabled configurations

    Inherits from BaseCheckpointManager for shared synapse extraction logic.
    """

    def __init__(self, striatum: Striatum):
        """Initialize checkpoint manager.

        Args:
            striatum: The Striatum instance to manage
        """
        super().__init__(format_version="1.0.0")
        self.striatum = striatum

    def collect_state(self) -> Dict[str, Any]:
        """Collect complete striatum state for checkpointing.

        Returns:
            Dict containing all state needed to restore striatum
        """
        s = self.striatum

        # 1. NEURON STATE - Use base class utility for common extraction
        neuron_state = self.extract_neuron_state_common(
            neurons=s.d1_pathway.neurons, n_neurons=s.d1_size + s.d2_size, device=s.device
        )
        # Add elastic tensor metadata
        neuron_state.update(
            self.extract_elastic_tensor_metadata(
                n_active=s.n_neurons_active, n_capacity=s.n_neurons_capacity
            )
        )
        # Add striatum-specific fields
        neuron_state.update(
            {
                "n_actions": s.n_actions,
                "total_input": s.input_size,
                "total_neurons": s.d1_size + s.d2_size,
            }
        )

        # 2. PATHWAY STATE (Multi-source weights, eligibility, etc.)
        # PHASE 5: Multi-source architecture - save per-source weights and eligibility
        # Note: D1/D2 pathway weights are stored in parent's synaptic_weights dict,
        # so we save their neuron state separately instead of calling pathway.get_state()
        pathway_state = {
            # D1 pathway neuron state
            "d1_neuron_membrane": (
                s.d1_pathway.neurons.membrane.clone()
                if s.d1_pathway.neurons.membrane is not None
                else None
            ),
            "d1_neuron_g_E": (
                s.d1_pathway.neurons.g_E.clone() if s.d1_pathway.neurons.g_E is not None else None
            ),
            "d1_neuron_g_I": (
                s.d1_pathway.neurons.g_I.clone() if s.d1_pathway.neurons.g_I is not None else None
            ),
            "d1_neuron_refractory": (
                s.d1_pathway.neurons.refractory.clone()
                if s.d1_pathway.neurons.refractory is not None
                else None
            ),
            # D2 pathway neuron state
            "d2_neuron_membrane": (
                s.d2_pathway.neurons.membrane.clone()
                if s.d2_pathway.neurons.membrane is not None
                else None
            ),
            "d2_neuron_g_E": (
                s.d2_pathway.neurons.g_E.clone() if s.d2_pathway.neurons.g_E is not None else None
            ),
            "d2_neuron_g_I": (
                s.d2_pathway.neurons.g_I.clone() if s.d2_pathway.neurons.g_I is not None else None
            ),
            "d2_neuron_refractory": (
                s.d2_pathway.neurons.refractory.clone()
                if s.d2_pathway.neurons.refractory is not None
                else None
            ),
            # Multi-source weights (Phase 5)
            "synaptic_weights": {
                key: tensor.detach().clone() for key, tensor in s.synaptic_weights.items()
            },
            # Multi-source eligibility traces (Phase 3)
            "eligibility_d1": (
                {key: tensor.detach().clone() for key, tensor in s._eligibility_d1.items()}
                if hasattr(s, "_eligibility_d1")
                else {}
            ),
            "eligibility_d2": (
                {key: tensor.detach().clone() for key, tensor in s._eligibility_d2.items()}
                if hasattr(s, "_eligibility_d2")
                else {}
            ),
            # Per-source STP modules (Phase 5)
            "stp_modules": (
                {
                    key: {
                        "u": stp.u.detach().clone() if stp.u is not None else None,
                        "x": stp.x.detach().clone() if stp.x is not None else None,
                    }
                    for key, stp in s.stp_modules.items()
                }
                if hasattr(s, "stp_modules")
                else {}
            ),
            # Input source tracking
            "input_sources": dict(s.input_sources) if hasattr(s, "input_sources") else {},
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
            "homeostasis_manager_state": (
                s.homeostasis.unified_homeostasis.get_state()
                if (s.homeostasis is not None and s.homeostasis.unified_homeostasis is not None)
                else None
            ),
        }

        # 4. EXPLORATION STATE (delegate to ExplorationManager)
        exploration_state = {
            "exploring": s.state_tracker.exploring,
            "last_uncertainty": s.state_tracker._last_uncertainty,
            "last_exploration_prob": s.state_tracker._last_exploration_prob,
            # Get exploration manager state (includes action_counts, recent_rewards, etc.)
            "manager_state": s.exploration.get_state(),
        }

        # 6. GOAL MODULATION STATE (if enabled)
        goal_state = {}
        if hasattr(s, "pfc_modulation_d1") and s.pfc_modulation_d1 is not None:
            assert s.pfc_modulation_d2 is not None, "pfc_modulation_d2 must exist if d1 exists"
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
            "d1_delay_buffer": (
                s._d1_delay_buffer.detach().clone() if s._d1_delay_buffer is not None else None
            ),
            "d2_delay_buffer": (
                s._d2_delay_buffer.detach().clone() if s._d2_delay_buffer is not None else None
            ),
            "d1_delay_ptr": s._d1_delay_ptr,
            "d2_delay_ptr": s._d2_delay_ptr,
        }

        return {
            "neuron_state": neuron_state,
            "pathway_state": pathway_state,
            "learning_state": learning_state,
            "exploration_state": exploration_state,
            "goal_state": goal_state,
            "action_state": action_state,
            "delay_state": delay_state,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore complete striatum state from checkpoint.

        Supports elastic tensor format (Phase 1):
        - Auto-grows if checkpoint has more neurons than current brain
        - Partial restore if checkpoint has fewer neurons
        - Validates capacity metadata

        Args:
            state: Dict from collect_state()
        """
        s = self.striatum

        # PHASE 1: ELASTIC TENSOR CAPACITY HANDLING - Use base class utilities
        neuron_state = state["neuron_state"]

        # Validate elastic metadata if present
        is_valid, error_msg = self.validate_elastic_metadata(neuron_state)
        if not is_valid:
            raise ValueError(f"Invalid elastic tensor metadata: {error_msg}")

        if "n_neurons_active" in neuron_state and "n_neurons_capacity" in neuron_state:
            # Handle elastic tensor growth/shrinkage using base class utility
            # With D1/D2 pathways, neurons_per_action applies to EACH pathway
            # So neurons_per_unit for growth calculation = neurons_per_action * 2
            neurons_per_unit = s.neurons_per_action * 2  # D1 + D2
            should_grow, n_grow_actions, warning_msg = self.handle_elastic_tensor_growth(
                checkpoint_active=neuron_state["n_neurons_active"],
                current_active=s.n_neurons_active,
                neurons_per_unit=neurons_per_unit,
                region_name="Striatum",
            )

            if warning_msg:
                warnings.warn(warning_msg, UserWarning)

            if should_grow:
                s.grow_output(n_new=n_grow_actions)

        # 1. RESTORE NEURON STATE
        if s.d1_pathway.neurons is not None and neuron_state.get("membrane_potential") is not None:
            # Ensure neurons are initialized
            if s.d1_pathway.neurons.membrane is None:
                s.d1_pathway.neurons.reset_state()

            membrane = neuron_state["membrane_potential"].to(s.device)
            # Partial restore: only copy up to checkpoint size
            n_restore = min(membrane.shape[0], s.d1_pathway.neurons.membrane.shape[0])
            s.d1_pathway.neurons.membrane[:n_restore] = membrane[:n_restore]

        # 2. RESTORE PATHWAY STATE (Multi-source architecture)
        pathway_state = state["pathway_state"]

        # Restore D1 pathway neuron state (elastic: only restore up to checkpoint size)
        if pathway_state.get("d1_neuron_membrane") is not None:
            if s.d1_pathway.neurons.membrane is None:
                s.d1_pathway.neurons.reset_state()
            checkpoint_membrane = pathway_state["d1_neuron_membrane"].to(s.device)
            n_restore = min(checkpoint_membrane.shape[0], s.d1_pathway.neurons.membrane.shape[0])
            s.d1_pathway.neurons.membrane[:n_restore] = checkpoint_membrane[:n_restore]

        if pathway_state.get("d1_neuron_g_E") is not None:
            if s.d1_pathway.neurons.g_E is None:
                s.d1_pathway.neurons.reset_state()
            checkpoint_g_E = pathway_state["d1_neuron_g_E"].to(s.device)
            n_restore = min(checkpoint_g_E.shape[0], s.d1_pathway.neurons.g_E.shape[0])
            s.d1_pathway.neurons.g_E[:n_restore] = checkpoint_g_E[:n_restore]

        if pathway_state.get("d1_neuron_g_I") is not None:
            if s.d1_pathway.neurons.g_I is None:
                s.d1_pathway.neurons.reset_state()
            checkpoint_g_I = pathway_state["d1_neuron_g_I"].to(s.device)
            n_restore = min(checkpoint_g_I.shape[0], s.d1_pathway.neurons.g_I.shape[0])
            s.d1_pathway.neurons.g_I[:n_restore] = checkpoint_g_I[:n_restore]

        if pathway_state.get("d1_neuron_refractory") is not None:
            if s.d1_pathway.neurons.refractory is None:
                s.d1_pathway.neurons.reset_state()
            checkpoint_refr = pathway_state["d1_neuron_refractory"].to(s.device)
            n_restore = min(checkpoint_refr.shape[0], s.d1_pathway.neurons.refractory.shape[0])
            s.d1_pathway.neurons.refractory[:n_restore] = checkpoint_refr[:n_restore]

        # Restore D2 pathway neuron state (elastic: only restore up to checkpoint size)
        if pathway_state.get("d2_neuron_membrane") is not None:
            if s.d2_pathway.neurons.membrane is None:
                s.d2_pathway.neurons.reset_state()
            checkpoint_membrane = pathway_state["d2_neuron_membrane"].to(s.device)
            n_restore = min(checkpoint_membrane.shape[0], s.d2_pathway.neurons.membrane.shape[0])
            s.d2_pathway.neurons.membrane[:n_restore] = checkpoint_membrane[:n_restore]

        if pathway_state.get("d2_neuron_g_E") is not None:
            if s.d2_pathway.neurons.g_E is None:
                s.d2_pathway.neurons.reset_state()
            checkpoint_g_E = pathway_state["d2_neuron_g_E"].to(s.device)
            n_restore = min(checkpoint_g_E.shape[0], s.d2_pathway.neurons.g_E.shape[0])
            s.d2_pathway.neurons.g_E[:n_restore] = checkpoint_g_E[:n_restore]

        if pathway_state.get("d2_neuron_g_I") is not None:
            if s.d2_pathway.neurons.g_I is None:
                s.d2_pathway.neurons.reset_state()
            checkpoint_g_I = pathway_state["d2_neuron_g_I"].to(s.device)
            n_restore = min(checkpoint_g_I.shape[0], s.d2_pathway.neurons.g_I.shape[0])
            s.d2_pathway.neurons.g_I[:n_restore] = checkpoint_g_I[:n_restore]

        if pathway_state.get("d2_neuron_refractory") is not None:
            if s.d2_pathway.neurons.refractory is None:
                s.d2_pathway.neurons.reset_state()
            checkpoint_refr = pathway_state["d2_neuron_refractory"].to(s.device)
            n_restore = min(checkpoint_refr.shape[0], s.d2_pathway.neurons.refractory.shape[0])
            s.d2_pathway.neurons.refractory[:n_restore] = checkpoint_refr[:n_restore]

        # Multi-source weights
        for key, weights in pathway_state["synaptic_weights"].items():
            if key in s.synaptic_weights:
                # Restore weight data
                s.synaptic_weights[key].data = weights.to(s.device)
            else:
                warnings.warn(
                    f"Checkpoint has source '{key}' but current brain doesn't. "
                    "Skipping this source weight.",
                    UserWarning,
                )

        # Restore multi-source eligibility traces
        if "eligibility_d1" in pathway_state and hasattr(s, "_eligibility_d1"):
            for key, elig in pathway_state["eligibility_d1"].items():
                if key in s._eligibility_d1:
                    s._eligibility_d1[key] = elig.to(s.device)

        if "eligibility_d2" in pathway_state and hasattr(s, "_eligibility_d2"):
            for key, elig in pathway_state["eligibility_d2"].items():
                if key in s._eligibility_d2:
                    s._eligibility_d2[key] = elig.to(s.device)

        # Restore per-source STP modules
        if "stp_modules" in pathway_state and hasattr(s, "stp_modules"):
            for key, stp_state in pathway_state["stp_modules"].items():
                if key in s.stp_modules:
                    # Initialize STP state if needed (in case reset_state() wasn't called)
                    if stp_state["u"] is not None:
                        if s.stp_modules[key].u is None:
                            # STP module exists but state not initialized - do it now
                            s.stp_modules[key].reset_state()
                        s.stp_modules[key].u.data = stp_state["u"].to(s.device)  # type: ignore[union-attr]
                    if stp_state["x"] is not None:
                        if s.stp_modules[key].x is None:
                            # STP module exists but state not initialized - do it now
                            s.stp_modules[key].reset_state()
                        s.stp_modules[key].x.data = stp_state["x"].to(s.device)  # type: ignore[union-attr]

        # Restore input source tracking
        if "input_sources" in pathway_state:
            s.input_sources = pathway_state["input_sources"]

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

        # Homeostasis manager
        if (
            s.homeostasis is not None
            and "homeostasis_manager_state" in learning_state
            and learning_state["homeostasis_manager_state"] is not None
        ):
            s.homeostasis.unified_homeostasis.load_state(
                learning_state["homeostasis_manager_state"]
            )

        # 4. RESTORE EXPLORATION STATE
        exploration_state = state["exploration_state"]
        s.state_tracker.exploring = exploration_state["exploring"]
        s.state_tracker._last_uncertainty = exploration_state["last_uncertainty"]
        s.state_tracker._last_exploration_prob = exploration_state["last_exploration_prob"]

        # Load exploration manager state
        if "manager_state" in exploration_state:
            s.exploration.load_state(exploration_state["manager_state"])

        # 6. RESTORE RPE STATE (if present)
        if "rpe_state" in state and state["rpe_state"]:
            rpe_state = state["rpe_state"]
            s.value_estimates = rpe_state["value_estimates"].to(s.device)
            s.state_tracker._last_rpe = rpe_state["last_rpe"]
            s.state_tracker._last_expected = rpe_state["last_expected"]

        # 6. RESTORE GOAL MODULATION STATE (if present)
        if "goal_state" in state and state["goal_state"]:
            goal_state = state["goal_state"]
            if hasattr(s, "pfc_modulation_d1") and s.pfc_modulation_d1 is not None:
                assert s.pfc_modulation_d2 is not None, "pfc_modulation_d2 must exist"
                s.pfc_modulation_d1.data = goal_state["pfc_modulation_d1"].to(s.device)  # type: ignore[union-attr]
                s.pfc_modulation_d2.data = goal_state["pfc_modulation_d2"].to(s.device)  # type: ignore[union-attr]

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

        # 10. RE-LINK PATHWAYS TO PARENT AFTER STATE RESTORATION
        # Pathways need weak references to access weights from parent's synaptic_weights dict
        if pathway_state["synaptic_weights"]:
            # Get first D1 and D2 source keys
            d1_keys = [k for k in pathway_state["synaptic_weights"].keys() if k.endswith("_d1")]
            d2_keys = [k for k in pathway_state["synaptic_weights"].keys() if k.endswith("_d2")]

            if d1_keys and s.d1_pathway._parent_striatum_ref is None:
                s.d1_pathway._parent_striatum_ref = weakref.ref(s)
                s.d1_pathway._weight_source = d1_keys[0]

            if d2_keys and s.d2_pathway._parent_striatum_ref is None:
                s.d2_pathway._parent_striatum_ref = weakref.ref(s)
                s.d2_pathway._weight_source = d2_keys[0]

    # =========================================================================
    # NEUROMORPHIC FORMAT (Phase 2) - Neuron-Centric Checkpoints
    # =========================================================================

    def _extract_synapses_for_neuron(
        self, neuron_idx: int, weights: torch.Tensor, eligibility: torch.Tensor, source_prefix: str
    ) -> list[Dict[str, Any]]:
        """Extract synapses for a single neuron (striatum-specific with eligibility).

        Args:
            neuron_idx: Index of target neuron
            weights: Weight matrix [n_output, n_input]
            eligibility: Eligibility traces [n_output, n_input]
            source_prefix: Prefix for source neuron IDs (e.g., "cortex_l4_neuron")

        Returns:
            List of synapse dicts with {from, weight, eligibility}
        """
        # Use base class method for weight extraction
        synapses = self.extract_synapses_for_neuron(neuron_idx, weights, source_prefix)

        # Add eligibility traces to each synapse
        if eligibility is not None:
            neuron_eligibility = eligibility[neuron_idx]
            for synapse in synapses:
                # Parse source index from ID (format: "{source_prefix}_{idx}")
                source_idx = int(synapse["from"].split("_")[-1])
                synapse["eligibility"] = neuron_eligibility[source_idx].item()
        else:
            # Default to zero eligibility
            for synapse in synapses:
                synapse["eligibility"] = 0.0

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
        # Use abstract method implementations to extract state
        neurons = self._get_neurons_data()
        learning_state = self._get_learning_state()
        neuromodulator_state = self._get_neuromodulator_state()
        region_state = self._get_region_state()

        # Package using base class method
        return self.package_neuromorphic_state(
            neurons=neurons,
            learning_state=learning_state,
            neuromodulator_state=neuromodulator_state,
            region_state=region_state,
        )

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

            # Restore synapses (weights and eligibility) - Multi-source architecture
            for synapse in neuron_data["incoming_synapses"]:
                # Parse source neuron ID to determine source and index
                # Format: "{source_name}_neuron_{idx}" (e.g., "cortex:l5_neuron_42")
                source_id = synapse["from"]
                if "_neuron_" in source_id:
                    try:
                        # Split into source_name and neuron index
                        parts = source_id.rsplit("_neuron_", 1)
                        source_name = parts[0]
                        input_idx = int(parts[1])

                        # Restore weight in appropriate source-pathway weight matrix
                        if neuron_type == "D1-MSN":
                            d1_key = f"{source_name}_d1"
                            if d1_key in s.synaptic_weights:
                                weights = s.synaptic_weights[d1_key]
                                if idx < weights.shape[0] and input_idx < weights.shape[1]:
                                    weights.data[idx, input_idx] = synapse["weight"]
                                    # Restore eligibility
                                    if d1_key in s._eligibility_d1:
                                        s._eligibility_d1[d1_key][idx, input_idx] = synapse[
                                            "eligibility"
                                        ]
                        elif neuron_type == "D2-MSN":
                            d2_key = f"{source_name}_d2"
                            if d2_key in s.synaptic_weights:
                                weights = s.synaptic_weights[d2_key]
                                if idx < weights.shape[0] and input_idx < weights.shape[1]:
                                    weights.data[idx, input_idx] = synapse["weight"]
                                    # Restore eligibility
                                    if d2_key in s._eligibility_d2:
                                        s._eligibility_d2[d2_key][idx, input_idx] = synapse[
                                            "eligibility"
                                        ]
                    except (ValueError, IndexError, KeyError):
                        # Invalid source ID format or missing source, skip
                        pass

            restored_count += 1

        # Restore global state
        if "exploration_state" in state and "manager_state" in state["exploration_state"]:
            s.exploration.load_state(state["exploration_state"]["manager_state"])

        if "action_state" in state:
            s.state_tracker.last_action = state["action_state"]["last_action"]

        # Log warnings
        if missing_count > 0:
            warnings.warn(
                f"Checkpoint has {missing_count} neurons not in current brain. "
                f"Skipped those neurons. Restored {restored_count} neurons.",
                UserWarning,
            )

        # Check for extra neurons in brain
        checkpoint_ids = set(neurons_by_id.keys())
        current_ids = set(s.neuron_ids)
        extra_ids = current_ids - checkpoint_ids

        if len(extra_ids) > 0:
            warnings.warn(
                f"Brain has {len(extra_ids)} neurons not in checkpoint. "
                f"Keeping their current state.",
                UserWarning,
            )

    # =========================================================================
    # Phase 3: Hybrid Format (Auto-Selection)
    # =========================================================================

    def _get_region(self) -> Any:
        """Get the region instance managed by this checkpoint manager."""
        return self.striatum

    def _get_selection_criteria(self) -> Dict[str, Any]:
        """Get region-specific criteria used for format selection."""
        return {
            "n_neurons": self.striatum.d1_size + self.striatum.d2_size,
            "growth_enabled": self.striatum.config.growth_enabled,
        }

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
        n_neurons = s.d1_size + s.d2_size

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

    # save() and load() methods inherited from BaseCheckpointManager

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (from BaseCheckpointManager)
    # =========================================================================

    def _get_neurons_data(self) -> list[Dict[str, Any]]:
        """Extract per-neuron data with incoming synapses for D1 and D2 neurons.

        **Multi-source architecture (Phase 5)**: Synapses are now per-source, so
        each neuron will have synapses grouped by source (e.g., "cortex:l5", "hippocampus").

        Returns:
            List of neuron dicts with membrane, synapses (grouped by source), and eligibility
        """
        s = self.striatum
        neurons = []

        # Extract D1 pathway neurons
        n_d1 = s.d1_size

        for i in range(n_d1):
            neuron_id = (
                s.neuron_ids[i] if i < len(s.neuron_ids) else f"striatum_d1_neuron_{i}_step0"
            )

            # Extract creation step from ID (format: "..._step{N}")
            created_step = 0
            if "_step" in neuron_id:
                created_step = int(neuron_id.split("_step")[1])

            # Collect all incoming synapses from all sources for this D1 neuron
            incoming_synapses = []
            for source_name, source_size in s.input_sources.items():
                # Source format: "source_d1" or "source_d2"
                if source_name.endswith("_d1"):
                    d1_key = source_name
                    base_source = source_name.rsplit("_d1", 1)[0]
                elif source_name.endswith("_d2"):
                    # Skip D2 sources when processing D1 neurons
                    continue
                else:
                    # Source without pathway suffix - add D1 suffix
                    d1_key = f"{source_name}_d1"
                    base_source = source_name

                if d1_key in s.synaptic_weights:
                    weights = s.synaptic_weights[d1_key]
                    eligibility = (
                        s._eligibility_d1.get(d1_key, torch.zeros_like(weights))
                        if hasattr(s, "_eligibility_d1")
                        else torch.zeros_like(weights)
                    )

                    # Extract synapses for this neuron from this source
                    source_prefix = f"{base_source}_neuron"
                    synapses = self._extract_synapses_for_neuron(
                        i, weights, eligibility, source_prefix
                    )
                    incoming_synapses.extend(synapses)

            neuron_data = {
                "id": neuron_id,
                "type": "D1-MSN",
                "region": "striatum",
                "created_step": created_step,
                "membrane": (
                    s.d1_pathway.neurons.membrane[i].item()
                    if s.d1_pathway.neurons.membrane is not None
                    else 0.0
                ),
                "incoming_synapses": incoming_synapses,
            }
            neurons.append(neuron_data)

        # Extract D2 pathway neurons
        n_d2 = s.d2_size

        for i in range(n_d2):
            neuron_idx = n_d1 + i
            neuron_id = (
                s.neuron_ids[neuron_idx]
                if neuron_idx < len(s.neuron_ids)
                else f"striatum_d2_neuron_{i}_step0"
            )

            # Extract creation step from ID
            created_step = 0
            if "_step" in neuron_id:
                created_step = int(neuron_id.split("_step")[1])

            # Collect all incoming synapses from all sources for this D2 neuron
            incoming_synapses = []
            for source_name, source_size in s.input_sources.items():
                # Source format: "source_d1" or "source_d2"
                if source_name.endswith("_d2"):
                    d2_key = source_name
                    base_source = source_name.rsplit("_d2", 1)[0]
                elif source_name.endswith("_d1"):
                    # Skip D1 sources when processing D2 neurons
                    continue
                else:
                    # Source without pathway suffix - add D2 suffix
                    d2_key = f"{source_name}_d2"
                    base_source = source_name

                if d2_key in s.synaptic_weights:
                    weights = s.synaptic_weights[d2_key]
                    eligibility = (
                        s._eligibility_d2.get(d2_key, torch.zeros_like(weights))
                        if hasattr(s, "_eligibility_d2")
                        else torch.zeros_like(weights)
                    )

                    # Extract synapses for this neuron from this source
                    source_prefix = f"{base_source}_neuron"
                    synapses = self._extract_synapses_for_neuron(
                        i, weights, eligibility, source_prefix
                    )
                    incoming_synapses.extend(synapses)

            neuron_data = {
                "id": neuron_id,
                "type": "D2-MSN",
                "region": "striatum",
                "created_step": created_step,
                "membrane": (
                    s.d2_pathway.neurons.membrane[i].item()
                    if s.d2_pathway.neurons.membrane is not None
                    else 0.0
                ),
                "incoming_synapses": incoming_synapses,
            }
            neurons.append(neuron_data)

        return neurons

    def _get_learning_state(self) -> Dict[str, Any]:
        """Extract learning-related state for striatum (homeostasis, trial stats)."""
        s = self.striatum
        return {
            # Trial accumulators (managed by state_tracker)
            "d1_votes_accumulated": s.state_tracker._d1_votes_accumulated.detach().clone(),
            "d2_votes_accumulated": s.state_tracker._d2_votes_accumulated.detach().clone(),
            # Homeostatic state
            "activity_ema": s._activity_ema,
            "trial_spike_count": s._trial_spike_count,
            "trial_timesteps": s._trial_timesteps,
            "homeostatic_scaling_applied": s._homeostatic_scaling_applied,
            # Homeostasis manager state (if enabled)
            "homeostasis_manager_state": (
                s.homeostasis.unified_homeostasis.get_state()
                if (s.homeostasis is not None and s.homeostasis.unified_homeostasis is not None)
                else None
            ),
        }

    def _get_neuromodulator_state(self) -> Dict[str, Any]:
        """Extract neuromodulator state for striatum (dopamine)."""
        s = self.striatum
        # Use centralized neuromodulator API from LearnableComponent
        return {
            "dopamine": (
                s.get_neuromodulator("dopamine") if hasattr(s, "get_neuromodulator") else 0.0
            ),
        }

    def _get_region_state(self) -> Dict[str, Any]:
        """Extract striatum-specific state (exploration, action selection, delays)."""
        s = self.striatum
        return {
            # Exploration state
            "exploring": s.state_tracker.exploring,
            "last_uncertainty": s.state_tracker._last_uncertainty,
            "last_exploration_prob": s.state_tracker._last_exploration_prob,
            "exploration_manager": s.exploration.get_state(),
            # Action selection state
            "last_action": s.state_tracker.last_action,
            "recent_spikes": s.state_tracker.recent_spikes.detach().clone(),
            "last_rpe": s.state_tracker._last_rpe,
            "last_expected": s.state_tracker._last_expected,
            # Goal modulation state (if enabled)
            "pfc_modulation_d1": (
                s.pfc_modulation_d1.detach().clone()
                if hasattr(s, "pfc_modulation_d1") and s.pfc_modulation_d1 is not None
                else None
            ),
            "pfc_modulation_d2": (
                s.pfc_modulation_d2.detach().clone()
                if hasattr(s, "pfc_modulation_d2") and s.pfc_modulation_d2 is not None
                else None
            ),
            # D1/D2 pathway delay buffers (temporal competition)
            "d1_delay_buffer": (
                s._d1_delay_buffer.detach().clone() if s._d1_delay_buffer is not None else None
            ),
            "d2_delay_buffer": (
                s._d2_delay_buffer.detach().clone() if s._d2_delay_buffer is not None else None
            ),
            "d1_delay_ptr": s._d1_delay_ptr,
            "d2_delay_ptr": s._d2_delay_ptr,
        }
