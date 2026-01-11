"""
Prefrontal Checkpoint Manager - State Serialization with Neuromorphic Support

This component manages checkpoint serialization and deserialization for the
Prefrontal cortex region, supporting both elastic tensor and neuromorphic formats.

**Responsibilities:**
- Serialize complete PFC state (feedforward, recurrent, inhibition weights)
- Support elastic tensor format (matrix-based)
- Support neuromorphic format (neuron-centric with IDs)
- Hybrid auto-selection based on region size/growth frequency
- Handle neuron-level growth (neurogenesis) gracefully
- Preserve working memory and rule representations

**Why Neuromorphic for Prefrontal:**
The prefrontal cortex benefits from neuromorphic checkpoints because:
1. **Growth Support**: Has grow_output() for expanding working memory capacity
2. **Small Scale**: Typically 50-200 neurons (working memory slots)
3. **Semantic Neurons**: Rule neurons and WM slots have meaningful identities
4. **Inspectability**: Can track which neurons encode which rules/memories

**Format Comparison:**

Elastic Tensor (good for static large regions):
```python
checkpoint = {
    "feedforward": torch.Tensor[n_output, n_input],
    "recurrent": torch.Tensor[n_output, n_output],
    "inhibition": torch.Tensor[n_output, n_output],
}
```

Neuromorphic (good for dynamic small regions):
```python
checkpoint = {
    "neurons": [
        {
            "id": "pfc_neuron_0_step0",
            "type": "rule_neuron",  # or "wm_neuron"
            "created_step": 0,
            "membrane": 0.5,
            "working_memory": 0.7,
            "incoming_synapses": [
                {"from": "input_42", "weight": 0.3, "type": "feedforward"},
                {"from": "pfc_neuron_5", "weight": 0.4, "type": "recurrent"},
            ]
        },
        # ... one per neuron
    ]
}
```

Author: Thalia Project
Date: December 13, 2025 (prefrontal neuromorphic checkpoint support)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any

import torch

from thalia.managers import BaseCheckpointManager

if TYPE_CHECKING:
    from thalia.regions.prefrontal import Prefrontal


class PrefrontalCheckpointManager(BaseCheckpointManager):
    """Manages state checkpointing for Prefrontal cortex.

    Handles:
    - Elastic tensor format for stable configurations
    - Neuromorphic format for growth-enabled configurations
    - Hybrid auto-selection based on region properties

    Inherits from BaseCheckpointManager for shared synapse extraction logic.
    """

    def __init__(self, prefrontal: Prefrontal):
        """Initialize checkpoint manager.

        Args:
            prefrontal: The Prefrontal instance to manage
        """
        super().__init__(format_version="2.0.0")
        self.prefrontal = prefrontal

    # =========================================================================
    # NEUROMORPHIC FORMAT - Neuron-Centric Checkpoints
    # =========================================================================

    def _extract_synapses_for_neuron(
        self,
        neuron_idx: int,
        feedforward_weights: torch.Tensor,
        recurrent_weights: torch.Tensor,
        inhibition_weights: torch.Tensor,
        input_prefix: str = "input"
    ) -> list[Dict[str, Any]]:
        """Extract incoming synapses for a single PFC neuron.

        Args:
            neuron_idx: Index of target neuron
            feedforward_weights: Feedforward weight matrix [n_output, n_input]
            recurrent_weights: Recurrent weight matrix [n_output, n_output]
            inhibition_weights: Inhibition weight matrix [n_output, n_output]
            input_prefix: Prefix for input neuron IDs

        Returns:
            List of synapse dicts with {from, weight, type}
        """
        # Use base class method for typed synapse extraction
        return self.extract_typed_synapses(
            neuron_idx,
            typed_weights=[
                (feedforward_weights, input_prefix, "feedforward"),
                (recurrent_weights, "pfc_neuron", "recurrent"),
                (inhibition_weights, "pfc_neuron", "inhibitory"),
            ],
        )

    def get_neuromorphic_state(self) -> Dict[str, Any]:
        """Get PFC state in neuromorphic (neuron-centric) format.

        Stores per-neuron data with explicit synapses for feedforward,
        recurrent, and inhibitory connections. Includes working memory
        and rule state per neuron.

        Returns:
            Dict with format: {
                "format": "neuromorphic",
                "neurons": [
                    {
                        "id": "pfc_neuron_0_step0",
                        "type": "rule_neuron",
                        "created_step": 0,
                        "membrane": 0.5,
                        "working_memory": 0.7,
                        "incoming_synapses": [...]
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
        """Load PFC state from neuromorphic format.

        Matches neurons by ID. Handles:
        - Missing neurons (checkpoint neurons not in current brain) → skip with warning
        - Extra neurons (current brain neurons not in checkpoint) → keep current state
        - Growth (new neurons added after checkpoint) → initialized normally

        Args:
            state: Dict from get_neuromorphic_state()
        """
        pfc = self.prefrontal

        # Build ID -> neuron data mapping from checkpoint
        checkpoint_neurons = {n["id"]: n for n in state["neurons"]}

        # Count neurons
        n_neurons = pfc.n_output

        # Restore weights
        ff_weights_restored = torch.zeros_like(pfc.synaptic_weights["default"])
        rec_weights_restored = torch.zeros_like(pfc.rec_weights)
        inhib_weights_restored = torch.zeros_like(pfc.inhib_weights)

        # Restore per-neuron state
        membrane_restored = torch.zeros(n_neurons, device=pfc.device)
        wm_restored = torch.zeros(n_neurons, device=pfc.device)
        update_gate_restored = torch.zeros(n_neurons, device=pfc.device)

        for i in range(n_neurons):
            neuron_id = f"pfc_neuron_{i}_step0"

            if neuron_id in checkpoint_neurons:
                neuron_data = checkpoint_neurons[neuron_id]

                # Restore membrane potential and working memory
                membrane_restored[i] = neuron_data["membrane"]
                wm_restored[i] = neuron_data["working_memory"]
                update_gate_restored[i] = neuron_data["update_gate"]

                # Restore synapses
                for synapse in neuron_data["incoming_synapses"]:
                    source_id = synapse["from"]
                    weight = synapse["weight"]
                    synapse_type = synapse["type"]

                    if synapse_type == "feedforward":
                        # Parse input index
                        if source_id.startswith("input_"):
                            source_idx = int(source_id.split("_")[-1])
                            if source_idx < ff_weights_restored.shape[1]:
                                ff_weights_restored[i, source_idx] = weight

                    elif synapse_type == "recurrent":
                        # Parse PFC neuron index
                        if source_id.startswith("pfc_neuron_"):
                            source_idx = int(source_id.split("_")[2])  # Extract index from "pfc_neuron_X_step0"
                            if source_idx < rec_weights_restored.shape[1]:
                                rec_weights_restored[i, source_idx] = weight

                    elif synapse_type == "inhibitory":
                        # Parse PFC neuron index
                        if source_id.startswith("pfc_neuron_"):
                            source_idx = int(source_id.split("_")[2])
                            if source_idx < inhib_weights_restored.shape[1]:
                                inhib_weights_restored[i, source_idx] = weight

        # Apply restored weights
        pfc.synaptic_weights["default"].data = ff_weights_restored
        pfc.rec_weights.data = rec_weights_restored
        pfc.inhib_weights.data = inhib_weights_restored

        # Apply restored neuron state
        pfc.state.membrane = membrane_restored
        pfc.state.working_memory = wm_restored
        pfc.state.update_gate = update_gate_restored

        # Also update neurons module state
        if pfc.neurons.membrane is not None:
            pfc.neurons.membrane.data = membrane_restored

        # Restore learning state
        learning_state = state["learning_state"]
        if "stdp_strategy" in learning_state and hasattr(pfc, 'learning_strategy'):
            if hasattr(pfc.learning_strategy, 'load_state'):
                pfc.learning_strategy.load_state(learning_state["stdp_strategy"])

        if "stp_recurrent" in learning_state and pfc.stp_recurrent is not None:
            pfc.stp_recurrent.load_state(learning_state["stp_recurrent"])

        # Restore neuromodulator state
        neuromodulator_state = state["neuromodulator_state"]
        pfc.state.dopamine = neuromodulator_state["dopamine"]
        pfc.dopamine_system.load_state(neuromodulator_state["dopamine_system"])

        # Restore region state
        region_state = state["region_state"]
        pfc.state.spikes = region_state["spikes"].to(pfc.device) if region_state["spikes"] is not None else None
        pfc.state.active_rule = region_state["active_rule"].to(pfc.device) if region_state["active_rule"] is not None else None

    # =========================================================================
    # HYBRID FORMAT - Auto-Selection Between Elastic and Neuromorphic
    # =========================================================================

    def _get_region(self) -> Any:
        """Get the region instance managed by this checkpoint manager."""
        return self.prefrontal

    def _get_selection_criteria(self) -> Dict[str, Any]:
        """Get region-specific criteria used for format selection."""
        pfc = self.prefrontal
        return {
            "n_neurons": pfc.n_output,
            "has_growth": hasattr(pfc, 'grow_output'),
        }

    def _should_use_neuromorphic(self) -> bool:
        """Determine if neuromorphic format should be used.

        Decision criteria:
        - Small regions (<200 neurons): Use neuromorphic (more inspectable)
        - Growth enabled: Use neuromorphic (ID-based matching handles growth)
        - Otherwise: Use elastic tensor (more efficient)

        Returns:
            bool: True if neuromorphic format should be used
        """
        pfc = self.prefrontal

        # Count neurons
        n_neurons = pfc.n_output

        # Threshold: small regions benefit from neuromorphic format
        SIZE_THRESHOLD = 200

        # If region is small, use neuromorphic for better inspectability
        if n_neurons < SIZE_THRESHOLD:
            return True

        # If growth enabled, use neuromorphic
        if hasattr(pfc, 'grow_output'):  # Has growth capability
            return True

        # For large stable regions, elastic tensor is more efficient
        return False

    # save() and load() methods inherited from BaseCheckpointManager

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (from BaseCheckpointManager)
    # =========================================================================

    def _get_neurons_data(self) -> list[Dict[str, Any]]:
        """Extract per-neuron data with incoming synapses."""
        pfc = self.prefrontal
        neurons = []

        n_neurons = pfc.n_neurons  # Use instance variable, not config
        membrane = pfc.state.membrane if pfc.state.membrane is not None else torch.zeros(n_neurons, device=pfc.device)
        wm = pfc.state.working_memory if pfc.state.working_memory is not None else torch.zeros(n_neurons, device=pfc.device)
        update_gate = pfc.state.update_gate if pfc.state.update_gate is not None else torch.zeros(n_neurons, device=pfc.device)

        for i in range(n_neurons):
            # Generate stable neuron ID with actual creation timestep
            birth_step = pfc._neuron_birth_steps[i].item() if hasattr(pfc, '_neuron_birth_steps') else 0
            neuron_id = f"pfc_neuron_{i}_step{birth_step}"

            # Determine neuron type (rule vs working memory)
            # For now, simple heuristic: if update_gate is high, it's a WM neuron
            neuron_type = "wm_neuron" if update_gate[i].item() > 0.5 else "rule_neuron"

            neuron_data = {
                "id": neuron_id,
                "type": neuron_type,
                "region": "prefrontal",
                "created_step": birth_step,
                "membrane": membrane[i].item(),
                "working_memory": wm[i].item(),
                "update_gate": update_gate[i].item(),
                "incoming_synapses": self._extract_synapses_for_neuron(
                    i, pfc.synaptic_weights["default"], pfc.rec_weights, pfc.inhib_weights
                ),
            }
            neurons.append(neuron_data)

        return neurons

    def _get_learning_state(self) -> Dict[str, Any]:
        """Extract learning-related state (STDP, STP, etc.)."""
        pfc = self.prefrontal
        learning_state = {}

        # STDP eligibility traces
        if hasattr(pfc, 'learning_strategy') and pfc.learning_strategy is not None:
            if hasattr(pfc.learning_strategy, 'get_state'):
                learning_state["stdp_strategy"] = pfc.learning_strategy.get_state()

        # STP state for recurrent connections
        if pfc.stp_recurrent is not None:
            learning_state["stp_recurrent"] = pfc.stp_recurrent.get_state()

        return learning_state

    def _get_neuromodulator_state(self) -> Dict[str, Any]:
        """Extract neuromodulator-related state."""
        pfc = self.prefrontal
        return {
            "dopamine": pfc.state.dopamine,
            "dopamine_system": pfc.dopamine_system.get_state(),
        }

    def _get_region_state(self) -> Dict[str, Any]:
        """Extract region-specific state (spikes, rules, etc.)."""
        pfc = self.prefrontal

        return {
            "spikes": pfc.state.spikes.detach().clone() if pfc.state.spikes is not None else None,
            "active_rule": pfc.state.active_rule.detach().clone() if pfc.state.active_rule is not None else None,
        }
