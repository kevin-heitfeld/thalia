"""
Hippocampus Checkpoint Manager - State Serialization with Neuromorphic Support

This component manages checkpoint serialization and deserialization for the
Trisynaptic Hippocampus region (DG→CA3→CA1), supporting both elastic tensor
and neuromorphic formats.

**Responsibilities:**
- Serialize complete hippocampus state (all three layers: DG, CA3, CA1)
- Support elastic tensor format (matrix-based)
- Support neuromorphic format (neuron-centric with IDs)
- Hybrid auto-selection based on region size/growth frequency
- Handle neuron-level growth (neurogenesis) gracefully

**Why Neuromorphic for Hippocampus:**
The hippocampus is a prime candidate for neuromorphic checkpoints because:
1. **Neurogenesis**: DG adds new neurons throughout life (granule cell proliferation)
2. **Small Scale**: DG/CA3/CA1 are relatively small (~100-500 neurons each)
3. **Episodic Memory**: Neuron identities matter for memory traces
4. **Inspectability**: Can track which neurons encode which episodes

**Format Comparison:**

Elastic Tensor (good for static regions):
```python
checkpoint = {
    "w_ec_dg": torch.Tensor[n_dg, n_input],
    "w_dg_ca3": torch.Tensor[n_ca3, n_dg],
    # ... 7 weight matrices total
}
```

Neuromorphic (good for dynamic regions):
```python
checkpoint = {
    "neurons": [
        {
            "id": "hippo_dg_neuron_0_step0",
            "layer": "DG",
            "created_step": 0,
            "membrane": 0.5,
            "incoming_synapses": [
                {"from": "ec_neuron_42", "weight": 0.3}
            ]
        },
        # ... one per neuron in DG, CA3, CA1
    ]
}
```

Author: Thalia Project
Date: December 13, 2025 (hippocampus neuromorphic checkpoint support)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any

import torch

from thalia.managers import BaseCheckpointManager

if TYPE_CHECKING:
    from thalia.regions.hippocampus.trisynaptic import TrisynapticHippocampus


class HippocampusCheckpointManager(BaseCheckpointManager):
    """Manages state checkpointing for Trisynaptic Hippocampus.

    Handles:
    - Elastic tensor format for stable configurations
    - Neuromorphic format for neurogenesis-enabled configurations
    - Hybrid auto-selection based on region properties

    Inherits from BaseCheckpointManager for shared synapse extraction logic.
    """

    def __init__(self, hippocampus: TrisynapticHippocampus):
        """Initialize checkpoint manager.

        Args:
            hippocampus: The TrisynapticHippocampus instance to manage
        """
        super().__init__(format_version="2.0.0")
        self.hippocampus = hippocampus

    # =========================================================================
    # NEUROMORPHIC FORMAT - Neuron-Centric Checkpoints
    # Note: _extract_synapses_for_neuron is provided by BaseCheckpointManager
    # =========================================================================

    def get_neuromorphic_state(self) -> Dict[str, Any]:
        """Get hippocampus state in neuromorphic (neuron-centric) format.

        Stores per-neuron data with explicit synapses for all three layers:
        DG, CA3, and CA1. This format supports neurogenesis gracefully.

        Returns:
            Dict with format: {
                "format": "neuromorphic",
                "neurons": [
                    {
                        "id": "hippo_dg_neuron_0_step0",
                        "layer": "DG",
                        "created_step": 0,
                        "membrane": 0.5,
                        "incoming_synapses": [...]
                    }
                ]
            }
        """
        h = self.hippocampus

        # Use abstract method implementations to extract state
        neurons = self._get_neurons_data()
        learning_state = self._get_learning_state()
        neuromodulator_state = self._get_neuromodulator_state()
        region_state = self._get_region_state()

        # Get episode buffer state (hippocampus-specific)
        episode_buffer_state = []
        for ep in h.episode_buffer:
            ep_state = {
                "state": ep.state.detach().clone(),
                "context": ep.context.detach().clone() if ep.context is not None else None,
                "action": ep.action,
                "reward": ep.reward,
                "correct": ep.correct,
                "metadata": ep.metadata,
                "priority": ep.priority,
                "timestamp": ep.timestamp,
                "sequence": [s.detach().clone() for s in ep.sequence] if ep.sequence is not None else None,
            }
            episode_buffer_state.append(ep_state)

        # Get oscillator state (hippocampus-specific)
        oscillator_state = {}
        if h.replay_engine is not None:
            oscillator_state["replay_engine"] = h.replay_engine.get_state()

        # Package using base class method with additional hippocampus-specific state
        return self.package_neuromorphic_state(
            neurons=neurons,
            learning_state=learning_state,
            neuromodulator_state=neuromodulator_state,
            region_state=region_state,
            additional_state={
                "episode_buffer": episode_buffer_state,
                "oscillator_state": oscillator_state,
            },
        )

    def _get_learning_state(self) -> Dict[str, Any]:
        """Extract learning-related state (STP, traces, etc.)."""
        h = self.hippocampus
        learning_state = {}

        if h.stp_mossy is not None:
            learning_state["stp_mossy"] = h.stp_mossy.get_state()
        if h.stp_schaffer is not None:
            learning_state["stp_schaffer"] = h.stp_schaffer.get_state()
        if h.stp_ec_ca1 is not None:
            learning_state["stp_ec_ca1"] = h.stp_ec_ca1.get_state()
        if h.stp_ca3_recurrent is not None:
            learning_state["stp_ca3_recurrent"] = h.stp_ca3_recurrent.get_state()

        return learning_state

    def _get_oscillator_state(self) -> Dict[str, Any]:
        """Extract oscillator-related state."""
        h = self.hippocampus
        oscillator_state = {}

        if h.replay_engine is not None:
            oscillator_state["replay_engine"] = h.replay_engine.get_state()

        return oscillator_state

    def _get_region_state(self) -> Dict[str, Any]:
        """Extract region-specific state (traces, buffers, etc.)."""
        h = self.hippocampus

        return {
            "ca3_activity_trace": h._ca3_activity_trace.detach().clone() if h._ca3_activity_trace is not None else None,
            "pending_theta_reset": h._pending_theta_reset,
            "sequence_position": h._sequence_position,
            "ca3_threshold_offset": h._ca3_threshold_offset.detach().clone() if h._ca3_threshold_offset is not None else None,
            "ca3_activity_history": h._ca3_activity_history.detach().clone() if h._ca3_activity_history is not None else None,
            "ca3_slot_assignment": h._ca3_slot_assignment.detach().clone() if h._ca3_slot_assignment is not None else None,
            "dg_ca3_delay_buffer": h._dg_ca3_delay_buffer.detach().clone() if h._dg_ca3_delay_buffer is not None else None,
            "dg_ca3_delay_pointer": h._dg_ca3_delay_ptr,
            "ca3_ca1_delay_buffer": h._ca3_ca1_delay_buffer.detach().clone() if h._ca3_ca1_delay_buffer is not None else None,
            "ca3_ca1_delay_pointer": h._ca3_ca1_delay_ptr,
            "trisynaptic_state": {
                "dg_spikes": h.state.dg_spikes.detach().clone() if h.state.dg_spikes is not None else None,
                "ca3_spikes": h.state.ca3_spikes.detach().clone() if h.state.ca3_spikes is not None else None,
                "ca1_spikes": h.state.ca1_spikes.detach().clone() if h.state.ca1_spikes is not None else None,
                "ca3_membrane": h.state.ca3_membrane.detach().clone() if h.state.ca3_membrane is not None else None,
                "ca3_persistent": h.state.ca3_persistent.detach().clone() if h.state.ca3_persistent is not None else None,
                "sample_trace": h.state.sample_trace.detach().clone() if h.state.sample_trace is not None else None,
                "dg_trace": h.state.dg_trace.detach().clone() if h.state.dg_trace is not None else None,
                "ca3_trace": h.state.ca3_trace.detach().clone() if h.state.ca3_trace is not None else None,
                "nmda_trace": h.state.nmda_trace.detach().clone() if h.state.nmda_trace is not None else None,
                "stored_dg_pattern": h.state.stored_dg_pattern.detach().clone() if h.state.stored_dg_pattern is not None else None,
                "ffi_strength": h.state.ffi_strength,
            }
        }

    def load_neuromorphic_state(self, state: Dict[str, Any]) -> None:
        """Load hippocampus state from neuromorphic format.

        Matches neurons by ID and layer. Handles:
        - Missing neurons (checkpoint neurons not in current brain) → skip with warning
        - Extra neurons (current brain neurons not in checkpoint) → keep current state
        - Neurogenesis (new neurons added after checkpoint) → initialized normally

        Args:
            state: Dict from get_neuromorphic_state()
        """
        h = self.hippocampus

        # Build ID -> neuron data mapping from checkpoint
        checkpoint_neurons = {n["id"]: n for n in state["neurons"]}

        # Restore DG neurons
        n_dg = h.dg_size
        dg_weights_restored = torch.zeros_like(h.synaptic_weights["ec_dg"])

        for i in range(n_dg):
            neuron_id = f"hippo_dg_neuron_{i}_step0"

            if neuron_id in checkpoint_neurons:
                neuron_data = checkpoint_neurons[neuron_id]

                # Restore membrane potential
                if h.dg_neurons.membrane is not None:
                    h.dg_neurons.membrane[i] = neuron_data["membrane"]

                # Restore synapses
                for synapse in neuron_data["incoming_synapses"]:
                    # Parse source ID to get index
                    source_id = synapse["from"]
                    if source_id.startswith("ec_neuron_"):
                        source_idx = int(source_id.split("_")[-1])
                        if source_idx < h.synaptic_weights["ec_dg"].shape[1]:
                            dg_weights_restored[i, source_idx] = synapse["weight"]

        h.synaptic_weights["ec_dg"].data = dg_weights_restored

        # Restore CA3 neurons
        n_ca3 = h.ca3_size
        ca3_dg_weights_restored = torch.zeros_like(h.w_dg_ca3)
        ca3_ca3_weights_restored = torch.zeros_like(h.w_ca3_ca3)

        for i in range(n_ca3):
            neuron_id = f"hippo_ca3_neuron_{i}_step0"

            if neuron_id in checkpoint_neurons:
                neuron_data = checkpoint_neurons[neuron_id]

                # Restore membrane potential
                if h.ca3_neurons.membrane is not None:
                    h.ca3_neurons.membrane[i] = neuron_data["membrane"]

                # Restore synapses (both DG and recurrent CA3)
                for synapse in neuron_data["incoming_synapses"]:
                    source_id = synapse["from"]

                    if source_id.startswith("hippo_dg_neuron_"):
                        source_idx = int(source_id.split("_")[-2])  # Extract index before "_step0"
                        if source_idx < h.w_dg_ca3.shape[1]:
                            ca3_dg_weights_restored[i, source_idx] = synapse["weight"]

                    elif source_id.startswith("hippo_ca3_neuron_"):
                        source_idx = int(source_id.split("_")[-2])
                        if source_idx < h.w_ca3_ca3.shape[1]:
                            ca3_ca3_weights_restored[i, source_idx] = synapse["weight"]

        h.w_dg_ca3.data = ca3_dg_weights_restored
        h.w_ca3_ca3.data = ca3_ca3_weights_restored

        # Restore CA1 neurons
        n_ca1 = h.ca1_size
        ca1_ca3_weights_restored = torch.zeros_like(h.w_ca3_ca1)
        ca1_ec_weights_restored = torch.zeros_like(h.synaptic_weights["ec_ca1"])
        w_ec_l3_ca1 = h.synaptic_weights.get("ec_l3_ca1", None)
        ca1_ec_l3_weights_restored = torch.zeros_like(w_ec_l3_ca1) if w_ec_l3_ca1 is not None else None

        for i in range(n_ca1):
            neuron_id = f"hippo_ca1_neuron_{i}_step0"

            if neuron_id in checkpoint_neurons:
                neuron_data = checkpoint_neurons[neuron_id]

                # Restore membrane potential
                if h.ca1_neurons.membrane is not None:
                    h.ca1_neurons.membrane[i] = neuron_data["membrane"]

                # Restore synapses (CA3, EC, and optionally EC L3)
                for synapse in neuron_data["incoming_synapses"]:
                    source_id = synapse["from"]

                    if source_id.startswith("hippo_ca3_neuron_"):
                        source_idx = int(source_id.split("_")[-2])
                        if source_idx < h.w_ca3_ca1.shape[1]:
                            ca1_ca3_weights_restored[i, source_idx] = synapse["weight"]

                    elif source_id.startswith("ec_neuron_"):
                        source_idx = int(source_id.split("_")[-1])
                        if source_idx < h.synaptic_weights["ec_ca1"].shape[1]:
                            ca1_ec_weights_restored[i, source_idx] = synapse["weight"]

                    elif source_id.startswith("ec_l3_neuron_") and ca1_ec_l3_weights_restored is not None:
                        source_idx = int(source_id.split("_")[-1])
                        if w_ec_l3_ca1 is not None and source_idx < w_ec_l3_ca1.shape[1]:
                            ca1_ec_l3_weights_restored[i, source_idx] = synapse["weight"]

        h.w_ca3_ca1.data = ca1_ca3_weights_restored
        h.synaptic_weights["ec_ca1"].data = ca1_ec_weights_restored
        if w_ec_l3_ca1 is not None and ca1_ec_l3_weights_restored is not None:
            h.synaptic_weights["ec_l3_ca1"].data = ca1_ec_l3_weights_restored

        # Update weight reference
        h.weights = h.w_ca3_ca1

        # Restore episode buffer
        from thalia.regions.hippocampus.config import Episode
        h.episode_buffer = []
        for ep_state in state["episode_buffer"]:
            episode = Episode(
                state=ep_state["state"].to(h.device),
                context=ep_state["context"].to(h.device) if ep_state["context"] is not None else None,
                action=ep_state["action"],
                reward=ep_state["reward"],
                correct=ep_state["correct"],
                metadata=ep_state["metadata"],
                priority=ep_state["priority"],
                timestamp=ep_state["timestamp"],
                sequence=[s.to(h.device) for s in ep_state["sequence"]] if ep_state["sequence"] is not None else None,
            )
            h.episode_buffer.append(episode)

        # Restore learning state
        learning_state = state["learning_state"]
        if "stp_mossy" in learning_state and h.stp_mossy is not None:
            h.stp_mossy.load_state(learning_state["stp_mossy"])
        if "stp_schaffer" in learning_state and h.stp_schaffer is not None:
            h.stp_schaffer.load_state(learning_state["stp_schaffer"])
        if "stp_ec_ca1" in learning_state and h.stp_ec_ca1 is not None:
            h.stp_ec_ca1.load_state(learning_state["stp_ec_ca1"])
        if "stp_ca3_recurrent" in learning_state and h.stp_ca3_recurrent is not None:
            h.stp_ca3_recurrent.load_state(learning_state["stp_ca3_recurrent"])

        # Restore oscillator state
        oscillator_state = state["oscillator_state"]
        if "replay_engine" in oscillator_state and h.replay_engine is not None:
            h.replay_engine.load_state(oscillator_state["replay_engine"])

        # Restore neuromodulator state
        neuromodulator_state = state["neuromodulator_state"]
        h.state.dopamine = neuromodulator_state["dopamine"]
        h.state.acetylcholine = neuromodulator_state["acetylcholine"]
        h.state.norepinephrine = neuromodulator_state["norepinephrine"]

        # Restore region state
        region_state = state["region_state"]
        if region_state["ca3_activity_trace"] is not None:
            h._ca3_activity_trace = region_state["ca3_activity_trace"].to(h.device)
        h._pending_theta_reset = region_state["pending_theta_reset"]
        h._sequence_position = region_state["sequence_position"]
        if region_state["ca3_threshold_offset"] is not None:
            h._ca3_threshold_offset = region_state["ca3_threshold_offset"].to(h.device)
        if region_state["ca3_activity_history"] is not None:
            h._ca3_activity_history = region_state["ca3_activity_history"].to(h.device)
        if region_state["ca3_slot_assignment"] is not None:
            h._ca3_slot_assignment = region_state["ca3_slot_assignment"].to(h.device)
        if region_state["dg_ca3_delay_buffer"] is not None:
            h._dg_ca3_delay_buffer = region_state["dg_ca3_delay_buffer"].to(h.device)
            h._dg_ca3_delay_ptr = region_state["dg_ca3_delay_pointer"]
        if region_state["ca3_ca1_delay_buffer"] is not None:
            h._ca3_ca1_delay_buffer = region_state["ca3_ca1_delay_buffer"].to(h.device)
            h._ca3_ca1_delay_ptr = region_state["ca3_ca1_delay_pointer"]

        # Restore trisynaptic state
        tri_state = region_state["trisynaptic_state"]
        h.state.dg_spikes = tri_state["dg_spikes"].to(h.device) if tri_state["dg_spikes"] is not None else None
        h.state.ca3_spikes = tri_state["ca3_spikes"].to(h.device) if tri_state["ca3_spikes"] is not None else None
        h.state.ca1_spikes = tri_state["ca1_spikes"].to(h.device) if tri_state["ca1_spikes"] is not None else None
        h.state.ca3_membrane = tri_state["ca3_membrane"].to(h.device) if tri_state["ca3_membrane"] is not None else None
        h.state.ca3_persistent = tri_state["ca3_persistent"].to(h.device) if tri_state["ca3_persistent"] is not None else None
        h.state.sample_trace = tri_state["sample_trace"].to(h.device) if tri_state["sample_trace"] is not None else None
        h.state.dg_trace = tri_state["dg_trace"].to(h.device) if tri_state["dg_trace"] is not None else None
        h.state.ca3_trace = tri_state["ca3_trace"].to(h.device) if tri_state["ca3_trace"] is not None else None
        h.state.nmda_trace = tri_state["nmda_trace"].to(h.device) if tri_state["nmda_trace"] is not None else None
        h.state.stored_dg_pattern = tri_state["stored_dg_pattern"].to(h.device) if tri_state["stored_dg_pattern"] is not None else None
        h.state.ffi_strength = tri_state["ffi_strength"]

    # =========================================================================
    # HYBRID FORMAT - Auto-Selection Between Elastic and Neuromorphic
    # =========================================================================

    def _should_use_neuromorphic(self) -> bool:
        """Determine if neuromorphic format should be used.

        Decision criteria:
        - Small regions (<300 total neurons): Use neuromorphic (more inspectable)
        - Neurogenesis enabled: Use neuromorphic (ID-based matching handles growth)
        - Otherwise: Use elastic tensor (more efficient)

        Returns:
            bool: True if neuromorphic format should be used
        """
        h = self.hippocampus

        # Count total neurons across all three layers
        n_total = h.dg_size + h.ca3_size + h.ca1_size

        # Threshold: small regions benefit from neuromorphic format
        SIZE_THRESHOLD = 300

        # If region is small, use neuromorphic for better inspectability
        if n_total < SIZE_THRESHOLD:
            return True

        # If neurogenesis enabled (DG growth), use neuromorphic
        # TODO: Check actual neurogenesis config when available
        # For now, default to neuromorphic for all hippocampus (it's usually small)
        return True

    def save(self, path: str | Path) -> Dict[str, Any]:
        """Save checkpoint with automatic format selection.

        Automatically chooses between elastic tensor and neuromorphic formats
        based on region size and neurogenesis settings.

        Args:
            path: Path where checkpoint will be saved

        Returns:
            Dict with checkpoint metadata
        """
        path = Path(path)

        # Auto-select format
        use_neuromorphic = self._should_use_neuromorphic()

        if use_neuromorphic:
            state = self.get_neuromorphic_state()
            format_name = "neuromorphic"
        else:
            state = self.hippocampus.get_full_state()
            format_name = "elastic_tensor"

        # Add hybrid format metadata
        h = self.hippocampus
        state["hybrid_metadata"] = {
            "auto_selected": True,
            "selected_format": format_name,
            "selection_criteria": {
                "n_total_neurons": h.dg_size + h.ca3_size + h.ca1_size,
                "n_dg": h.dg_size,
                "n_ca3": h.ca3_size,
                "n_ca1": h.ca1_size,
            }
        }

        # Save to disk
        torch.save(state, path)

        return {
            "path": str(path),
            "format": format_name,
            "n_total_neurons": h.dg_size + h.ca3_size + h.ca1_size,
            "file_size": path.stat().st_size if path.exists() else 0,
        }

    def load(self, path: str | Path) -> None:
        """Load checkpoint with automatic format detection.

        Detects format from hybrid_metadata and loads accordingly.

        Args:
            path: Path to checkpoint file
        """
        path = Path(path)

        # Load checkpoint
        state = torch.load(path, weights_only=False)

        # Check hybrid metadata for format
        if "hybrid_metadata" not in state:
            raise ValueError("Checkpoint missing hybrid_metadata - not a valid hybrid format checkpoint")

        # Load based on selected format
        selected_format = state["hybrid_metadata"]["selected_format"]
        if selected_format == "neuromorphic":
            self.load_neuromorphic_state(state)
        else:
            self.hippocampus.load_full_state(state)

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (from BaseCheckpointManager)
    # =========================================================================

    def _get_neurons_data(self) -> list[Dict[str, Any]]:
        """Extract per-neuron data for all three layers (DG, CA3, CA1)."""
        h = self.hippocampus
        neurons = []

        # Extract DG neurons (Dentate Gyrus)
        n_dg = h.dg_size
        dg_membrane = h.dg_neurons.membrane if h.dg_neurons.membrane is not None else torch.zeros(n_dg, device=h.device)

        for i in range(n_dg):
            neuron_id = f"hippo_dg_neuron_{i}_step0"

            neuron_data = {
                "id": neuron_id,
                "layer": "DG",
                "region": "hippocampus",
                "created_step": 0,
                "membrane": dg_membrane[i].item(),
                "incoming_synapses": self.extract_synapses_for_neuron(
                    i, h.synaptic_weights["ec_dg"], "ec_neuron"
                ),
            }
            neurons.append(neuron_data)

        # Extract CA3 neurons
        n_ca3 = h.ca3_size
        ca3_membrane = h.ca3_neurons.membrane if h.ca3_neurons.membrane is not None else torch.zeros(n_ca3, device=h.device)

        for i in range(n_ca3):
            neuron_id = f"hippo_ca3_neuron_{i}_step0"

            # CA3 has two sets of incoming synapses: from DG and from CA3 (recurrent)
            all_synapses = self.extract_multi_source_synapses(
                i,
                weight_source_pairs=[
                    (h.w_dg_ca3, "hippo_dg_neuron"),
                    (h.w_ca3_ca3, "hippo_ca3_neuron"),
                ],
            )

            neuron_data = {
                "id": neuron_id,
                "layer": "CA3",
                "region": "hippocampus",
                "created_step": 0,
                "membrane": ca3_membrane[i].item(),
                "incoming_synapses": all_synapses,
            }
            neurons.append(neuron_data)

        # Extract CA1 neurons
        n_ca1 = h.ca1_size
        ca1_membrane = h.ca1_neurons.membrane if h.ca1_neurons.membrane is not None else torch.zeros(n_ca1, device=h.device)

        for i in range(n_ca1):
            neuron_id = f"hippo_ca1_neuron_{i}_step0"

            # CA1 has multiple input sources
            weight_source_pairs = [
                (h.w_ca3_ca1, "hippo_ca3_neuron"),
                (h.synaptic_weights["ec_ca1"], "ec_neuron"),
            ]

            # Add EC L3 synapses if present
            w_ec_l3_ca1 = h.synaptic_weights.get("ec_l3_ca1", None)
            if w_ec_l3_ca1 is not None:
                weight_source_pairs.append((w_ec_l3_ca1, "ec_l3_neuron"))

            all_synapses = self.extract_multi_source_synapses(i, weight_source_pairs)

            neuron_data = {
                "id": neuron_id,
                "layer": "CA1",
                "region": "hippocampus",
                "created_step": 0,
                "membrane": ca1_membrane[i].item(),
                "incoming_synapses": all_synapses,
            }
            neurons.append(neuron_data)

        return neurons

    def _get_neuromodulator_state(self) -> Dict[str, Any]:
        """Extract neuromodulator state (delegate to hippocampus)."""
        return self.hippocampus.get_neuromodulator_state()
