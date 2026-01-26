"""
Cortex Checkpoint Manager - State Serialization for Layered Cortex

This component manages checkpoint serialization and deserialization for the
LayeredCortex region (L4→L2/3→L5→L6a→L6b), following the established pattern
from Striatum, Hippocampus, and Cerebellum checkpoint managers.

**Responsibilities:**
- Serialize complete layered cortex state (all 5 layers: L4, L2/3, L5, L6a, L6b)
- Support multi-layer architecture with inter-layer connections
- Handle recurrent connections in L2/3 with gap junctions
- Manage BCM and STDP learning traces per layer
- Support neuromorphic format for layer-specific neuron tracking

**Used By:**
- `LayeredCortex` (main region class)
- Training scripts that save/load checkpoints
- Curriculum learning system for stage transitions

**Coordinates With:**
- Layer-specific neuron populations (L4, L2/3, L5, L6a, L6b)
- Short-term plasticity (STP) for L2/3 recurrent pathway
- Gap junction coupling for L2/3 synchronization
- Composite learning strategies (BCM + STDP)

**Why Extracted:**
- Orthogonal concern: State management is separate from forward/learning logic
- Complexity reduction: Checkpoint code isolated from main class
- Testability: Can test serialization/deserialization independently
- Maintainability: Checkpoint format changes isolated to single module

**Checkpoint Format:**
The full state dict contains:
- `neuron_state`: Per-layer neuron states (L4, L2/3, L5, L6a, L6b)
- `learning_state`: STDP traces, BCM thresholds, STP state
- `recurrent_state`: L2/3 recurrent activity and gap junction state
- `attention_state`: Gamma attention gating, alpha suppression
- `layer_weights`: Inter-layer connection weights

Author: Thalia Project
Date: January 26, 2026 (Architecture Review Task 1.11 / 2.7)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from thalia.managers import BaseCheckpointManager

if TYPE_CHECKING:
    from .layered_cortex import LayeredCortex


class LayeredCortexCheckpointManager(BaseCheckpointManager):
    """Manages state checkpointing for LayeredCortex.

    Handles:
    - Full state serialization (5 layers with distinct properties)
    - Multi-layer weight matrices and traces
    - L2/3 recurrent state and gap junctions
    - Attention and oscillatory gating state
    - Neuromorphic format for layer-specific neuron tracking

    Inherits from BaseCheckpointManager for shared synapse extraction logic.
    """

    def __init__(self, cortex: LayeredCortex):
        """Initialize checkpoint manager.

        Args:
            cortex: The LayeredCortex instance to manage
        """
        super().__init__(format_version="1.0.0")
        self.cortex = cortex

    def collect_state(self) -> Dict[str, Any]:
        """Collect complete layered cortex state for checkpointing.

        Returns:
            Dict containing all state needed to restore cortex
        """
        c = self.cortex

        # 1. PER-LAYER NEURON STATES
        neuron_state = {
            "l4_size": c.l4_size,
            "l23_size": c.l23_size,
            "l5_size": c.l5_size,
            "l6a_size": c.l6a_size,
            "l6b_size": c.l6b_size,
            "total_neurons": c.l4_size + c.l23_size + c.l5_size + c.l6a_size + c.l6b_size,
        }

        # Extract each layer's neuron state
        if hasattr(c, "l4_neurons") and c.l4_neurons is not None:
            neuron_state["l4_neurons"] = {
                "membrane": c.l4_neurons.membrane.clone(),
                "g_E": c.l4_neurons.g_E.clone(),
                "g_I": c.l4_neurons.g_I.clone(),
                "refractory": c.l4_neurons.refractory.clone(),
            }

        if hasattr(c, "l23_neurons") and c.l23_neurons is not None:
            neuron_state["l23_neurons"] = {
                "membrane": c.l23_neurons.membrane.clone(),
                "g_E": c.l23_neurons.g_E.clone(),
                "g_I": c.l23_neurons.g_I.clone(),
                "refractory": c.l23_neurons.refractory.clone(),
            }

        if hasattr(c, "l5_neurons") and c.l5_neurons is not None:
            neuron_state["l5_neurons"] = {
                "membrane": c.l5_neurons.membrane.clone(),
                "g_E": c.l5_neurons.g_E.clone(),
                "g_I": c.l5_neurons.g_I.clone(),
                "refractory": c.l5_neurons.refractory.clone(),
            }

        if hasattr(c, "l6a_neurons") and c.l6a_neurons is not None:
            neuron_state["l6a_neurons"] = {
                "membrane": c.l6a_neurons.membrane.clone(),
                "g_E": c.l6a_neurons.g_E.clone(),
                "g_I": c.l6a_neurons.g_I.clone(),
                "refractory": c.l6a_neurons.refractory.clone(),
            }

        if hasattr(c, "l6b_neurons") and c.l6b_neurons is not None:
            neuron_state["l6b_neurons"] = {
                "membrane": c.l6b_neurons.membrane.clone(),
                "g_E": c.l6b_neurons.g_E.clone(),
                "g_I": c.l6b_neurons.g_I.clone(),
                "refractory": c.l6b_neurons.refractory.clone(),
            }

        # 2. LEARNING STATE
        learning_state = self._get_learning_state()

        # 3. RECURRENT STATE (L2/3 specific)
        recurrent_state = self._get_recurrent_state()

        # 4. ATTENTION STATE
        attention_state = self._get_attention_state()

        # 5. NEUROMODULATOR STATE
        neuromodulator_state = self._get_neuromodulator_state()

        # 6. REGION-SPECIFIC STATE (weights, routing, etc.)
        region_state = self._get_region_state()

        return {
            "format_version": self.format_version,
            "neuron_state": neuron_state,
            "learning_state": learning_state,
            "recurrent_state": recurrent_state,
            "attention_state": attention_state,
            "neuromodulator_state": neuromodulator_state,
            "region_state": region_state,
        }

    def _get_learning_state(self) -> Dict[str, Any]:
        """Extract learning-related state (traces, strategies, STP)."""
        c = self.cortex
        learning_state = {}

        # Per-layer STDP traces
        if hasattr(c, "l4_trace"):
            learning_state["l4_trace"] = c.l4_trace.clone()
        if hasattr(c, "l23_trace"):
            learning_state["l23_trace"] = c.l23_trace.clone()
        if hasattr(c, "l5_trace"):
            learning_state["l5_trace"] = c.l5_trace.clone()
        if hasattr(c, "l6a_trace"):
            learning_state["l6a_trace"] = c.l6a_trace.clone()
        if hasattr(c, "l6b_trace"):
            learning_state["l6b_trace"] = c.l6b_trace.clone()

        # Learning strategy state
        if hasattr(c, "learning_strategy") and c.learning_strategy is not None:
            learning_state["learning_strategy"] = c.learning_strategy.get_state()

        # BCM thresholds (if using BCM)
        if hasattr(c, "bcm_threshold"):
            learning_state["bcm_threshold"] = c.bcm_threshold.clone()

        # Short-term plasticity for L2/3 recurrent
        if hasattr(c, "stp_l23_recurrent") and c.stp_l23_recurrent is not None:
            learning_state["stp_l23_recurrent"] = c.stp_l23_recurrent.get_state()

        # Homeostasis state
        if hasattr(c, "homeostasis") and c.homeostasis is not None:
            if hasattr(c.homeostasis, "get_state"):
                learning_state["homeostasis"] = c.homeostasis.get_state()
            else:
                # UnifiedHomeostasis doesn't have get_state(), just skip it
                learning_state["homeostasis"] = None

        # E/I balance state
        if hasattr(c, "ei_balance") and c.ei_balance is not None:
            learning_state["ei_balance"] = c.ei_balance.get_state()

        return learning_state

    def _get_recurrent_state(self) -> Dict[str, Any]:
        """Extract L2/3 recurrent-specific state."""
        c = self.cortex
        recurrent_state = {}

        # L2/3 recurrent activity accumulator
        if hasattr(c, "l23_recurrent_activity"):
            recurrent_state["l23_recurrent_activity"] = c.l23_recurrent_activity.clone()

        # Gap junction coupling state
        if hasattr(c, "gap_junctions") and c.gap_junctions is not None:
            recurrent_state["gap_junctions"] = c.gap_junctions.get_state()

        # L2/3 recurrent weights (if stored separately)
        if hasattr(c, "l23_recurrent_weights"):
            recurrent_state["l23_recurrent_weights"] = c.l23_recurrent_weights.clone()

        return recurrent_state

    def _get_attention_state(self) -> Dict[str, Any]:
        """Extract attention and oscillatory gating state."""
        c = self.cortex
        attention_state = {}

        # Gamma attention gating
        if hasattr(c, "gamma_attention_gate"):
            attention_state["gamma_attention_gate"] = c.gamma_attention_gate.clone()
        if hasattr(c, "gamma_attention_phase"):
            attention_state["gamma_attention_phase"] = c.gamma_attention_phase

        # Alpha suppression
        if hasattr(c, "alpha_suppression"):
            attention_state["alpha_suppression"] = c.alpha_suppression

        # Feedforward inhibition
        if hasattr(c, "ffi_strength"):
            attention_state["ffi_strength"] = c.ffi_strength

        # Top-down modulation
        if hasattr(c, "top_down_modulation"):
            attention_state["top_down_modulation"] = c.top_down_modulation.clone()

        return attention_state

    def _get_neuromodulator_state(self) -> Dict[str, Any]:
        """Extract neuromodulator levels."""
        c = self.cortex
        return {
            "dopamine": c.dopamine if hasattr(c, "dopamine") else 0.0,
            "acetylcholine": c.acetylcholine if hasattr(c, "acetylcholine") else 0.0,
            "norepinephrine": c.norepinephrine if hasattr(c, "norepinephrine") else 0.0,
        }

    def _get_region_state(self) -> Dict[str, Any]:
        """Extract region-specific state (inter-layer weights, routing, etc.)."""
        c = self.cortex
        region_state = {}

        # Multi-source synaptic weights (Phase 5 architecture)
        region_state["synaptic_weights"] = {
            key: tensor.detach().clone() for key, tensor in c.synaptic_weights.items()
        }

        # Inter-layer connection weights
        if hasattr(c, "w_l4_l23"):
            region_state["w_l4_l23"] = c.w_l4_l23.clone()
        if hasattr(c, "w_l23_l5"):
            region_state["w_l23_l5"] = c.w_l23_l5.clone()
        if hasattr(c, "w_l23_l6a"):
            region_state["w_l23_l6a"] = c.w_l23_l6a.clone()
        if hasattr(c, "w_l23_l6b"):
            region_state["w_l23_l6b"] = c.w_l23_l6b.clone()

        # Input routing state
        if hasattr(c, "input_router") and c.input_router is not None:
            region_state["input_router"] = c.input_router.get_state()

        # Oscillator phase preferences
        if hasattr(c, "phase_preferences"):
            region_state["phase_preferences"] = c.phase_preferences.clone()

        # Stimulus gating state (StimulusGating doesn't have get_state method)
        if hasattr(c, "stimulus_gating") and c.stimulus_gating is not None:
            region_state["stimulus_gating"] = {
                "threshold": c.stimulus_gating.threshold,
                "max_inhibition": c.stimulus_gating.max_inhibition,
                "decay_rate": c.stimulus_gating.decay_rate,
                "steepness": c.stimulus_gating.steepness,
                "current_inhibition": c.stimulus_gating._current_inhibition,
                "prev_input": c.stimulus_gating._prev_input.clone() if c.stimulus_gating._prev_input is not None else None,
            }

        return region_state

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore complete layered cortex state from checkpoint.

        Args:
            state: State dict from collect_state()
        """
        c = self.cortex

        # Validate checkpoint compatibility
        if "format_version" in state:
            self.validate_checkpoint_compatibility(
                checkpoint_version=state["format_version"], current_version=self.format_version
            )

        # 1. Restore per-layer neuron states
        if "neuron_state" in state:
            neuron_state = state["neuron_state"]

            for layer_name in ["l4", "l23", "l5", "l6a", "l6b"]:
                layer_key = f"{layer_name}_neurons"
                if layer_key in neuron_state:
                    layer_data = neuron_state[layer_key]
                    neurons = getattr(c, layer_key, None)
                    if neurons is not None:
                        if "membrane" in layer_data:
                            neurons.membrane.copy_(layer_data["membrane"])
                        if "g_E" in layer_data:
                            neurons.g_E.copy_(layer_data["g_E"])
                        if "g_I" in layer_data:
                            neurons.g_I.copy_(layer_data["g_I"])
                        if "refractory" in layer_data:
                            neurons.refractory.copy_(layer_data["refractory"])

        # 2. Restore learning state
        if "learning_state" in state:
            learning_state = state["learning_state"]

            # Restore traces
            for trace_name in ["l4_trace", "l23_trace", "l5_trace", "l6a_trace", "l6b_trace"]:
                if trace_name in learning_state and hasattr(c, trace_name):
                    getattr(c, trace_name).copy_(learning_state[trace_name])

            # Restore learning strategy
            if "learning_strategy" in learning_state and hasattr(c, "learning_strategy"):
                c.learning_strategy.load_state(learning_state["learning_strategy"])

            # Restore BCM threshold
            if "bcm_threshold" in learning_state and hasattr(c, "bcm_threshold"):
                c.bcm_threshold.copy_(learning_state["bcm_threshold"])

            # Restore STP
            if "stp_l23_recurrent" in learning_state and hasattr(c, "stp_l23_recurrent"):
                c.stp_l23_recurrent.load_state(learning_state["stp_l23_recurrent"])

            # Restore homeostasis
            if "homeostasis" in learning_state and hasattr(c, "homeostasis"):
                c.homeostasis.load_state(learning_state["homeostasis"])

            # Restore E/I balance
            if "ei_balance" in learning_state and hasattr(c, "ei_balance"):
                c.ei_balance.load_state(learning_state["ei_balance"])

        # 3. Restore recurrent state
        if "recurrent_state" in state:
            recurrent_state = state["recurrent_state"]

            if "l23_recurrent_activity" in recurrent_state and hasattr(c, "l23_recurrent_activity"):
                c.l23_recurrent_activity.copy_(recurrent_state["l23_recurrent_activity"])

            if "gap_junctions" in recurrent_state and hasattr(c, "gap_junctions"):
                c.gap_junctions.load_state(recurrent_state["gap_junctions"])

            if "l23_recurrent_weights" in recurrent_state and hasattr(c, "l23_recurrent_weights"):
                c.l23_recurrent_weights.copy_(recurrent_state["l23_recurrent_weights"])

        # 4. Restore attention state
        if "attention_state" in state:
            attention_state = state["attention_state"]

            if "gamma_attention_gate" in attention_state and hasattr(c, "gamma_attention_gate"):
                c.gamma_attention_gate.copy_(attention_state["gamma_attention_gate"])

            if "gamma_attention_phase" in attention_state:
                c.gamma_attention_phase = attention_state["gamma_attention_phase"]

            if "alpha_suppression" in attention_state:
                c.alpha_suppression = attention_state["alpha_suppression"]

            if "ffi_strength" in attention_state:
                c.ffi_strength = attention_state["ffi_strength"]

            if "top_down_modulation" in attention_state and hasattr(c, "top_down_modulation"):
                c.top_down_modulation.copy_(attention_state["top_down_modulation"])

        # 5. Restore neuromodulators
        if "neuromodulator_state" in state:
            nm_state = state["neuromodulator_state"]
            c.set_neuromodulators(
                dopamine=nm_state.get("dopamine", c.dopamine),
                acetylcholine=nm_state.get("acetylcholine", c.acetylcholine),
                norepinephrine=nm_state.get("norepinephrine", c.norepinephrine),
            )

        # 6. Restore region-specific state
        if "region_state" in state:
            region_state = state["region_state"]

            # Restore multi-source weights
            if "synaptic_weights" in region_state:
                for key, tensor in region_state["synaptic_weights"].items():
                    if key in c.synaptic_weights:
                        c.synaptic_weights[key].copy_(tensor)

            # Restore inter-layer weights
            for weight_name in ["w_l4_l23", "w_l23_l5", "w_l23_l6a", "w_l23_l6b"]:
                if weight_name in region_state and hasattr(c, weight_name):
                    getattr(c, weight_name).copy_(region_state[weight_name])

            # Restore routing
            if "input_router" in region_state and hasattr(c, "input_router"):
                c.input_router.load_state(region_state["input_router"])

            # Restore oscillator state
            if "phase_preferences" in region_state and hasattr(c, "phase_preferences"):
                c.phase_preferences.copy_(region_state["phase_preferences"])

            # Restore stimulus gating
            if "stimulus_gating" in region_state and hasattr(c, "stimulus_gating"):
                c.stimulus_gating.load_state(region_state["stimulus_gating"])

    def _get_neurons_data(self) -> list[Dict[str, Any]]:
        """Extract per-neuron data for neuromorphic format.

        Returns:
            List of neuron dicts from all layers (L4, L2/3, L5, L6a, L6b)
        """
        c = self.cortex
        neurons = []

        # Helper to extract neurons from a layer
        def extract_layer(layer_name: str, layer_neurons, n_neurons: int):
            for i in range(n_neurons):
                neuron_data = {
                    "id": f"cortex_{layer_name}_neuron_{i}",
                    "layer": layer_name,
                    "type": "pyramidal" if layer_name in ["l23", "l5"] else "stellate",
                    "membrane": layer_neurons.membrane[i].item() if layer_neurons else 0.0,
                    "incoming_synapses": [],
                }

                # Extract synapses for this neuron
                # This would need layer-specific weight matrices
                # For multi-source architecture:
                for source_name, weights in c.synaptic_weights.items():
                    if weights.size(0) > i:
                        synapses = self.extract_synapses_for_neuron(
                            neuron_idx=i,
                            weights=weights,
                            source_prefix=source_name,
                            synapse_type="excitatory",
                            sparsity_threshold=1e-8,
                        )
                        neuron_data["incoming_synapses"].extend(synapses)

                neurons.append(neuron_data)

        # Extract all layers
        if hasattr(c, "l4_neurons"):
            extract_layer("L4", c.l4_neurons, c.l4_size)
        if hasattr(c, "l23_neurons"):
            extract_layer("L2/3", c.l23_neurons, c.l23_size)
        if hasattr(c, "l5_neurons"):
            extract_layer("L5", c.l5_neurons, c.l5_size)
        if hasattr(c, "l6a_neurons"):
            extract_layer("L6a", c.l6a_neurons, c.l6a_size)
        if hasattr(c, "l6b_neurons"):
            extract_layer("L6b", c.l6b_neurons, c.l6b_size)

        return neurons

    def get_neuromorphic_state(self) -> Dict[str, Any]:
        """Get cortex state in neuromorphic (neuron-centric) format.

        Stores per-neuron data with explicit layer membership and synapses.
        Useful for tracking layer-specific learning and connectivity patterns.

        Returns:
            Dict with neuromorphic format
        """
        neurons = self._get_neurons_data()
        learning_state = self._get_learning_state()
        neuromodulator_state = self._get_neuromodulator_state()
        region_state = self._get_region_state()

        return self.package_neuromorphic_state(
            neurons=neurons,
            learning_state=learning_state,
            neuromodulator_state=neuromodulator_state,
            region_state=region_state,
            additional_state={
                "recurrent_state": self._get_recurrent_state(),
                "attention_state": self._get_attention_state(),
            },
        )

    # ==================== REQUIRED ABSTRACT METHODS ====================

    def _get_region(self) -> Any:
        """Get the region instance managed by this checkpoint manager."""
        return self.cortex

    def _get_selection_criteria(self) -> Dict[str, Any]:
        """Get region-specific criteria for format selection."""
        total_neurons = (
            self.cortex.n_l4
            + self.cortex.n_l23
            + self.cortex.n_l5
            + self.cortex.n_l6a
            + self.cortex.n_l6b
        )
        return {
            "n_neurons": total_neurons,
            "growth_enabled": False,  # LayeredCortex currently doesn't support growth
            "region_type": "layered_cortex",
        }

    def _should_use_neuromorphic(self) -> bool:
        """Determine if neuromorphic format should be used.

        For cortex: Use elastic tensor format (more efficient for large layers).
        """
        return False  # Use elastic tensor format

    def load_neuromorphic_state(self, state: Dict[str, Any]) -> None:
        """Load cortical state from neuromorphic format.

        Currently not implemented as cortex uses elastic tensor format.
        If needed in future, would handle layer-by-layer neuron restoration.

        Args:
            state: Neuromorphic checkpoint dict
        """
        raise NotImplementedError(
            "LayeredCortex uses elastic tensor format. "
            "Use collect_state()/restore_state() for checkpointing."
        )
