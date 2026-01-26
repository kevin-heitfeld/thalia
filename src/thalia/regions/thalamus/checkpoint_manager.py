"""
Thalamus Checkpoint Manager - State Serialization for Thalamic Relay

This component manages checkpoint serialization and deserialization for the
ThalamicRelay region (relay neurons + TRN gating), following the established
pattern from Striatum, Hippocampus, Cerebellum, and Cortex checkpoint managers.

**Responsibilities:**
- Serialize complete thalamic relay state (relay neurons + TRN)
- Support dual-population architecture (relay + reticular nucleus)
- Handle burst/tonic mode transitions and tracking
- Manage alpha gating for attentional modulation
- Support neuromorphic format for relay-TRN interaction tracking

**Used By:**
- `ThalamicRelay` (main region class)
- Training scripts that save/load checkpoints
- Curriculum learning system for stage transitions

**Coordinates With:**
- Relay neuron population (sensory relay)
- TRN neuron population (attentional gating)
- STP for sensory input and L6 feedback pathways
- Burst/tonic mode controller
- Alpha gating system

**Why Extracted:**
- Orthogonal concern: State management is separate from forward/learning logic
- Complexity reduction: Checkpoint code isolated from main class
- Testability: Can test serialization/deserialization independently
- Maintainability: Checkpoint format changes isolated to single module

**Checkpoint Format:**
The full state dict contains:
- `neuron_state`: Relay and TRN neuron states
- `learning_state`: STDP traces, STP state, homeostasis
- `gating_state`: Alpha gating, burst/tonic mode, inhibition strength
- `relay_state`: Sensory routing, L6 feedback integration

Author: Thalia Project
Date: January 26, 2026 (Architecture Review Task 1.11 / 2.7)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from thalia.managers import BaseCheckpointManager

if TYPE_CHECKING:
    from .thalamus import ThalamicRelay


class ThalamicCheckpointManager(BaseCheckpointManager):
    """Manages state checkpointing for ThalamicRelay.

    Handles:
    - Full state serialization (relay + TRN populations)
    - Dual-population weight matrices and traces
    - Burst/tonic mode tracking and transition state
    - Alpha gating for attentional modulation
    - Neuromorphic format for relay-TRN interaction tracking

    Inherits from BaseCheckpointManager for shared synapse extraction logic.
    """

    def __init__(self, thalamus: ThalamicRelay):
        """Initialize checkpoint manager.

        Args:
            thalamus: The ThalamicRelay instance to manage
        """
        super().__init__(format_version="1.0.0")
        self.thalamus = thalamus

    def collect_state(self) -> Dict[str, Any]:
        """Collect complete thalamic relay state for checkpointing.

        Returns:
            Dict containing all state needed to restore thalamus
        """
        t = self.thalamus

        # 1. NEURON STATES (relay + TRN)
        neuron_state = {
            "relay_size": t.relay_size,
            "trn_size": t.trn_size,
            "total_neurons": t.total_neurons,
        }

        # Extract relay neuron state
        if hasattr(t, "relay_neurons") and t.relay_neurons is not None:
            neuron_state["relay_neurons"] = {
                "membrane": t.relay_neurons.membrane.clone(),
                "g_E": t.relay_neurons.g_E.clone(),
                "g_I": t.relay_neurons.g_I.clone(),
                "refractory": t.relay_neurons.refractory.clone(),
            }

        # Extract TRN neuron state
        if hasattr(t, "trn_neurons") and t.trn_neurons is not None:
            neuron_state["trn_neurons"] = {
                "membrane": t.trn_neurons.membrane.clone(),
                "g_E": t.trn_neurons.g_E.clone(),
                "g_I": t.trn_neurons.g_I.clone(),
                "refractory": t.trn_neurons.refractory.clone(),
            }

        # 2. LEARNING STATE
        learning_state = self._get_learning_state()

        # 3. GATING STATE (alpha, burst/tonic, inhibition)
        gating_state = self._get_gating_state()

        # 4. RELAY STATE (routing, feedback integration)
        relay_state = self._get_relay_state()

        # 5. NEUROMODULATOR STATE
        neuromodulator_state = self._get_neuromodulator_state()

        # 6. REGION-SPECIFIC STATE (weights, routing, etc.)
        region_state = self._get_region_state()

        return {
            "format_version": self.format_version,
            "neuron_state": neuron_state,
            "learning_state": learning_state,
            "gating_state": gating_state,
            "relay_state": relay_state,
            "neuromodulator_state": neuromodulator_state,
            "region_state": region_state,
        }

    def _get_learning_state(self) -> Dict[str, Any]:
        """Extract learning-related state (traces, strategies, STP)."""
        t = self.thalamus
        learning_state = {}

        # Relay neuron STDP traces
        if hasattr(t, "relay_trace"):
            learning_state["relay_trace"] = t.relay_trace.clone()

        # TRN neuron STDP traces
        if hasattr(t, "trn_trace"):
            learning_state["trn_trace"] = t.trn_trace.clone()

        # Learning strategy state
        if hasattr(t, "learning_strategy") and t.learning_strategy is not None:
            learning_state["learning_strategy"] = t.learning_strategy.get_state()

        # Short-term plasticity for sensory input
        if hasattr(t, "stp_sensory") and t.stp_sensory is not None:
            learning_state["stp_sensory"] = t.stp_sensory.get_state()

        # Short-term plasticity for L6 feedback
        if hasattr(t, "stp_l6_feedback") and t.stp_l6_feedback is not None:
            learning_state["stp_l6_feedback"] = t.stp_l6_feedback.get_state()

        # Homeostasis state
        if hasattr(t, "homeostasis") and t.homeostasis is not None:
            learning_state["homeostasis"] = t.homeostasis.get_state()

        return learning_state

    def _get_gating_state(self) -> Dict[str, Any]:
        """Extract gating state (alpha, burst/tonic, TRN inhibition)."""
        t = self.thalamus
        gating_state = {}

        # Alpha gating (attentional modulation)
        if hasattr(t, "alpha_gate"):
            gating_state["alpha_gate"] = t.alpha_gate.clone()
        if hasattr(t, "alpha_phase"):
            gating_state["alpha_phase"] = t.alpha_phase
        if hasattr(t, "alpha_strength"):
            gating_state["alpha_strength"] = t.alpha_strength

        # Burst/tonic mode tracking
        if hasattr(t, "burst_mode"):
            gating_state["burst_mode"] = t.burst_mode.clone()  # Per-neuron boolean
        if hasattr(t, "burst_threshold"):
            gating_state["burst_threshold"] = t.burst_threshold
        if hasattr(t, "tonic_threshold"):
            gating_state["tonic_threshold"] = t.tonic_threshold

        # TRN inhibition strength
        if hasattr(t, "trn_inhibition_strength"):
            gating_state["trn_inhibition_strength"] = t.trn_inhibition_strength

        # TRN activity for gating
        if hasattr(t, "trn_activity"):
            gating_state["trn_activity"] = t.trn_activity.clone()

        return gating_state

    def _get_relay_state(self) -> Dict[str, Any]:
        """Extract relay-specific state (sensory routing, feedback)."""
        t = self.thalamus
        relay_state = {}

        # Sensory relay accumulator
        if hasattr(t, "relay_accumulator"):
            relay_state["relay_accumulator"] = t.relay_accumulator.clone()

        # L6 feedback modulation
        if hasattr(t, "l6_feedback_modulation"):
            relay_state["l6_feedback_modulation"] = t.l6_feedback_modulation.clone()

        # Corticothalamic feedback strength
        if hasattr(t, "ct_feedback_strength"):
            relay_state["ct_feedback_strength"] = t.ct_feedback_strength

        # Sensory gain control
        if hasattr(t, "sensory_gain"):
            relay_state["sensory_gain"] = t.sensory_gain

        return relay_state

    def _get_neuromodulator_state(self) -> Dict[str, Any]:
        """Extract neuromodulator levels."""
        t = self.thalamus
        return {
            "dopamine": t.dopamine if hasattr(t, "dopamine") else 0.0,
            "acetylcholine": t.acetylcholine if hasattr(t, "acetylcholine") else 0.0,
            "norepinephrine": t.norepinephrine if hasattr(t, "norepinephrine") else 0.0,
        }

    def _get_region_state(self) -> Dict[str, Any]:
        """Extract region-specific state (weights, routing, etc.)."""
        t = self.thalamus
        region_state = {}

        # Multi-source synaptic weights (Phase 5 architecture)
        region_state["synaptic_weights"] = {
            key: tensor.detach().clone() for key, tensor in t.synaptic_weights.items()
        }

        # TRN inhibitory weights (TRN → relay)
        if hasattr(t, "w_trn_relay"):
            region_state["w_trn_relay"] = t.w_trn_relay.clone()

        # L6 feedback weights (cortical L6 → thalamus)
        if hasattr(t, "w_l6_feedback"):
            region_state["w_l6_feedback"] = t.w_l6_feedback.clone()

        # Sensory input weights (sensory pathway → relay)
        if hasattr(t, "w_sensory_relay"):
            region_state["w_sensory_relay"] = t.w_sensory_relay.clone()

        # Input routing state
        if hasattr(t, "input_router") and t.input_router is not None:
            region_state["input_router"] = t.input_router.get_state()

        # Oscillator phase preferences
        if hasattr(t, "phase_preferences"):
            region_state["phase_preferences"] = t.phase_preferences.clone()

        return region_state

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore complete thalamic relay state from checkpoint.

        Args:
            state: State dict from collect_state()
        """
        t = self.thalamus

        # Validate checkpoint compatibility
        if "format_version" in state:
            self.validate_checkpoint_compatibility(
                checkpoint_version=state["format_version"], current_version=self.format_version
            )

        # 1. Restore neuron states
        if "neuron_state" in state:
            neuron_state = state["neuron_state"]

            # Restore relay neurons
            if "relay_neurons" in neuron_state and hasattr(t, "relay_neurons"):
                relay_data = neuron_state["relay_neurons"]
                if "membrane" in relay_data:
                    t.relay_neurons.membrane.copy_(relay_data["membrane"])
                if "g_E" in relay_data:
                    t.relay_neurons.g_E.copy_(relay_data["g_E"])
                if "g_I" in relay_data:
                    t.relay_neurons.g_I.copy_(relay_data["g_I"])
                if "refractory" in relay_data:
                    t.relay_neurons.refractory.copy_(relay_data["refractory"])

            # Restore TRN neurons
            if "trn_neurons" in neuron_state and hasattr(t, "trn_neurons"):
                trn_data = neuron_state["trn_neurons"]
                if "membrane" in trn_data:
                    t.trn_neurons.membrane.copy_(trn_data["membrane"])
                if "g_E" in trn_data:
                    t.trn_neurons.g_E.copy_(trn_data["g_E"])
                if "g_I" in trn_data:
                    t.trn_neurons.g_I.copy_(trn_data["g_I"])
                if "refractory" in trn_data:
                    t.trn_neurons.refractory.copy_(trn_data["refractory"])

        # 2. Restore learning state
        if "learning_state" in state:
            learning_state = state["learning_state"]

            # Restore traces
            if "relay_trace" in learning_state and hasattr(t, "relay_trace"):
                t.relay_trace.copy_(learning_state["relay_trace"])
            if "trn_trace" in learning_state and hasattr(t, "trn_trace"):
                t.trn_trace.copy_(learning_state["trn_trace"])

            # Restore learning strategy
            if "learning_strategy" in learning_state and hasattr(t, "learning_strategy"):
                t.learning_strategy.load_state(learning_state["learning_strategy"])

            # Restore STP
            if "stp_sensory" in learning_state and hasattr(t, "stp_sensory"):
                t.stp_sensory.load_state(learning_state["stp_sensory"])
            if "stp_l6_feedback" in learning_state and hasattr(t, "stp_l6_feedback"):
                t.stp_l6_feedback.load_state(learning_state["stp_l6_feedback"])

            # Restore homeostasis
            if "homeostasis" in learning_state and hasattr(t, "homeostasis"):
                t.homeostasis.load_state(learning_state["homeostasis"])

        # 3. Restore gating state
        if "gating_state" in state:
            gating_state = state["gating_state"]

            # Restore alpha gating
            if "alpha_gate" in gating_state and hasattr(t, "alpha_gate"):
                t.alpha_gate.copy_(gating_state["alpha_gate"])
            if "alpha_phase" in gating_state:
                t.alpha_phase = gating_state["alpha_phase"]
            if "alpha_strength" in gating_state:
                t.alpha_strength = gating_state["alpha_strength"]

            # Restore burst/tonic mode
            if "burst_mode" in gating_state and hasattr(t, "burst_mode"):
                t.burst_mode.copy_(gating_state["burst_mode"])
            if "burst_threshold" in gating_state:
                t.burst_threshold = gating_state["burst_threshold"]
            if "tonic_threshold" in gating_state:
                t.tonic_threshold = gating_state["tonic_threshold"]

            # Restore TRN inhibition
            if "trn_inhibition_strength" in gating_state:
                t.trn_inhibition_strength = gating_state["trn_inhibition_strength"]
            if "trn_activity" in gating_state and hasattr(t, "trn_activity"):
                t.trn_activity.copy_(gating_state["trn_activity"])

        # 4. Restore relay state
        if "relay_state" in state:
            relay_state = state["relay_state"]

            if "relay_accumulator" in relay_state and hasattr(t, "relay_accumulator"):
                t.relay_accumulator.copy_(relay_state["relay_accumulator"])

            if "l6_feedback_modulation" in relay_state and hasattr(t, "l6_feedback_modulation"):
                t.l6_feedback_modulation.copy_(relay_state["l6_feedback_modulation"])

            if "ct_feedback_strength" in relay_state:
                t.ct_feedback_strength = relay_state["ct_feedback_strength"]

            if "sensory_gain" in relay_state:
                t.sensory_gain = relay_state["sensory_gain"]

        # 5. Restore neuromodulators
        if "neuromodulator_state" in state:
            nm_state = state["neuromodulator_state"]
            t.set_neuromodulators(
                dopamine=nm_state.get("dopamine", t.dopamine),
                acetylcholine=nm_state.get("acetylcholine", t.acetylcholine),
                norepinephrine=nm_state.get("norepinephrine", t.norepinephrine),
            )

        # 6. Restore region-specific state
        if "region_state" in state:
            region_state = state["region_state"]

            # Restore multi-source weights
            if "synaptic_weights" in region_state:
                for key, tensor in region_state["synaptic_weights"].items():
                    if key in t.synaptic_weights:
                        t.synaptic_weights[key].copy_(tensor)

            # Restore TRN and feedback weights
            for weight_name in ["w_trn_relay", "w_l6_feedback", "w_sensory_relay"]:
                if weight_name in region_state and hasattr(t, weight_name):
                    getattr(t, weight_name).copy_(region_state[weight_name])

            # Restore routing
            if "input_router" in region_state and hasattr(t, "input_router"):
                t.input_router.load_state(region_state["input_router"])

            # Restore oscillator state
            if "phase_preferences" in region_state and hasattr(t, "phase_preferences"):
                t.phase_preferences.copy_(region_state["phase_preferences"])

    def _get_neurons_data(self) -> list[Dict[str, Any]]:
        """Extract per-neuron data for neuromorphic format.

        Returns:
            List of neuron dicts from both populations (relay + TRN)
        """
        t = self.thalamus
        neurons = []

        # Helper to extract neurons from a population
        def extract_population(pop_name: str, pop_neurons, n_neurons: int, neuron_type: str):
            for i in range(n_neurons):
                neuron_data = {
                    "id": f"thalamus_{pop_name}_neuron_{i}",
                    "population": pop_name,
                    "type": neuron_type,
                    "membrane": pop_neurons.membrane[i].item() if pop_neurons else 0.0,
                    "incoming_synapses": [],
                }

                # Extract synapses for this neuron
                # For multi-source architecture:
                for source_name, weights in t.synaptic_weights.items():
                    if weights.size(0) > i:
                        synapses = self.extract_synapses_for_neuron(
                            neuron_idx=i,
                            weights=weights,
                            source_prefix=source_name,
                            synapse_type="excitatory" if pop_name == "relay" else "inhibitory",
                            sparsity_threshold=1e-8,
                        )
                        neuron_data["incoming_synapses"].extend(synapses)

                neurons.append(neuron_data)

        # Extract relay neurons
        if hasattr(t, "relay_neurons"):
            extract_population("relay", t.relay_neurons, t.relay_size, "relay")

        # Extract TRN neurons
        if hasattr(t, "trn_neurons"):
            extract_population("trn", t.trn_neurons, t.trn_size, "interneuron")

        return neurons

    def get_neuromorphic_state(self) -> Dict[str, Any]:
        """Get thalamus state in neuromorphic (neuron-centric) format.

        Stores per-neuron data with explicit population membership and synapses.
        Useful for tracking relay-TRN gating interactions and burst/tonic modes.

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
                "gating_state": self._get_gating_state(),
                "relay_state": self._get_relay_state(),
            },
        )
