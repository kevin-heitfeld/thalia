"""
Cerebellum Checkpoint Manager - State Serialization and Restoration

This component manages checkpoint serialization and deserialization for the
Cerebellum region, following the established pattern from Striatum and Hippocampus.

**Responsibilities:**
- Serialize complete cerebellum state (weights, error signals, microcircuit components)
- Deserialize and restore state from checkpoints
- Support both classic and enhanced microcircuit modes
- Handle neuromorphic format for error-corrective learning

**Used By:**
- `Cerebellum` (main region class)
- Training scripts that save/load checkpoints
- Curriculum learning system for stage transitions

**Coordinates With:**
- `GranuleCellLayer`: Serializes granule cell state (if enhanced mode)
- `EnhancedPurkinjeCell`: Serializes Purkinje cell dendrites (if enhanced)
- `DeepCerebellarNuclei`: Serializes DCN state (if enhanced)
- `ClimbingFiberSystem`: Serializes error signals
- `EligibilityTraceManager`: Serializes learning traces

**Why Extracted:**
- Orthogonal concern: State management is separate from forward/learning logic
- Complexity reduction: Checkpoint code isolated from main class
- Testability: Can test serialization/deserialization independently
- Maintainability: Checkpoint format changes isolated to single module

**Checkpoint Format:**
The full state dict contains:
- `neuron_state`: Purkinje cell state (classic or enhanced microcircuit)
- `learning_state`: Eligibility traces, error signals, STP state
- `climbing_fiber_state`: Error signal from inferior olive
- `microcircuit_state`: Granule, Purkinje, DCN components (if enhanced)

Author: Thalia Project
Date: January 26, 2026 (Architecture Review Task 1.11 / 2.7)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict

from thalia.managers import BaseCheckpointManager

if TYPE_CHECKING:
    from .cerebellum import Cerebellum


class CerebellumCheckpointManager(BaseCheckpointManager):
    """Manages state checkpointing for Cerebellum.

    Handles:
    - Full state serialization (weights, traces, error signals)
    - State restoration with elastic tensor support
    - Support for both classic and enhanced microcircuit modes
    - Neuromorphic format for growth-enabled configurations

    Inherits from BaseCheckpointManager for shared synapse extraction logic.
    """

    def __init__(self, cerebellum: Cerebellum):
        """Initialize checkpoint manager.

        Args:
            cerebellum: The Cerebellum instance to manage
        """
        super().__init__(format_version="1.0.0")
        self.cerebellum = cerebellum

    def collect_state(self) -> Dict[str, Any]:
        """Collect complete cerebellum state for checkpointing.

        Returns:
            Dict containing all state needed to restore cerebellum
        """
        c = self.cerebellum

        # 1. NEURON STATE - Purkinje cells (primary output neurons)
        neuron_state = self.extract_neuron_state_common(
            neurons=c.purkinje_neurons if hasattr(c, "purkinje_neurons") else None,
            n_neurons=c.purkinje_size,
            device=c.device,
        )
        neuron_state.update(
            {
                "purkinje_size": c.purkinje_size,
                "granule_size": c.granule_size,
                "use_enhanced_microcircuit": c.config.use_enhanced_microcircuit,
            }
        )

        # 2. LEARNING STATE
        learning_state = self._get_learning_state()

        # 3. CLIMBING FIBER STATE
        climbing_fiber_state = c.climbing_fiber.get_state() if c.climbing_fiber else {}

        # 4. ENHANCED MICROCIRCUIT STATE (if applicable)
        microcircuit_state = {}
        if c.config.use_enhanced_microcircuit:
            microcircuit_state = {
                "granule_layer": c.granule_layer.get_state() if hasattr(c, "granule_layer") else {},
                "purkinje_cells": (
                    [cell.get_state() for cell in c.purkinje_cells]
                    if hasattr(c, "purkinje_cells")
                    else []
                ),
                "deep_nuclei": c.deep_nuclei.get_state() if hasattr(c, "deep_nuclei") else {},
            }

        # 5. NEUROMODULATOR STATE
        neuromodulator_state = self._get_neuromodulator_state()

        # 6. REGION-SPECIFIC STATE
        region_state = self._get_region_state()

        return {
            "format_version": self.format_version,
            "config": asdict(c.config),
            "neuron_state": neuron_state,
            "learning_state": learning_state,
            "climbing_fiber_state": climbing_fiber_state,
            "microcircuit_state": microcircuit_state,
            "neuromodulator_state": neuromodulator_state,
            "region_state": region_state,
        }

    def _get_learning_state(self) -> Dict[str, Any]:
        """Extract learning-related state (traces, eligibility, STP)."""
        c = self.cerebellum
        learning_state = {}

        # Eligibility trace manager state
        if hasattr(c, "trace_manager") and c.trace_manager is not None:
            learning_state["trace_manager"] = c.trace_manager.get_state()

        # Short-term plasticity state
        if hasattr(c, "stp_pf_purkinje") and c.stp_pf_purkinje is not None:
            learning_state["stp_pf_purkinje"] = c.stp_pf_purkinje.get_state()

        if hasattr(c, "stp_mf_granule") and c.stp_mf_granule is not None:
            learning_state["stp_mf_granule"] = c.stp_mf_granule.get_state()

        # Homeostasis state
        if hasattr(c, "homeostasis") and c.homeostasis is not None:
            if hasattr(c.homeostasis, "get_state"):
                learning_state["homeostasis"] = c.homeostasis.get_state()
            else:
                # UnifiedHomeostasis doesn't have get_state(), just skip it
                learning_state["homeostasis"] = None

        # Learning rate and error history
        learning_state["learning_rate"] = c.config.learning_rate
        if hasattr(c, "error_history"):
            learning_state["error_history"] = c.error_history.clone()

        return learning_state

    def _get_neuromodulator_state(self) -> Dict[str, Any]:
        """Extract neuromodulator levels."""
        c = self.cerebellum
        return {
            "dopamine": c.dopamine if hasattr(c, "dopamine") else 0.0,
            "acetylcholine": c.acetylcholine if hasattr(c, "acetylcholine") else 0.0,
            "norepinephrine": c.norepinephrine if hasattr(c, "norepinephrine") else 0.0,
        }

    def _get_region_state(self) -> Dict[str, Any]:
        """Extract region-specific state."""
        c = self.cerebellum
        region_state = {}

        # Multi-source weights (Phase 5 architecture)
        region_state["synaptic_weights"] = {
            key: tensor.detach().clone() for key, tensor in c.synaptic_weights.items()
        }

        # Input routing state
        if hasattr(c, "input_router") and c.input_router is not None:
            region_state["input_router"] = c.input_router.get_state()

        # Oscillator state (if applicable)
        if hasattr(c, "phase_preferences"):
            region_state["phase_preferences"] = c.phase_preferences.clone()

        return region_state

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore complete cerebellum state from checkpoint.

        Args:
            state: State dict from collect_state()
        """
        c = self.cerebellum

        # Validate checkpoint compatibility
        if "format_version" in state:
            self.validate_checkpoint_compatibility(
                checkpoint_version=state["format_version"], current_version=self.format_version
            )

        # 1. Restore neuron state
        if "neuron_state" in state:
            neuron_state = state["neuron_state"]
            # Restore to Purkinje neurons
            if hasattr(c, "purkinje_neurons") and c.purkinje_neurons is not None:
                if "voltage" in neuron_state:
                    c.purkinje_neurons.membrane.copy_(neuron_state["voltage"])
                if "conductance_E" in neuron_state:
                    c.purkinje_neurons.g_E.copy_(neuron_state["conductance_E"])
                if "conductance_I" in neuron_state:
                    c.purkinje_neurons.g_I.copy_(neuron_state["conductance_I"])

        # 2. Restore learning state
        if "learning_state" in state:
            learning_state = state["learning_state"]

            if "trace_manager" in learning_state and hasattr(c, "trace_manager"):
                c.trace_manager.load_state(learning_state["trace_manager"])

            if "stp_pf_purkinje" in learning_state and hasattr(c, "stp_pf_purkinje"):
                c.stp_pf_purkinje.load_state(learning_state["stp_pf_purkinje"])

            if "stp_mf_granule" in learning_state and hasattr(c, "stp_mf_granule"):
                c.stp_mf_granule.load_state(learning_state["stp_mf_granule"])

            if "homeostasis" in learning_state and hasattr(c, "homeostasis"):
                c.homeostasis.load_state(learning_state["homeostasis"])

        # 3. Restore climbing fiber state
        if "climbing_fiber_state" in state and c.climbing_fiber is not None:
            c.climbing_fiber.load_state(state["climbing_fiber_state"])

        # 4. Restore enhanced microcircuit state (if applicable)
        if "microcircuit_state" in state and c.config.use_enhanced_microcircuit:
            microcircuit = state["microcircuit_state"]

            if "granule_layer" in microcircuit and hasattr(c, "granule_layer"):
                c.granule_layer.load_state(microcircuit["granule_layer"])

            if "purkinje_cells" in microcircuit and hasattr(c, "purkinje_cells"):
                for i, cell_state in enumerate(microcircuit["purkinje_cells"]):
                    if i < len(c.purkinje_cells):
                        c.purkinje_cells[i].load_state(cell_state)

            if "deep_nuclei" in microcircuit and hasattr(c, "deep_nuclei"):
                c.deep_nuclei.load_state(microcircuit["deep_nuclei"])

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

            # Restore input routing
            if "input_router" in region_state and hasattr(c, "input_router"):
                c.input_router.load_state(region_state["input_router"])

            # Restore oscillator state
            if "phase_preferences" in region_state and hasattr(c, "phase_preferences"):
                c.phase_preferences.copy_(region_state["phase_preferences"])

    def _get_neurons_data(self) -> list[Dict[str, Any]]:
        """Extract per-neuron data for neuromorphic format.

        Returns:
            List of neuron dicts with synapses
        """
        c = self.cerebellum
        neurons = []

        # Extract Purkinje cell data
        for i in range(c.purkinje_size):
            neuron_data = {
                "id": f"cerebellum_purkinje_{i}",
                "type": "purkinje",
                "membrane": (
                    c.purkinje_neurons.membrane[i].item()
                    if hasattr(c, "purkinje_neurons")
                    else 0.0
                ),
                "incoming_synapses": [],
            }

            # Extract synapses from parallel fibers (or granule cells)
            for source_name, weights in c.synaptic_weights.items():
                if weights.size(0) > i:  # Check bounds
                    synapses = self.extract_synapses_for_neuron(
                        neuron_idx=i,
                        weights=weights,
                        source_prefix=source_name,
                        synapse_type="excitatory",
                        sparsity_threshold=1e-8,
                    )
                    neuron_data["incoming_synapses"].extend(synapses)

            neurons.append(neuron_data)

        return neurons

    def get_neuromorphic_state(self) -> Dict[str, Any]:
        """Get cerebellum state in neuromorphic (neuron-centric) format.

        Stores per-neuron data with explicit synapses. Useful for
        inspecting learned motor patterns and error-corrective adjustments.

        Returns:
            Dict with format: {
                "format": "neuromorphic",
                "neurons": [...],
                "learning_state": {...},
                ...
            }
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
                "climbing_fiber_state": (
                    self.cerebellum.climbing_fiber.get_state()
                    if self.cerebellum.climbing_fiber
                    else {}
                ),
            },
        )

    # ==================== REQUIRED ABSTRACT METHODS ====================

    def _get_region(self) -> Any:
        """Get the region instance managed by this checkpoint manager."""
        return self.cerebellum

    def _get_selection_criteria(self) -> Dict[str, Any]:
        """Get region-specific criteria for format selection."""
        return {
            "n_neurons": self.cerebellum.n_granule + self.cerebellum.n_purkinje,
            "growth_enabled": False,  # Cerebellum currently doesn't support growth
            "region_type": "cerebellum",
        }

    def _should_use_neuromorphic(self) -> bool:
        """Determine if neuromorphic format should be used.

        For cerebellum: Use elastic tensor format (more efficient for large granule layer).
        """
        return False  # Use elastic tensor format

    def load_neuromorphic_state(self, state: Dict[str, Any]) -> None:
        """Load cerebellar state from neuromorphic format.

        Currently not implemented as cerebellum uses elastic tensor format.
        If needed in future, would handle neuron-by-neuron restoration.

        Args:
            state: Neuromorphic checkpoint dict
        """
        raise NotImplementedError(
            "Cerebellum uses elastic tensor format. "
            "Use collect_state()/restore_state() for checkpointing."
        )
