"""Base checkpoint manager with shared serialization logic.

Provides common patterns for neuromorphic (neuron-centric) checkpointing:
- Synapse extraction with sparsity
- State packaging with versioning
- Elastic tensor capacity handling
- Shared utility methods for state extraction

Region-specific managers should inherit from this base and implement:
- _get_neurons_data(): Extract per-neuron data and synapses
- _get_learning_state(): Extract learning-related state (STP, STDP, etc.)
- _get_neuromodulator_state(): Extract neuromodulator levels
- _get_region_state(): Extract region-specific state (traces, buffers, etc.)
- load_neuromorphic_state(): Restore state from neuromorphic format

Design Rationale:
- Eliminates ~400-500 lines of duplication across 3 checkpoint managers
- Provides single source of truth for neuromorphic encoding
- Makes it easy to add new checkpoint formats or features
- Preserves region-specific customization through abstract methods

Biological Context:
Different brain regions require different checkpoint strategies:
- Striatum: D1/D2 pathway separation, eligibility traces, exploration state
- Hippocampus: 3-layer circuit (DG→CA3→CA1), episode buffer, replay state
- Prefrontal: Feedforward/recurrent/inhibitory separation, working memory
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseCheckpointManager(ABC):
    """Base class for region-specific checkpoint managers.

    Provides shared logic for neuromorphic checkpointing while allowing
    region-specific customization through abstract methods.

    Usage:
        class MyRegionCheckpointManager(BaseCheckpointManager):
            def __init__(self, region):
                super().__init__(format_version="2.0.0")
                self.region = region

            def _get_neurons_data(self) -> list[Dict[str, Any]]:
                # Extract per-neuron data with incoming synapses
                ...

            def _get_learning_state(self) -> Dict[str, Any]:
                # Extract STP, STDP, etc.
                ...
    """

    def __init__(self, format_version: str = "2.0.0"):
        """Initialize base checkpoint manager.

        Args:
            format_version: Checkpoint format version (semantic versioning)
        """
        self.format_version = format_version

    # ==================== SHARED UTILITY METHODS ====================

    def extract_synapses_for_neuron(
        self,
        neuron_idx: int,
        weights: torch.Tensor,
        source_prefix: str,
        synapse_type: Optional[str] = None,
        sparsity_threshold: float = 1e-8,
    ) -> list[Dict[str, Any]]:
        """Extract incoming synapses for a single neuron.

        Stores only non-zero synapses (sparse format) to reduce checkpoint size.

        Args:
            neuron_idx: Index of target neuron
            weights: Weight matrix [n_target, n_source]
            source_prefix: Prefix for source neuron IDs (e.g., "input", "ec_neuron")
            synapse_type: Optional synapse type label (e.g., "feedforward", "recurrent")
            sparsity_threshold: Minimum weight magnitude to store (default: 1e-8)

        Returns:
            List of synapse dicts with {from, weight, type (optional)}
        """
        synapses = []
        neuron_weights = weights[neuron_idx]

        # Only store non-zero synapses (sparse format)
        nonzero_mask = neuron_weights.abs() > sparsity_threshold
        nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=True)[0]

        for source_idx in nonzero_indices:
            synapse = {
                "from": f"{source_prefix}_{source_idx.item()}",
                "weight": neuron_weights[source_idx].item(),
            }
            if synapse_type is not None:
                synapse["type"] = synapse_type
            synapses.append(synapse)

        return synapses

    def extract_multi_source_synapses(
        self,
        neuron_idx: int,
        weight_source_pairs: list[tuple[torch.Tensor, str]],
        sparsity_threshold: float = 1e-8,
    ) -> list[Dict[str, Any]]:
        """Extract synapses from multiple weight matrices.

        Useful for neurons with multiple input sources (e.g., CA1 from CA3 + EC).

        Args:
            neuron_idx: Index of target neuron
            weight_source_pairs: List of (weight_matrix, source_prefix) tuples
            sparsity_threshold: Minimum weight magnitude to store

        Returns:
            Combined list of synapses from all sources
        """
        all_synapses = []
        for weights, source_prefix in weight_source_pairs:
            synapses = self.extract_synapses_for_neuron(
                neuron_idx, weights, source_prefix, sparsity_threshold=sparsity_threshold
            )
            all_synapses.extend(synapses)
        return all_synapses

    def extract_typed_synapses(
        self,
        neuron_idx: int,
        typed_weights: list[tuple[torch.Tensor, str, str]],
        sparsity_threshold: float = 1e-8,
    ) -> list[Dict[str, Any]]:
        """Extract synapses with type labels from multiple weight matrices.

        Useful for regions with multiple synapse types (e.g., feedforward, recurrent, inhibitory).

        Args:
            neuron_idx: Index of target neuron
            typed_weights: List of (weight_matrix, source_prefix, synapse_type) tuples
            sparsity_threshold: Minimum weight magnitude to store

        Returns:
            Combined list of synapses with type labels
        """
        all_synapses = []
        for weights, source_prefix, synapse_type in typed_weights:
            synapses = self.extract_synapses_for_neuron(
                neuron_idx, weights, source_prefix, synapse_type, sparsity_threshold
            )
            all_synapses.extend(synapses)
        return all_synapses

    # ==================== STATE PACKAGING ====================

    def package_neuromorphic_state(
        self,
        neurons: list[Dict[str, Any]],
        learning_state: Dict[str, Any],
        neuromodulator_state: Dict[str, Any],
        region_state: Dict[str, Any],
        additional_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Package region state in standardized neuromorphic format.

        Args:
            neurons: List of per-neuron data dicts
            learning_state: Learning-related state (STP, STDP, etc.)
            neuromodulator_state: Neuromodulator levels and systems
            region_state: Region-specific state (traces, buffers, etc.)
            additional_state: Optional additional state sections (e.g., episode_buffer)

        Returns:
            Complete checkpoint dict with standardized structure
        """
        checkpoint = {
            "format": "neuromorphic",
            "format_version": self.format_version,
            "neurons": neurons,
            "learning_state": learning_state,
            "neuromodulator_state": neuromodulator_state,
            "region_state": region_state,
        }

        # Add optional additional state sections
        if additional_state is not None:
            checkpoint.update(additional_state)

        return checkpoint

    # ==================== ABSTRACT METHODS (region-specific) ====================

    @abstractmethod
    def _get_neurons_data(self) -> list[Dict[str, Any]]:
        """Extract per-neuron data with incoming synapses.

        Each neuron dict should contain:
        - id: Stable neuron identifier (e.g., "pfc_neuron_0_step0")
        - region: Region name (e.g., "prefrontal", "hippocampus")
        - created_step: Creation timestep (for neurogenesis tracking)
        - membrane: Current membrane potential
        - incoming_synapses: List of synapse dicts from extract_synapses_for_neuron()
        - Additional region-specific fields (e.g., "layer", "working_memory")

        Returns:
            List of per-neuron data dicts
        """
        ...

    @abstractmethod
    def _get_learning_state(self) -> Dict[str, Any]:
        """Extract learning-related state (STP, STDP, eligibility traces, etc.).

        Should include:
        - STP state: stp_*.get_state() for all STP instances
        - STDP state: stdp_strategy.get_state() if applicable
        - Eligibility traces: eligibility_d1, eligibility_d2, etc.
        - Homeostasis state: homeostasis_manager.get_state() if applicable

        Returns:
            Dict of learning-related state
        """
        ...

    @abstractmethod
    def _get_neuromodulator_state(self) -> Dict[str, Any]:
        """Extract neuromodulator-related state.

        Should include:
        - Current modulator levels: dopamine, acetylcholine, norepinephrine
        - Modulator system state: dopamine_system.get_state(), etc.

        Returns:
            Dict of neuromodulator state
        """
        ...

    @abstractmethod
    def _get_region_state(self) -> Dict[str, Any]:
        """Extract region-specific state (traces, buffers, etc.).

        Should include region-specific state that doesn't fit in other categories:
        - Activity traces: ca3_activity_trace, dg_trace, etc.
        - Delay buffers: d1_delay_buffer, d2_delay_buffer, etc.
        - Region-specific flags: pending_theta_reset, exploring, etc.
        - Spike history: recent_spikes, spikes, etc.

        Returns:
            Dict of region-specific state
        """
        ...

    @abstractmethod
    def collect_state(self) -> Dict[str, Any]:
        """Collect region-specific state for checkpointing (elastic tensor format).

        This method should extract all state needed to restore the region,
        organized into logical sections (neuron_state, pathway_state, learning_state, etc.).

        This is the elastic tensor format counterpart to _get_neurons_data() +
        _get_learning_state() + _get_neuromodulator_state() + _get_region_state().

        Returns:
            Dict with region-specific state sections
        """
        ...

    @abstractmethod
    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore region-specific state from checkpoint (elastic tensor format).

        This method should restore all state extracted by collect_state(),
        handling backward compatibility and graceful degradation for missing fields.

        This is the elastic tensor format counterpart to load_neuromorphic_state().

        Args:
            state: Dict from collect_state()
        """
        ...

    @abstractmethod
    def get_neuromorphic_state(self) -> Dict[str, Any]:
        """Get complete region state in neuromorphic (neuron-centric) format.

        Typically implemented as:
            neurons = self._get_neurons_data()
            learning = self._get_learning_state()
            neuromodulator = self._get_neuromodulator_state()
            region = self._get_region_state()
            return self.package_neuromorphic_state(
                neurons, learning, neuromodulator, region
            )

        Returns:
            Complete checkpoint dict in neuromorphic format
        """
        ...

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete region state for checkpointing (elastic tensor format).

        This is a concrete method that calls collect_state() and wraps it with
        standardized metadata (format, version).

        Returns:
            Dict containing all state needed to restore region
        """
        state = self.collect_state()

        # Add format metadata
        state["format"] = "elastic_tensor"
        state["format_version"] = self.format_version

        return state

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Restore complete region state from checkpoint (elastic tensor format).

        This is a concrete method that validates the state and calls restore_state().

        Args:
            state: Dict from get_full_state()
        """
        # Validate format (basic check)
        if "format" not in state:
            import warnings
            warnings.warn(
                "Checkpoint missing 'format' field. Assuming elastic_tensor format.",
                UserWarning
            )
        elif state["format"] != "elastic_tensor":
            raise ValueError(
                f"Expected elastic_tensor format, got {state['format']}. "
                f"Use load_neuromorphic_state() for neuromorphic format."
            )

        # Validate version compatibility (optional)
        is_compatible, error_msg = self.validate_checkpoint_compatibility(state)
        if not is_compatible:
            import warnings
            warnings.warn(
                f"Checkpoint version compatibility warning: {error_msg}",
                UserWarning
            )

        # Delegate to region-specific restore logic
        self.restore_state(state)

    @abstractmethod
    def load_neuromorphic_state(self, state: Dict[str, Any]) -> None:
        """Load region state from neuromorphic format.

        Should handle:
        - Neuron matching by ID
        - Missing neurons (checkpoint → skip with warning)
        - Extra neurons (current brain → keep current state)
        - Weight restoration from synapses
        - Learning state restoration
        - Neuromodulator state restoration
        - Region-specific state restoration

        Args:
            state: Dict from get_neuromorphic_state()
        """
        ...

    # ==================== OPTIONAL METHODS ====================

    def _get_elastic_tensor_metadata(self, n_active: int, n_capacity: int) -> Dict[str, Any]:
        """Get metadata for elastic tensor format (optional).

        Elastic tensor format supports brain growth by tracking active vs capacity neurons.

        Args:
            n_active: Number of currently active neurons
            n_capacity: Total neuron capacity (including inactive/reserved)

        Returns:
            Metadata dict for capacity tracking
        """
        return {
            "n_neurons_active": n_active,
            "n_neurons_capacity": n_capacity,
        }

    def validate_checkpoint_compatibility(self, state: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate checkpoint format and version compatibility (optional).

        Args:
            state: Checkpoint dict to validate

        Returns:
            (is_compatible, error_message) tuple
        """
        # Check format
        if "format" not in state:
            return False, "Missing 'format' field in checkpoint"

        # Check version (basic semantic versioning)
        if "format_version" not in state:
            return False, "Missing 'format_version' field in checkpoint"

        checkpoint_version = state["format_version"]
        current_version = self.format_version

        # Simple major version check (e.g., "2.0.0" vs "1.0.0")
        checkpoint_major = int(checkpoint_version.split(".")[0])
        current_major = int(current_version.split(".")[0])

        if checkpoint_major != current_major:
            return False, f"Incompatible major version: checkpoint={checkpoint_version}, current={current_version}"

        return True, None

    # ==================== HYBRID FORMAT SUPPORT ====================

    @abstractmethod
    def _should_use_neuromorphic(self) -> bool:
        """Determine if neuromorphic format should be used for this region.

        Decision criteria (region-specific):
        - Small regions: Use neuromorphic (more inspectable)
        - Growth-enabled: Use neuromorphic (ID-based matching)
        - Large stable regions: Use elastic tensor (more efficient)

        Returns:
            bool: True if neuromorphic format should be used
        """
        ...

    @abstractmethod
    def _get_region(self) -> Any:
        """Get the region instance managed by this checkpoint manager.

        Returns:
            The region instance (e.g., self.striatum, self.hippocampus, self.prefrontal)
        """
        ...

    @abstractmethod
    def _get_selection_criteria(self) -> Dict[str, Any]:
        """Get region-specific criteria used for format selection.

        Returns:
            Dict with selection criteria (e.g., {"n_neurons": 100, "growth_enabled": True})
        """
        ...

    def save(self, path: str) -> Dict[str, Any]:
        """Save checkpoint with automatic format selection.

        Automatically chooses between elastic tensor and neuromorphic formats
        based on region size and properties. This is the primary save method
        that should be used by all regions.

        Args:
            path: Path where checkpoint will be saved

        Returns:
            Dict with checkpoint metadata (path, format, size, etc.)
        """
        from pathlib import Path
        path = Path(path)

        # Auto-select format using region-specific logic
        use_neuromorphic = self._should_use_neuromorphic()

        if use_neuromorphic:
            state = self.get_neuromorphic_state()
            format_name = "neuromorphic"
        else:
            # Delegate to region's get_full_state() for elastic tensor format
            region = self._get_region()
            state = region.get_full_state()
            format_name = "elastic_tensor"

        # Add hybrid format metadata for automatic detection during load
        state["hybrid_metadata"] = {
            "auto_selected": True,
            "selected_format": format_name,
            "selection_criteria": self._get_selection_criteria(),
        }

        # Save to disk
        torch.save(state, path)

        # Get region for metadata
        region = self._get_region()
        n_neurons = getattr(region.config, "n_output", None)

        return {
            "path": str(path),
            "format": format_name,
            "n_neurons": n_neurons,
            "file_size": path.stat().st_size if path.exists() else 0,
        }

    def load(self, path: str) -> None:
        """Load checkpoint with automatic format detection.

        Detects format from hybrid_metadata and loads accordingly.
        This is the primary load method that should be used by all regions.

        Args:
            path: Path to checkpoint file

        Raises:
            ValueError: If checkpoint is missing hybrid_metadata or has invalid format
        """
        from pathlib import Path
        path = Path(path)

        # Load checkpoint from disk
        state = torch.load(path, weights_only=False)

        # Validate hybrid metadata presence
        if "hybrid_metadata" not in state:
            raise ValueError(
                f"Checkpoint missing hybrid_metadata - not a valid hybrid format checkpoint. "
                f"Checkpoint may be from an older version. Keys present: {list(state.keys())}"
            )

        # Detect format and dispatch to appropriate loader
        selected_format = state["hybrid_metadata"]["selected_format"]
        if selected_format == "neuromorphic":
            self.load_neuromorphic_state(state)
        elif selected_format == "elastic_tensor":
            # Delegate to region's load_full_state() for elastic tensor format
            region = self._get_region()
            region.load_full_state(state)
        else:
            raise ValueError(
                f"Unknown checkpoint format: '{selected_format}'. "
                f"Expected 'neuromorphic' or 'elastic_tensor'."
            )
