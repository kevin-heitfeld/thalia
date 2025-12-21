"""Checkpoint Protocol for State Serialization.

Defines the standard interface for components that support checkpointing.
This protocol enables unified checkpoint management across all brain regions
and components, ensuring consistent state serialization and restoration.

Design Rationale:
=================
Prior to this protocol, each region implemented checkpointing differently:
- Different method names (get_full_state vs get_checkpoint_state)
- Different state dict structures
- Inconsistent metadata handling
- No version compatibility guarantees

The Checkpointable protocol standardizes:
1. Method signatures (get_checkpoint_state, load_checkpoint_state)
2. State dict structure (state + metadata + version)
3. Metadata requirements (version, architecture info)
4. Backward compatibility contracts

Usage:
======
    from thalia.core.protocols.checkpoint import Checkpointable

    class MyRegion(NeuralRegion, Checkpointable):
        def get_checkpoint_state(self) -> Dict[str, Any]:
            return {
                "weights": self.weights,
                "neuron_state": self.neurons.get_state(),
                # ... more state
            }

        def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
            self.weights = state["weights"]
            self.neurons.load_state(state["neuron_state"])
            # ... more restoration

        def get_checkpoint_metadata(self) -> Dict[str, Any]:
            return {
                "version": "2.0.0",
                "n_neurons": self.n_neurons,
                "architecture": "layered_cortex",
            }

Benefits:
=========
- Type-safe: Static type checkers can verify implementation
- Duck-typed: No inheritance required (runtime_checkable)
- Composable: Can mix with other protocols
- Testable: Easy to verify checkpoint contracts
- Documented: Clear expectations for implementers

Author: Thalia Project
Date: December 21, 2025 (Tier 3 architectural improvements)
"""

from typing import Protocol, Dict, Any, runtime_checkable


@runtime_checkable
class Checkpointable(Protocol):
    """Protocol for components that support state checkpointing.

    Components implementing this protocol can be saved to and restored from
    checkpoint files with guaranteed version compatibility and metadata tracking.

    **Contract Requirements**:

    1. **Completeness**: `get_checkpoint_state()` must capture ALL state needed
       to restore the component to an equivalent functional state

    2. **Idempotency**: Calling `load_checkpoint_state(get_checkpoint_state())`
       should restore the component to the same state

    3. **Independence**: State dict should be self-contained (no external refs)

    4. **Versioning**: Metadata must include version for migration support

    **State Dict Structure**:

    The state dict returned by `get_checkpoint_state()` should contain:
    - Tensor state (weights, membrane potentials, etc.)
    - Configuration state (sizes, parameters)
    - Learning state (traces, eligibility, etc.)
    - Runtime state (last spikes, buffers, etc.)

    **Metadata Dict Structure**:

    The metadata dict returned by `get_checkpoint_metadata()` should contain:
    - `version`: Semantic version string (e.g., "2.0.0")
    - `architecture`: Component architecture identifier
    - `n_neurons`: Number of neurons (if applicable)
    - Additional component-specific metadata

    **Example Implementation**:

    ```python
    class MyRegion(NeuralRegion):
        def get_checkpoint_state(self) -> Dict[str, Any]:
            return {
                "weights": self.weights.detach().clone(),
                "neuron_state": self.neurons.get_state(),
                "config": {
                    "n_neurons": self.n_neurons,
                    "n_input": self.n_input,
                },
            }

        def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
            self.weights.data = state["weights"].to(self.device)
            self.neurons.load_state(state["neuron_state"])
            # Config validation
            assert state["config"]["n_neurons"] == self.n_neurons

        def get_checkpoint_metadata(self) -> Dict[str, Any]:
            return {
                "version": "2.0.0",
                "architecture": "my_region",
                "n_neurons": self.n_neurons,
                "created_at": self._creation_timestamp,
            }
    ```
    """

    def get_checkpoint_state(self) -> Dict[str, Any]:
        """Return complete state dict for checkpointing.

        **Must capture ALL state needed to restore component functionality.**

        Returns:
            Dict containing:
            - Tensor state (weights, potentials, buffers)
            - Configuration (sizes, parameters)
            - Learning state (traces, eligibility)
            - Runtime state (last values, accumulators)

        **Contract**:
        - State must be self-contained (no external references)
        - Tensors should be detached and cloned
        - Must work with `load_checkpoint_state()` for full restoration

        Example:
        --------
        >>> state = region.get_checkpoint_state()
        >>> state.keys()
        dict_keys(['weights', 'neuron_state', 'config', 'learning_state'])
        """
        ...

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """Restore component from checkpoint state dict.

        **Must restore component to functionally equivalent state.**

        Args:
            state: State dict from `get_checkpoint_state()`

        **Contract**:
        - Handle version migration if needed
        - Validate state dict structure
        - Move tensors to correct device
        - Update all internal state

        **Raises**:
        - ValueError: If state dict is invalid or incompatible
        - RuntimeError: If restoration fails

        Example:
        --------
        >>> state = region.get_checkpoint_state()
        >>> region2 = MyRegion(config)
        >>> region2.load_checkpoint_state(state)
        >>> # region2 now functionally equivalent to original region
        """
        ...

    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        """Return metadata for checkpoint inspection.

        **Metadata enables version checking without full state loading.**

        Returns:
            Dict containing:
            - `version`: Semantic version (e.g., "2.0.0")
            - `architecture`: Component type identifier
            - `n_neurons`: Number of neurons (if applicable)
            - Additional component-specific metadata

        **Contract**:
        - Must include `version` field
        - Must be fast (no tensor operations)
        - Should enable compatibility checks before loading

        Example:
        --------
        >>> meta = region.get_checkpoint_metadata()
        >>> meta
        {
            'version': '2.0.0',
            'architecture': 'layered_cortex',
            'n_neurons': 500,
            'layers': ['l4', 'l23', 'l5', 'l6'],
        }
        """
        ...


@runtime_checkable
class CheckpointableWithNeuromorphic(Checkpointable, Protocol):
    """Extended protocol for neuromorphic (neuron-centric) checkpoints.

    This protocol extends Checkpointable with neuromorphic format support,
    which stores per-neuron data with explicit synapses. This format is
    ideal for regions with neurogenesis or dynamic structure.

    **When to Use Neuromorphic Format**:
    - Region supports neurogenesis (adding neurons during training)
    - Small region size (<1000 neurons)
    - Need to inspect individual neuron properties
    - Neuron identities matter (e.g., episodic memory traces)

    **Neuromorphic State Structure**:

    ```python
    {
        "format": "neuromorphic",
        "version": "2.0.0",
        "neurons": [
            {
                "id": "region_neuron_0_step1000",
                "created_step": 1000,
                "membrane": -0.65,
                "threshold": -0.50,
                "incoming_synapses": [
                    {"from": "input_neuron_42", "weight": 0.35},
                    {"from": "input_neuron_87", "weight": 0.22},
                    # ... sparse format (only non-zero weights)
                ],
            },
            # ... one entry per neuron
        ],
    }
    ```

    **Example Implementation**:

    ```python
    class HippocampusCA3(NeuralRegion, CheckpointableWithNeuromorphic):
        def get_neuromorphic_state(self) -> Dict[str, Any]:
            neurons = []
            for i in range(self.n_neurons):
                neurons.append({
                    "id": f"ca3_neuron_{i}_step{self.timestep}",
                    "created_step": self._neuron_birth_steps[i],
                    "membrane": self.membrane[i].item(),
                    "incoming_synapses": self._extract_synapses(i),
                })
            return {"format": "neuromorphic", "neurons": neurons}

        def load_neuromorphic_state(self, state: Dict[str, Any]) -> None:
            neurons_data = state["neurons"]
            self.n_neurons = len(neurons_data)
            self.membrane = torch.tensor([n["membrane"] for n in neurons_data])
            # ... restore synapses, etc.
    ```
    """

    def get_neuromorphic_state(self) -> Dict[str, Any]:
        """Return checkpoint in neuromorphic (neuron-centric) format.

        **Stores per-neuron data with explicit synapses.**

        Returns:
            Dict with format:
            {
                "format": "neuromorphic",
                "version": "2.0.0",
                "neurons": [
                    {
                        "id": unique neuron identifier,
                        "created_step": birth timestep,
                        "membrane": current membrane potential,
                        "incoming_synapses": [{from, weight}, ...],
                    },
                    # ... one per neuron
                ],
            }

        **Contract**:
        - One entry per neuron
        - Synapses in sparse format (only non-zero weights)
        - Neuron IDs must be unique and stable
        """
        ...

    def load_neuromorphic_state(self, state: Dict[str, Any]) -> None:
        """Restore component from neuromorphic checkpoint.

        **Reconstructs neurons and synapses from neuron-centric format.**

        Args:
            state: Neuromorphic state dict from `get_neuromorphic_state()`

        **Contract**:
        - Must handle variable neuron count (neurogenesis support)
        - Reconstruct weight matrices from sparse synapses
        - Validate neuron IDs and connectivity
        """
        ...
