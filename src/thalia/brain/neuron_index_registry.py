"""Brain-wide global neuron index registry.

Maps every ``(region_name, population_name)`` to a contiguous global index
range, unifying the currently fragmented index spaces.

Two index spaces:

- **target_registry**: ConductanceLIF + subclass neurons only (sparse matrix
  rows, conductance output buffer indices).  Ordering: reuse
  ``ConductanceLIFBatch.registry`` for batched populations (deterministic
  sorted order), then append subclass populations sorted by (region, pop).

- **source_registry**: All neurons (for future steps and global spike vector).
  Ordering: all populations sorted by (region, pop).

The hierarchical ``(region_name, population_name, neuron_id)`` scheme
mirrors the existing Philox RNG seeding in ``_create_neuron_seeds()``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

import torch

from thalia.typing import PopulationKey, PopulationName, RegionName

if TYPE_CHECKING:
    from thalia.brain.neurons.conductance_lif_batch import ConductanceLIFBatch
    from thalia.brain.neurons.conductance_lif_neuron import ConductanceLIF
    from thalia.brain.neurons.two_compartment_lif_neuron import TwoCompartmentLIF
    from thalia.brain.regions.neural_region import NeuralRegion

logger = logging.getLogger(__name__)


class NeuronIndexRegistry:
    """Brain-wide global neuron index for all populations.

    Attributes:
        target_registry: ``PopulationKey → (start, end)`` for eligible targets
            (ConductanceLIF and subclasses).  Row indices in the global sparse
            synaptic matrices.
        source_registry: ``PopulationKey → (start, end)`` for ALL neurons.
        total_target_neurons: Total number of target neurons.
        total_source_neurons: Total number of source neurons.
    """

    def __init__(
        self,
        regions: dict[RegionName, NeuralRegion],
        neuron_batch: ConductanceLIFBatch,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        from thalia.brain.neurons.conductance_lif_neuron import ConductanceLIF
        from thalia.brain.neurons.two_compartment_lif_neuron import TwoCompartmentLIF

        self.device = torch.device(device)

        # =================================================================
        # TARGET REGISTRY (ConductanceLIF + subclasses, NOT TwoCompartmentLIF)
        # =================================================================
        # Start with batched ConductanceLIF populations (preserve their order)
        self.target_registry: dict[PopulationKey, tuple[int, int]] = {}
        for key, (start, end) in neuron_batch.registry.items():
            self.target_registry[key] = (start, end)

        # Append subclass neurons (not in ConductanceLIFBatch but still ConductanceLIF subclasses)
        target_offset = neuron_batch.total_neurons
        subclass_entries: list[tuple[PopulationKey, ConductanceLIF]] = []

        for region_name, region in sorted(regions.items()):
            for pop_name in sorted(region.neuron_populations.keys()):
                key: PopulationKey = (region_name, pop_name)
                neuron = region.neuron_populations[pop_name]

                # Skip if already in the batch (pure ConductanceLIF)
                if key in neuron_batch.registry:
                    continue

                # Include ConductanceLIF subclasses (Serotonin, NE, ACh)
                # Exclude TwoCompartmentLIF
                if isinstance(neuron, ConductanceLIF) and not isinstance(neuron, TwoCompartmentLIF):
                    n = neuron.n_neurons
                    self.target_registry[key] = (target_offset, target_offset + n)
                    subclass_entries.append((key, neuron))
                    target_offset += n

        self.total_target_neurons = target_offset
        self._subclass_entries = subclass_entries

        # =================================================================
        # SOURCE REGISTRY (ALL neurons)
        # =================================================================
        self.source_registry: dict[PopulationKey, tuple[int, int]] = {}
        source_offset = 0

        for region_name, region in sorted(regions.items()):
            for pop_name in sorted(region.neuron_populations.keys()):
                key = (region_name, pop_name)
                neuron = region.neuron_populations[pop_name]
                n = neuron.n_neurons
                self.source_registry[key] = (source_offset, source_offset + n)
                source_offset += n

        self.total_source_neurons = source_offset

        logger.info(
            "NeuronIndexRegistry: %d target neurons (%d batched + %d subclass), "
            "%d total source neurons",
            self.total_target_neurons,
            neuron_batch.total_neurons,
            sum(n.n_neurons for _, n in subclass_entries),
            self.total_source_neurons,
        )

    def is_eligible_target(self, pop_key: PopulationKey) -> bool:
        """True if this population receives conductances via the global sparse matrix."""
        return pop_key in self.target_registry

    def get_target_slice(self, pop_key: PopulationKey) -> tuple[int, int]:
        """Return (start, end) in the target index for a population."""
        return self.target_registry[pop_key]

    def get_source_slice(self, pop_key: PopulationKey) -> tuple[int, int]:
        """Return (start, end) in the source index for a population."""
        return self.source_registry[pop_key]
