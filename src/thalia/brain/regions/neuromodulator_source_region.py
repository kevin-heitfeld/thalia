"""NeuromodulatorSourceRegion - Base class for broadcast neuromodulator regions.

This mixin base class captures the shared circuit topology present in VTA, SNc,
Locus Coeruleus, Nucleus Basalis, and Dorsal Raphe Nucleus:

    primary population (DA / NE / ACh / 5-HT)
        |
        +--[collateral]--→ GABA interneurons  (homeostatic feedback)
        |
        ↓
    neuromodulator broadcast

Each region has one or more principal neuron populations that broadcast a
neuromodulator, plus a small pool of local GABA interneurons that provide
homeostatic feedback inhibition (preventing runaway activity).

Subclasses are responsible for:
- Creating and registering their primary neuron population(s)
- Calling ``_init_gaba_interneurons()`` in ``__init__``
- Calling ``_step_gaba_interneurons(primary_activity)`` in ``forward()``
- Overriding ``_compute_gaba_drive()`` if they need non-default baseline / gain

Excluded regions
----------------
``MedialSeptum`` is intentionally **not** a subclass: its GABA population is a
co-equal principal (theta-phase output), not a homeostatic feedback pool, and it
contains ionic pacemaker buffers and explicit recurrent weight matrices between
ACh and GABA populations.
"""

from __future__ import annotations

from typing import Generic, TypeVar

import torch

from thalia.brain.configs import NeuralRegionConfig
from thalia.components import NeuronFactory, NeuronType
from thalia.typing import (
    ConductanceTensor,
    PopulationName,
    PopulationPolarity,
)
from thalia.utils import split_excitatory_conductance

from .neural_region import NeuralRegion

ConfigT = TypeVar("ConfigT", bound=NeuralRegionConfig)


class NeuromodulatorSourceRegion(NeuralRegion[ConfigT], Generic[ConfigT]):
    """Base class for regions that broadcast a neuromodulator signal.

    Provides the shared primary-population + local-GABA-interneuron pattern.
    """

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_gaba_interneurons(
        self,
        gaba_population_name: PopulationName,
        gaba_size: int,
    ) -> None:
        """Construct and register the local GABA interneuron pool.

        Call this **once** from the subclass ``__init__``, after setting up the
        primary population(s).

        Args:
            gaba_population_name: Population enum member identifying GABA neurons
                                  (used for registration and weight routing).
            gaba_size:            Number of GABA neurons.
        """
        self._gaba_population_name: PopulationName = gaba_population_name
        self.gaba_neurons_size: int = gaba_size

        from thalia.components import ConductanceLIF  # imported lazily to avoid cycles

        self.gaba_neurons = NeuronFactory.create(
            region_name=self.region_name,
            population_name=gaba_population_name,
            neuron_type=NeuronType.FAST_SPIKING,
            n_neurons=gaba_size,
            device=self.device,
        )
        self._register_neuron_population(
            gaba_population_name,
            self.gaba_neurons,
            polarity=PopulationPolarity.INHIBITORY,
        )

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _compute_gaba_drive(self, primary_activity: float) -> torch.Tensor:
        """Compute excitatory drive for GABA interneurons.

        Default implementation: tonic baseline plus proportional feedback from
        the primary population activity (``baseline + activity × 0.8``).

        Subclasses should override when they need a different baseline or gain.

        Args:
            primary_activity: Mean firing rate (spikes per step) of the primary
                              population(s), in the range [0, 1].

        Returns:
            Drive conductance tensor of shape ``[gaba_neurons_size]``.
        """
        baseline = 0.3 if self.config.baseline_noise_conductance_enabled else 0.0
        return torch.full(
            (self.gaba_neurons_size,),
            baseline + primary_activity * 0.8,
            device=self.device,
        )

    def _step_gaba_interneurons(self, primary_activity: float) -> torch.Tensor:
        """Advance GABA interneurons one timestep and return their spikes.

        Call at the end of the subclass ``forward()`` after the primary
        population(s) have been stepped.

        Args:
            primary_activity: Mean firing rate of the primary population(s),
                              used to drive the GABA interneurons.

        Returns:
            GABA spike tensor of shape ``[gaba_neurons_size]``.
        """
        gaba_drive = self._compute_gaba_drive(primary_activity)
        gaba_g_ampa, gaba_g_nmda = split_excitatory_conductance(gaba_drive, nmda_ratio=0.3)
        gaba_spikes, _ = self.gaba_neurons.forward(
            g_ampa_input=ConductanceTensor(gaba_g_ampa),
            g_nmda_input=ConductanceTensor(gaba_g_nmda),
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )
        return gaba_spikes
