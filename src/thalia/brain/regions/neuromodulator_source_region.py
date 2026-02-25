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
        # Delay buffer: GABA spikes from the previous timestep, used as:
        #   (a) inhibitory feedback to primary neurons next step
        #   (b) self-inhibition within the GABA pool
        # Initialised to zeros (silent at t=0).
        self._prev_gaba_spikes: torch.Tensor = torch.zeros(
            gaba_size, dtype=torch.bool, device=self.device
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
        # No tonic baseline: a constant 0.3 drive saturates fast-spiking GABA
        # neurons at ~400 Hz (V* = 2.25, far above threshold 0.9), which then
        # completely silences primary neurons through feedback.  GABA should only
        # fire proportionally when the primary population overshoots its target.
        # Gain 2.0 keeps GABA below threshold at normal primary rates (<10 Hz)
        # but drives it strongly when the primary runs away (>50 Hz).
        return torch.full(
            (self.gaba_neurons_size,),
            primary_activity * 2.0,
            device=self.device,
        )

    def _step_gaba_interneurons(self, primary_activity: float) -> torch.Tensor:
        """Advance GABA interneurons one timestep and return their spikes.

        The GABA pool receives excitatory drive proportional to the primary
        population's activity.  The pool also receives self-inhibition from its
        own spikes at the previous timestep, preventing runaway saturation.

        The returned spikes are stored in ``_prev_gaba_spikes`` for use on the
        **next** call to this method and as inhibitory feedback to the primary
        population via ``_get_gaba_feedback_conductance()``.

        Call at the end of the subclass ``forward()`` **after** the primary
        population(s) have been stepped.

        Args:
            primary_activity: Mean firing rate of the primary population(s),
                              used to drive the GABA interneurons.

        Returns:
            GABA spike tensor of shape ``[gaba_neurons_size]``.
        """
        gaba_drive = self._compute_gaba_drive(primary_activity)
        gaba_g_ampa, gaba_g_nmda = split_excitatory_conductance(gaba_drive, nmda_ratio=0.3)

        # Self-inhibition: GABA neurons inhibit each other (recurrent lateral inh.)
        # Uses mean activity of the pool rather than a full weight matrix —
        # equivalent to uniform all-to-all inhibition with unit weight.
        prev_gaba_rate = self._prev_gaba_spikes.float().mean().item()
        gaba_self_inh = torch.full(
            (self.gaba_neurons_size,),
            prev_gaba_rate * 0.5,   # moderate self-inhibition to prevent saturation
            device=self.device,
        )

        gaba_spikes, _ = self.gaba_neurons.forward(
            g_ampa_input=ConductanceTensor(gaba_g_ampa),
            g_nmda_input=ConductanceTensor(gaba_g_nmda),
            g_gaba_a_input=ConductanceTensor(gaba_self_inh),
            g_gaba_b_input=None,
        )
        # Store for next timestep (used by _get_gaba_feedback_conductance)
        self._prev_gaba_spikes = gaba_spikes
        return gaba_spikes

    def _get_gaba_feedback_conductance(self, primary_size: int, gain: float) -> torch.Tensor:
        """Return a GABA-A conductance tensor to apply to the primary population.

        Uses the spikes from the **previous** timestep so there are no circular
        dependencies within a single forward pass.  The inhibitory conductance is
        broadcast uniformly to all primary neurons (models diffuse volume
        transmission from the local GABA pool).

        Args:
            primary_size: Number of primary neurons to inhibit.
            gain:         Scaling factor (conductance units per spike fraction).

        Returns:
            Float tensor of shape ``[primary_size]``.
        """
        prev_gaba_rate = self._prev_gaba_spikes.float().mean().item()
        return torch.full(
            (primary_size,),
            prev_gaba_rate * gain,
            device=self.device,
        )
