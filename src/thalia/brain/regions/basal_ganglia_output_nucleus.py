"""Shared base class for basal ganglia output nuclei (SNr, GPe, GPi).

All three major BG output nuclei share the same biophysical footprint:
- Tonically active GABAergic neurons (50–80 Hz baseline)
- Conductance-LIF with g_L=0.10, τ_E=5ms, τ_I=10ms
- Per-step tonic baseline drive from ``config.baseline_drive``
- A common integrate→split→forward pattern per population

This module extracts those commonalities into helper methods so each
nucleus stays focused on its own population layout and connectivity.
"""

from __future__ import annotations

from typing import TypeVar

import torch

from thalia.brain.configs.basal_ganglia import TonicPacemakerConfig
from thalia.brain.neurons import (
    ConductanceLIF,
    split_excitatory_conductance,
)
from thalia.typing import (
    ConductanceTensor,
    SynapticInput,
)

from .neural_region import NeuralRegion

ConfigT = TypeVar("ConfigT", bound=TonicPacemakerConfig)


class BasalGangliaOutputNucleus(NeuralRegion[ConfigT]):
    """Abstract base class for basal ganglia output nuclei."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_tonic_baseline(self, size: int, factor: float = 1.0) -> torch.Tensor:
        """Create a tonic baseline drive tensor for one population.

        Returns ``torch.full((size,), config.baseline_drive * factor)`` on
        ``self.device``.  Secondary populations (GPe arkypallidal, GPi border
        cells) pass a ``factor < 1.0`` to keep them sub-threshold at rest.

        Args:
            size: Number of neurons (tensor length).
            factor: Scaling relative to ``config.baseline_drive``.

        Returns:
            1-D float tensor of per-step AMPA conductance values.
        """
        return torch.full(
            (size,), self.config.baseline_drive * factor, device=self.device
        )

    def _bg_step_single(
        self,
        synaptic_inputs: SynapticInput,
        n_neurons: int,
        population_name: str,
        neurons: ConductanceLIF,
        baseline: torch.Tensor,
        *,
        nmda_ratio: float = 0.05,
    ) -> torch.Tensor:
        """Run one timestep for a single BG output population.

        Pattern:
            1. Integrate synaptic inputs at dendrites (filtered to this population).
            2. Add tonic baseline to g_AMPA.
            3. Split combined excitatory conductance into AMPA + NMDA components.
            4. Forward through the ConductanceLIF neuron model.

        Args:
            synaptic_inputs: Region-wide synaptic input dict for this timestep.
            n_neurons: Population size (passed to dendrite integration).
            population_name: Used to filter incoming synaptic weights to this pop.
            neurons: The :class:`~thalia.brain.neurons.ConductanceLIF` for this pop.
            baseline: Tonic baseline tensor of shape ``(n_neurons,)``.
            nmda_ratio: Fraction of excitatory conductance assigned to NMDA.
                GPe and GPi use 0.05 (sparse NMDA at pallidal synapses).
                SNr uses 0.01 (near-soma synapses with minimal NMDA involvement).

        Returns:
            Spike tensor of shape ``(n_neurons,)``.
        """
        dendrite = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=n_neurons,
            filter_by_target_population=population_name,
        )
        g_exc = baseline.clone() + dendrite.g_ampa
        g_inh = dendrite.g_gaba_a

        g_ampa, g_nmda = split_excitatory_conductance(g_exc, nmda_ratio=nmda_ratio)
        spikes, _ = neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa),
            g_nmda_input=ConductanceTensor(g_nmda),
            g_gaba_a_input=ConductanceTensor(g_inh),
            g_gaba_b_input=None,
        )
        return spikes
