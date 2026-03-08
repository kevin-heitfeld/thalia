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
from thalia.brain.neurons import ConductanceLIF, ConductanceLIFConfig
from thalia.typing import (
    ConductanceTensor,
    SynapticInput,
)
from thalia.utils import split_excitatory_conductance

from .neural_region import NeuralRegion

ConfigT = TypeVar("ConfigT", bound=TonicPacemakerConfig)


class BasalGangliaOutputNucleus(NeuralRegion[ConfigT]):
    """Abstract base class for basal ganglia output nuclei.

    Provides three lightweight factory / step helpers that eliminate the
    boilerplate repeated identically in SNr, GPe, and GPi:

    ``_make_bg_neurons(n, pop_name, noise_std)``
        Constructs a :class:`~thalia.brain.neurons.ConductanceLIF` with the
        canonical BG output-nucleus biophysical parameters (g_L=0.10, reversals,
        tau_E/I) drawn from ``self.config``.

    ``_make_tonic_baseline(size, factor)``
        Returns a ``torch.full`` tensor of per-step AMPA conductance for tonic
        pacemaking, scaled by ``config.baseline_drive * factor``.

    ``_bg_step_single(inputs, n, pop_name, neurons, baseline, nmda_ratio)``
        Runs one complete dendrite-integrate → AMPA/NMDA split → neuron-forward
        cycle for a single BG population.  SNr passes ``nmda_ratio=0.01``;
        GPe and GPi use the default 0.05.

    Subclasses must still implement :meth:`_step` (calling these helpers) and
    register their populations in ``__init__`` via
    :meth:`~NeuralRegion._register_neuron_population`.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_bg_neurons(
        self,
        n_neurons: int,
        population_name: str,
        noise_std: float = 0.007,
    ) -> ConductanceLIF:
        """Create a standard BG output-nucleus ConductanceLIF population.

        Parameters drawn from ``self.config``: ``tau_mem``, ``v_threshold``,
        ``tau_ref``.  Fixed biophysical constants (``g_L=0.10``, reversal
        potentials, ``tau_E=5ms``, ``tau_I=10ms``) are shared by all BG
        output nuclei and therefore hardcoded here.

        Args:
            n_neurons: Number of neurons.
            population_name: Stored in the neuron config (used for diagnostics).
            noise_std: Membrane voltage noise σ.  Prototypic / principal
                populations typically use 0.007; secondary populations 0.005.

        Returns:
            Initialized :class:`~thalia.brain.neurons.ConductanceLIF` placed on
            ``self.device``.
        """
        return ConductanceLIF(
            n_neurons=n_neurons,
            config=ConductanceLIFConfig(
                region_name=self.region_name,
                population_name=population_name,
                tau_mem=self.config.tau_mem,
                v_threshold=self.config.v_threshold,
                v_reset=0.0,
                tau_ref=self.config.tau_ref,
                g_L=0.10,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                noise_std=noise_std,
            ),
            device=self.device,
        )

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
