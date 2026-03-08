"""Shared base class for amygdala nuclei (BLA and CeA).

Extracts the canonical biophysical constants and neuron-factory helper that are
common across amygdala sub-regions:

- Reversal potentials: ``E_L = 0``, ``E_E = 3``, ``E_I = -0.5``
- Membrane noise: ``noise_std = 0.03``
- ``_make_amygdala_neuron`` — builds a ``ConductanceLIF`` population with the
  canonical amygdala synaptic parameters, reading ``tau_ref`` from config.
"""

from __future__ import annotations

from typing import Generic, Optional, TypeVar

from thalia.brain.neurons import ConductanceLIF, ConductanceLIFConfig
from thalia.brain.configs.amygdala import AmygdalaNucleusConfig
from thalia.typing import PopulationName

from .neural_region import NeuralRegion

ConfigT = TypeVar("ConfigT", bound=AmygdalaNucleusConfig)


class AmygdalaNucleus(NeuralRegion[ConfigT], Generic[ConfigT]):
    """Shared base for amygdala nuclei (BLA, CeA).

    Provides the canonical amygdala biophysical constants and a factory method
    for building ``ConductanceLIF`` neuron populations with those constants.

    Canonical amygdala biophysics (common across sub-nuclei):

    * ``E_L = 0.0, E_E = 3.0, E_I = -0.5, v_reset = 0.0`` — normalised reversal
      potentials consistent with all other Thalia neurons.
    * ``noise_std = 0.03`` — moderate membrane noise; empirically needed to prevent
      synchrony artefacts in the recurrent BLA/CeA circuits.

    The two major sub-regions differ in:

    * **BLA**: glutamatergic principal neurons (``TwoCompartmentLIF``), PV and SOM
      GABAergic interneurons; STDP fear-conditioning plasticity; homeostasis.
    * **CeA**: purely GABAergic lateral and medial populations; lateral-inhibition
      disinhibition circuit; no Hebbian plasticity.

    ``_make_amygdala_neuron`` covers the GABAergic ``ConductanceLIF`` populations
    used by both regions (CeL, CeM, BLA-SOM).
    """

    # -------------------------------------------------------------------------
    # Canonical amygdala biophysical constants
    # -------------------------------------------------------------------------
    _AMY_E_L: float = 0.0
    _AMY_E_E: float = 3.0
    _AMY_E_I: float = -0.5
    _AMY_NOISE_STD: float = 0.03

    def _make_amygdala_neuron(
        self,
        n_neurons: int,
        pop_name: PopulationName,
        tau_mem: float,
        v_threshold: float = 1.0,
        *,
        g_L: float = 0.06,
        tau_E: float = 6.0,
        tau_I: float = 12.0,
        adapt_increment: float = 0.06,
        tau_adapt: float = 120.0,
        tau_ref: Optional[float] = None,
    ) -> ConductanceLIF:
        """Create an amygdala ``ConductanceLIF`` population.

        The canonical biophysical constants (``E_L``, ``E_E``, ``E_I``,
        ``v_reset``, ``noise_std``) are set from class-level attributes.
        ``tau_ref`` defaults to ``self.config.tau_ref``.

        Args:
            n_neurons:       Number of neurons.
            pop_name:        Population name, used for per-population diagnostics.
            tau_mem:         Membrane time constant (ms).
            v_threshold:     Firing threshold (normalised units).
            g_L:             Leak conductance.
            tau_E:           AMPA/NMDA time constant (ms).  Default 6 ms for CeA;
                             pass 8 ms for BLA SOM interneurons.
            tau_I:           GABA time constant (ms).  Default 12 ms for CeA;
                             pass 15 ms for BLA SOM interneurons.
            adapt_increment: Spike-triggered adaptation conductance increment.
            tau_adapt:       Adaptation conductance time constant (ms).
            tau_ref:         Refractory period (ms).  Defaults to
                             ``self.config.tau_ref`` when *None*.

        Returns:
            A ``ConductanceLIF`` instance on ``self.device``.
        """
        effective_tau_ref = self.config.tau_ref if tau_ref is None else tau_ref
        return ConductanceLIF(
            n_neurons=n_neurons,
            config=ConductanceLIFConfig(
                region_name=self.region_name,
                population_name=pop_name,
                tau_mem=tau_mem,
                v_threshold=v_threshold,
                v_reset=0.0,
                tau_ref=effective_tau_ref,
                g_L=g_L,
                E_L=self._AMY_E_L,
                E_E=self._AMY_E_E,
                E_I=self._AMY_E_I,
                tau_E=tau_E,
                tau_I=tau_I,
                adapt_increment=adapt_increment,
                tau_adapt=tau_adapt,
                E_adapt=-0.5,
                noise_std=self._AMY_NOISE_STD,
            ),
            device=self.device,
        )
