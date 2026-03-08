"""Shared base class for dopaminergic pacemaker regions (VTA, SNc).

Extracts common infrastructure shared by VTA and SNc:

- ``_make_da_neurons`` — builds a ``ConductanceLIF`` DA population with the
  canonical I_h (HCN) pacemaker parameters.
- ``_compute_gaba_drive`` — tonic GABA drive ``0.004 + 0.05 × activity``.
- ``_step_gaba_interneurons`` — AMPA-only GABA interneuron step (no NMDA),
  consistent with the low NMDA-receptor expression in midbrain GABA neurons.
"""

from __future__ import annotations

from typing import Generic, Optional, TypeVar

import torch

from thalia.brain.neurons import ConductanceLIF, ConductanceLIFConfig
from thalia.brain.configs.basal_ganglia import DopaminePacemakerConfig
from thalia.typing import ConductanceTensor, PopulationName

from .neuromodulator_source_region import NeuromodulatorSourceRegion

ConfigT = TypeVar("ConfigT", bound=DopaminePacemakerConfig)


class DopaminePacemakerBase(NeuromodulatorSourceRegion[ConfigT], Generic[ConfigT]):
    """Shared base for dopaminergic pacemaker regions (VTA, SNc).

    Provides three methods that are identical between VTA and SNc, factored out
    here to keep each concrete class focused on its unique biology:

    * ``_make_da_neurons(n, pop_name, *, adapt_increment)`` — creates a
      ``ConductanceLIF`` population with the canonical DA I_h pacemaker
      parameters. Callers may override ``adapt_increment`` to match
      sub-population-specific adaptation (e.g. VTA mesocortical neurons).

    * ``_compute_gaba_drive(primary_activity)`` — returns the tonic GABA drive
      tensor ``0.004 + 0.05 × primary_activity`` (Tepper & Lee 2007). Overrides
      the ``NeuromodulatorSourceRegion`` default, which is calibrated for
      different neuromodulator sources.

    * ``_step_gaba_interneurons(primary_activity)`` — steps GABA interneurons
      with AMPA-only input (``g_nmda_input=None``). Midbrain GABA neurons express
      few functional NMDA receptors; passing NMDA would cause voltage-dependent
      Mg²⁺-block artefacts and spurious conductance build-up.
    """

    # -------------------------------------------------------------------------
    # I_h (HCN) pacemaker channel constants — shared by all DA sub-populations.
    # Source: Neuhoff et al. (2002), J Neurophysiol 88:1689–1700.
    #
    # _IH_E_H = +0.9 (normalised):
    #     HCN reversal potential ≈ -45 mV biological; in the Thalia
    #     E_L = 0, E_E = 3 normalisation this maps to +0.9.
    #     Previously hard-coded as -0.3 in VTA/SNc, which placed E_h *below*
    #     E_L (i.e. hyperpolarising at rest) — the opposite of biology.
    #
    # _IH_V_HALF_H = -0.35 (normalised):
    #     Half-activation voltage ≈ -75 mV biological maps to -0.35 normalised.
    # -------------------------------------------------------------------------
    _IH_G_H_MAX: float = 0.03
    _IH_E_H: float = 0.9
    _IH_V_HALF_H: float = -0.35
    _IH_K_H: float = 0.08
    _IH_TAU_H_MS: float = 150.0

    def _make_da_neurons(
        self,
        n_neurons: int,
        pop_name: PopulationName,
        *,
        adapt_increment: Optional[float] = None,
    ) -> ConductanceLIF:
        """Create a DA neuron population with I_h (HCN) pacemaker channels.

        All biophysical parameters (``tau_mem``, ``g_L``, ``tau_ref``,
        ``noise_std``, ``adapt_increment``, ``tau_adapt``) are drawn from
        ``self.config``.  I_h parameters use the class-level constants.

        Args:
            n_neurons:       Number of DA neurons.
            pop_name:        Population name for per-population diagnostics.
            adapt_increment: Spike-triggered adaptation increment.
                             Defaults to ``self.config.adapt_increment`` when
                             *None*; pass an explicit value to override for
                             sub-populations with different adaptation dynamics
                             (e.g. VTA mesocortical neurons use a higher value).

        Returns:
            A ``ConductanceLIF`` instance on ``self.device``.
        """
        cfg = self.config
        inc = cfg.adapt_increment if adapt_increment is None else adapt_increment
        return ConductanceLIF(
            n_neurons=n_neurons,
            config=ConductanceLIFConfig(
                region_name=self.region_name,
                population_name=pop_name,
                tau_mem=cfg.tau_mem,
                v_threshold=1.0,
                v_reset=0.0,
                tau_ref=cfg.tau_ref,
                g_L=cfg.g_L,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                noise_std=cfg.noise_std,
                adapt_increment=inc,
                tau_adapt=cfg.tau_adapt,
                E_adapt=-0.5,
                # I_h (HCN) pacemaker — see class-level constants for rationale.
                enable_ih=True,
                g_h_max=self._IH_G_H_MAX,
                E_h=self._IH_E_H,
                V_half_h=self._IH_V_HALF_H,
                k_h=self._IH_K_H,
                tau_h_ms=self._IH_TAU_H_MS,
            ),
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Overrides of NeuromodulatorSourceRegion defaults
    # ------------------------------------------------------------------

    def _compute_gaba_drive(self, primary_activity: float) -> torch.Tensor:
        """GABA interneuron drive: tonic baseline + DA auto-inhibition.

        Returns ``0.004 + 0.05 × primary_activity``, empirically calibrated for
        DA pacemaker regions (Tepper & Lee 2007).  Overrides the base-class
        default, which uses a formula calibrated for other neuromodulator sources.

        Args:
            primary_activity: Mean spike rate of the DA population (spikes/step).

        Returns:
            Drive conductance tensor of shape ``[gaba_neurons_size]``.
        """
        return torch.full(
            (self.gaba_neurons_size,),
            0.004 + primary_activity * 0.05,
            device=self.device,
        )

    def _step_gaba_interneurons(self, primary_activity: float) -> torch.Tensor:
        """Step GABA interneurons with AMPA-only drive (no NMDA).

        Midbrain GABA interneurons express few functional NMDA receptors.
        Using AMPA-only avoids the voltage-dependent Mg²⁺-block artefact and
        prevents spurious NMDA conductance build-up during tonic pacemaking.

        Args:
            primary_activity: Mean spike rate of the DA population (spikes/step).

        Returns:
            GABA spike tensor of shape ``[gaba_neurons_size]``.
        """
        gaba_drive = self._compute_gaba_drive(primary_activity)
        gaba_spikes, _ = self.gaba_neurons.forward(
            g_ampa_input=ConductanceTensor(gaba_drive),
            g_nmda_input=None,
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )
        return gaba_spikes
