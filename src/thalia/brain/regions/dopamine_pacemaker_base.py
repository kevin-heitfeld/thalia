"""Shared base class for dopaminergic pacemaker regions (VTA, SNc)."""

from __future__ import annotations

from typing import Generic, TypeVar

import torch

from thalia.brain.configs.basal_ganglia import DopaminePacemakerConfig
from thalia.typing import ConductanceTensor

from .neuromodulator_source_region import NeuromodulatorSourceRegion

ConfigT = TypeVar("ConfigT", bound=DopaminePacemakerConfig)


class DopaminePacemakerBase(NeuromodulatorSourceRegion[ConfigT], Generic[ConfigT]):
    """Shared base for dopaminergic pacemaker regions (VTA, SNc)."""

    def _compute_gaba_drive(self, primary_activity: float) -> torch.Tensor:
        """GABA interneuron drive: tonic baseline + DA auto-inhibition.

        Returns ``0.004 + 0.05 × primary_activity``, empirically calibrated for
        DA pacemaker regions (Tepper & Lee 2007).  Overrides the base-class
        default, which uses a formula calibrated for other neuromodulator sources.

        Args:
            primary_activity: Mean spike rate of the DA population (spikes/step).

        Returns:
            Drive conductance tensor of shape ``[gaba_size]``.
        """
        return torch.full(
            (self.gaba_size,),
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
            GABA spike tensor of shape ``[gaba_size]``.
        """
        gaba_drive = self._compute_gaba_drive(primary_activity)
        gaba_spikes, _ = self.gaba_neurons.forward(
            g_ampa_input=ConductanceTensor(gaba_drive),
            g_nmda_input=None,
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )
        return gaba_spikes
