"""Base class for subcortical neuromodulator projection nuclei.

All neuromodulator source regions (VTA, SNc, DRN, LC, NB) share a common
structural pattern:

    [Projection neurons (DA/NE/ACh/5-HT)] + [Local GABA interneurons]
    → Region-specific drive computation
    → Homeostatic GABA feedback loop

This base class centralizes:
- GABA interneuron creation with standardized biophysical parameters
- One-step causal GABA feedback to projection neurons
- GABA interneuron update driven by projection neuron activity
- ``_prev_gaba_spikes`` buffer management

Subclasses create their projection neurons in ``__init__()`` and implement
the full ``_step()`` method, calling GABA helpers for the shared portion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import torch

from thalia.brain.neurons import (
    ConductanceLIFConfig,
    heterogeneous_dendrite_coupling,
    heterogeneous_g_L,
    heterogeneous_noise_std,
    heterogeneous_tau_mem,
    heterogeneous_v_threshold,
    split_excitatory_conductance,
)
from thalia.typing import (
    ConductanceTensor,
    PopulationName,
    PopulationPolarity,
)

from .neural_region import NeuralRegion, ConfigT

if TYPE_CHECKING:
    from thalia.brain.neurons import ConductanceLIF


class NeuromodulatorSourceRegion(NeuralRegion[ConfigT]):
    """Base class for subcortical neuromodulator projection nuclei.

    Provides standardized GABA interneuron infrastructure shared by all
    neuromodulator source regions.  Subclasses create projection neurons
    and implement ``_step()`` using the helpers below.
    """

    # ── GABA interneuron helpers ─────────────────────────────────────────

    def _init_gaba_interneurons(
        self,
        population_name: PopulationName,
        n_neurons: int,
        device: Union[str, torch.device],
        *,
        tau_mem_ms: float = 8.0,
        noise_std: float = 0.08,
    ) -> None:
        """Create local GABAergic interneurons with standardised biophysical config.

        Parameters match the conserved profile across VTA, SNc, DRN, LC, NB:
        fast membrane (8 ms default), low threshold heterogeneity, and
        standard reversal potentials for all four receptor types.

        Args:
            population_name: StrEnum member for this GABA population.
            n_neurons: Number of GABA interneurons.
            device: Torch device.
            tau_mem_ms: Membrane time constant (default 8.0 ms; LC uses 10.0).
            noise_std: Membrane voltage noise (default 0.08; LC uses 0.02).
        """
        self.gaba_size = n_neurons

        self.gaba_neurons: ConductanceLIF
        self.gaba_neurons = self._create_and_register_neuron_population(
            population_name=population_name,
            n_neurons=n_neurons,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(tau_mem_ms, n_neurons, device=device, cv=0.10),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(1.0, n_neurons, device=device, cv=0.06),
                tau_ref=2.5,
                g_L=heterogeneous_g_L(0.10, n_neurons, device=device, cv=0.08),
                E_E=3.0,
                E_I=-0.5,
                tau_E=3.0,
                tau_I=3.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(noise_std, n_neurons, device=device),
                noise_tau_ms=3.0,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(
                    0.2, n_neurons, device, cv=0.20
                ),
            ),
        )

        self._prev_gaba_spikes: torch.Tensor
        self.register_buffer(
            "_prev_gaba_spikes",
            torch.zeros(n_neurons, dtype=torch.float32, device=device),
            persistent=False,
        )

    def _gaba_feedback(self, n_target_neurons: int, scale: float = 0.01) -> ConductanceTensor:
        """Compute GABA feedback conductance from previous timestep's interneuron spikes.

        Returns a uniform inhibitory conductance tensor sized for the target
        projection population.  Uses mean GABA activity × *scale* to set the
        conductance level.  One-step causal delay is inherent (uses stored
        ``_prev_gaba_spikes``).

        Args:
            n_target_neurons: Size of the target projection population.
            scale: Gain applied to mean GABA spike rate (default 0.01).
        """
        return ConductanceTensor(
            torch.full(
                (n_target_neurons,),
                self._prev_gaba_spikes.mean().item() * scale,
                device=self._prev_gaba_spikes.device,
            )
        )

    def _step_gaba_interneurons(
        self,
        projection_activity: float,
        *,
        drive_scale: float,
        drive_baseline: float = 0.0,
        nmda_ratio: float = 0.0,
        self_inhibition_scale: float = 0.0,
    ) -> torch.Tensor:
        """Update GABA interneurons driven by projection neuron activity.

        Implements the homeostatic feedback loop: projection neuron activity
        excites local GABA interneurons, which (on the next timestep via
        ``_gaba_feedback``) inhibit projection neurons.

        Args:
            projection_activity: Mean spike rate of the projection population
                (pre-computed by the subclass, may be low-pass filtered).
            drive_scale: Multiplicative gain on projection_activity.
            drive_baseline: Additive tonic baseline drive (default 0.0).
            nmda_ratio: Fraction of excitatory drive split to NMDA (default 0.0).
            self_inhibition_scale: If > 0, recurrent lateral inhibition among
                GABA neurons using previous timestep's spikes (DRN/NB pattern).

        Returns:
            GABA interneuron spike tensor (also stored in ``_prev_gaba_spikes``).
        """
        drive = drive_baseline + projection_activity * drive_scale
        gaba_drive = torch.full((self.gaba_size,), drive, device=self.device)

        g_nmda: Optional[ConductanceTensor] = None
        if nmda_ratio > 0:
            gaba_drive, g_nmda_raw = split_excitatory_conductance(gaba_drive, nmda_ratio=nmda_ratio)
            g_nmda = ConductanceTensor(g_nmda_raw)

        g_gaba_a: Optional[ConductanceTensor] = None
        if self_inhibition_scale > 0:
            g_gaba_a = ConductanceTensor(
                torch.full(
                    (self.gaba_size,),
                    self._prev_gaba_spikes.mean().item() * self_inhibition_scale,
                    device=self._prev_gaba_spikes.device,
                )
            )

        gaba_spikes, _ = self.gaba_neurons.forward(
            g_ampa_input=ConductanceTensor(gaba_drive),
            g_nmda_input=g_nmda,
            g_gaba_a_input=g_gaba_a,
            g_gaba_b_input=None,
        )

        self._prev_gaba_spikes = gaba_spikes.float()
        return gaba_spikes
