"""Nucleus Basalis (NB) - Acetylcholine Attention and Encoding System.

The NB is the brain's primary source of cortical acetylcholine (ACh), broadcasting
attention and encoding/retrieval mode signals that modulate learning, sensory
processing, and memory consolidation. NB cholinergic neurons exhibit fast bursts
in response to prediction errors and attention-grabbing events.
"""

from __future__ import annotations

from typing import ClassVar, Dict, Optional, Union

import torch

from thalia import GlobalConfig
from thalia.brain.adaptive_normalization import AdaptiveNormalization
from thalia.brain.configs import NeuralRegionConfig
from thalia.brain.neurons import (
    AcetylcholineNeuronConfig,
    AcetylcholineNeuron,
    heterogeneous_dendrite_coupling,
    heterogeneous_g_L,
    heterogeneous_noise_std,
    heterogeneous_tau_mem,
    heterogeneous_v_reset,
    heterogeneous_v_threshold,
)
from thalia.typing import (
    NeuromodulatorInput,
    NeuromodulatorChannel,
    PopulationName,
    PopulationPolarity,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)
from thalia.utils import CircularDelayBuffer

from .neuromodulator_source_region import NeuromodulatorSourceRegion
from .population_names import CortexPopulation, NucleusBasalisPopulation
from .region_registry import register_region


@register_region(
    "nucleus_basalis",
    aliases=["nb", "acetylcholine_system", "basal_forebrain"],
    description="Nucleus basalis - acetylcholine attention and encoding system",
)
class NucleusBasalis(NeuromodulatorSourceRegion[NeuralRegionConfig]):
    """Nucleus Basalis - Acetylcholine Attention and Encoding System.

    Computes prediction errors and attention signals, broadcasting acetylcholine
    via fast, brief bursts. Switches between encoding and retrieval modes,
    modulates attention, and coordinates cortical-hippocampal learning.
    """

    # Declarative neuromodulator output registry.
    neuromodulator_outputs: ClassVar[Dict[NeuromodulatorChannel, PopulationName]] = {
        NeuromodulatorChannel.ACH: NucleusBasalisPopulation.ACH,
    }

    def __init__(
        self,
        config: NeuralRegionConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        super().__init__(config, population_sizes, region_name, device=device)

        # Store sizes
        self.ach_neurons_size = population_sizes[NucleusBasalisPopulation.ACH]
        self.gaba_size = population_sizes[NucleusBasalisPopulation.GABA]

        # Acetylcholine neurons (fast bursts, brief duration)
        self.ach_neurons: AcetylcholineNeuron
        self.ach_neurons = self._create_and_register_neuron_population(
            population_name=NucleusBasalisPopulation.ACH,
            n_neurons=self.ach_neurons_size,
            polarity=PopulationPolarity.ANY,
            config=AcetylcholineNeuronConfig(
                prediction_error_to_current_gain=25.0,
                tau_mem_ms=heterogeneous_tau_mem(12.0, self.ach_neurons_size, device, cv=0.20),
                v_reset=heterogeneous_v_reset(-0.08, self.ach_neurons_size, device),
                v_threshold=heterogeneous_v_threshold(0.90, self.ach_neurons_size, device, cv=0.12, clamp_fraction=0.25),
                g_L=heterogeneous_g_L(0.083, self.ach_neurons_size, device),
                noise_std=heterogeneous_noise_std(0.07, self.ach_neurons_size, device),
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.ach_neurons_size, device, cv=0.25),
            ),
        )

        # GABAergic interneurons (local inhibition, homeostasis)
        self._init_gaba_interneurons(NucleusBasalisPopulation.GABA, self.gaba_size, device)

        # Prediction error computation state - use CircularDelayBuffer for history
        self._pfc_activity_buffer = CircularDelayBuffer(
            max_delay=10,  # Track last 10 timesteps for variance computation
            size=1,  # Single scalar value per timestep
            dtype=torch.float32,
            device=device,
        )

        # Adaptive normalization
        self._pe_norm = AdaptiveNormalization(clip_range=(0.0, 2.0))

        # Ensure all tensors are on the correct device
        self.to(device)

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Compute prediction error and drive acetylcholine neurons to burst.

        Note: neuromodulator_inputs is not used - NB is a neuromodulator source region.
        """
        # Find PFC executive spikes via SynapseId (SynapticInput is Dict[SynapseId, Tensor])
        pfc_spikes: Optional[torch.Tensor] = None
        amygdala_spikes: Optional[torch.Tensor] = None
        for sid, spikes in synaptic_inputs.items():
            if sid.source_region == "prefrontal_cortex" and sid.source_population == CortexPopulation.L5_PYR:
                pfc_spikes = spikes
            elif sid.source_region == "basolateral_amygdala":
                # BLA → NB: emotional salience / aversive surprise drives ACh bursts
                amygdala_spikes = spikes

        # Compute prediction error signal from inputs
        prediction_error = self._compute_prediction_error(pfc_spikes, amygdala_spikes)

        # Normalize PE to prevent saturation
        prediction_error = self._pe_norm(prediction_error)

        # Update ACh neurons with PE drive
        # High |PE| → depolarization → fast burst (10-20 Hz for 50-100ms)
        # ACh responds to magnitude, not sign (|PE|)
        # Apply GABA feedback from the previous timestep (closes homeostatic loop).
        gaba_feedback = self._gaba_feedback(self.ach_neurons_size, scale=0.05)
        ach_spikes, _ = self.ach_neurons.forward(
            g_ampa_input=None,    # No direct excitatory input; drive is via prediction error modulation
            g_nmda_input=None,    # NMDA not used for ACh neurons
            g_gaba_a_input=gaba_feedback,
            g_gaba_b_input=None,  # No GABA_B for fast ACh bursting
            prediction_error_drive=prediction_error,
        )
        # Update GABA interneurons (homeostatic control)
        ach_activity = ach_spikes.float().mean().item()
        gaba_spikes = self._step_gaba_interneurons(
            ach_activity,
            drive_scale=0.3,
            nmda_ratio=0.3,
            self_inhibition_scale=0.5,
        )

        region_outputs: RegionOutput = {
            NucleusBasalisPopulation.ACH: ach_spikes,
            NucleusBasalisPopulation.GABA: gaba_spikes,
        }

        self._pfc_activity_buffer.advance()

        return region_outputs

    def _compute_prediction_error(
        self,
        pfc_spikes: Optional[torch.Tensor],
        amygdala_spikes: Optional[torch.Tensor],
    ) -> float:
        """Compute prediction error magnitude from PFC and amygdala inputs.

        Prediction error heuristic:
        - Sudden PFC activity change → high PE
        - High amygdala activity → salient event → high PE
        - Combined: max(pfc_pe, amygdala_pe)

        Args:
            pfc_spikes: PFC spike tensor [n_pfc_neurons]
            amygdala_spikes: Amygdala spike tensor [n_amygdala_neurons]

        Returns:
            Prediction error magnitude in range [0, 1]
        """
        pe_components = []

        # PFC prediction error: sudden activity change
        if pfc_spikes is not None and pfc_spikes.sum() > 0:
            pfc_rate = pfc_spikes.float().mean().item()
            self._pfc_activity_buffer.write(torch.tensor([pfc_rate], device=self.device))

            # Read history for variance computation (last 10 timesteps)
            # Note: read returns zeros for uninitialized timesteps
            history_values = []
            for i in range(1, 11):  # Read delays 1-10
                val = self._pfc_activity_buffer.read(delay=i)
                if val.abs().sum() > 1e-8:  # Only include if buffer has data
                    history_values.append(val.item())

            # Sudden change → high PE
            if len(history_values) >= 2:
                recent = history_values[0]  # Most recent (delay=1)
                previous = sum(history_values[1:]) / len(history_values[1:])  # Average of older values
                change = abs(recent - previous)
                pfc_pe = min(1.0, change * 10.0)  # Scale to [0, 1]
                pe_components.append(pfc_pe)

        # Amygdala prediction error: high activity → salient/emotional event
        if amygdala_spikes is not None and amygdala_spikes.sum() > 0:
            amygdala_rate = amygdala_spikes.float().mean().item()
            # High amygdala activity → high PE
            amygdala_pe = min(1.0, amygdala_rate * 5.0)
            pe_components.append(amygdala_pe)

        # Combine: take maximum (any source triggers ACh burst)
        if pe_components:
            prediction_error = max(pe_components)
        else:
            # No inputs → low PE (predictable environment)
            prediction_error = 0.0

        return prediction_error
