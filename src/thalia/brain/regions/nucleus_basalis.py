"""Nucleus Basalis (NB) - Acetylcholine Attention and Encoding System.

The NB is the brain's primary source of cortical acetylcholine (ACh), broadcasting
attention and encoding/retrieval mode signals that modulate learning, sensory
processing, and memory consolidation. NB cholinergic neurons exhibit fast bursts
in response to prediction errors and attention-grabbing events.
"""

from __future__ import annotations

from typing import ClassVar, Dict, Optional

import torch

from thalia.brain.configs import NucleusBasalisConfig
from thalia.components import (
    AcetylcholineNeuron,
    AcetylcholineNeuronConfig,
)
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorInput,
    NeuromodulatorType,
    PopulationName,
    PopulationPolarity,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)
from thalia.utils import CircularDelayBuffer

from .neuromodulator_source_region import NeuromodulatorSourceRegion
from .population_names import NucleusBasalisPopulation, PrefrontalPopulation
from .region_registry import register_region


@register_region(
    "nucleus_basalis",
    aliases=["nb", "acetylcholine_system", "basal_forebrain"],
    description="Nucleus basalis - acetylcholine attention and encoding system",
    version="1.0",
    author="Thalia Project",
    config_class=NucleusBasalisConfig,
)
class NucleusBasalis(NeuromodulatorSourceRegion[NucleusBasalisConfig]):
    """Nucleus Basalis - Acetylcholine Attention and Encoding System.

    Computes prediction errors and attention signals, broadcasting acetylcholine
    via fast, brief bursts. Switches between encoding and retrieval modes,
    modulates attention, and coordinates cortical-hippocampal learning.
    """

    # Declarative neuromodulator output registry.
    neuromodulator_outputs: ClassVar[Dict[NeuromodulatorType, PopulationName]] = {
        'ach': NucleusBasalisPopulation.ACH,
    }

    def __init__(self, config: NucleusBasalisConfig, population_sizes: PopulationSizes, region_name: RegionName):
        super().__init__(config, population_sizes, region_name)

        # Store sizes
        self.ach_neurons_size = population_sizes[NucleusBasalisPopulation.ACH]
        self.gaba_neurons_size = population_sizes[NucleusBasalisPopulation.GABA]

        # Acetylcholine neurons (fast bursts, brief duration)
        self.ach_neurons = AcetylcholineNeuron(
            n_neurons=self.ach_neurons_size,
            config=AcetylcholineNeuronConfig(
                region_name=self.region_name,
                population_name=NucleusBasalisPopulation.ACH,
                prediction_error_to_current_gain=self.config.pe_gain,
            ),
            device=self.device,
        )

        # GABAergic interneurons (local inhibition, homeostasis)
        self._init_gaba_interneurons(NucleusBasalisPopulation.GABA, self.gaba_neurons_size)

        # Prediction error computation state - use CircularDelayBuffer for history
        self._pfc_activity_buffer = CircularDelayBuffer(
            max_delay=10,  # Track last 10 timesteps for variance computation
            size=1,  # Single scalar value per timestep
            dtype=torch.float32,
            device=self.device,
        )
        self._prediction_error_buffer = CircularDelayBuffer(
            max_delay=1000,  # Track last 1000 timesteps for analysis
            size=1,  # Single scalar value per timestep
            dtype=torch.float32,
            device=self.device,
        )

        # Adaptive normalization
        if config.pe_normalization:
            self._avg_pe = 0.5
            self._pe_count = 0

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(NucleusBasalisPopulation.ACH, self.ach_neurons, polarity=PopulationPolarity.ANY)

        # Ensure all tensors are on the correct device
        self.to(self.device)

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Compute prediction error and drive acetylcholine neurons to burst.

        Note: neuromodulator_inputs is not used - NB is a neuromodulator source region.
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        # Find PFC executive spikes via SynapseId (SynapticInput is Dict[SynapseId, Tensor])
        pfc_spikes: Optional[torch.Tensor] = None
        amygdala_spikes: Optional[torch.Tensor] = None
        for sid, spikes in synaptic_inputs.items():
            if sid.source_region == "prefrontal" and sid.source_population == PrefrontalPopulation.EXECUTIVE:
                pfc_spikes = spikes
            elif sid.source_region == "basolateral_amygdala":
                # BLA → NB: emotional salience / aversive surprise drives ACh bursts
                amygdala_spikes = spikes

        # Compute prediction error signal from inputs
        prediction_error = self._compute_prediction_error(pfc_spikes, amygdala_spikes)

        # Normalize PE to prevent saturation
        if self.config.pe_normalization:
            prediction_error = self._normalize_pe(prediction_error)

        # Track history for analysis using CircularDelayBuffer
        self._prediction_error_buffer.write(torch.tensor([prediction_error], device=self.device))

        # Update ACh neurons with PE drive
        # High |PE| → depolarization → fast burst (10-20 Hz for 50-100ms)
        # ACh responds to magnitude, not sign (|PE|)
        # Apply GABA feedback from the previous timestep (closes homeostatic loop).
        gaba_feedback = self._get_gaba_feedback_conductance(self.ach_neurons_size, gain=0.01)
        ach_spikes, _ = self.ach_neurons.forward(
            g_ampa_input=None,    # No direct excitatory input; drive is via prediction error modulation
            g_nmda_input=None,    # NMDA not used for ACh neurons
            g_gaba_a_input=ConductanceTensor(gaba_feedback),
            g_gaba_b_input=None,  # No GABA_B for fast ACh bursting
            prediction_error_drive=prediction_error,
        )
        # Update GABA interneurons (homeostatic control)
        ach_activity = ach_spikes.float().mean().item()
        gaba_spikes = self._step_gaba_interneurons(ach_activity)

        region_outputs: RegionOutput = {
            NucleusBasalisPopulation.ACH: ach_spikes,
            NucleusBasalisPopulation.GABA: gaba_spikes,
        }

        return self._post_forward(region_outputs)

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

    def _normalize_pe(self, pe: float) -> float:
        """Adaptive prediction error normalization.

        Tracks running average and normalizes to maintain stable dynamics.

        Args:
            pe: Raw prediction error value

        Returns:
            Normalized PE in range [0, 2]
        """
        # Update running average
        self._pe_count += 1
        alpha = 1.0 / min(self._pe_count, 100)
        self._avg_pe = (1 - alpha) * self._avg_pe + alpha * pe

        # Normalize
        epsilon = 0.1
        normalized = pe / (self._avg_pe + epsilon)

        # Clip
        normalized = max(0.0, min(2.0, normalized))

        return normalized

    def _compute_gaba_drive(self, primary_activity: float) -> torch.Tensor:
        """Compute drive for GABAergic interneurons.

        No tonic baseline: GABA only fires proportionally when ACh overshoots
        its target (gain=2.0).  The previous baseline=0.4 saturated GABA at
        ~420 Hz which completely silenced ACh neurons via feedback.

        Returns:
            Drive conductance for GABA neurons [gaba_neurons_size]
        """
        # Tonic baseline REMOVED: a constant 0.4 drive pushes GABA neurons to V*=2.67
        # (threshold=0.9), saturating them at ~420 Hz and fully silencing ACh neurons
        # via massive GABA feedback. GABA should only fire when ACh overshoots its
        # target. Match the base-class convention: gain 2.0, no baseline.
        feedback = primary_activity * 2.0  # Proportional feedback, no tonic component

        return torch.full((self.gaba_neurons_size,), feedback, device=self.device)
