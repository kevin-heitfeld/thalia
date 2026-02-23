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
    NeuronFactory,
    NeuronType,
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
from thalia.utils import CircularDelayBuffer, split_excitatory_conductance

from .neural_region import NeuralRegion
from .population_names import NucleusBasalisPopulation
from .region_registry import register_region


@register_region(
    "nucleus_basalis",
    aliases=["nb", "acetylcholine_system", "basal_forebrain"],
    description="Nucleus basalis - acetylcholine attention and encoding system",
    version="1.0",
    author="Thalia Project",
    config_class=NucleusBasalisConfig,
)
class NucleusBasalis(NeuralRegion[NucleusBasalisConfig]):
    """Nucleus Basalis - Acetylcholine Attention and Encoding System.

    Computes prediction errors and attention signals, broadcasting acetylcholine
    via fast, brief bursts. Switches between encoding and retrieval modes,
    modulates attention, and coordinates cortical-hippocampal learning.

    Input Populations:
    ------------------
    - "pfc_input": Prefrontal cortex activity (prediction errors, attention)
    - "amygdala_input": Amygdala activity (emotional salience) [future]

    Output Populations:
    -------------------
    - "ach_neurons": Acetylcholine neuron spikes (to cortex/hippocampus)

    Computational Function:
    -----------------------
    1. Compute prediction error magnitude from PFC activity
    2. Drive ACh neurons: high |PE| → fast burst
    3. Fast SK adaptation limits burst to 50-100ms
    4. Broadcast ACh spikes to cortex and hippocampus
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
                device=self.device,
                prediction_error_to_current_gain=self.config.pe_gain,
                i_h_conductance=AcetylcholineNeuronConfig.i_h_conductance if self.config.baseline_noise_conductance_enabled else 0.0,
                noise_std=AcetylcholineNeuronConfig.noise_std if self.config.baseline_noise_conductance_enabled else 0.0,
            ),
        )

        # GABAergic interneurons (local inhibition, homeostasis)
        self.gaba_neurons = NeuronFactory.create(
            region_name=self.region_name,
            population_name=NucleusBasalisPopulation.GABA,
            neuron_type=NeuronType.FAST_SPIKING,
            n_neurons=self.gaba_neurons_size,
            device=self.device,
        )

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
        self._register_neuron_population(NucleusBasalisPopulation.GABA, self.gaba_neurons, polarity=PopulationPolarity.INHIBITORY)

        self.__post_init__()

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Compute prediction error and drive acetylcholine neurons to burst.

        Note: neuromodulator_inputs is not used - NB is a neuromodulator source region.
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        # Get inputs using routing keys (target_pop:source_region:source_pop)
        pfc_spikes = synaptic_inputs.get("ach_neurons:pfc:executive")
        amygdala_spikes = synaptic_inputs.get("ach_neurons:amygdala:...")  # Not connected yet

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
        ach_spikes, _ = self.ach_neurons.forward(
            g_ampa_input=None,  # No direct excitatory input; drive is via prediction error modulation
            g_gaba_a_input=None,  # No direct inhibitory input; homeostasis via interneurons
            g_nmda_input=None,  # NMDA not used for ACh neurons
            prediction_error_drive=prediction_error,
        )
        self._current_ach_spikes = ach_spikes  # Store for GABA computation

        # Update GABA interneurons (homeostatic control)
        gaba_drive = self._compute_gaba_drive()

        gaba_g_ampa, gaba_g_nmda = split_excitatory_conductance(gaba_drive, nmda_ratio=0.3)
        self.gaba_neurons.forward(
            g_ampa_input=ConductanceTensor(gaba_g_ampa),
            g_gaba_a_input=None,
            g_nmda_input=ConductanceTensor(gaba_g_nmda),
        )

        region_outputs: RegionOutput = {
            NucleusBasalisPopulation.ACH: ach_spikes,
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

    def _compute_gaba_drive(self) -> torch.Tensor:
        """Compute drive for GABAergic interneurons.

        GABA interneurons provide homeostatic control, preventing
        runaway ACh bursting through local inhibition.

        Returns:
            Drive conductance for GABA neurons [gaba_neurons_size]
        """
        assert self._current_ach_spikes is not None, "ACh spikes must be computed before GABA drive"

        # Tonic baseline conductance
        # BIOLOGY: NB-GABA neurons have intrinsic pacemaker activity
        # CONDUCTANCE: tonic drive
        baseline = 0.4 if self.config.baseline_noise_conductance_enabled else 0.0

        # Increase during ACh bursts (negative feedback)
        ach_activity = self._current_ach_spikes.float().mean().item()
        feedback = ach_activity * 1.2  # CONDUCTANCE: strong feedback

        total_drive = baseline + feedback

        return torch.full(
            (self.gaba_neurons_size,), total_drive, device=self.device
        )
