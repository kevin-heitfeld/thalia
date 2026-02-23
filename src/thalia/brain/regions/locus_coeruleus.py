"""Locus Coeruleus (LC) - Norepinephrine Arousal and Uncertainty System.

The LC is the brain's primary source of norepinephrine (NE), broadcasting arousal
and uncertainty signals that modulate attention, gain, and exploratory behavior.
LC norepinephrine neurons exhibit synchronized bursting in response to task
difficulty and novel/unexpected events.
"""

from __future__ import annotations

from typing import ClassVar, Dict, Optional

import torch

from thalia.brain.configs import LocusCoeruleusConfig
from thalia.components import (
    NorepinephrineNeuron,
    NorepinephrineNeuronConfig,
)
from thalia.typing import (
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
from .population_names import LocusCoeruleusPopulation
from .region_registry import register_region


@register_region(
    "locus_coeruleus",
    aliases=["lc", "norepinephrine_system"],
    description="Locus coeruleus - norepinephrine arousal and uncertainty system",
    version="1.0",
    author="Thalia Project",
    config_class=LocusCoeruleusConfig,
)
class LocusCoeruleus(NeuromodulatorSourceRegion[LocusCoeruleusConfig]):
    """Locus Coeruleus - Norepinephrine Arousal and Uncertainty System."""

    # Declarative neuromodulator output registry.
    neuromodulator_outputs: ClassVar[Dict[NeuromodulatorType, PopulationName]] = {
        'ne': LocusCoeruleusPopulation.NE,
    }

    def __init__(self, config: LocusCoeruleusConfig, population_sizes: PopulationSizes, region_name: RegionName):
        super().__init__(config, population_sizes, region_name)

        self.ne_neurons_size = population_sizes[LocusCoeruleusPopulation.NE]
        self.gaba_neurons_size = population_sizes[LocusCoeruleusPopulation.GABA]

        # Norepinephrine neurons (gap junction coupled, synchronized bursts)
        self.ne_neurons = NorepinephrineNeuron(
            n_neurons=self.ne_neurons_size,
            config=NorepinephrineNeuronConfig(
                region_name=self.region_name,
                population_name=LocusCoeruleusPopulation.NE,
                device=self.device,
                uncertainty_to_current_gain=self.config.uncertainty_gain,
                gap_junction_strength=self.config.gap_junction_strength,
                gap_junction_neighbor_radius=self.config.gap_junction_radius,
                i_h_conductance=NorepinephrineNeuronConfig.i_h_conductance if self.config.baseline_noise_conductance_enabled else 0.0,
                noise_std=NorepinephrineNeuronConfig.noise_std if self.config.baseline_noise_conductance_enabled else 0.0,
            ),
        )

        # GABAergic interneurons (local inhibition, homeostasis)
        self._init_gaba_interneurons(LocusCoeruleusPopulation.GABA, self.gaba_neurons_size)

        # Uncertainty computation state - use CircularDelayBuffer for history
        self._pfc_activity_buffer = CircularDelayBuffer(
            max_delay=10,  # Track last 10 timesteps for variance computation
            size=1,  # Single scalar value per timestep
            dtype=torch.float32,
            device=self.device,
        )
        self._hippocampus_activity_buffer = CircularDelayBuffer(
            max_delay=10,  # Track last 10 timesteps for novelty detection
            size=1,  # Single scalar value per timestep
            dtype=torch.float32,
            device=self.device,
        )
        self._uncertainty_buffer = CircularDelayBuffer(
            max_delay=1000,  # Track last 1000 timesteps for analysis
            size=1,  # Single scalar value per timestep
            dtype=torch.float32,
            device=self.device,
        )

        # Adaptive normalization
        if config.uncertainty_normalization:
            self._avg_uncertainty = 0.5
            self._uncertainty_count = 0

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(LocusCoeruleusPopulation.NE, self.ne_neurons, polarity=PopulationPolarity.ANY)

        # Ensure all tensors are on the correct device
        self.to(self.device)

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Compute uncertainty and drive norepinephrine neurons to burst.

        Note: neuromodulator_inputs is not used - LC is a neuromodulator source region.
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        # Extract PFC and hippocampus spikes from registered synaptic inputs.
        # Iterate over all inputs and identify by source_region so that any
        # BrainBuilder-registered PFC→LC or HPC→LC connection is automatically
        # picked up without constructing a hardcoded SynapseId at runtime.
        # (Constructing a SynapseId at runtime and using it as a dict key is
        # fragile: the key will never match unless BrainBuilder registered the
        # exact same connection, and it completely bypasses weight matrices.)
        pfc_spikes: Optional[torch.Tensor] = None
        hippocampus_spikes: Optional[torch.Tensor] = None
        for sid, spikes in synaptic_inputs.items():
            if sid.source_region == "prefrontal":
                pfc_spikes = spikes
            elif sid.source_region == "hippocampus":
                hippocampus_spikes = spikes

        # Compute uncertainty signal from inputs
        uncertainty = self._compute_uncertainty(pfc_spikes, hippocampus_spikes)

        # Normalize uncertainty to prevent saturation
        if self.config.uncertainty_normalization:
            uncertainty = self._normalize_uncertainty(uncertainty)

        # Track history for analysis using CircularDelayBuffer
        self._uncertainty_buffer.write(torch.tensor([uncertainty], device=self.device))

        # Update NE neurons with uncertainty drive
        # High uncertainty → depolarization → synchronized burst
        # Gap junctions → population synchronization
        ne_spikes, _ = self.ne_neurons.forward(
            g_ampa_input=None,  # No direct AMPA input to NE neurons (modulated by uncertainty drive instead)
            g_nmda_input=None,  # NE neurons do not receive NMDA input
            g_gaba_a_input=None,  # No direct GABA input to NE neurons (inhibition via gap junctions and interneurons)
            g_gaba_b_input=None,
            uncertainty_drive=uncertainty,
        )
        # Update GABA interneurons (homeostatic control)
        ne_activity = ne_spikes.float().mean().item()
        self._step_gaba_interneurons(ne_activity)

        region_outputs: RegionOutput = {
            LocusCoeruleusPopulation.NE: ne_spikes,
        }

        return self._post_forward(region_outputs)

    def _compute_uncertainty(
        self,
        pfc_spikes: Optional[torch.Tensor],
        hippocampus_spikes: Optional[torch.Tensor],
    ) -> float:
        """Compute uncertainty signal from PFC and hippocampus inputs.

        Uncertainty heuristic:
        - High PFC variance → high task difficulty/conflict → high uncertainty
        - High hippocampus activity → novelty → high uncertainty
        - Combined: max(pfc_uncertainty, hippocampus_uncertainty)

        Args:
            pfc_spikes: PFC spike tensor [n_pfc_neurons]
            hippocampus_spikes: Hippocampus spike tensor [n_hpc_neurons]

        Returns:
            Uncertainty signal in range [0, 1]
        """
        uncertainty_components = []

        # PFC uncertainty: variance of activity (conflict detection)
        if pfc_spikes is not None and pfc_spikes.sum() > 0:
            pfc_rate = pfc_spikes.float().mean().item()
            self._pfc_activity_buffer.write(torch.tensor([pfc_rate], device=self.device))

            # Read history for variance computation (last 10 timesteps)
            history_values = []
            for i in range(1, 11):  # Read delays 1-10
                val = self._pfc_activity_buffer.read(delay=i)
                if val.abs().sum() > 1e-8:  # Only include if buffer has data
                    history_values.append(val.item())

            # High variance → high uncertainty
            if len(history_values) >= 3:
                import math

                mean = sum(history_values) / len(history_values)
                variance = sum((x - mean) ** 2 for x in history_values) / len(history_values)
                std = math.sqrt(variance)
                pfc_uncertainty = min(1.0, std * 5.0)  # Scale to [0, 1]
                uncertainty_components.append(pfc_uncertainty)

        # Hippocampus uncertainty: high activity → novelty
        if hippocampus_spikes is not None and hippocampus_spikes.sum() > 0:
            hpc_rate = hippocampus_spikes.float().mean().item()
            self._hippocampus_activity_buffer.write(torch.tensor([hpc_rate], device=self.device))

            # Read history for baseline computation (last 10 timesteps)
            history_values = []
            for i in range(1, 11):  # Read delays 1-10
                val = self._hippocampus_activity_buffer.read(delay=i)
                if val.abs().sum() > 1e-8:  # Only include if buffer has data
                    history_values.append(val.item())

            # Deviation from baseline → novelty
            if len(history_values) >= 3:
                baseline = sum(history_values) / len(history_values)
                deviation = abs(hpc_rate - baseline)
                hpc_uncertainty = min(1.0, deviation * 10.0)  # Scale to [0, 1]
                uncertainty_components.append(hpc_uncertainty)

        # Combine: take maximum (any source of uncertainty triggers burst)
        if uncertainty_components:
            uncertainty = max(uncertainty_components)
        else:
            # No inputs → low uncertainty (stable environment)
            uncertainty = 0.0

        return uncertainty

    def _normalize_uncertainty(self, uncertainty: float) -> float:
        """Adaptive uncertainty normalization.

        Tracks running average and normalizes to maintain stable dynamics.

        Args:
            uncertainty: Raw uncertainty value

        Returns:
            Normalized uncertainty in range [0, 2]
        """
        # Update running average
        self._uncertainty_count += 1
        alpha = 1.0 / min(self._uncertainty_count, 100)
        self._avg_uncertainty = (
            1 - alpha
        ) * self._avg_uncertainty + alpha * uncertainty

        # Normalize
        epsilon = 0.1
        normalized = uncertainty / (self._avg_uncertainty + epsilon)

        # Clip
        normalized = max(0.0, min(2.0, normalized))

        return normalized
