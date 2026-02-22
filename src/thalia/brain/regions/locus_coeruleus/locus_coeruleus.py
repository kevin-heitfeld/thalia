"""Locus Coeruleus (LC) - Norepinephrine Arousal and Uncertainty System.

The LC is the brain's primary source of norepinephrine (NE), broadcasting arousal
and uncertainty signals that modulate attention, gain, and exploratory behavior.
LC norepinephrine neurons exhibit synchronized bursting in response to task
difficulty and novel/unexpected events.

Biological Background:
======================
**Anatomy:**
- Location: Dorsal pons (brainstem)
- ~1,600 norepinephrine neurons in humans (tiny but highly influential)
- Dense gap junction coupling → synchronized population activity
- ~300-500 GABAergic interneurons for local inhibition

**Norepinephrine Neuron Firing Patterns:**
1. **Tonic Baseline** (1-3 Hz):
   - Low arousal, alert-but-relaxed state
   - Moderate exploration, stable task performance
   - Provides background NE tone

2. **Phasic Bursts** (10-15 Hz, 500ms):
   - High uncertainty, task difficulty, novelty
   - Triggered by unexpected events or errors
   - Network reset, increased exploration
   - Longer duration than DA bursts (500ms vs 200ms)

3. **High Tonic** (5-8 Hz):
   - Stress, high arousal, anxiety
   - Impairs performance (inverted-U relationship)

**Computational Role:**
=======================
LC implements **adaptive gain theory** by modulating:

1. **Neural Gain:** Amplifies signal-to-noise ratio in cortex
2. **Exploration-Exploitation:** High NE → explore, Low NE → exploit
3. **Network Reset:** High NE bursts clear working memory, enable belief updating
4. **Plasticity:** High NE → increased learning rate

Uncertainty signal derived from:
- Task difficulty (high conflict → high uncertainty)
- Prediction errors (unexpected outcomes)
- Novelty (unfamiliar stimuli)

**Inputs:**
- PFC: Task difficulty, conflict detection
- Hippocampus: Novelty detection, context mismatch
- Amygdala: Emotional salience

**Outputs:**
- **Global projections:** Cortex, hippocampus, striatum, thalamus, cerebellum
- **Effects:**
  - α1 receptors: Increase excitability, amplify responses
  - α2 receptors: Decrease noise, sharpen tuning
  - β receptors: Enhance plasticity, memory consolidation

**Implementation Notes:**
=========================
Phase 2 (Current):
- Uncertainty derived from simple heuristic (PFC activity variance)
- Single NE output (broadcast to all regions)
- Gap junction coupling for synchronized bursts

Phase 3 (Future):
- Sophisticated uncertainty estimation (prediction error magnitude)
- Separate NE projections by target region
- Interaction with VTA (DA × NE coordination)
- Stress modulation (cortisol, CRF inputs)
"""

from __future__ import annotations

from typing import ClassVar, Dict, Optional

import torch

from thalia.brain.configs import LCConfig
from thalia.brain.regions.population_names import HippocampusPopulation, LocusCoeruleusPopulation, PrefrontalPopulation
from thalia.components.neurons.neuron_factory import NeuronFactory, NeuronType
from thalia.components.neurons.norepinephrine_neuron import (
    NorepinephrineNeuron,
    NorepinephrineNeuronConfig,
)
from thalia.typing import (
    NeuromodulatorInput,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.units import ConductanceTensor
from thalia.utils import CircularDelayBuffer

from ..neural_region import NeuralRegion
from ..region_registry import register_region


@register_region(
    "locus_coeruleus",
    aliases=["lc", "norepinephrine_system"],
    description="Locus coeruleus - norepinephrine arousal and uncertainty system",
    version="1.0",
    author="Thalia Project",
    config_class=LCConfig,
)
class LocusCoeruleus(NeuralRegion[LCConfig]):
    """Locus Coeruleus - Norepinephrine Arousal and Uncertainty System."""

    # Declarative neuromodulator output registry.
    neuromodulator_outputs: ClassVar[Dict[str, str]] = {'ne': 'ne'}

    def __init__(self, config: LCConfig, population_sizes: PopulationSizes, region_name: RegionName):
        super().__init__(config, population_sizes, region_name)

        self.ne_neurons_size = population_sizes[LocusCoeruleusPopulation.NE.value]
        self.gaba_neurons_size = population_sizes[LocusCoeruleusPopulation.GABA.value]

        # Norepinephrine neurons (gap junction coupled, synchronized bursts)
        ne_config = NorepinephrineNeuronConfig(
            region_name=self.region_name,
            population_name=LocusCoeruleusPopulation.NE.value,
            device=self.device,
            uncertainty_to_current_gain=self.config.uncertainty_gain,
            gap_junction_strength=self.config.gap_junction_strength,
            gap_junction_neighbor_radius=self.config.gap_junction_radius,
            i_h_conductance=NorepinephrineNeuronConfig.i_h_conductance if self.config.baseline_noise_conductance_enabled else 0.0,
            noise_std=NorepinephrineNeuronConfig.noise_std if self.config.baseline_noise_conductance_enabled else 0.0,
        )
        self.ne_neurons = NorepinephrineNeuron(
            n_neurons=self.ne_neurons_size,
            config=ne_config,
        )

        # GABAergic interneurons (local inhibition, homeostasis)
        self.gaba_neurons = NeuronFactory.create(
            region_name=self.region_name,
            population_name=LocusCoeruleusPopulation.GABA.value,
            neuron_type=NeuronType.FAST_SPIKING,
            n_neurons=self.gaba_neurons_size,
            device=self.device,
        )

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
        self._register_neuron_population(LocusCoeruleusPopulation.NE.value, self.ne_neurons)
        self._register_neuron_population(LocusCoeruleusPopulation.GABA.value, self.gaba_neurons)

        self.__post_init__()

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Compute uncertainty and drive norepinephrine neurons to burst.

        Note: neuromodulator_inputs is not used - LC is a neuromodulator source region.
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        pfc_ne_synapse = SynapseId(
            source_region="prefrontal",
            source_population=PrefrontalPopulation.EXECUTIVE.value,
            target_region=self.region_name,
            target_population=LocusCoeruleusPopulation.NE.value,
        )
        ca1_ne_synapse = SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1.value,  # For simplicity, use overall CA1 activity as novelty signal
            target_region=self.region_name,
            target_population=LocusCoeruleusPopulation.NE.value,
        )

        pfc_spikes = synaptic_inputs.get(pfc_ne_synapse, None)
        hippocampus_spikes = synaptic_inputs.get(ca1_ne_synapse, None)

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
            g_gaba_a_input=None,  # No direct GABA input to NE neurons (inhibition via gap junctions and interneurons)
            g_nmda_input=None,  # NE neurons do not receive NMDA input
            uncertainty_drive=uncertainty,
        )
        self._current_ne_spikes = ne_spikes  # Store for GABA computation

        # Update GABA interneurons (homeostatic control)
        gaba_drive = self._compute_gaba_drive()
        # Split excitatory conductance: 70% AMPA (fast), 30% NMDA (slow)
        gaba_g_ampa = ConductanceTensor(gaba_drive * 0.7)
        gaba_g_nmda = ConductanceTensor(gaba_drive * 0.3)

        self.gaba_neurons.forward(
            g_ampa_input=gaba_g_ampa,
            g_gaba_a_input=None,
            g_nmda_input=gaba_g_nmda,
        )

        region_outputs: RegionOutput = {
            LocusCoeruleusPopulation.NE.value: ne_spikes,
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

    def _compute_gaba_drive(self) -> torch.Tensor:
        """Compute drive for GABAergic interneurons.

        GABA interneurons provide homeostatic control, preventing
        runaway NE bursting through local inhibition.

        Returns:
            Drive conductance for GABA neurons [gaba_neurons_size]
        """
        assert self._current_ne_spikes is not None, "NE spikes must be computed before GABA drive"

        # Tonic baseline conductance
        # BIOLOGY: LC-GABA neurons have intrinsic pacemaker activity
        # CONDUCTANCE: tonic drive
        baseline = 0.3 if self.config.baseline_noise_conductance_enabled else 0.0

        # Increase during NE bursts (negative feedback)
        ne_activity = self._current_ne_spikes.float().mean().item()
        feedback = ne_activity * 0.8  # CONDUCTANCE: proportional

        total_drive = baseline + feedback

        return torch.full((self.gaba_neurons_size,), total_drive, device=self.device)
