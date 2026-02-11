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

from typing import Dict, Optional

import torch

from thalia.brain.configs import LCConfig
from thalia.components.neurons.neuron_factory import NeuronFactory, NeuronType
from thalia.components.neurons.norepinephrine_neuron import (
    NorepinephrineNeuron,
    NorepinephrineNeuronConfig,
)
from thalia.typing import PopulationName, PopulationSizes, RegionSpikesDict
from thalia.units import ConductanceTensor

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
    """Locus Coeruleus - Norepinephrine Arousal and Uncertainty System.

    Computes uncertainty signals and broadcasts norepinephrine via long,
    synchronized bursts. Modulates attention, gain, and exploratory behavior
    across the brain.

    Input Populations:
    ------------------
    - "pfc_input": Prefrontal cortex activity (task difficulty, conflict)
    - "hippocampus_input": Hippocampus activity (novelty detection)

    Output Populations:
    -------------------
    - "ne_output": Norepinephrine neuron spikes (broadcast to all targets)

    Computational Function:
    -----------------------
    1. Compute uncertainty from PFC/hippocampus activity
    2. Drive NE neurons: high uncertainty → synchronized burst
    3. Gap junction coupling → population synchronization
    4. Broadcast NE spikes to all target regions
    """

    OUTPUT_POPULATIONS: Dict[PopulationName, str] = {
        "ne_output": "n_ne_neurons",
    }

    def __init__(self, config: LCConfig, population_sizes: PopulationSizes):
        super().__init__(config, population_sizes)

        # Store sizes for test compatibility
        self.n_ne_neurons = config.n_ne_neurons
        self.n_gaba_neurons = config.n_gaba_neurons

        # Store input layer sizes
        self.pfc_input_size = population_sizes.get("pfc_input", 0)
        self.hippocampus_input_size = population_sizes.get("hippocampus_input", 0)

        # Norepinephrine neurons (gap junction coupled, synchronized bursts)
        self.ne_neurons = self._create_ne_neurons()

        # GABAergic interneurons (local inhibition, homeostasis)
        self.gaba_neurons = NeuronFactory.create(
            NeuronType.FAST_SPIKING,
            n_neurons=config.n_gaba_neurons,
            device=self.device,
        )

        # Uncertainty computation state
        self._pfc_activity_history: list[float] = []
        self._hippocampus_activity_history: list[float] = []
        self._uncertainty_history: list[float] = []

        # Adaptive normalization
        if config.uncertainty_normalization:
            self._avg_uncertainty = 0.5
            self._uncertainty_count = 0

        self.__post_init__()

    def _create_ne_neurons(self) -> NorepinephrineNeuron:
        """Create norepinephrine neuron population with gap junction coupling."""
        if self.config.ne_neuron_config is not None:
            ne_config = self.config.ne_neuron_config
        else:
            # Use default configuration
            ne_config = NorepinephrineNeuronConfig(
                device=self.config.device,
                uncertainty_to_current_gain=self.config.uncertainty_gain,
                gap_junction_strength=self.config.gap_junction_strength,
                gap_junction_neighbor_radius=self.config.gap_junction_radius,
            )

        return NorepinephrineNeuron(
            n_neurons=self.config.n_ne_neurons, config=ne_config, device=self.device
        )

    def forward(self, region_inputs: RegionSpikesDict) -> RegionSpikesDict:
        """Compute uncertainty and drive norepinephrine neurons to burst.

        Args:
            region_inputs: Dictionary of input spike tensors:
                - "pfc_input": PFC activity (task difficulty) [n_pfc_neurons]
                - "hippocampus_input": Hippocampus activity (novelty) [n_hpc_neurons]
        """
        self._pre_forward(region_inputs)

        # Get inputs (via connections from BrainBuilder)
        pfc_spikes = region_inputs.get("pfc_input")
        hippocampus_spikes = region_inputs.get("hippocampus_input")

        # Compute uncertainty signal from inputs
        uncertainty = self._compute_uncertainty(pfc_spikes, hippocampus_spikes)

        # Normalize uncertainty to prevent saturation
        if self.config.uncertainty_normalization:
            uncertainty = self._normalize_uncertainty(uncertainty)

        # Track history for analysis
        self._uncertainty_history.append(uncertainty)
        if len(self._uncertainty_history) > 1000:
            self._uncertainty_history.pop(0)

        # Update NE neurons with uncertainty drive
        # High uncertainty → depolarization → synchronized burst
        # Gap junctions → population synchronization
        ne_spikes, _ = self.ne_neurons.forward(
            g_exc_input=ConductanceTensor(torch.zeros(self.config.n_ne_neurons, device=self.device)),
            g_inh_input=ConductanceTensor(torch.zeros(self.config.n_ne_neurons, device=self.device)),
            uncertainty_drive=uncertainty,
        )
        self._current_ne_spikes = ne_spikes  # Store for GABA computation

        # Update GABA interneurons (homeostatic control)
        gaba_drive = self._compute_gaba_drive()
        g_gaba_exc = ConductanceTensor(gaba_drive / 10.0)  # Already a tensor
        self.gaba_neurons.forward(g_gaba_exc, None)

        region_outputs: RegionSpikesDict = {
            "ne_output": ne_spikes,
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
            self._pfc_activity_history.append(pfc_rate)
            if len(self._pfc_activity_history) > 10:
                self._pfc_activity_history.pop(0)

            # High variance → high uncertainty
            if len(self._pfc_activity_history) >= 3:
                import math

                mean = sum(self._pfc_activity_history) / len(
                    self._pfc_activity_history
                )
                variance = sum(
                    (x - mean) ** 2 for x in self._pfc_activity_history
                ) / len(self._pfc_activity_history)
                std = math.sqrt(variance)
                pfc_uncertainty = min(1.0, std * 5.0)  # Scale to [0, 1]
                uncertainty_components.append(pfc_uncertainty)

        # Hippocampus uncertainty: high activity → novelty
        if hippocampus_spikes is not None and hippocampus_spikes.sum() > 0:
            hpc_rate = hippocampus_spikes.float().mean().item()
            self._hippocampus_activity_history.append(hpc_rate)
            if len(self._hippocampus_activity_history) > 10:
                self._hippocampus_activity_history.pop(0)

            # Deviation from baseline → novelty
            if len(self._hippocampus_activity_history) >= 3:
                baseline = sum(self._hippocampus_activity_history) / len(
                    self._hippocampus_activity_history
                )
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
            Drive current for GABA neurons [n_gaba_neurons]
        """
        # Tonic baseline drive
        baseline = 3.0

        # Increase during NE bursts (negative feedback)
        if hasattr(self, "_current_ne_spikes"):
            ne_activity = self._current_ne_spikes.float().mean().item()
        else:
            ne_activity = 0.02  # Default tonic rate (1-3 Hz)
        feedback = ne_activity * 8.0  # Proportional to NE activity

        total_drive = baseline + feedback

        return torch.full(
            (self.config.n_gaba_neurons,), total_drive, device=self.device
        )

    def get_mean_uncertainty(self, window: int = 100) -> float:
        """Get mean uncertainty over recent history.

        Args:
            window: Number of timesteps to average over

        Returns:
            Mean uncertainty, or 0.0 if no history
        """
        if not self._uncertainty_history:
            return 0.0
        recent = self._uncertainty_history[-window:]
        return sum(recent) / len(recent)

    def get_ne_firing_rate_hz(self) -> float:
        """Get current NE neuron population firing rate in Hz.

        Returns:
            Firing rate in Hz
        """
        return self.ne_neurons.get_firing_rate_hz()
