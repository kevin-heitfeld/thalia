"""Nucleus Basalis (NB) - Acetylcholine Attention and Encoding System.

The NB is the brain's primary source of cortical acetylcholine (ACh), broadcasting
attention and encoding/retrieval mode signals that modulate learning, sensory
processing, and memory consolidation. NB cholinergic neurons exhibit fast bursts
in response to prediction errors and attention-grabbing events.

Biological Background:
======================
**Anatomy:**
- Location: Basal forebrain (nucleus basalis of Meynert)
- ~3,000-5,000 cholinergic neurons in humans
- Selective projections to cortex and hippocampus (not striatum)
- ~500-1,000 GABAergic interneurons for local inhibition

**Acetylcholine Neuron Firing Patterns:**
1. **Tonic Baseline** (2-5 Hz):
   - Retrieval mode, low attention demands
   - Pattern completion in hippocampus
   - Recurrent processing in cortex

2. **Phasic Bursts** (10-20 Hz, 50-100ms):
   - Prediction errors, attention shifts
   - Brief, fast bursts (faster than DA/NE)
   - Switch to encoding mode
   - Enhanced sensory processing

**Computational Role:**
=======================
NB implements **encoding/retrieval mode switching**:

**High ACh (Encoding Mode):**
- Enhance feedforward processing
- Suppress recurrent activity (reduce interference)
- Strengthen sensory → cortex weights
- Form new memories

**Low ACh (Retrieval Mode):**
- Enable pattern completion
- Strengthen recurrent activity
- Access existing memories
- Consolidate during sleep

**ACh also modulates:**
- Attention: Amplify attended features, suppress distractors
- Plasticity: High ACh → increased learning rate
- Cortical state: Desynchronize for processing vs synchronize for consolidation

**Inputs:**
- PFC: Prediction errors, attention signals
- Amygdala: Emotional salience, arousal
- Hypothalamus: Arousal, sleep-wake state

**Outputs:**
- **Cortex:** Sensory cortex, association cortex (encoding/attention modulation)
- **Hippocampus:** CA1, CA3, dentate gyrus (encoding/retrieval switching)
- **Effects:**
  - Nicotinic receptors: Fast excitation, attention
  - M1 receptors: Enhance feedforward, suppress feedback
  - M2 receptors: Reduce recurrence, enhance encoding

**Implementation Notes:**
=========================
Phase 2 (Current):
- Prediction error from simple heuristic (activity mismatch)
- Single ACh output (broadcast to targets)
- Fast burst dynamics with rapid SK adaptation

Phase 3 (Future):
- Sophisticated PE estimation (compare predictions to sensory input)
- Separate ACh projections by cortical layer
- Interaction with VTA/LC (DA × NE × ACh coordination)
- Sleep-wake modulation

References:
- Hasselmo & McGaughy (2004): ACh and cortical function
- Sarter & Parikh (2005): Cholinergic attention system
- Hangya et al. (2015): BF cholinergic neuron properties
- Gu & Yakel (2011): Cholinergic coordination of prefrontal cortex

Author: Thalia Project
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from thalia.brain.configs import NBConfig
from thalia.components.neurons.acetylcholine_neuron import (
    AcetylcholineNeuron,
    AcetylcholineNeuronConfig,
)
from thalia.components.neurons.neuron_factory import NeuronFactory, NeuronType
from thalia.typing import PopulationName, PopulationSizes, RegionSpikesDict

from ..neural_region import NeuralRegion
from ..region_registry import register_region


@register_region(
    "nucleus_basalis",
    aliases=["nb", "acetylcholine_system", "basal_forebrain"],
    description="Nucleus basalis - acetylcholine attention and encoding system",
    version="1.0",
    author="Thalia Project",
    config_class=NBConfig,
)
class NucleusBasalis(NeuralRegion[NBConfig]):
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
    - "ach_output": Acetylcholine neuron spikes (to cortex/hippocampus)

    Computational Function:
    -----------------------
    1. Compute prediction error magnitude from PFC activity
    2. Drive ACh neurons: high |PE| → fast burst
    3. Fast SK adaptation limits burst to 50-100ms
    4. Broadcast ACh spikes to cortex and hippocampus
    """

    OUTPUT_POPULATIONS: Dict[PopulationName, str] = {
        "ach_output": "n_ach_neurons",
    }

    def __init__(self, config: NBConfig, population_sizes: PopulationSizes):
        super().__init__(config, population_sizes)

        # Store sizes for test compatibility
        self.n_ach_neurons = config.n_ach_neurons
        self.n_gaba_neurons = config.n_gaba_neurons

        # Store input layer sizes
        self.pfc_input_size = population_sizes.get("pfc_input", 0)
        self.amygdala_input_size = population_sizes.get("amygdala_input", 0)

        # Acetylcholine neurons (fast bursts, brief duration)
        self.ach_neurons = self._create_ach_neurons()

        # GABAergic interneurons (local inhibition, homeostasis)
        self.gaba_neurons = NeuronFactory.create(
            NeuronType.FAST_SPIKING,
            n_neurons=config.n_gaba_neurons,
            device=self.device,
        )

        # Prediction error computation state
        self._pfc_activity_history: list[float] = []
        self._prediction_error_history: list[float] = []

        # Adaptive normalization
        if config.pe_normalization:
            self._avg_pe = 0.5
            self._pe_count = 0

        self.__post_init__()

    def _create_ach_neurons(self) -> AcetylcholineNeuron:
        """Create acetylcholine neuron population with fast burst dynamics."""
        if self.config.ach_neuron_config is not None:
            ach_config = self.config.ach_neuron_config
        else:
            # Use default configuration
            ach_config = AcetylcholineNeuronConfig(
                device=self.config.device,
                prediction_error_to_current_gain=self.config.pe_gain,
            )

        return AcetylcholineNeuron(
            n_neurons=self.config.n_ach_neurons, config=ach_config, device=self.device
        )

    def forward(self, region_inputs: RegionSpikesDict) -> RegionSpikesDict:
        """Compute prediction error and drive acetylcholine neurons to burst.

        Args:
            region_inputs: Dictionary of input spike tensors:
                - "pfc_input": PFC activity (prediction errors) [n_pfc_neurons]
                - "amygdala_input": Amygdala activity (salience) [n_amygdala_neurons]
        """
        self._pre_forward(region_inputs)

        # Get inputs (via connections from BrainBuilder)
        pfc_spikes = region_inputs.get("pfc_input")
        amygdala_spikes = region_inputs.get("amygdala_input")

        # Compute prediction error signal from inputs
        prediction_error = self._compute_prediction_error(pfc_spikes, amygdala_spikes)

        # Normalize PE to prevent saturation
        if self.config.pe_normalization:
            prediction_error = self._normalize_pe(prediction_error)

        # Track history for analysis
        self._prediction_error_history.append(prediction_error)
        if len(self._prediction_error_history) > 1000:
            self._prediction_error_history.pop(0)

        # Update ACh neurons with PE drive
        # High |PE| → depolarization → fast burst (10-20 Hz for 50-100ms)
        # ACh responds to magnitude, not sign (|PE|)
        ach_spikes = self.ach_neurons.forward(
            i_synaptic=0.0, prediction_error_drive=prediction_error
        )
        self._current_ach_spikes = ach_spikes  # Store for GABA computation

        # Update GABA interneurons (homeostatic control)
        gaba_drive = self._compute_gaba_drive()
        self.gaba_neurons.forward(gaba_drive)

        region_outputs: RegionSpikesDict = {
            "ach_output": ach_spikes,
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
            self._pfc_activity_history.append(pfc_rate)
            if len(self._pfc_activity_history) > 10:
                self._pfc_activity_history.pop(0)

            # Sudden change → high PE
            if len(self._pfc_activity_history) >= 2:
                recent = self._pfc_activity_history[-1]
                previous = sum(self._pfc_activity_history[:-1]) / (
                    len(self._pfc_activity_history) - 1
                )
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
            Drive current for GABA neurons [n_gaba_neurons]
        """
        # Tonic baseline drive
        baseline = 4.0

        # Increase during ACh bursts (negative feedback)
        if hasattr(self, "_current_ach_spikes"):
            ach_activity = self._current_ach_spikes.float().mean().item()
        else:
            ach_activity = 0.03  # Default tonic rate (2-5 Hz)
        feedback = ach_activity * 12.0  # Strong feedback (fast adaptation)

        total_drive = baseline + feedback

        return torch.full(
            (self.config.n_gaba_neurons,), total_drive, device=self.device
        )

    def get_mean_prediction_error(self, window: int = 100) -> float:
        """Get mean prediction error over recent history.

        Args:
            window: Number of timesteps to average over

        Returns:
            Mean PE, or 0.0 if no history
        """
        if not self._prediction_error_history:
            return 0.0
        recent = self._prediction_error_history[-window:]
        return sum(recent) / len(recent)

    def get_ach_firing_rate_hz(self) -> float:
        """Get current ACh neuron population firing rate in Hz.

        Returns:
            Firing rate in Hz
        """
        return self.ach_neurons.get_firing_rate_hz()

    def is_encoding_mode(self, threshold: float = 0.5) -> bool:
        """Determine if brain is in encoding mode (high ACh) or retrieval mode (low ACh).

        Args:
            threshold: ACh threshold for encoding vs retrieval

        Returns:
            True if encoding mode (ACh > threshold), False if retrieval mode
        """
        firing_rate = self.get_ach_firing_rate_hz()
        return firing_rate > threshold

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for this region."""
        diagnostics = {
            "ach_firing_rate_hz": self.get_ach_firing_rate_hz(),
            "mean_prediction_error": self.get_mean_prediction_error(window=100),
            "is_encoding_mode": self.is_encoding_mode(),
            "ach_mean_membrane_potential": self.ach_neurons.v_mem.mean().item(),
            "ach_mean_calcium": self.ach_neurons.ca_concentration.mean().item(),
            "ach_mean_sk_activation": self.ach_neurons.sk_activation.mean().item(),
        }

        if self._pfc_activity_history:
            diagnostics["mean_pfc_activity"] = sum(
                self._pfc_activity_history[-100:]
            ) / min(100, len(self._pfc_activity_history))

        if self.config.pe_normalization:
            diagnostics["avg_pe"] = self._avg_pe

        return diagnostics
