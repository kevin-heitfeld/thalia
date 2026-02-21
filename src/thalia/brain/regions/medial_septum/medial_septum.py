"""
Medial Septum - Theta Pacemaker for Hippocampal Circuits.

The medial septum is the primary theta rhythm generator, driving 4-10 Hz
oscillations in the hippocampus through two distinct cell populations:

1. **Cholinergic neurons** (ChAT+): Excite hippocampal pyramidal cells
2. **GABAergic neurons** (PV+): Inhibit hippocampal interneurons

**Key Biological Features**:
============================

1. **Intrinsic Bursting**: Neurons have intrinsic pacemaker properties
   - Slow calcium currents create rhythmic bursting
   - Frequency: 4-10 Hz (theta band)
   - Not driven by external input - self-sustaining oscillation

2. **Phase-Locked Populations**:
   - ACh neurons fire at theta peaks (0°, encoding phase)
   - GABA neurons fire at theta troughs (180°, retrieval phase)
   - 180° phase offset creates encoding/retrieval separation

3. **Pulsed Output** (not sinusoidal):
   - Burst phase: High firing rate (~50 Hz within burst)
   - Inter-burst: Silent or low firing
   - Hippocampal neurons phase-lock to these pulses

**Circuit Function**:
====================

Septal Output → Hippocampal Phase-Locking:

- **ACh → CA3 pyramidal**: Excite during encoding
- **GABA → OLM interneurons**: Inhibit during retrieval
  → OLM cells fire at theta troughs (rebound from inhibition)
  → OLM→CA1 suppresses apical dendrites (blocks retrieval)

Result: **Theta rhythm emerges from circuit dynamics, not hardcoded sinusoid**

**Neuromodulation**:
===================

- **Acetylcholine** (self-produced): Speeds up theta (7→11 Hz)
- **Norepinephrine**: Increases burst amplitude (arousal)
- **Dopamine**: Modulates burst frequency (motivation)
"""

from __future__ import annotations

import torch
import numpy as np

from thalia.brain.configs import MedialSeptumConfig
from thalia.brain.regions.population_names import HippocampusPopulation, MedialSeptumPopulation
from thalia.components.neurons import ConductanceLIF, ConductanceLIFConfig
from thalia.typing import (
    NeuromodulatorInput,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.units import ConductanceTensor

from ..neural_region import NeuralRegion
from ..region_registry import register_region


@register_region(
    "medial_septum",
    description="Theta pacemaker with cholinergic and GABAergic neurons",
    version="1.0",
    author="Thalia Project",
    config_class=MedialSeptumConfig,
)
class MedialSeptum(NeuralRegion[MedialSeptumConfig]):
    """
    Medial septum theta pacemaker with intrinsic bursting dynamics.

    Generates theta rhythm (4-10 Hz) through two phase-locked populations:
    - Cholinergic neurons (excite hippocampal pyramidal)
    - GABAergic neurons (inhibit hippocampal interneurons → OLM rebound)

    No external oscillator needed - theta emerges from intrinsic properties.
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: MedialSeptumConfig, population_sizes: PopulationSizes, region_name: RegionName):
        """Initialize medial septum with pacemaker neurons."""
        super().__init__(config, population_sizes, region_name)

        # Store sizes
        self.ach_size = population_sizes[MedialSeptumPopulation.ACH.value]
        self.gaba_size = population_sizes[MedialSeptumPopulation.GABA.value]

        # =====================================================================
        # NEURON POPULATIONS
        # =====================================================================

        # Cholinergic neurons (excite hippocampal pyramidal)
        # Properties: Slow bursting (~8 Hz), adaptation-driven burst termination
        ach_config = ConductanceLIFConfig(
            region_name=self.region_name,
            population_name=MedialSeptumPopulation.ACH.value,
            device=self.device,
            tau_mem=config.ach_tau_mem,
            v_threshold=config.ach_threshold,
            v_reset=config.ach_reset,
            tau_adapt=config.ach_adaptation_tau,
            adapt_increment=config.ach_adaptation_increment,
            tau_ref=5.0,  # SHORT refractory (5ms) allows multiple spikes per burst
        )
        self.ach_neurons = ConductanceLIF(
            n_neurons=self.ach_size,
            config=ach_config,
        )

        # GABAergic neurons (inhibit hippocampal interneurons)
        # Properties: Phase-locked to ACh but 180° offset, similar burst dynamics
        gaba_config = ConductanceLIFConfig(
            region_name=self.region_name,
            population_name=MedialSeptumPopulation.GABA.value,
            device=self.device,
            tau_mem=config.gaba_tau_mem,
            v_threshold=config.gaba_threshold,
            v_reset=config.gaba_reset,
            tau_adapt=config.gaba_adaptation_tau,
            adapt_increment=config.gaba_adaptation_increment,
            tau_ref=5.0,  # SHORT refractory (5ms) allows multiple spikes per burst
        )
        self.gaba_neurons = ConductanceLIF(
            n_neurons=self.gaba_size,
            config=gaba_config,
        )

        # =====================================================================
        # PACEMAKER DYNAMICS
        # =====================================================================

        # Intrinsic bursting is driven by slow calcium currents
        # Modeled as sinusoidal drive current (biological simplification)
        self.base_frequency_hz = config.base_frequency_hz
        self.pacemaker_phase = 0.0  # Tracks oscillator phase [0, 2π)

        # Burst parameters
        self.burst_duty_cycle = config.burst_duty_cycle  # Fraction of cycle spent bursting
        self.burst_amplitude = config.burst_amplitude     # Peak current during burst
        self.inter_burst_amplitude = config.inter_burst_amplitude  # Baseline current

        # =====================================================================
        # RECURRENT CONNECTIONS (for synchrony)
        # =====================================================================

        # Cholinergic neurons are weakly coupled (gap junctions + chemical)
        # CRITICAL: Strong enough for synchrony but not drive amplification
        # Recurrence synchronizes firing during burst window
        ach_recurrent_weights = torch.randn(self.ach_size, self.ach_size, device=self.device) * 0.03 / np.sqrt(self.ach_size)
        ach_recurrent_weights.fill_diagonal_(0.0)  # Zero self-connections
        self._add_internal_connection(
            source_population=MedialSeptumPopulation.ACH.value,
            target_population=MedialSeptumPopulation.ACH.value,
            weights=ach_recurrent_weights,
            stp_config=None,
            is_inhibitory=False,
        )

        # GABAergic neurons have stronger coupling (fast synchronization)
        # CRITICAL: Strong enough for tight synchrony
        gaba_recurrent_weights = torch.randn(self.gaba_size, self.gaba_size, device=self.device) * 0.04 / np.sqrt(self.gaba_size)
        gaba_recurrent_weights.fill_diagonal_(0.0)  # Zero self-connections
        self._add_internal_connection(
            source_population=MedialSeptumPopulation.GABA.value,
            target_population=MedialSeptumPopulation.GABA.value,
            weights=gaba_recurrent_weights,
            stp_config=None,
            is_inhibitory=False,
        )

        # Initialize state variables for spikes (for recurrent connections)
        # TODO: Use CircularBuffer for longer history if needed for more complex dynamics
        self._last_ach_spikes: torch.Tensor = torch.zeros(self.ach_size, dtype=torch.bool, device=self.device)
        self._last_gaba_spikes: torch.Tensor = torch.zeros(self.gaba_size, dtype=torch.bool, device=self.device)

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(MedialSeptumPopulation.ACH.value, self.ach_neurons)
        self._register_neuron_population(MedialSeptumPopulation.GABA.value, self.gaba_neurons)

        # =====================================================================
        # POST-INITIALIZATION
        # =====================================================================
        self.__post_init__()

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Generate theta rhythm through intrinsic bursting.

        Note: neuromodulator_inputs is not used - medial septum is a neuromodulator source region.
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        # =====================================================================
        # NEUROMODULATION OF PACEMAKER
        # =====================================================================
        # Acetylcholine: Speed up theta (7→11 Hz range)
        # TEMPORARY: Clamp to baseline 0.5 to prevent frequency upmodulation
        # Previously causing 16 Hz instead of 8 Hz theta due to neuromodulator scaling
        # TODO: Re-enable with proper neuromodulator dynamics and scaling after stabilizing theta rhythm
        ach_level = 0.5  # Clamped to baseline (was: self._ach_concentration.mean().item())
        frequency_mod = 1.0 + (ach_level - 0.5) * 0.5  # ±25% frequency
        current_freq = self.base_frequency_hz * frequency_mod

        # Norepinephrine: Increase burst amplitude (arousal)
        # TODO: Re-enable with proper neuromodulator dynamics and scaling after stabilizing theta rhythm
        ne_level = 0.5  # Clamped to baseline (was: self._ne_concentration.mean().item())
        amplitude_mod = 1.0 + (ne_level - 0.5) * 0.4  # ±20% amplitude

        # Dopamine: Subtle frequency modulation (motivation)
        # TODO: Re-enable with proper neuromodulator dynamics and scaling after stabilizing theta rhythm
        da_level = 0.5  # Clamped to baseline (was: self._da_concentration.mean().item())
        frequency_mod *= 1.0 + (da_level - 0.5) * 0.2  # Additional ±10%

        # =====================================================================
        # ADVANCE PACEMAKER PHASE
        # =====================================================================
        # Phase increment per millisecond
        phase_increment = 2 * np.pi * current_freq * self.config.dt_ms / 1000.0
        self.pacemaker_phase = (self.pacemaker_phase + phase_increment) % (2 * np.pi)

        # =====================================================================
        # GENERATE BURST CURRENTS
        # =====================================================================
        # Cholinergic neurons burst at phase 0 (theta peak, encoding)
        ach_phase = self.pacemaker_phase
        # CORRECTED: For X% duty cycle, use cos(phase) > cos(X * π)
        # This ensures burst window is exactly X% of the cycle
        burst_threshold = np.cos(self.burst_duty_cycle * np.pi)
        ach_in_burst = np.cos(ach_phase) > burst_threshold
        if ach_in_burst:
            ach_drive = self.burst_amplitude
        else:
            ach_drive = self.inter_burst_amplitude

        # GABAergic neurons burst at phase π (theta trough, retrieval)
        gaba_phase = (self.pacemaker_phase + np.pi) % (2 * np.pi)
        gaba_in_burst = np.cos(gaba_phase) > burst_threshold
        if gaba_in_burst:
            gaba_drive = self.burst_amplitude
        else:
            gaba_drive = self.inter_burst_amplitude

        # =====================================================================
        # RECURRENT EXCITATION (synchrony)
        # =====================================================================
        ach_rec_synapse = SynapseId(
            source_region=self.region_name,
            source_population=MedialSeptumPopulation.ACH.value,
            target_region=self.region_name,
            target_population=MedialSeptumPopulation.ACH.value,
        )
        gaba_rec_synapse = SynapseId(
            source_region=self.region_name,
            source_population=MedialSeptumPopulation.GABA.value,
            target_region=self.region_name,
            target_population=MedialSeptumPopulation.GABA.value,
        )
        ach_recurrent_conductance = self.get_synaptic_weights(ach_rec_synapse) @ self._last_ach_spikes.float()
        gaba_recurrent_conductance = self.get_synaptic_weights(gaba_rec_synapse) @ self._last_gaba_spikes.float()

        # =====================================================================
        # HIPPOCAMPAL FEEDBACK INHIBITION
        # =====================================================================
        # CA1 → Septum GABAergic feedback creates closed-loop control
        # When hippocampus is hyperactive, feedback suppresses septal drive
        # This prevents runaway theta generation that leads to oscillations
        ca1_gaba_synapse = SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1.value,
            target_region=self.region_name,
            target_population=MedialSeptumPopulation.GABA.value,
        )
        ca1_feedback = synaptic_inputs.get(ca1_gaba_synapse, None)
        hippocampal_inhibition_conductance = torch.zeros(self.gaba_size, device=self.device)
        if ca1_feedback is not None:
            weights_ca1 = self.get_synaptic_weights(ca1_gaba_synapse)  # [gaba_size, n_ca1]
            hippocampal_inhibition_conductance = torch.matmul(weights_ca1, ca1_feedback.float())

        # =====================================================================
        # RUN NEURONS
        # =====================================================================
        # CRITICAL: Recurrent input provides synchronization during bursts
        # Pacemaker drive creates burst window, recurrence synchronizes population
        ach_input = ach_drive + ach_recurrent_conductance * 0.15  # Synchrony mechanism
        gaba_input = gaba_drive + gaba_recurrent_conductance * 0.20  # Stronger for GABA

        # Split excitatory conductance: 70% AMPA (fast), 30% NMDA (slow)
        ach_g_ampa, ach_g_nmda = ach_input * 0.7, ach_input * 0.3
        gaba_g_ampa, gaba_g_nmda = gaba_input * 0.7, gaba_input * 0.3

        ach_spikes, _ = self.ach_neurons.forward(
            g_ampa_input=ConductanceTensor(ach_g_ampa),
            g_gaba_a_input=None,
            g_nmda_input=ConductanceTensor(ach_g_nmda),
        )
        gaba_spikes, _ = self.gaba_neurons.forward(
            g_ampa_input=ConductanceTensor(gaba_g_ampa),
            g_gaba_a_input=ConductanceTensor(hippocampal_inhibition_conductance),
            g_nmda_input=ConductanceTensor(gaba_g_nmda),
        )

        # =====================================================================
        # UPDATE STATE
        # =====================================================================
        self._last_ach_spikes = ach_spikes
        self._last_gaba_spikes = gaba_spikes

        region_outputs: RegionOutput = {
            MedialSeptumPopulation.ACH.value: ach_spikes,
            MedialSeptumPopulation.GABA.value: gaba_spikes,
        }

        return self._post_forward(region_outputs)

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        super().update_temporal_parameters(dt_ms)

        # Update neurons
        self.ach_neurons.update_temporal_parameters(dt_ms)
        self.gaba_neurons.update_temporal_parameters(dt_ms)
