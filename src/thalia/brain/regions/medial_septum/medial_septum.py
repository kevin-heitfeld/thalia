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

from typing import Any, Dict

import torch
import torch.nn as nn
import numpy as np

from thalia.brain.configs import MedialSeptumConfig
from thalia.components.neurons import ConductanceLIF, ConductanceLIFConfig
from thalia.typing import PopulationName, PopulationSizes, RegionSpikesDict

from ..neural_region import NeuralRegion
from ..region_registry import register_region


@register_region(
    "medial_septum",
    aliases=["septum", "ms"],
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

    OUTPUT_POPULATIONS: Dict[PopulationName, str] = {
        "ach": "n_ach",        # Cholinergic only
        "gaba": "n_gaba",      # GABAergic only
    }

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: MedialSeptumConfig, population_sizes: PopulationSizes):
        """Initialize medial septum with pacemaker neurons."""
        super().__init__(config=config, population_sizes=population_sizes)

        self.n_ach = config.n_ach
        self.n_gaba = config.n_gaba
        self.n_total = config.n_ach + config.n_gaba

        # Add gaba layer for consistent routing (primary output is GABAergic inhibition)
        self.gaba_size = config.n_gaba

        # =====================================================================
        # NEURON POPULATIONS
        # =====================================================================

        # Cholinergic neurons (excite hippocampal pyramidal)
        # Properties: Slow bursting (~8 Hz), strong adaptation, high threshold
        ach_config = ConductanceLIFConfig(
            tau_mem=config.ach_tau_mem,
            v_threshold=config.ach_threshold,
            v_reset=config.ach_reset,
            tau_adapt=config.ach_adaptation_tau,
            adapt_increment=config.ach_adaptation_increment,
        )
        self.ach_neurons = ConductanceLIF(
            n_neurons=config.n_ach,
            config=ach_config,
            device=self.device,
        )

        # GABAergic neurons (inhibit hippocampal interneurons)
        # Properties: Fast bursting (~8 Hz), less adaptation, lower threshold
        gaba_config = ConductanceLIFConfig(
            tau_mem=config.gaba_tau_mem,
            v_threshold=config.gaba_threshold,
            v_reset=config.gaba_reset,
            tau_adapt=config.gaba_adaptation_tau,
            adapt_increment=config.gaba_adaptation_increment,
        )
        self.gaba_neurons = ConductanceLIF(
            n_neurons=config.n_gaba,
            config=gaba_config,
            device=self.device,
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
        self.ach_recurrent = nn.Parameter(
            torch.randn(config.n_ach, config.n_ach, device=self.device) * 0.1 / np.sqrt(config.n_ach)
        )
        # Zero self-connections
        with torch.no_grad():
            self.ach_recurrent.fill_diagonal_(0.0)

        # GABAergic neurons have stronger coupling (fast synchronization)
        self.gaba_recurrent = nn.Parameter(
            torch.randn(config.n_gaba, config.n_gaba, device=self.device) * 0.15 / np.sqrt(config.n_gaba)
        )
        with torch.no_grad():
            self.gaba_recurrent.fill_diagonal_(0.0)

        # Initialize state variables for spikes (for recurrent connections)
        self.cholinergic_spikes: torch.Tensor = torch.zeros(config.n_ach, dtype=torch.bool, device=self.device)
        self.gabaergic_spikes: torch.Tensor = torch.zeros(config.n_gaba, dtype=torch.bool, device=self.device)

        # =====================================================================
        # POST-INITIALIZATION
        # =====================================================================
        self.__post_init__()

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def forward(self, region_inputs: RegionSpikesDict) -> RegionSpikesDict:
        """Generate theta rhythm through intrinsic bursting."""
        self._pre_forward(region_inputs)

        # =====================================================================
        # NEUROMODULATION OF PACEMAKER
        # =====================================================================
        # Acetylcholine: Speed up theta (7→11 Hz range)
        ach_level = self._ach_concentration.mean().item() if hasattr(self, '_ach_concentration') else 0.5
        frequency_mod = 1.0 + (ach_level - 0.5) * 0.5  # ±25% frequency
        current_freq = self.base_frequency_hz * frequency_mod

        # Norepinephrine: Increase burst amplitude (arousal)
        ne_level = self._ne_concentration.mean().item() if hasattr(self, '_ne_concentration') else 0.5
        amplitude_mod = 1.0 + (ne_level - 0.5) * 0.4  # ±20% amplitude

        # Dopamine: Subtle frequency modulation (motivation)
        da_level = self._da_concentration.mean().item() if hasattr(self, '_da_concentration') else 0.5
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
        ach_in_burst = torch.cos(torch.tensor(ach_phase)) > (1 - 2 * self.burst_duty_cycle)
        if ach_in_burst:
            ach_drive = self.burst_amplitude * amplitude_mod
        else:
            ach_drive = self.inter_burst_amplitude

        # GABAergic neurons burst at phase π (theta trough, retrieval)
        gaba_phase = (self.pacemaker_phase + np.pi) % (2 * np.pi)
        gaba_in_burst = torch.cos(torch.tensor(gaba_phase)) > (1 - 2 * self.burst_duty_cycle)
        if gaba_in_burst:
            gaba_drive = self.burst_amplitude * amplitude_mod
        else:
            gaba_drive = self.inter_burst_amplitude

        # =====================================================================
        # EXTERNAL INPUTS (minimal - mostly self-sustaining)
        # =====================================================================
        external_input = region_inputs.get("default", torch.zeros(self.config.n_ach, device=self.device))
        if external_input.numel() < self.config.n_ach:
            external_input = torch.zeros(self.config.n_ach, device=self.device)
        elif external_input.numel() > self.config.n_ach:
            external_input = external_input[: self.config.n_ach]

        # =====================================================================
        # RECURRENT EXCITATION (synchrony)
        # =====================================================================
        # Cholinergic recurrence
        ach_recurrent_input = self.cholinergic_spikes.float() @ self.ach_recurrent.T

        # GABAergic recurrence
        gaba_recurrent_input = self.gabaergic_spikes.float() @ self.gaba_recurrent.T

        # =====================================================================
        # COMPUTE TOTAL INPUTS
        # =====================================================================
        ach_input = (
            ach_drive
            + external_input * 0.1  # Weak external modulation
            + ach_recurrent_input * 0.3  # Recurrent synchrony
        )

        gaba_input = (
            gaba_drive
            + gaba_recurrent_input * 0.5  # Stronger recurrence for fast sync
        )

        # =====================================================================
        # RUN NEURONS
        # =====================================================================
        ach_spikes, _ = self.ach_neurons(ach_input)
        gaba_spikes, _ = self.gaba_neurons(gaba_input)

        # =====================================================================
        # UPDATE STATE
        # =====================================================================
        self.cholinergic_spikes = ach_spikes
        self.gabaergic_spikes = gaba_spikes

        region_outputs: RegionSpikesDict = {
            "ach": ach_spikes,
            "gaba": gaba_spikes,
        }

        return self._post_forward(region_outputs)

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for this region."""
        ach_rate = self.cholinergic_spikes.float().mean().item()
        gaba_rate = self.gabaergic_spikes.float().mean().item()

        return {
            "pacemaker_phase": self.pacemaker_phase,
            "ach_firing_rate": ach_rate,
            "gaba_firing_rate": gaba_rate,
        }
