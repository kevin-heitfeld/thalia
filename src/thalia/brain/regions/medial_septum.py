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

from typing import ClassVar, Dict

import torch
import numpy as np

from thalia.brain.configs import MedialSeptumConfig
from thalia.components import ConductanceLIF, ConductanceLIFConfig, WeightInitializer
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorInput,
    NeuromodulatorType,
    PopulationName,
    PopulationPolarity,
    PopulationSizes,
    ReceptorType,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.utils import split_excitatory_conductance

from .neural_region import NeuralRegion
from .population_names import HippocampusPopulation, MedialSeptumPopulation
from .region_registry import register_region


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

    # Declarative neuromodulator output registry.
    # ACh from the septal cholinergic population modulates hippocampal CA1 via
    # muscarinic receptors (M1) — routed through NeuromodulatorHub, not AMPA.
    neuromodulator_outputs: ClassVar[Dict[NeuromodulatorType, PopulationName]] = {
        'ach_septal': MedialSeptumPopulation.ACH,
    }

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: MedialSeptumConfig, population_sizes: PopulationSizes, region_name: RegionName):
        """Initialize medial septum with pacemaker neurons."""
        super().__init__(config, population_sizes, region_name)

        # Store sizes
        self.ach_size = population_sizes[MedialSeptumPopulation.ACH]
        self.gaba_size = population_sizes[MedialSeptumPopulation.GABA]

        # =====================================================================
        # NEURON POPULATIONS
        # =====================================================================

        # Cholinergic neurons (excite hippocampal pyramidal)
        # Properties: Slow bursting (~8 Hz), adaptation-driven burst termination
        self.ach_neurons = ConductanceLIF(
            n_neurons=self.ach_size,
            config=ConductanceLIFConfig(
                region_name=self.region_name,
                population_name=MedialSeptumPopulation.ACH,
                device=self.device,
                tau_mem=config.ach_tau_mem,
                v_threshold=config.ach_threshold,
                v_reset=config.ach_reset,
                tau_adapt=config.ach_adaptation_tau,
                adapt_increment=config.ach_adaptation_increment,
                tau_ref=5.0,  # SHORT refractory (5ms) allows multiple spikes per burst
            ),
        )

        # GABAergic neurons (inhibit hippocampal interneurons)
        # Properties: Phase-locked to ACh but 180° offset, similar burst dynamics
        gaba_config = ConductanceLIFConfig(
            region_name=self.region_name,
            population_name=MedialSeptumPopulation.GABA,
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
        # IONIC PACEMAKER CURRENTS
        # =====================================================================
        # Theta (4-10 Hz) arises from ionic currents and reciprocal connectivity
        # rather than a hardcoded sinusoidal phase variable:
        #
        #   I_NaP : persistent Na⁺ depolarises ACh neurons tonically.
        #   ACh → GABA (excitatory): each ACh burst activates GABA neurons.
        #   GABA → ACh (inhibitory): GABA feedback hyperpolarises ACh neurons,
        #           terminating their burst (together with I_AHP).
        #   I_h / HCN : opens in quiescent GABA neurons → slow depolarising
        #           ramp that times the next GABA burst (≈180° vs ACh).
        #   I_AHP : slow K⁺ current accumulates per spike → burst termination.
        #
        # Frequency is modulated by NE (+amplitude) and ACh/DA (±frequency).

        # I_NaP : per-neuron tonic drive for ACh population
        self._i_nap_ach: torch.Tensor
        self.register_buffer(
            "_i_nap_ach",
            torch.full((self.ach_size,), config.i_nap_conductance, device=self.device),
        )
        # I_AHP : slow after-hyperpolarisation conductance (burst termination)
        self._i_ahp_ach: torch.Tensor
        self._i_ahp_gaba: torch.Tensor
        self.register_buffer("_i_ahp_ach",  torch.zeros(self.ach_size,  device=self.device))
        self.register_buffer("_i_ahp_gaba", torch.zeros(self.gaba_size, device=self.device))
        # I_h (HCN) : activation variable for GABA rebound (0 = closed, 1 = open)
        self._g_ih_gaba: torch.Tensor
        self.register_buffer("_g_ih_gaba",  torch.zeros(self.gaba_size, device=self.device))

        # =====================================================================
        # RECURRENT CONNECTIONS (for synchrony)
        # =====================================================================

        # Cholinergic neurons are weakly coupled (gap junctions + chemical)
        # Strong enough for synchrony but not drive amplification
        # Recurrence synchronizes firing during burst window
        self._add_internal_connection(
            source_population=MedialSeptumPopulation.ACH,
            target_population=MedialSeptumPopulation.ACH,
            weights=WeightInitializer.sparse_random_no_autapses(
                n_input=self.ach_size,
                n_output=self.ach_size,
                connectivity=1.0,  # Fully connected (sparse recurrence handled by small scale)
                weight_scale=0.03 / np.sqrt(self.ach_size),
                device=self.device,
            ),
            stp_config=None,
            receptor_type=ReceptorType.AMPA,
        )

        # GABAergic neurons have stronger coupling (fast synchronization)
        # Strong enough for tight synchrony
        self._add_internal_connection(
            source_population=MedialSeptumPopulation.GABA,
            target_population=MedialSeptumPopulation.GABA,
            weights=WeightInitializer.sparse_random_no_autapses(
                n_input=self.gaba_size,
                n_output=self.gaba_size,
                connectivity=1.0,  # Fully connected (sparse recurrence handled by small scale)
                weight_scale=0.04 / np.sqrt(self.gaba_size),
                device=self.device,
            ),
            stp_config=None,
            receptor_type=ReceptorType.AMPA,
        )

        # =====================================================================
        # INTER-POPULATION CONNECTIONS (oscillatory circuit)
        # =====================================================================

        # ACh → GABA (excitatory): ACh bursts drive GABA neurons
        # Each ACh spike provides brief excitatory drive to GABA population
        ach_to_gaba_weights = WeightInitializer.sparse_random(
            n_input=self.ach_size,
            n_output=self.gaba_size,
            connectivity=0.8,
            weight_scale=0.04 / np.sqrt(self.ach_size),
            device=self.device,
        )
        self._add_internal_connection(
            source_population=MedialSeptumPopulation.ACH,
            target_population=MedialSeptumPopulation.GABA,
            weights=ach_to_gaba_weights,
            stp_config=None,
            receptor_type=ReceptorType.AMPA,
        )

        # GABA → ACh (inhibitory): GABA feedback suppresses ACh burst
        # This is the KEY feedback loop that creates the oscillation
        gaba_to_ach_weights = WeightInitializer.sparse_random(
            n_input=self.gaba_size,
            n_output=self.ach_size,
            connectivity=0.8,
            weight_scale=0.06 / np.sqrt(self.gaba_size),
            device=self.device,
        )
        self._add_internal_connection(
            source_population=MedialSeptumPopulation.GABA,
            target_population=MedialSeptumPopulation.ACH,
            weights=gaba_to_ach_weights,
            stp_config=None,
            receptor_type=ReceptorType.GABA_A,
        )

        # Initialize state variables for spikes (for recurrent connections)
        # TODO: Use CircularBuffer for longer history if needed for more complex dynamics
        self._last_ach_spikes: torch.Tensor = torch.zeros(self.ach_size, dtype=torch.bool, device=self.device)
        self._last_gaba_spikes: torch.Tensor = torch.zeros(self.gaba_size, dtype=torch.bool, device=self.device)

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(MedialSeptumPopulation.ACH, self.ach_neurons, polarity=PopulationPolarity.ANY)
        self._register_neuron_population(MedialSeptumPopulation.GABA, self.gaba_neurons, polarity=PopulationPolarity.INHIBITORY)

        # =====================================================================
        # POST-INITIALIZATION
        # =====================================================================
        self.__post_init__()

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Generate theta rhythm through intrinsic ionic currents and reciprocal connectivity.

        Biological oscillatory mechanism:
        1. I_NaP (persistent Na⁺) tonically depolarises ACh neurons → burst.
        2. ACh → GABA excitation: ACh burst activates GABA neurons.
        3. GABA → ACh inhibition: GABA spikes suppress ACh → burst ends.
        4. ACh silence ⇒ I_h opens in GABA neurons (slow rebound), setting
           the ~180° phase offset (4–10 Hz theta period).
        5. I_AHP (slow K⁺) terminates both populations' bursts.

        Neuromodulators read from actual broadcast inputs (not hardcoded).
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        cfg = self.config
        dt_ms = cfg.dt_ms

        # =====================================================================
        # NEUROMODULATION
        # =====================================================================
        # Read spike-based neuromodulator broadcast; fallback to 0.5 (baseline)
        # when not connected to a source (DA/NE/ACh are all optional here).
        def _level(key: str) -> float:
            t = neuromodulator_inputs.get(key)
            return float(t.float().mean()) if t is not None else 0.5

        ne_level  = _level("norepinephrine")
        da_level  = _level("dopamine")
        ach_level = _level("acetylcholine")

        # I_NaP target amplitude scaled by neuromodulators:
        #   NE  → +40% (arousal increases burst drive)
        #   ACh → +30% (cholinergic auto-facilitation)
        #   DA  → ±20% (motivational modulation)
        i_nap_target = cfg.i_nap_conductance * (
            1.0
            + (ne_level  - 0.5) * 0.4
            + (ach_level - 0.5) * 0.3
            + (da_level  - 0.5) * 0.2
        )

        # =====================================================================
        # UPDATE IONIC CURRENTS
        # =====================================================================
        # I_NaP: slow tracking of modulated target (tau ~500ms)
        alpha_nap = dt_ms / 500.0
        self._i_nap_ach.add_(alpha_nap * (i_nap_target - self._i_nap_ach))

        # I_AHP: exponential decay + spike-driven accumulation (per-neuron, g_inh)
        decay_ahp = np.exp(-dt_ms / cfg.i_ahp_tau_ms)
        self._i_ahp_ach.mul_(decay_ahp).add_(self._last_ach_spikes.float()  * cfg.i_ahp_increment)
        self._i_ahp_gaba.mul_(decay_ahp).add_(self._last_gaba_spikes.float() * cfg.i_ahp_increment)

        # I_h (HCN): opens during GABA quiescence, closes when GABA fires (tau ~150ms)
        # Provides slow depolarising ramp → times GABA rebound burst at ~180°.
        alpha_ih = dt_ms / 150.0
        gaba_fired = self._last_gaba_spikes.float()
        ih_target = cfg.i_h_conductance * (1.0 - gaba_fired)  # Closes when spiking
        self._g_ih_gaba.add_(alpha_ih * (ih_target - self._g_ih_gaba))

        # =====================================================================
        # INTER-POPULATION CONDUCTANCES
        # =====================================================================
        # ACh → GABA (excitatory)
        ach_gaba_syn = SynapseId(
            source_region=self.region_name,
            source_population=MedialSeptumPopulation.ACH,
            target_region=self.region_name,
            target_population=MedialSeptumPopulation.GABA,
            receptor_type=ReceptorType.AMPA,
        )
        ach_to_gaba_exc = self.get_synaptic_weights(ach_gaba_syn) @ self._last_ach_spikes.float()

        # GABA → ACh (inhibitory)
        gaba_ach_syn = SynapseId(
            source_region=self.region_name,
            source_population=MedialSeptumPopulation.GABA,
            target_region=self.region_name,
            target_population=MedialSeptumPopulation.ACH,
            receptor_type=ReceptorType.GABA_A,
        )
        gaba_to_ach_inh = self.get_synaptic_weights(gaba_ach_syn) @ self._last_gaba_spikes.float()

        # Recurrent synchrony within each population
        ach_rec_syn = SynapseId(
            source_region=self.region_name,
            source_population=MedialSeptumPopulation.ACH,
            target_region=self.region_name,
            target_population=MedialSeptumPopulation.ACH,
            receptor_type=ReceptorType.AMPA,
        )
        gaba_rec_syn = SynapseId(
            source_region=self.region_name,
            source_population=MedialSeptumPopulation.GABA,
            target_region=self.region_name,
            target_population=MedialSeptumPopulation.GABA,
            receptor_type=ReceptorType.AMPA,
        )
        ach_recurrent_exc  = self.get_synaptic_weights(ach_rec_syn)  @ self._last_ach_spikes.float()
        gaba_recurrent_exc = self.get_synaptic_weights(gaba_rec_syn) @ self._last_gaba_spikes.float()

        # =====================================================================
        # HIPPOCAMPAL FEEDBACK EXCITATION (CA1 → Septum GABA)
        # =====================================================================
        # CA1 pyramidal cells (glutamatergic) project back to septal GABAergic neurons
        # via AMPA receptors.  This EXCITES the septal GABA neurons, which in turn
        # GABA-inhibit the ACh pacemakers (see GABA→ACh internal connection above).
        # Net effect of the closed loop:
        #   Septum drives hippocampus → hippocampus excites septal GABA → GABA
        #   suppresses septal ACh pacemakers → theta amplitude / frequency decreases.
        # This negative-feedback loop stabilises theta frequency against perturbations,
        # implementing the damped-oscillator behaviour described by Bland & Colom 1993.
        ca1_gaba_syn = SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1,
            target_region=self.region_name,
            target_population=MedialSeptumPopulation.GABA,
            receptor_type=ReceptorType.AMPA,
        )
        ca1_feedback = synaptic_inputs.get(ca1_gaba_syn, None)
        hippocampal_exc = torch.zeros(self.gaba_size, device=self.device)
        if ca1_feedback is not None:
            hippocampal_exc = self.get_synaptic_weights(ca1_gaba_syn) @ ca1_feedback.float()

        # =====================================================================
        # TOTAL CONDUCTANCES PER POPULATION
        # =====================================================================
        # ACh excitatory: I_NaP (tonic) + recurrent synchrony
        g_exc_ach = torch.clamp(self._i_nap_ach + ach_recurrent_exc, min=0.0)
        # ACh inhibitory: I_AHP (burst termination) + GABA feedback
        g_inh_ach = torch.clamp(self._i_ahp_ach + gaba_to_ach_inh, min=0.0)

        # GABA excitatory: I_h rebound + ACh→GABA drive + recurrent synchrony + hippocampal CA1 feedback (AMPA, excites septal GABA)
        g_exc_gaba = torch.clamp(self._g_ih_gaba + ach_to_gaba_exc + gaba_recurrent_exc + hippocampal_exc, min=0.0)
        # GABA inhibitory: I_AHP (burst termination) only
        g_inh_gaba = torch.clamp(self._i_ahp_gaba, min=0.0)

        # =====================================================================
        # RUN NEURONS
        # =====================================================================
        ach_g_ampa, ach_g_nmda = split_excitatory_conductance(g_exc_ach, nmda_ratio=0.3)
        gaba_g_ampa, gaba_g_nmda = split_excitatory_conductance(g_exc_gaba, nmda_ratio=0.3)

        ach_spikes, _ = self.ach_neurons.forward(
            g_ampa_input=ConductanceTensor(ach_g_ampa),
            g_gaba_a_input=ConductanceTensor(g_inh_ach),
            g_nmda_input=ConductanceTensor(ach_g_nmda),
        )
        gaba_spikes, _ = self.gaba_neurons.forward(
            g_ampa_input=ConductanceTensor(gaba_g_ampa),
            g_gaba_a_input=ConductanceTensor(g_inh_gaba),
            g_nmda_input=ConductanceTensor(gaba_g_nmda),
        )

        # =====================================================================
        # UPDATE STATE
        # =====================================================================
        self._last_ach_spikes  = ach_spikes
        self._last_gaba_spikes = gaba_spikes

        region_outputs: RegionOutput = {
            MedialSeptumPopulation.ACH:  ach_spikes,
            MedialSeptumPopulation.GABA: gaba_spikes,
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
