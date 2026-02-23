"""Ventral Tegmental Area (VTA) - Dopamine Reward Prediction Error System.

The VTA is the brain's primary source of dopamine, broadcasting reward prediction
error (RPE) signals that modulate learning across all brain regions. VTA dopamine
neurons exhibit characteristic burst/pause dynamics that encode positive/negative
prediction errors through firing rate modulation.

Biological Background:
======================
**Anatomy:**
- Location: Midbrain (ventral tegmental area)
- ~20,000-30,000 dopamine neurons in humans (~70% of VTA)
- ~5,000-10,000 GABAergic interneurons (~20% of VTA)
- ~10% glutamatergic neurons (not modeled in Phase 1)

**Dopamine Neuron Firing Patterns:**
1. **Tonic Pacemaking** (4-5 Hz baseline):
   - Intrinsic oscillation driven by I_h (HCN channels)
   - Represents background motivation/mood
   - Sets baseline learning rate

2. **Phasic Bursts** (15-20 Hz, 100-200 ms):
   - Unexpected reward (positive RPE)
   - Triggers long-term potentiation (LTP)
   - "Learn this!" signal

3. **Phasic Pauses** (<1 Hz, 100-200 ms):
   - Expected reward omitted (negative RPE)
   - Triggers long-term depression (LTD)
   - "Unlearn this!" signal

**Computational Role:**
=======================
VTA implements **temporal difference (TD) learning** by computing:

    RPE (δ) = r + γ·V(s') - V(s)

Where:
- r: Current reward (from reward encoder)
- V(s): Current state value (from SNr firing rate)
- V(s'): Next state value (simplified: not used in Phase 1)
- γ: Discount factor (0.99)

The basal ganglia loop creates a closed-loop TD system:
```
Striatum (learns Q-values) → SNr (encodes V) → VTA (computes RPE) → Striatum (updates)
```

**Inputs:**
- Reward signal (from RewardEncoder region)
- Value estimate (from SNr via firing rate)
- Contextual modulation (from PFC, amygdala - future)

**Outputs:**
- Mesocortical pathway: DA → Prefrontal cortex (executive function)
- Mesolimbic pathway: DA → Striatum, hippocampus, amygdala (motivation, memory)
- Nigrostriatal pathway: DA → Dorsal striatum (motor learning)

**Implementation Notes:**
=========================
Current State:
- Simplified RPE: δ ≈ r - V(s) (no next-state value)
- SNr provides V(s) estimate via firing rate
- Two DA sub-populations: mesolimbic (reward/motivation) + mesocortical (PFC arousal)
- Mesolimbic has D2 somatodendritic autoreceptors; mesocortical does not
- LHb → RMTg → VTA inhibitory pause pathway implemented

Planned Enhancements (Phase 2):
- Full TD: Include V(s') from eligibility traces
- PFC → VTA contextual modulation
"""

from __future__ import annotations

from typing import ClassVar, Dict, Optional

import torch

from thalia.brain.configs import VTAConfig
from thalia.components import (
    ConductanceLIFConfig,
    ConductanceLIF,
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
    SynapseId,
    SynapticInput,
)
from thalia.utils import split_excitatory_conductance

from .neural_region import NeuralRegion
from .population_names import VTAPopulation
from .region_registry import register_region


@register_region(
    "vta",
    aliases=["ventral_tegmental_area", "dopamine_system"],
    description="Ventral tegmental area - dopamine reward prediction error system",
    version="1.0",
    author="Thalia Project",
    config_class=VTAConfig,
)
class VTA(NeuralRegion[VTAConfig]):
    """Ventral Tegmental Area - Dopamine Reward Prediction Error System.

    Computes reward prediction errors (RPE) and broadcasts dopamine signals
    via burst/pause dynamics. Integrates reward signals and value estimates
    to drive reinforcement learning across the brain.
    """

    # Declarative neuromodulator output registry.
    # Two separate DA channels reflecting the two major VTA projection systems
    # (Lammel et al. 2008):  mesolimbic (reward/motivation) and mesocortical (PFC
    # executive function/arousal).  These were previously both pointing to the same
    # 'da' population — now each maps to a distinct sub-population with different
    # biophysical properties and autoreceptor feedback.
    neuromodulator_outputs: ClassVar[Dict[NeuromodulatorType, PopulationName]] = {
        'da_mesolimbic': VTAPopulation.DA_MESOLIMBIC,
        'da_mesocortical': VTAPopulation.DA_MESOCORTICAL,
    }

    def __init__(self, config: VTAConfig, population_sizes: PopulationSizes, region_name: RegionName):
        super().__init__(config, population_sizes, region_name)

        self.da_mesolimbic_size = population_sizes[VTAPopulation.DA_MESOLIMBIC]
        self.da_mesocortical_size = population_sizes[VTAPopulation.DA_MESOCORTICAL]
        self.gaba_neurons_size = population_sizes[VTAPopulation.GABA]

        # -----------------------------------------------------------------------
        # MESOLIMBIC DA NEURONS (~55% of VTA DA)
        # Project to ventral striatum, hippocampus, amygdala.
        # Have D2 somatodendritic autoreceptors (Beckstead et al. 2004).
        # Tonic 4-6 Hz, burst to 15-20 Hz for positive RPE.
        # -----------------------------------------------------------------------
        # I_h (HCN) contributes to the 4-6 Hz tonic rhythm and rebound after
        # inhibitory pauses by RMTg (negative RPE pathway).
        self.da_mesolimbic_neurons = ConductanceLIF(
            n_neurons=self.da_mesolimbic_size,
            config=ConductanceLIFConfig(
                region_name=region_name,
                population_name=VTAPopulation.DA_MESOLIMBIC,
                device=self.device,
                tau_mem=config.tau_mem,
                v_threshold=1.0,
                v_reset=0.0,
                v_rest=0.0,
                tau_ref=config.tau_ref,
                g_L=config.g_L,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                noise_std=config.noise_std,
                adapt_increment=config.adapt_increment,
                tau_adapt=config.tau_adapt,
                E_adapt=-0.5,
                # I_h (HCN) pacemaker — voltage-dependent rebound after RMTg-driven pauses
                enable_ih=True,
                g_h_max=0.03,
                E_h=-0.3,
                V_half_h=-0.35,
                k_h=0.08,
                tau_h_ms=150.0,  # Slow activation (~150ms, Neuhoff et al. 2002)
            ),
        )

        # -----------------------------------------------------------------------
        # MESOCORTICAL DA NEURONS (~35% of VTA DA)
        # Project to PFC (DLPFC, OFC, ACC).
        # Lack somatodendritic D2 autoreceptors (Lammel et al. 2008) → higher
        # baseline firing (~7-9 Hz) and unattenuated burst responses.
        # Respond more to aversive/novel stimuli than to pure reward.
        # -----------------------------------------------------------------------
        self.da_mesocortical_neurons = ConductanceLIF(
            n_neurons=self.da_mesocortical_size,
            config=ConductanceLIFConfig(
                region_name=region_name,
                population_name=VTAPopulation.DA_MESOCORTICAL,
                device=self.device,
                tau_mem=config.tau_mem,
                v_threshold=1.0,
                v_reset=0.0,
                v_rest=0.0,
                tau_ref=config.tau_ref,
                g_L=config.g_L,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                noise_std=config.noise_std,
                adapt_increment=config.mesocortical_adapt_increment,  # Faster adaptation
                tau_adapt=config.tau_adapt,
                E_adapt=-0.5,
                # I_h pacemaker — same kinetics as mesolimbic
                enable_ih=True,
                g_h_max=0.03,
                E_h=-0.3,
                V_half_h=-0.35,
                k_h=0.08,
                tau_h_ms=150.0,
            ),
        )

        # GABAergic interneurons (local inhibition, homeostasis)
        self.gaba_neurons = NeuronFactory.create(
            region_name=self.region_name,
            population_name=VTAPopulation.GABA,
            neuron_type=NeuronType.FAST_SPIKING,
            n_neurons=self.gaba_neurons_size,
            device=self.device,
        )

        # D2 somatodendritic autoreceptors — MESOLIMBIC ONLY
        # Exponential moving average of per-neuron DA firing rate (one-step lag).
        # Mesocortical neurons lack this feedback (Lammel et al. 2008).
        self.register_buffer(
            "_prev_mesolimbic_spike_rate",
            torch.zeros(self.da_mesolimbic_size, device=self.device),
        )

        # Adaptive normalization state
        if config.rpe_normalization:
            self._avg_abs_rpe = 0.5  # Running average of |RPE|
            self._rpe_count = 0

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(VTAPopulation.DA_MESOLIMBIC, self.da_mesolimbic_neurons, polarity=PopulationPolarity.ANY)
        self._register_neuron_population(VTAPopulation.DA_MESOCORTICAL, self.da_mesocortical_neurons, polarity=PopulationPolarity.ANY)
        self._register_neuron_population(VTAPopulation.GABA, self.gaba_neurons, polarity=PopulationPolarity.INHIBITORY)

        self.__post_init__()

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Compute RPE and drive both DA sub-populations to burst/pause.

        Mesolimbic and mesocortical neurons receive the same RPE-driven drive but differ
        in autoreceptor feedback and baseline rate.  Both are inhibited by RMTg (negative
        RPE pathway).  Only mesolimbic neurons have D2 autoreceptor state.

        Args:
            synaptic_inputs: Reward signal (injected by DynamicBrain.deliver_reward()), RMTg GABA
            neuromodulator_inputs: Not consumed (VTA is a neuromodulator source, not consumer)
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        # =====================================================================
        # DECODE REWARD AND COMPUTE RPE
        # =====================================================================
        reward_synapse = SynapseId.external_reward_to_vta_da(self.region_name)
        reward_spikes = synaptic_inputs.get(reward_synapse, None)
        reward = self._decode_reward(reward_spikes)

        # Positive RPE path: reward — negative RPE handled by RMTg→VTA GABA pause.
        rpe = reward

        if self.config.rpe_normalization:
            self._rpe_count += 1
            alpha = 1.0 / min(self._rpe_count, 100)
            self._avg_abs_rpe = (1 - alpha) * self._avg_abs_rpe + alpha * abs(rpe)
            epsilon = 1e-6
            rpe = rpe / (self._avg_abs_rpe + epsilon)

        rpe_clipped = max(-1.0, min(1.0, rpe))

        # =====================================================================
        # MESOLIMBIC DA NEURONS (reward/motivation channel)
        # Baseline 4-6 Hz tonic; burst on positive RPE; D2 autoreceptor feedback.
        # =====================================================================
        ml_baseline = torch.full((self.da_mesolimbic_size,), self.config.baseline_drive, device=self.device)

        if rpe_clipped > 0:
            rpe_exc = rpe_clipped * self.config.rpe_gain
            rpe_jitter = torch.clamp(
                1.0 + torch.randn(self.da_mesolimbic_size, device=self.device) * 0.2,
                0.5, 1.5,
            )
            ml_baseline = ml_baseline + rpe_exc * rpe_jitter

        # D2 somatodendritic autoreceptors — mesolimbic only
        if self.config.d2_autoreceptor_gain > 0.0:
            autoreceptor_suppression = self.config.d2_autoreceptor_gain * self._prev_mesolimbic_spike_rate
            ml_baseline = (ml_baseline * (1.0 - autoreceptor_suppression)).clamp(min=0.0)

        ml_g_ampa, ml_g_nmda = split_excitatory_conductance(ml_baseline, nmda_ratio=0.30)

        # RMTg inhibition targeting mesolimbic neurons specifically
        rmtg_ml = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.da_mesolimbic_size,
            filter_by_source_region="rostromedial_tegmentum",
            filter_by_target_population=VTAPopulation.DA_MESOLIMBIC,
        )

        ml_spikes, _ = self.da_mesolimbic_neurons.forward(
            g_ampa_input=ConductanceTensor(ml_g_ampa),
            g_gaba_a_input=ConductanceTensor(rmtg_ml.g_inh),
            g_nmda_input=ConductanceTensor(ml_g_nmda),
        )

        # Update D2 autoreceptor EMA (mesolimbic only; ~10ms tau at 1ms dt)
        self._prev_mesolimbic_spike_rate.mul_(0.9).add_(ml_spikes.float() * 0.1)

        # =====================================================================
        # MESOCORTICAL DA NEURONS (executive/arousal channel)
        # Baseline 7-9 Hz tonic; NO D2 autoreceptors; same RMTg pause pathway.
        # Receive same RPE burst drive (salience/uncertainty signal to PFC).
        # =====================================================================
        mc_baseline = torch.full(
            (self.da_mesocortical_size,), self.config.mesocortical_baseline_drive, device=self.device
        )

        if rpe_clipped > 0:
            rpe_exc = rpe_clipped * self.config.rpe_gain
            rpe_jitter = torch.clamp(
                1.0 + torch.randn(self.da_mesocortical_size, device=self.device) * 0.2,
                0.5, 1.5,
            )
            mc_baseline = mc_baseline + rpe_exc * rpe_jitter
        # No D2 autoreceptor suppression for mesocortical neurons

        mc_g_ampa, mc_g_nmda = split_excitatory_conductance(mc_baseline, nmda_ratio=0.30)

        # RMTg inhibition targeting mesocortical neurons specifically
        rmtg_mc = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.da_mesocortical_size,
            filter_by_source_region="rostromedial_tegmentum",
            filter_by_target_population=VTAPopulation.DA_MESOCORTICAL,
        )

        mc_spikes, _ = self.da_mesocortical_neurons.forward(
            g_ampa_input=ConductanceTensor(mc_g_ampa),
            g_gaba_a_input=ConductanceTensor(rmtg_mc.g_inh),
            g_nmda_input=ConductanceTensor(mc_g_nmda),
        )

        # =====================================================================
        # GABA INTERNEURONS (homeostatic control of both DA sub-populations)
        # =====================================================================
        baseline_gaba_drive = 0.5
        # Total DA activity drives GABA interneurons
        da_activity = (ml_spikes.float().mean() + mc_spikes.float().mean()).item() * 0.5
        gaba_drive = torch.full(
            (self.gaba_neurons_size,), baseline_gaba_drive + da_activity, device=self.device
        )
        gaba_g_ampa, gaba_g_nmda = split_excitatory_conductance(gaba_drive, nmda_ratio=0.3)
        gaba_spikes, _gaba_membrane = self.gaba_neurons.forward(
            g_ampa_input=ConductanceTensor(gaba_g_ampa),
            g_gaba_a_input=None,
            g_nmda_input=ConductanceTensor(gaba_g_nmda),
        )

        region_outputs: RegionOutput = {
            VTAPopulation.DA_MESOLIMBIC: ml_spikes,
            VTAPopulation.DA_MESOCORTICAL: mc_spikes,
            VTAPopulation.GABA: gaba_spikes,
        }

        return self._post_forward(region_outputs)

    def _decode_reward(self, reward_spikes: Optional[torch.Tensor]) -> float:
        """Decode reward value from population spike pattern.

        RewardEncoder uses population coding:
        - First N/2 neurons: Positive rewards
        - Last N/2 neurons: Negative rewards (punishments)

        Args:
            reward_spikes: Spike tensor [n_reward_neurons]

        Returns:
            Scalar reward in range [-1, +1]
        """
        if reward_spikes is None or reward_spikes.sum() == 0:
            return 0.0

        n_total = reward_spikes.shape[0]
        n_half = n_total // 2

        # Positive reward neurons (first half)
        positive_rate = reward_spikes[:n_half].float().mean().item()

        # Negative reward neurons (second half)
        negative_rate = reward_spikes[n_half:].float().mean().item()

        # Combine: positive - negative
        reward = positive_rate - negative_rate

        # Clamp to valid range
        reward = max(-1.0, min(1.0, reward))

        return reward

    def _decode_value(self, snr_spikes: Optional[torch.Tensor]) -> float:
        """Decode state value from SNr firing rate.

        SNr encodes value inversely (biological realism):
        - High SNr firing rate → low state value (action suppression)
        - Low SNr firing rate → high state value (action release)

        Args:
            snr_spikes: SNr neuron spikes [n_snr_neurons]

        Returns:
            State value estimate in range [0, 1]
        """
        if snr_spikes is None:
            return 0.0  # Neutral default

        # Convert spikes to firing rate
        snr_spike_rate = snr_spikes.float().mean().item()

        # SNr baseline: ~0.06 (60 Hz / 1000 ms)
        # High SNr rate → low value
        # Low SNr rate → high value
        baseline_rate = 0.06
        value = 1.0 - (snr_spike_rate / (2 * baseline_rate))

        # Clamp to [0, 1]
        value = max(0.0, min(1.0, value))

        return value
