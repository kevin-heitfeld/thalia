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
Phase 1 (Current):
- Simplified RPE: δ ≈ r - V(s) (no next-state value)
- SNr provides V(s) estimate via firing rate
- Single DA output (not separated by pathway yet)

Phase 2 (Future):
- Full TD: Include V(s') from eligibility traces
- Separate DA projections (mesocortical, mesolimbic, nigrostriatal)
- Lateral habenula → RMTg → VTA (aversive signals)
- PFC → VTA (contextual modulation)
"""

from __future__ import annotations

from typing import ClassVar, Dict, Optional

import torch

from thalia.brain.configs import VTAConfig
from thalia.brain.regions.population_names import RewardEncoderPopulation, SubstantiaNigraPopulation, VTAPopulation
from thalia.components.neurons import TonicDopamineNeuron, NeuronFactory, NeuronType
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

    Input Populations:
    ------------------
    - "reward_signal": Reward spikes from RewardEncoder
    - "snr_value": Value estimate from SNr (inhibitory spikes)

    Output Populations:
    -------------------
    - "da_neurons": Dopamine neuron spikes (broadcast to all targets)

    Future:
    - "da_mesocortical": DA → Prefrontal cortex
    - "da_mesolimbic": DA → Striatum, hippocampus, amygdala
    - "da_nigrostriatal": DA → Dorsal striatum

    Computational Function:
    -----------------------
    1. Decode reward from spike pattern (population coding)
    2. Decode value from SNr firing rate (inverse encoding)
    3. Compute RPE: δ = r - V(s)
    4. Drive DA neurons: positive RPE → burst, negative RPE → pause
    5. Broadcast DA spikes to target regions
    """

    # Declarative neuromodulator output registry.
    # DynamicBrain reads this ClassVar to build NeuromodulatorTract diffusion filters
    # and route outputs without any hardcoded region-name checks in brain.py.
    neuromodulator_outputs: ClassVar[Dict[str, str]] = {'da': 'da'}

    def __init__(self, config: VTAConfig, population_sizes: PopulationSizes, region_name: RegionName):
        super().__init__(config, population_sizes, region_name)

        self.da_neurons_size = population_sizes[VTAPopulation.DA.value]
        self.gaba_neurons_size = population_sizes[VTAPopulation.GABA.value]

        # Dopamine neurons (pacemakers with burst/pause)
        self.da_neurons = TonicDopamineNeuron(n_neurons=self.da_neurons_size, device=self.device)

        # GABAergic interneurons (local inhibition, homeostasis)
        self.gaba_neurons = NeuronFactory.create(
            region_name=self.region_name,
            population_name=VTAPopulation.GABA.value,
            neuron_type=NeuronType.FAST_SPIKING,
            n_neurons=self.gaba_neurons_size,
            device=self.device,
        )

        # Adaptive normalization state
        if config.rpe_normalization:
            self._avg_abs_rpe = 0.5  # Running average of |RPE|
            self._rpe_count = 0

        # GABA → DA inhibition state (delayed by 1 timestep for causality)
        self._gaba_to_da_inhibition = 0.0

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(VTAPopulation.DA.value, self.da_neurons)
        self._register_neuron_population(VTAPopulation.GABA.value, self.gaba_neurons)

        self.__post_init__()

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Compute RPE and drive dopamine neurons to burst/pause.

        Args:
            synaptic_inputs: Reward signal from RewardEncoder, value from SNr
            neuromodulator_inputs: Not used (source region, doesn't consume neuromodulators)
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        reward_synapse = SynapseId(
            source_region="reward_encoder",
            source_population=RewardEncoderPopulation.REWARD_SIGNAL.value,
            target_region=self.region_name,
            target_population=VTAPopulation.DA.value,
        )

        snr_synapse = SynapseId(
            source_region="substantia_nigra",
            source_population=SubstantiaNigraPopulation.VTA_FEEDBACK.value,
            target_region=self.region_name,
            target_population=VTAPopulation.DA.value,
        )

        reward_spikes = synaptic_inputs.get(reward_synapse, None)
        snr_spikes = synaptic_inputs.get(snr_synapse, None)

        # Decode reward from spike pattern
        reward = self._decode_reward(reward_spikes)

        # Decode value from SNr firing rate
        value = self._decode_value(snr_spikes)

        # Compute RPE: δ = r - V(s)
        # Simplified (Phase 1): No next-state value V(s')
        # Full TD (Phase 2): δ = r + γ·V(s') - V(s)
        rpe = reward - value

        # Normalize RPE to prevent saturation
        if self.config.rpe_normalization:
            # Update running average of |RPE|
            self._rpe_count += 1
            alpha = 1.0 / min(self._rpe_count, 100)  # Converge over 100 samples
            self._avg_abs_rpe = (1 - alpha) * self._avg_abs_rpe + alpha * abs(rpe)

            # Normalize by average magnitude
            epsilon = 1e-6  # Prevent division by zero
            normalized_rpe = rpe / (self._avg_abs_rpe + epsilon)

            rpe = normalized_rpe

        # Moderate excitatory noise for stochastic pacemaker firing
        # This noise just adds variability, not the main drive.
        noise_conductance = torch.randn(self.da_neurons_size, device=self.device) * 0.0025
        noise_conductance = torch.clamp(noise_conductance, min=0.0)

        # Tonic inhibition from SNr (can now use it with voltage-dependent HCN!)
        # With voltage-dependent HCN channels, tonic inhibition is SAFE.
        # When inhibition hyperpolarizes DA neurons → HCN activates MORE → compensates!
        # This creates natural negative feedback for homeostasis.
        #
        # SNr encodes action suppression via tonic firing:
        # - Low SNr activity → low inhibition → DA bursts easily
        # - High SNr activity → moderate inhibition → DA stabilizes
        #
        # Biology: SNr → VTA GABAergic projection provides tonic inhibition
        # that's modulated by striatal activity (direct/indirect pathways)

        snr_firing_rate = 0.0
        if snr_spikes is not None and snr_spikes.sum() > 0:
            snr_firing_rate = snr_spikes.float().mean().item()

        # Map SNr rate (0-1) to light tonic inhibition
        inhibition_strength = snr_firing_rate * 0.15
        inhibition_strength = min(inhibition_strength, 0.02)  # Cap max inhibition to prevent complete silencing

        baseline_inhibition = torch.full(
            (self.da_neurons_size,),
            inhibition_strength + self._gaba_to_da_inhibition,  # Add GABA inhibition from previous timestep
            device=self.device
        )

        # Add small heterogeneity to prevent synchronization
        inhibition_jitter = 1.0 + torch.randn(self.da_neurons_size, device=self.device) * 0.2
        inhibition_jitter = torch.clamp(inhibition_jitter, min=0.8, max=1.2)
        baseline_inhibition = baseline_inhibition * inhibition_jitter

        # Moderate RPE jitter per-neuron (±20% heterogeneity)
        # Clip RPE to prevent complete silencing during negative RPE
        # Biology: DA neurons maintain 4-5 Hz tonic baseline even with mild negative RPE
        rpe_clipped = max(-1.0, min(1.0, rpe))  # Prevent mild negative RPE from silencing

        if rpe_clipped != 0:
            rpe_jitter = 1.0 + torch.randn(self.da_neurons_size, device=self.device) * 0.2
            rpe_jitter = torch.clamp(rpe_jitter, min=0.5, max=1.5)
            # Convert scalar RPE to per-neuron tensor with jitter
            rpe_per_neuron = torch.full((self.da_neurons_size,), rpe_clipped, device=self.device) * rpe_jitter
        else:
            rpe_per_neuron = torch.zeros(self.da_neurons_size, device=self.device)

        # Update DA neurons with balanced E/I
        # Positive RPE → reduces inhibition → pacemaker increases → burst
        # Negative RPE → increases inhibition → pacemaker decreases → pause
        da_spikes, _ = self.da_neurons.forward(
            g_exc_input=ConductanceTensor(noise_conductance),  # Weak noise for variability
            g_inh_input=ConductanceTensor(baseline_inhibition),  # Tonic inhibition
            rpe_drive=rpe_per_neuron,  # Per-neuron jittered RPE modulation
        )

        # Update GABA interneurons (homeostatic control)
        # Provide tonic inhibition to prevent runaway bursting
        baseline_gaba_drive = 0.5  # Tonic baseline conductance
        # Increase during DA bursts (negative feedback)
        da_activity = da_spikes.float().mean().item()
        gaba_drive = torch.full((self.gaba_neurons_size,), baseline_gaba_drive + da_activity, device=self.device)

        # Split excitatory conductance: 70% AMPA (fast), 30% NMDA (slow)
        gaba_g_ampa = ConductanceTensor(gaba_drive * 0.7)
        gaba_g_nmda = ConductanceTensor(gaba_drive * 0.3)

        gaba_spikes, _gaba_membrane = self.gaba_neurons.forward(
            g_ampa_input=gaba_g_ampa,
            g_gaba_a_input=None,
            g_nmda_input=gaba_g_nmda,
        )

        # GABA → DA inhibition (tonic control)
        # VTA GABA interneurons provide tonic inhibition to DA neurons
        # Prevents runaway bursting and maintains homeostasis
        if gaba_spikes.any():
            # GABA spikes → inhibitory conductance for DA neurons
            gaba_to_da_strength = 0.0015
            gaba_inh_contrib = gaba_spikes.float().sum() * gaba_to_da_strength / self.da_neurons_size
            # Store for next timestep (causal: t GABA activity → t+1 DA suppression)
            self._gaba_to_da_inhibition = float(gaba_inh_contrib)
        else:
            self._gaba_to_da_inhibition = 0.0

        region_outputs: RegionOutput = {
            VTAPopulation.DA.value: da_spikes,
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
