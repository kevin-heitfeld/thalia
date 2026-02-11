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

from typing import Dict, Optional

import torch

from thalia.brain.configs import VTAConfig
from thalia.components.neurons.dopamine_neuron import (
    DopamineNeuron,
    DopamineNeuronConfig,
)
from thalia.components.neurons.neuron_factory import NeuronFactory, NeuronType
from thalia.typing import PopulationName, PopulationSizes, RegionSpikesDict
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
    - "da_output": Dopamine neuron spikes (broadcast to all targets)

    Future (Phase 2):
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

    OUTPUT_POPULATIONS: Dict[PopulationName, str] = {
        "da_output": "n_da_neurons",
    }

    def __init__(self, config: VTAConfig, population_sizes: PopulationSizes):
        super().__init__(config, population_sizes)

        # Store sizes for test compatibility
        self.n_da_neurons = config.n_da_neurons
        self.n_gaba_neurons = config.n_gaba_neurons

        # Store input layer sizes as attributes for connection routing
        self.snr_value_size = population_sizes.get("snr_value", 0)
        self.reward_signal_size = population_sizes.get("reward_signal", 0)

        # Dopamine neurons (pacemakers with burst/pause)
        self.da_neurons = self._create_da_neurons()

        # GABAergic interneurons (local inhibition, homeostasis)
        self.gaba_neurons = NeuronFactory.create(
            NeuronType.FAST_SPIKING,
            n_neurons=config.n_gaba_neurons,
            device=self.device,
        )

        # RPE computation state
        self._reward_history: list[float] = []
        self._value_history: list[float] = []
        self._rpe_history: list[float] = []

        # Adaptive normalization state
        if config.rpe_normalization:
            self._avg_abs_rpe = 0.5  # Running average of |RPE|
            self._rpe_count = 0

        self.__post_init__()

    def _create_da_neurons(self) -> DopamineNeuron:
        """Create dopamine neuron population with burst/pause dynamics."""
        if self.config.da_neuron_config is not None:
            da_config = self.config.da_neuron_config
        else:
            # Use default configuration
            da_config = DopamineNeuronConfig(
                device=self.config.device,
                rpe_to_current_gain=self.config.rpe_gain,
            )

        return DopamineNeuron(
            n_neurons=self.config.n_da_neurons, config=da_config, device=self.device
        )

    def forward(self, region_inputs: RegionSpikesDict) -> RegionSpikesDict:
        """Compute RPE and drive dopamine neurons to burst/pause.

        Args:
            region_inputs: Dictionary of input spike tensors:
                - "reward_signal": Reward encoding from RewardEncoder [n_reward_neurons]
                - "snr_value": Value estimate from SNr [n_snr_neurons]
        """
        self._pre_forward(region_inputs)

        # Get inputs (via connections from BrainBuilder)
        reward_spikes = region_inputs.get("reward_signal")
        snr_spikes = region_inputs.get("snr_value")

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
            rpe = self._normalize_rpe(rpe)

        # Track history for analysis
        self._reward_history.append(reward)
        self._value_history.append(value)
        self._rpe_history.append(rpe)
        if len(self._rpe_history) > 1000:
            self._reward_history.pop(0)
            self._value_history.pop(0)
            self._rpe_history.pop(0)

        # Update DA neurons with RPE drive
        # Positive RPE → depolarization → burst
        # Negative RPE → hyperpolarization → pause
        da_spikes, _ = self.da_neurons.forward(
            g_exc_input=ConductanceTensor(torch.zeros(self.n_da_neurons, device=self.device)),
            g_inh_input=ConductanceTensor(torch.zeros(self.n_da_neurons, device=self.device)),
            rpe_drive=rpe,
        )
        self._current_da_spikes = da_spikes  # Store for GABA computation

        # Update GABA interneurons (homeostatic control)
        # Provide tonic inhibition to prevent runaway bursting
        gaba_drive = self._compute_gaba_drive()
        # Convert drive to excitatory conductance (scale down from current-like values)
        g_gaba_exc = ConductanceTensor(gaba_drive / 10.0)  # Already a tensor from _compute_gaba_drive
        self.gaba_neurons.forward(g_gaba_exc, None)

        region_outputs: RegionSpikesDict = {
            "da_output": da_spikes,
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
            # No value estimate → use reward history as proxy
            if len(self._reward_history) > 0:
                return max(0.0, sum(self._reward_history[-10:]) / 10.0)
            return 0.5  # Neutral default

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

    def _normalize_rpe(self, rpe: float) -> float:
        """Adaptive RPE normalization to prevent saturation.

        Tracks running average of |RPE| and normalizes current RPE
        to maintain stable learning dynamics.

        Args:
            rpe: Raw RPE value

        Returns:
            Normalized RPE in approximate range [-2, +2]
        """
        # Update running average of |RPE|
        abs_rpe = abs(rpe)
        self._rpe_count += 1
        alpha = 1.0 / min(self._rpe_count, 100)  # Converge over 100 samples
        self._avg_abs_rpe = (1 - alpha) * self._avg_abs_rpe + alpha * abs_rpe

        # Normalize by average magnitude
        epsilon = 0.1  # Prevent division by zero
        normalized_rpe = rpe / (self._avg_abs_rpe + epsilon)

        # Clip to reasonable range
        normalized_rpe = max(-2.0, min(2.0, normalized_rpe))

        return normalized_rpe

    def _compute_gaba_drive(self) -> torch.Tensor:
        """Compute drive for GABAergic interneurons.

        GABA interneurons provide homeostatic control, preventing
        runaway DA bursting through local inhibition.

        Returns:
            Drive current for GABA neurons [n_gaba_neurons]
        """
        # Tonic baseline drive
        baseline = 5.0

        # Increase during DA bursts (negative feedback)
        if hasattr(self, '_current_da_spikes'):
            da_activity = self._current_da_spikes.float().mean().item()
        else:
            da_activity = 0.05  # Default tonic rate
        feedback = da_activity * 10.0  # Proportional to DA activity

        total_drive = baseline + feedback

        return torch.full(
            (self.config.n_gaba_neurons,), total_drive, device=self.device
        )

    def get_mean_rpe(self, window: int = 100) -> float:
        """Get mean RPE over recent history.

        Args:
            window: Number of timesteps to average over

        Returns:
            Mean RPE, or 0.0 if no history
        """
        if not self._rpe_history:
            return 0.0
        recent = self._rpe_history[-window:]
        return sum(recent) / len(recent)

    def get_da_firing_rate_hz(self) -> float:
        """Get current DA neuron population firing rate in Hz.

        Returns:
            Firing rate in Hz
        """
        return self.da_neurons.get_firing_rate_hz()
