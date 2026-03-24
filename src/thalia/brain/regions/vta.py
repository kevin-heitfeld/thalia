"""Ventral Tegmental Area (VTA) - Dopamine Reward Prediction Error System.

The VTA is the brain's primary source of dopamine, broadcasting reward prediction
error (RPE) signals that modulate learning across all brain regions. VTA dopamine
neurons exhibit characteristic burst/pause dynamics that encode positive/negative
prediction errors through firing rate modulation.
"""

from __future__ import annotations

from typing import ClassVar, Dict, List, Optional, Union

import torch
from torch import nn

from thalia import GlobalConfig
from thalia.brain.adaptive_normalization import AdaptiveNormalization
from thalia.brain.configs import VTAConfig
from thalia.brain.synapses import WeightInitializer
from thalia.brain.neurons import (
    ConductanceLIF,
    ConductanceLIFConfig,
    heterogeneous_adapt_increment,
    heterogeneous_dendrite_coupling,
    heterogeneous_g_L,
    heterogeneous_noise_std,
    heterogeneous_tau_adapt,
    heterogeneous_tau_mem,
    heterogeneous_v_threshold,
    split_excitatory_conductance,
)
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorInput,
    NeuromodulatorChannel,
    PopulationName,
    PopulationPolarity,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.utils import CircularDelayBuffer

from .neuromodulator_source_region import NeuromodulatorSourceRegion
from .population_names import VTAPopulation
from .region_registry import register_region


@register_region(
    "vta",
    aliases=["ventral_tegmental_area", "dopamine_system"],
    description="Ventral tegmental area - dopamine reward prediction error system",
)
class VTA(NeuromodulatorSourceRegion[VTAConfig]):
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
    neuromodulator_outputs: ClassVar[Dict[NeuromodulatorChannel, PopulationName]] = {
        NeuromodulatorChannel.DA_MESOLIMBIC:   VTAPopulation.DA_MESOLIMBIC,
        NeuromodulatorChannel.DA_MESOCORTICAL: VTAPopulation.DA_MESOCORTICAL,
    }

    # 5-HT1A receptors on VTA DA neurons: DRN serotonin attenuates phasic burst
    # amplitude of mesolimbic DA neurons (inhibitory Gi-coupled hyperpolarisation).
    # Mesocortical neurons are less sensitive (Gervais & Bhagya 2007).
    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = [NeuromodulatorChannel.SHT]

    def __init__(
        self,
        config: VTAConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        super().__init__(config, population_sizes, region_name, device=device)

        self.da_mesolimbic_size = population_sizes[VTAPopulation.DA_MESOLIMBIC]
        self.da_mesocortical_size = population_sizes[VTAPopulation.DA_MESOCORTICAL]
        self.gaba_size = population_sizes[VTAPopulation.GABA]

        # -----------------------------------------------------------------------
        # MESOLIMBIC DA NEURONS (~55% of VTA DA)
        # Project to ventral striatum, hippocampus, amygdala.
        # Have D2 somatodendritic autoreceptors (Beckstead et al. 2004).
        # Tonic 4-6 Hz, burst to 15-20 Hz for positive RPE.
        # -----------------------------------------------------------------------
        self.da_mesolimbic_neurons: ConductanceLIF
        self.da_mesolimbic_neurons = self._create_and_register_neuron_population(
            VTAPopulation.DA_MESOLIMBIC,
            self.da_mesolimbic_size,
            polarity=PopulationPolarity.ANY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(config.tau_mem_ms, self.da_mesolimbic_size, self.device, cv=0.20),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(1.0, self.da_mesolimbic_size, self.device, cv=0.25, clamp_fraction=0.25),
                tau_ref=config.tau_ref,
                g_L=heterogeneous_g_L(config.g_L, self.da_mesolimbic_size, self.device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(config.noise_std, self.da_mesolimbic_size, self.device),
                noise_tau_ms=3.0,
                tau_adapt_ms=heterogeneous_tau_adapt(config.tau_adapt_ms, self.da_mesolimbic_size, self.device, cv=0.40),
                adapt_increment=heterogeneous_adapt_increment(config.adapt_increment, self.da_mesolimbic_size, self.device),
                E_adapt=-0.5,
                # I_h (HCN) pacemaker — see class-level constants for rationale.
                # g_h_max reduced 0.03→0.015→0.008: prevents I_h rebound-driven epileptiform bursting.
                enable_ih=True,
                g_h_max=0.008,
                E_h=0.9,
                V_half_h=-0.35,
                k_h=0.08,
                tau_h_ms=150.0,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.da_mesolimbic_size, self.device, cv=0.25),
            ),
        )

        # -----------------------------------------------------------------------
        # MESOCORTICAL DA NEURONS (~35% of VTA DA)
        # Project to PFC (DLPFC, OFC, ACC).
        # Lack somatodendritic D2 autoreceptors (Lammel et al. 2008) → higher
        # baseline firing (~7-9 Hz) and unattenuated burst responses.
        # Respond more to aversive/novel stimuli than to pure reward.
        # Uses config.mesocortical_adapt_increment for faster adaptation.
        # -----------------------------------------------------------------------
        self.da_mesocortical_neurons: ConductanceLIF
        self.da_mesocortical_neurons = self._create_and_register_neuron_population(
            VTAPopulation.DA_MESOCORTICAL,
            self.da_mesocortical_size,
            polarity=PopulationPolarity.ANY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(config.tau_mem_ms, self.da_mesocortical_size, self.device, cv=0.20),
                v_reset=0.0,
                # v_threshold CV raised 0.12→0.25→0.35: without D2 autoreceptors,
                # mesocortical needs wider threshold heterogeneity than mesolimbic
                # to break pacemaker phase-lock.  At CV=0.25, still 100% epileptiform.
                v_threshold=heterogeneous_v_threshold(1.0, self.da_mesocortical_size, self.device, cv=0.35, clamp_fraction=0.25),
                tau_ref=config.tau_ref,
                g_L=heterogeneous_g_L(config.g_L, self.da_mesocortical_size, self.device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(config.mesocortical_noise_std, self.da_mesocortical_size, self.device),
                noise_tau_ms=3.0,
                tau_adapt_ms=heterogeneous_tau_adapt(config.tau_adapt_ms, self.da_mesocortical_size, self.device, cv=0.40),
                adapt_increment=heterogeneous_adapt_increment(config.mesocortical_adapt_increment, self.da_mesocortical_size, self.device),
                E_adapt=-0.5,
                # I_h (HCN) pacemaker — g_h_max 0.03→0.015→0.008→0.004: combined
                # with higher CV (0.35) and stronger adaptation (0.18), reducing
                # g_h_max weakens the pacemaker drive that synchronises neurons,
                # allowing noise and heterogeneity to break phase-lock.
                enable_ih=True,
                g_h_max=0.004,
                E_h=0.9,
                V_half_h=-0.35,
                k_h=0.08,
                tau_h_ms=150.0,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.da_mesocortical_size, self.device, cv=0.25),
            ),
        )

        # GABAergic interneurons (local inhibition, homeostasis)
        self._init_gaba_interneurons(VTAPopulation.GABA, self.gaba_size, device)

        # D2 somatodendritic autoreceptors — MESOLIMBIC ONLY
        # Exponential moving average of per-neuron DA firing rate (one-step lag).
        # Mesocortical neurons lack this feedback.
        self._prev_mesolimbic_spike_rate: torch.Tensor
        self.register_buffer("_prev_mesolimbic_spike_rate", torch.zeros(self.da_mesolimbic_size, device=device))

        # -----------------------------------------------------------------------
        # MESOCORTICAL SPARSE RECURRENT GABA INHIBITION
        # -----------------------------------------------------------------------
        # Mesocortical DA neurons lack D2 autoreceptors (Lammel et al. 2008), so
        # they have no self-inhibitory feedback to break pacemaker synchrony.
        # This sparse inhibition matrix models heterogeneous VTA GABA interneuron
        # projections back onto mesocortical DA neurons: when multiple MC neurons
        # co-fire, each receives different-strength GABA feedback next timestep,
        # creating competitive dynamics that desynchronise the population.
        # Connectivity 30%, scale 0.012 (g_L=0.08 → shunt ratio ~0.15).
        # Raised 0.006→0.012 and 20%→30%: with ~263 neighbors at 4 Hz,
        # recurrent GABA ≈ 0.25 per 20ms — strong enough to break I_h sync.
        self._mc_recurrent_inhib_weights = nn.Parameter(
            WeightInitializer.sparse_random_no_autapses(
                n_input=self.da_mesocortical_size,
                n_output=self.da_mesocortical_size,
                connectivity=0.30,
                weight_scale=0.012,
                device=device,
            ),
            requires_grad=False,
        )

        # -----------------------------------------------------------------------
        # V(s') ring buffer — full TD error: δ = r + γ·V(s') − V(s)
        # Stores the SNr-decoded scalar value estimate as a float over time.
        # V(s') = buffer read ``value_lag_steps`` steps ago.
        # -----------------------------------------------------------------------
        self._value_lag_steps: int = max(1, int(config.value_lag_ms / config.dt_ms))
        self._snr_rate_buffer = CircularDelayBuffer(
            max_delay=self._value_lag_steps,
            size=1,
            device=device,
            dtype=torch.float32,
        )

        # -----------------------------------------------------------------------
        # DA ramp — anticipatory reward signal (Howe et al. 2013)
        # Slowly builds up each step without reward; resets on reward delivery.
        # -----------------------------------------------------------------------
        self._da_ramp_signal: torch.Tensor
        self.register_buffer("_da_ramp_signal", torch.zeros(1, device=device))

        # Adaptive normalization
        self._rpe_norm = AdaptiveNormalization(epsilon=1e-6, track_abs=True)

        # =====================================================================
        # SEROTONIN RECEPTOR (5-HT1A on mesolimbic DA neurons — from DRN)
        # =====================================================================
        # Biology: DRN projects 5-HT to VTA DA neurons via 5-HT1A (inhibitory,
        # Gi-coupled).  High 5-HT hyperpolarises DA neurons, attenuating phasic
        # burst amplitude and reducing DA released into the mesolimbic pathway.
        # Mesocortical neurons have weaker 5-HT1A expression (Gervais & Bhagya 2007).
        # Kinetics: tau_rise ~10 ms, tau_decay ~200 ms (SERT reuptake).
        # 5-HT1A (Gi → GIRK): inhibitory on DA-mesolimbic neurons
        self._init_receptors_from_config(device)

        # Ensure all tensors are on the correct device
        self.to(device)

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Compute full TD RPE and drive both DA sub-populations to burst/pause.

        Implements the canonical temporal-difference error:

            δ = r + γ·V(s’) − V(s)

        where V(s) is decoded from the current SNr firing rate (high SNr = low value)
        and V(s’) is the same quantity lagged by ``config.value_lag_ms``.  If no SNr
        connection is present, the system falls back to δ ≈ r.

        An anticipatory DA ramp (Howe et al. 2013) slowly builds tonic drive across
        unrewarded steps and resets on reward delivery, providing additional temporal
        credit beyond eligibility-trace windows.
        """
        device = self.device
        config = self.config

        # =====================================================================
        # SEROTONIN RECEPTOR UPDATE (DRN Spikes → 5-HT1A Concentration)
        # =====================================================================
        # Update before RPE computation so the attenuation factor is ready for
        # both sub-populations in the same timestep.
        self._update_receptors(neuromodulator_inputs)
        sht_level = self._sht_concentration.mean().item()

        # =====================================================================
        # DECODE REWARD AND COMPUTE RPE
        # =====================================================================
        reward_synapse = SynapseId.external_reward_to_vta_da(self.region_name)
        reward_spikes = synaptic_inputs.get(reward_synapse, None)
        reward = self._decode_reward(reward_spikes)

        # =====================================================================
        # FULL TD ERROR: δ = r + γ·V(s’) − V(s)
        # V(s) is decoded from current SNr firing rate (high SNr = suppressed
        # action = low value).  V(s’) is the same signal ``value_lag_steps`` ago.
        # Falls back to δ ≈ r if no SNr connection is present.
        # =====================================================================

        # Extract SNr spikes (any SynapseId whose source_region is "substantia_nigra")
        snr_spikes: Optional[torch.Tensor] = None
        for sid, spikes in synaptic_inputs.items():
            if sid.source_region == "substantia_nigra":
                snr_spikes = spikes
                break

        V_s = self._decode_value(snr_spikes)  # Current state value [0, 1]

        if self._value_lag_steps > 0:
            # V(s’): read lagged value BEFORE writing current, so the buffer always
            # returns the state estimate from ``value_lag_steps`` timesteps ago.
            V_s_prime_tensor = self._snr_rate_buffer.read(self._value_lag_steps)  # [1]
            V_s_prime = float(V_s_prime_tensor.item())
            # Now write current V(s) into buffer
            self._snr_rate_buffer.write(
                torch.tensor([V_s], dtype=torch.float32, device=device)
            )
        else:
            V_s_prime = 0.0

        # 5-HT patience modulation: high serotonin → higher effective gamma → more
        # future-oriented TD learning (willing to wait for delayed rewards).
        # Biology: DRN 5-HT neurons encode patience; optogenetic activation of DRN
        # promotes waiting for delayed rewards (Miyazaki et al. 2014; Fonseca et al. 2015).
        # Range: gamma linearly interpolated from config.gamma to 0.999 by sht_level.
        # At sht=0 → gamma=0.99 (100-step horizon); sht=1 → gamma=0.999 (1000-step horizon).
        effective_gamma = self.config.gamma + sht_level * (0.999 - self.config.gamma)

        # Full temporal-difference error
        rpe = reward + effective_gamma * V_s_prime - V_s

        # =====================================================================
        # DA RAMP — anticipatory reward signal
        # =====================================================================
        ramp: float = 0.0
        if reward > 0:
            # Reward received: reset ramp (dopamine ramp collapses post-reward)
            self._da_ramp_signal.fill_(0.0)
        else:
            # No reward: build anticipatory ramp (bounded exponential rise).
            # V(s')-scaled: ramp increments proportional to learned value anticipation.
            # When striatum hasn't learned to predict reward (V_s_prime ≈ 0) the ramp
            # stays flat; once it strongly predicts upcoming reward (V_s_prime → 1)
            # the ramp builds at full rate.  This makes the ramp emerge from
            # corticostriatal learning rather than being a fixed-clock timer.
            # Biological basis: in Howe et al. 2013 ramp amplitude correlates with
            # learned value; Hamid et al. 2016 shows DA ramp scales with Q(s).
            decay = 1.0 - config.dt_ms / self.config.da_ramp_tau_ms
            value_confidence = max(V_s_prime, 0.0)  # scaled by learned V(s')
            self._da_ramp_signal.mul_(decay).add_(
                torch.tensor(
                    [self.config.da_ramp_gain * config.dt_ms * value_confidence],
                    dtype=torch.float32, device=device,
                )
            )
        ramp = float(self._da_ramp_signal.item())

        rpe = self._rpe_norm(rpe)

        rpe_clipped = max(-1.0, min(1.0, rpe))

        # =====================================================================
        # MESOLIMBIC DA NEURONS (reward/motivation channel)
        # Tonic 4-6 Hz firing is driven by intrinsic I_h pacemaker (enable_ih=True).
        # Phasic burst on positive RPE; pause via RMTg GABA on negative RPE.
        # D2 autoreceptor feedback suppresses phasic burst amplitude.
        # =====================================================================
        ml_phasic = torch.zeros(self.da_mesolimbic_size, device=device)

        if rpe_clipped > 0:
            rpe_exc = rpe_clipped * self.config.rpe_gain
            rpe_jitter = torch.clamp(
                1.0 + torch.randn(self.da_mesolimbic_size, device=device) * 0.2,
                0.5, 1.5,
            )
            ml_phasic = ml_phasic + rpe_exc * rpe_jitter

        # D2 somatodendritic autoreceptors — mesolimbic only
        if self.config.d2_autoreceptor_gain > 0.0:
            autoreceptor_suppression = self.config.d2_autoreceptor_gain * self._prev_mesolimbic_spike_rate
            ml_phasic = (ml_phasic * (1.0 - autoreceptor_suppression)).clamp(min=0.0)

        # Anticipatory DA ramp (mesolimbic: full ramp amplitude)
        if ramp > 0.0:
            ml_phasic = ml_phasic + ramp

        # 5-HT1A: attenuate mesolimbic DA burst amplitude under high serotonin.
        # Biology: Gi-coupled hyperpolarisation reduces AP rate; max ~35% suppression.
        if sht_level > 0.01:
            ml_phasic = ml_phasic * (1.0 - 0.35 * sht_level)

        # Add tonic baseline drive (mesolimbic pacemaker floor, ~5-6 Hz via
        # adaptation-rebound mechanism — see VTAConfig.mesolimbic_baseline_drive).
        ml_baseline = torch.full((self.da_mesolimbic_size,), config.mesolimbic_baseline_drive, device=device)
        ml_phasic = ml_phasic + ml_baseline

        # Phasic RPE/ramp conductance split: 5% NMDA to avoid steady-state accumulation.
        ml_g_ampa, ml_g_nmda = split_excitatory_conductance(ml_phasic, nmda_ratio=0.05)

        # RMTg inhibition targeting mesolimbic neurons specifically
        rmtg_ml = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.da_mesolimbic_size,
            filter_by_source_region="rostromedial_tegmentum",
            filter_by_target_population=VTAPopulation.DA_MESOLIMBIC,
        )

        ml_spikes, _ = self.da_mesolimbic_neurons.forward(
            g_ampa_input=ConductanceTensor(ml_g_ampa),
            g_nmda_input=ConductanceTensor(ml_g_nmda),
            g_gaba_a_input=ConductanceTensor(rmtg_ml.g_gaba_a),
            g_gaba_b_input=None,
        )
        ml_spikes_float = ml_spikes.float()

        # Update D2 autoreceptor EMA (mesolimbic only; ~10ms tau at 1ms dt)
        self._prev_mesolimbic_spike_rate.mul_(0.9).add_(ml_spikes_float * 0.1)

        # =====================================================================
        # MESOCORTICAL DA NEURONS (executive/arousal channel)
        # Tonic 7-9 Hz firing driven by intrinsic I_h pacemaker (enable_ih=True).
        # NO D2 autoreceptors; same RPE burst drive and RMTg pause pathway.
        # =====================================================================
        mc_phasic = torch.zeros(self.da_mesocortical_size, device=device)

        if rpe_clipped > 0:
            rpe_exc = rpe_clipped * self.config.rpe_gain
            rpe_jitter = torch.clamp(
                1.0 + torch.randn(self.da_mesocortical_size, device=device) * 0.2,
                0.5, 1.5,
            )
            mc_phasic = mc_phasic + rpe_exc * rpe_jitter

        # No D2 autoreceptor suppression for mesocortical neurons

        # Anticipatory DA ramp (mesocortical: 50% amplitude, models weaker ramp
        # in PFC-projecting neurons; less reward-driven than mesolimbic)
        if ramp > 0.0:
            mc_phasic = mc_phasic + ramp * 0.5

        # 5-HT1A: mesocortical DA neurons have weaker 5-HT1A expression (~15% attenuation).
        if sht_level > 0.01:
            mc_phasic = mc_phasic * (1.0 - 0.15 * sht_level)

        # Add tonic baseline drive (mesocortical pacemaker floor, ~7-8 Hz via
        # adaptation-rebound mechanism — see VTAConfig.mesocortical_baseline_drive).
        mc_baseline = torch.full((self.da_mesocortical_size,), config.mesocortical_baseline_drive, device=device)
        mc_phasic = mc_phasic + mc_baseline

        # Phasic RPE/ramp conductance split: 5% NMDA to avoid steady-state accumulation.
        mc_g_ampa, mc_g_nmda = split_excitatory_conductance(mc_phasic, nmda_ratio=0.05)

        # RMTg inhibition targeting mesocortical neurons specifically
        rmtg_mc = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.da_mesocortical_size,
            filter_by_source_region="rostromedial_tegmentum",
            filter_by_target_population=VTAPopulation.DA_MESOCORTICAL,
        )

        # Sparse recurrent GABA inhibition: previous-timestep MC spikes →
        # heterogeneous GABA_A conductance.  Each MC neuron receives a
        # different weighted sum of neighbors' spikes, breaking co-activation.
        mc_recurrent_gaba = torch.matmul(
            self._mc_recurrent_inhib_weights,
            self._prev_spikes(VTAPopulation.DA_MESOCORTICAL),
        )
        mc_total_gaba_a = rmtg_mc.g_gaba_a + mc_recurrent_gaba

        mc_spikes, _ = self.da_mesocortical_neurons.forward(
            g_ampa_input=ConductanceTensor(mc_g_ampa),
            g_nmda_input=ConductanceTensor(mc_g_nmda),
            g_gaba_a_input=ConductanceTensor(mc_total_gaba_a),
            g_gaba_b_input=None,
        )
        mc_spikes_float = mc_spikes.float()

        # =====================================================================
        # GABA INTERNEURONS (homeostatic control of both DA sub-populations)
        # =====================================================================
        da_activity = (ml_spikes_float.mean() + mc_spikes_float.mean()).item() * 0.5
        gaba_spikes = self._step_gaba_interneurons(
            da_activity, drive_scale=0.05, drive_baseline=0.004,
        )

        region_outputs: RegionOutput = {
            VTAPopulation.DA_MESOLIMBIC: ml_spikes,
            VTAPopulation.DA_MESOCORTICAL: mc_spikes,
            VTAPopulation.GABA: gaba_spikes,
        }

        self._snr_rate_buffer.advance()

        return region_outputs

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

        reward_spikes_float = reward_spikes.float()

        # Positive reward neurons (first half)
        positive_rate = reward_spikes_float[:n_half].mean().item()

        # Negative reward neurons (second half)
        negative_rate = reward_spikes_float[n_half:].mean().item()

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
