"""Locus Coeruleus (LC) - Norepinephrine Arousal and Uncertainty System.

The LC is the brain's primary source of norepinephrine (NE), broadcasting arousal
and uncertainty signals that modulate attention, gain, and exploratory behavior.
LC norepinephrine neurons exhibit synchronized bursting in response to task
difficulty and novel/unexpected events.

Uncertainty is computed from four biological sources:
- **PFC variance**: Cognitive conflict / decision difficulty (Aston-Jones & Cohen 2005)
- **Hippocampus novelty**: Mismatch between current and expected input (Sara 2009)
- **CeA salience**: Emotional arousal from amygdala fear/threat signals (Berridge & Waterhouse 2003)
- **VTA RPE**: Reward prediction error surprise via dopamine (Bouret & Sara 2004)
"""

from __future__ import annotations

import math
from typing import ClassVar, Dict, List, Optional, Union

import torch

from thalia import GlobalConfig
from thalia.brain.adaptive_normalization import AdaptiveNormalization
from thalia.brain.configs import LocusCoeruleusConfig
from thalia.brain.neurons import (
    NorepinephrineNeuronConfig,
    NorepinephrineNeuron,
    heterogeneous_dendrite_coupling,
    heterogeneous_noise_std,
    heterogeneous_tau_mem,
    heterogeneous_v_reset,
    heterogeneous_v_threshold,
    heterogeneous_g_L,
)
from thalia.typing import (
    NeuromodulatorInput,
    NeuromodulatorChannel,
    PopulationName,
    PopulationPolarity,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)
from thalia.utils import CircularDelayBuffer

from .neuromodulator_source_region import NeuromodulatorSourceRegion
from .population_names import LocusCoeruleusPopulation
from .region_registry import register_region


@register_region(
    "locus_coeruleus",
    aliases=["lc", "norepinephrine_system"],
    description="Locus coeruleus - norepinephrine arousal and uncertainty system",
)
class LocusCoeruleus(NeuromodulatorSourceRegion[LocusCoeruleusConfig]):
    """Locus Coeruleus - Norepinephrine Arousal and Uncertainty System."""

    # Declarative neuromodulator output registry.
    neuromodulator_outputs: ClassVar[Dict[NeuromodulatorChannel, PopulationName]] = {
        NeuromodulatorChannel.NE: LocusCoeruleusPopulation.NE,
    }

    # Subscribe to DA_MESOLIMBIC to detect reward prediction error (RPE) surprise.
    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = [
        NeuromodulatorChannel.DA_MESOLIMBIC,
    ]

    def __init__(
        self,
        config: LocusCoeruleusConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        super().__init__(config, population_sizes, region_name, device=device)

        self.ne_neurons_size = population_sizes[LocusCoeruleusPopulation.NE]
        self.gaba_size = population_sizes[LocusCoeruleusPopulation.GABA]

        # Norepinephrine neurons (gap junction coupled, synchronized bursts)
        self.ne_neurons: NorepinephrineNeuron
        self.ne_neurons = self._create_and_register_neuron_population(
            population_name=LocusCoeruleusPopulation.NE,
            n_neurons=self.ne_neurons_size,
            polarity=PopulationPolarity.ANY,
            config=NorepinephrineNeuronConfig(
                uncertainty_to_current_gain=20.0,
                gap_junction_strength=self.config.gap_junctions.coupling_strength,
                gap_junction_neighbor_radius=self.config.gap_junction_radius,
                tau_mem_ms=heterogeneous_tau_mem(18.0, self.ne_neurons_size, device, cv=0.20),
                v_reset=heterogeneous_v_reset(-0.12, self.ne_neurons_size, device),
                v_threshold=heterogeneous_v_threshold(1.08, self.ne_neurons_size, device, cv=0.12, clamp_fraction=0.25),
                g_L=heterogeneous_g_L(0.056, self.ne_neurons_size, device),
                noise_std=heterogeneous_noise_std(0.08, self.ne_neurons_size, device),
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.ne_neurons_size, device, cv=0.25),
            ),
        )

        # GABAergic interneurons (local inhibition, homeostasis).
        self._init_gaba_interneurons(
            LocusCoeruleusPopulation.GABA, self.gaba_size, device,
            tau_mem_ms=10.0, noise_std=0.02,
        )

        # Low-pass filtered NE activity (τ=20ms).
        # Smoothing converts the sparse instantaneous ne_activity signal into a
        # rate-coded signal that can drive GABA into the subthreshold/noise-driven regime.
        self._ne_lp_activity: float = 0.0

        # Uncertainty computation state - use CircularDelayBuffer for history
        self._pfc_activity_buffer = CircularDelayBuffer(
            max_delay=10,  # Track last 10 timesteps for variance computation
            size=1,  # Single scalar value per timestep
            dtype=torch.float32,
            device=device,
        )
        self._hippocampus_activity_buffer = CircularDelayBuffer(
            max_delay=10,  # Track last 10 timesteps for novelty detection
            size=1,  # Single scalar value per timestep
            dtype=torch.float32,
            device=device,
        )
        self._cea_activity_buffer = CircularDelayBuffer(
            max_delay=10,  # Track last 10 timesteps for emotional salience
            size=1,
            dtype=torch.float32,
            device=device,
        )

        # DA baseline tracker for RPE surprise detection
        self._da_baseline: float = 0.0
        self._da_baseline_count: int = 0

        # Adaptive normalization
        self._uncertainty_norm = AdaptiveNormalization(clip_range=(0.0, 2.0))

        # Ensure all tensors are on the correct device
        self.to(device)

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Compute uncertainty and drive norepinephrine neurons to burst."""
        # Extract source-specific spikes from registered synaptic inputs.
        # Iterate over all inputs and identify by source_region so that any
        # BrainBuilder-registered connection is automatically picked up.
        pfc_spikes: Optional[torch.Tensor] = None
        hippocampus_spikes: Optional[torch.Tensor] = None
        cea_spikes: Optional[torch.Tensor] = None
        for sid, spikes in synaptic_inputs.items():
            if sid.source_region == "prefrontal_cortex":
                pfc_spikes = spikes
            elif sid.source_region == "hippocampus":
                hippocampus_spikes = spikes
            elif sid.source_region == "central_amygdala":
                cea_spikes = spikes

        # Extract DA signal for RPE surprise detection
        da_signal = self._extract_neuromodulator(
            neuromodulator_inputs, NeuromodulatorChannel.DA_MESOLIMBIC
        )

        # Compute uncertainty signal from all sources
        uncertainty = self._compute_uncertainty(
            pfc_spikes, hippocampus_spikes, cea_spikes, da_signal
        )

        # Normalize uncertainty to prevent saturation
        uncertainty = self._uncertainty_norm(uncertainty)

        # Update NE neurons with uncertainty drive
        # High uncertainty → depolarization → synchronized burst
        # Gap junctions → population synchronization
        # Apply GABA feedback from the previous timestep (closes homeostatic loop).
        gaba_feedback = self._gaba_feedback(self.ne_neurons_size, scale=0.01)
        ne_spikes, _ = self.ne_neurons.forward(
            g_ampa_input=None,  # No direct AMPA input to NE neurons (modulated by uncertainty drive instead)
            g_nmda_input=None,  # NE neurons do not receive NMDA input
            g_gaba_a_input=gaba_feedback,
            g_gaba_b_input=None,
            uncertainty_drive=uncertainty,
        )
        # Update GABA interneurons (homeostatic control)
        ne_activity = ne_spikes.float().mean().item()
        _LC_LP_DECAY: float = 0.9512  # exp(-1 / 20ms)
        self._ne_lp_activity = self._ne_lp_activity * _LC_LP_DECAY + ne_activity
        gaba_spikes = self._step_gaba_interneurons(
            self._ne_lp_activity, drive_scale=0.15,
        )

        region_outputs: RegionOutput = {
            LocusCoeruleusPopulation.NE: ne_spikes,
            LocusCoeruleusPopulation.GABA: gaba_spikes,
        }

        self._pfc_activity_buffer.advance()
        self._hippocampus_activity_buffer.advance()
        self._cea_activity_buffer.advance()

        return region_outputs

    def _compute_uncertainty(
        self,
        pfc_spikes: Optional[torch.Tensor],
        hippocampus_spikes: Optional[torch.Tensor],
        cea_spikes: Optional[torch.Tensor],
        da_signal: Optional[torch.Tensor],
    ) -> float:
        """Compute uncertainty signal from multiple biological sources.

        Four uncertainty components (Aston-Jones & Cohen 2005; Sara 2009):
        - **PFC variance**: High temporal variance in PFC activity signals
          cognitive conflict or decision difficulty.
        - **Hippocampus novelty**: Deviation from baseline CA1 firing rate
          signals environmental novelty / mismatch.
        - **CeA salience**: High amygdala output signals emotional arousal
          from threat or salient stimuli (Berridge & Waterhouse 2003).
        - **VTA RPE**: Large dopamine deviations (burst or pause) from
          baseline signal unexpected reward outcomes (Bouret & Sara 2004).

        Combined via max-plus-additive: the dominant source sets the floor,
        and secondary sources contribute additively (capped at 1.0).

        Returns:
            Uncertainty signal in range [0, 1]
        """
        uncertainty_components: Dict[str, float] = {}

        # ── PFC uncertainty: variance of activity (conflict detection) ─────────
        if pfc_spikes is not None and pfc_spikes.sum() > 0:
            pfc_rate = pfc_spikes.float().mean().item()
            self._pfc_activity_buffer.write(torch.tensor([pfc_rate], device=self.device))

            history_values = []
            for i in range(1, 11):
                val = self._pfc_activity_buffer.read(delay=i)
                if val.abs().sum() > 1e-8:
                    history_values.append(val.item())

            if len(history_values) >= 3:
                mean = sum(history_values) / len(history_values)
                variance = sum((x - mean) ** 2 for x in history_values) / len(history_values)
                std = math.sqrt(variance)
                uncertainty_components["pfc"] = min(1.0, std * 5.0)

        # ── Hippocampus uncertainty: deviation from baseline → novelty ─────────
        if hippocampus_spikes is not None and hippocampus_spikes.sum() > 0:
            hpc_rate = hippocampus_spikes.float().mean().item()
            self._hippocampus_activity_buffer.write(torch.tensor([hpc_rate], device=self.device))

            history_values = []
            for i in range(1, 11):
                val = self._hippocampus_activity_buffer.read(delay=i)
                if val.abs().sum() > 1e-8:
                    history_values.append(val.item())

            if len(history_values) >= 3:
                baseline = sum(history_values) / len(history_values)
                deviation = abs(hpc_rate - baseline)
                uncertainty_components["hpc"] = min(1.0, deviation * 10.0)

        # ── CeA salience: emotional arousal from amygdala output ──────────────
        # High CeM firing signals threat detection or conditioned fear.
        # CeA→LC is the primary pathway for fear-driven NE release.
        if cea_spikes is not None and cea_spikes.sum() > 0:
            cea_rate = cea_spikes.float().mean().item()
            self._cea_activity_buffer.write(torch.tensor([cea_rate], device=self.device))

            history_values = []
            for i in range(1, 11):
                val = self._cea_activity_buffer.read(delay=i)
                if val.abs().sum() > 1e-8:
                    history_values.append(val.item())

            if len(history_values) >= 3:
                # Both absolute rate and deviation matter for emotional arousal:
                # sustained CeA activity = tonic fear, CeA burst = phasic alarm
                baseline = sum(history_values) / len(history_values)
                phasic = abs(cea_rate - baseline)  # Sudden change
                tonic = cea_rate  # Sustained level
                salience = max(phasic * 8.0, tonic * 4.0)
                uncertainty_components["cea"] = min(1.0, salience)

        # ── VTA RPE: reward prediction error surprise ─────────────────────────
        # DA burst (positive RPE) or pause (negative RPE) both signal unexpected
        # outcomes. The absolute deviation from baseline DA level drives LC.
        if da_signal is not None:
            da_rate = da_signal.float().mean().item()

            # Update DA baseline with slow exponential moving average
            self._da_baseline_count += 1
            alpha = 1.0 / min(self._da_baseline_count, 200)  # Slow adaptation
            self._da_baseline = (1 - alpha) * self._da_baseline + alpha * da_rate

            # Absolute deviation from baseline: both burst and pause are surprising
            if self._da_baseline_count >= 10:  # Wait for baseline to stabilize
                da_surprise = abs(da_rate - self._da_baseline)
                uncertainty_components["rpe"] = min(1.0, da_surprise * 15.0)

        # ── Combine sources ───────────────────────────────────────────────────
        # Max-plus-additive: dominant source sets the floor, secondary sources
        # add a fraction. This ensures any single strong signal triggers a burst
        # while multiple moderate signals combine.
        if uncertainty_components:
            values = list(uncertainty_components.values())
            primary = max(values)
            secondary_sum = sum(v for v in values if v < primary) * 0.3
            uncertainty = min(1.0, primary + secondary_sum)
        else:
            uncertainty = 0.0

        return uncertainty
