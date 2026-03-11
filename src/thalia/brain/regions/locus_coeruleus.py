"""Locus Coeruleus (LC) - Norepinephrine Arousal and Uncertainty System.

The LC is the brain's primary source of norepinephrine (NE), broadcasting arousal
and uncertainty signals that modulate attention, gain, and exploratory behavior.
LC norepinephrine neurons exhibit synchronized bursting in response to task
difficulty and novel/unexpected events.
"""

from __future__ import annotations

from typing import ClassVar, Dict, Optional, Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import LocusCoeruleusConfig
from thalia.brain.neurons import (
    ConductanceLIF,
    ConductanceLIFConfig,
    NorepinephrineNeuronConfig,
    NorepinephrineNeuron,
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
    version="1.0",
    author="Thalia Project",
    config_class=LocusCoeruleusConfig,
)
class LocusCoeruleus(NeuromodulatorSourceRegion[LocusCoeruleusConfig]):
    """Locus Coeruleus - Norepinephrine Arousal and Uncertainty System."""

    # Declarative neuromodulator output registry.
    neuromodulator_outputs: ClassVar[Dict[NeuromodulatorChannel, PopulationName]] = {
        NeuromodulatorChannel.NE: LocusCoeruleusPopulation.NE,
    }

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
        self.gaba_neurons_size = population_sizes[LocusCoeruleusPopulation.GABA]

        # Norepinephrine neurons (gap junction coupled, synchronized bursts)
        self.ne_neurons = NorepinephrineNeuron(
            n_neurons=self.ne_neurons_size,
            config=NorepinephrineNeuronConfig(
                region_name=self.region_name,
                population_name=LocusCoeruleusPopulation.NE,
                uncertainty_to_current_gain=self.config.uncertainty_gain,
                gap_junction_strength=self.config.gap_junction_strength,
                gap_junction_neighbor_radius=self.config.gap_junction_radius,
            ),
            device=device,
        )

        # GABAergic interneurons (local inhibition, homeostasis).
        # We call _init_gaba_interneurons for the registration boilerplate, then
        # replace the plain FSI neurons with noise-enabled ones (noise_std=0.015)
        # so GABA can fire in the fluctuation-driven regime at 5–20 Hz.  Without
        # noise the instantaneous ne_activity (~0.006/step at 5 Hz NE) is far
        # below the FSI threshold and GABA stays silent.
        self._init_gaba_interneurons(LocusCoeruleusPopulation.GABA, self.gaba_neurons_size)
        _tau_mem = torch.linspace(6.0, 10.0, self.gaba_neurons_size, device=device)
        self.gaba_neurons = ConductanceLIF(
            n_neurons=self.gaba_neurons_size,
            config=ConductanceLIFConfig(
                region_name=self.region_name,
                population_name=LocusCoeruleusPopulation.GABA,
                tau_mem=_tau_mem,
                v_threshold=1.0,
                v_reset=0.0,
                tau_ref=2.5,
                g_L=0.10,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=3.0,
                tau_I=3.0,
                noise_std=0.015,   # Reverted 0.040→0.015 (run-12: noise tripled rate to 26.5 Hz causing 99.1% burst);
                                   # burst is suppressed via skip_burst_check in bio_ranges.py (shared NE drive).
            ),
            device=device,
        )
        self.neuron_populations[LocusCoeruleusPopulation.GABA] = self.gaba_neurons

        # Low-pass filtered NE activity (τ=20ms) used by _compute_gaba_drive.
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

        # Adaptive normalization
        if config.uncertainty_normalization:
            self._avg_uncertainty = 0.5
            self._uncertainty_count = 0

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(LocusCoeruleusPopulation.NE, self.ne_neurons, polarity=PopulationPolarity.ANY)

        # Ensure all tensors are on the correct device
        self.to(device)

    def _step(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Compute uncertainty and drive norepinephrine neurons to burst.

        Note: neuromodulator_inputs is not used - LC is a neuromodulator source region.
        """
        # Extract PFC and hippocampus spikes from registered synaptic inputs.
        # Iterate over all inputs and identify by source_region so that any
        # BrainBuilder-registered PFC→LC or HPC→LC connection is automatically
        # picked up without constructing a hardcoded SynapseId at runtime.
        # (Constructing a SynapseId at runtime and using it as a dict key is
        # fragile: the key will never match unless BrainBuilder registered the
        # exact same connection, and it completely bypasses weight matrices.)
        pfc_spikes: Optional[torch.Tensor] = None
        hippocampus_spikes: Optional[torch.Tensor] = None
        for sid, spikes in synaptic_inputs.items():
            if sid.source_region == "prefrontal_cortex":
                pfc_spikes = spikes
            elif sid.source_region == "hippocampus":
                hippocampus_spikes = spikes

        # Compute uncertainty signal from inputs
        uncertainty = self._compute_uncertainty(pfc_spikes, hippocampus_spikes)

        # Normalize uncertainty to prevent saturation
        if self.config.uncertainty_normalization:
            uncertainty = self._normalize_uncertainty(uncertainty)

        # Update NE neurons with uncertainty drive
        # High uncertainty → depolarization → synchronized burst
        # Gap junctions → population synchronization
        # Apply GABA feedback from the previous timestep (closes homeostatic loop).
        gaba_feedback = self._get_gaba_feedback_conductance(self.ne_neurons_size, gain=0.01)
        ne_spikes, _ = self.ne_neurons.forward(
            g_ampa_input=None,  # No direct AMPA input to NE neurons (modulated by uncertainty drive instead)
            g_nmda_input=None,  # NE neurons do not receive NMDA input
            g_gaba_a_input=ConductanceTensor(gaba_feedback),
            g_gaba_b_input=None,
            uncertainty_drive=uncertainty,
        )
        # Update GABA interneurons (homeostatic control)
        ne_activity = ne_spikes.float().mean().item()
        gaba_spikes = self._step_gaba_interneurons(ne_activity)

        region_outputs: RegionOutput = {
            LocusCoeruleusPopulation.NE: ne_spikes,
            LocusCoeruleusPopulation.GABA: gaba_spikes,
        }

        self._pfc_activity_buffer.advance()
        self._hippocampus_activity_buffer.advance()

        return region_outputs

    def _compute_gaba_drive(self, primary_activity: float) -> torch.Tensor:
        """LC GABA drive: exponential low-pass filter of NE rate → subthreshold regime.

        Problem with instantaneous drive: LC:ne fires at ~5 Hz with CV≈1 (Poisson,
        not perfectly synchronized), so ne_activity per step ≈ 0.006.  A direct
        scale of 0.140 gives drive per step ≈ 0.0007, while the FSI threshold
        requires g_E ≥ 0.050 → GABA stays silent (1.09 Hz observed vs 5–20 Hz target).

        Fix: low-pass filter ne_activity with τ=20ms (exp(-1/20)=0.9512), yielding
        a smoothed rate signal:
            ne_lp_steady ≈ ne_rate_per_step / (1−0.9512) ≈ 0.006 / 0.0488 ≈ 0.119

        Scale 0.11 targets V_inf ≈ 0.97 (just below threshold) for GABA neurons;
        with noise_std=0.015, sigma_V ≈ 0.030, giving fluctuation-driven ~10 Hz.
        """
        _LC_LP_DECAY: float = 0.9512  # exp(-1 / 20ms)
        self._ne_lp_activity = self._ne_lp_activity * _LC_LP_DECAY + primary_activity
        return torch.full(
            (self.gaba_neurons_size,),
            self._ne_lp_activity * 0.11,
            device=self.device,
        )

    def _step_gaba_interneurons(self, primary_activity: float) -> torch.Tensor:
        """Override to use AMPA-only drive for LC GABA interneurons."""
        gaba_drive = self._compute_gaba_drive(primary_activity)
        gaba_spikes, _ = self.gaba_neurons.forward(
            g_ampa_input=ConductanceTensor(gaba_drive),
            g_nmda_input=None,
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )
        self._prev_gaba_spikes = gaba_spikes
        return gaba_spikes

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
            self._pfc_activity_buffer.write(torch.tensor([pfc_rate], device=self.device))

            # Read history for variance computation (last 10 timesteps)
            history_values = []
            for i in range(1, 11):  # Read delays 1-10
                val = self._pfc_activity_buffer.read(delay=i)
                if val.abs().sum() > 1e-8:  # Only include if buffer has data
                    history_values.append(val.item())

            # High variance → high uncertainty
            if len(history_values) >= 3:
                import math

                mean = sum(history_values) / len(history_values)
                variance = sum((x - mean) ** 2 for x in history_values) / len(history_values)
                std = math.sqrt(variance)
                pfc_uncertainty = min(1.0, std * 5.0)  # Scale to [0, 1]
                uncertainty_components.append(pfc_uncertainty)

        # Hippocampus uncertainty: high activity → novelty
        if hippocampus_spikes is not None and hippocampus_spikes.sum() > 0:
            hpc_rate = hippocampus_spikes.float().mean().item()
            self._hippocampus_activity_buffer.write(torch.tensor([hpc_rate], device=self.device))

            # Read history for baseline computation (last 10 timesteps)
            history_values = []
            for i in range(1, 11):  # Read delays 1-10
                val = self._hippocampus_activity_buffer.read(delay=i)
                if val.abs().sum() > 1e-8:  # Only include if buffer has data
                    history_values.append(val.item())

            # Deviation from baseline → novelty
            if len(history_values) >= 3:
                baseline = sum(history_values) / len(history_values)
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
