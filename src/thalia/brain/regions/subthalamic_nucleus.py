"""Subthalamic Nucleus (STN) - Basal Ganglia Glutamatergic Pacemaker.

The STN is the sole glutamatergic (excitatory) nucleus within the basal ganglia.
It occupies a pivotal position receiving the cortical hyperdirect pathway and
forming the reciprocal GPe-STN oscillatory loop, providing net excitatory drive
to the output nuclei (SNr and GPi).
"""

from __future__ import annotations

import torch

from thalia.brain.configs import SubthalamicNucleusConfig
from thalia.components import ConductanceLIF, ConductanceLIFConfig
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorInput,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)

from .neural_region import NeuralRegion
from .population_names import STNPopulation
from .region_registry import register_region


@register_region(
    "subthalamic_nucleus",
    aliases=["stn"],
    description="Subthalamic nucleus - glutamatergic basal ganglia pacemaker",
    version="1.0",
    author="Thalia Project",
    config_class=SubthalamicNucleusConfig,
)
class SubthalamicNucleus(NeuralRegion[SubthalamicNucleusConfig]):
    """Subthalamic Nucleus - Glutamatergic Basal Ganglia Pacemaker.

    Autonomous ~20 Hz pacemaker neurons that receive hyperdirect cortical
    input and GPe inhibition, and project excitatory output to SNr and GPe.
    """

    def __init__(self, config: SubthalamicNucleusConfig, population_sizes: PopulationSizes, region_name: RegionName):
        super().__init__(config, population_sizes, region_name)

        self.stn_size = population_sizes[STNPopulation.STN]

        # STN glutamatergic neurons with I_h pacemaking via HCN channels
        # enable_ih=True: voltage-dependent HCN current activates on hyperpolarisation,
        # providing the depolarising sag and rebound burst that drives ~20 Hz pacemaking.
        # This replaces the old constant i_h_drive scalar offset.
        self.stn_neurons = ConductanceLIF(
            n_neurons=self.stn_size,
            config=ConductanceLIFConfig(
                region_name=self.region_name,
                population_name=STNPopulation.STN,
                tau_mem=self.config.tau_mem,
                v_threshold=self.config.v_threshold,
                v_reset=0.0,
                tau_ref=self.config.tau_ref,
                g_L=0.08,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                noise_std=0.006,
                # I_h (HCN) pacemaker — voltage-dependent, activates during hyperpolarisation
                enable_ih=True,
                g_h_max=self.config.i_h_conductance * 10.0,  # Scale from old constant to max-conductance
                E_h=-0.3,      # Depolarising (between E_L=0 and E_I=-0.5)
                V_half_h=-0.3, # Halfway between rest and inhibitory reversal
                k_h=0.10,      # Steep voltage-dependence
                tau_h_ms=80.0, # Slightly faster than default for ~20 Hz pacemaking
            ),
            device=self.device,
        )

        # Baseline drive for autonomous pacemaking (tonic excitatory conductance)
        # I_h is now handled by enable_ih in the neuron config (voltage-dependent).
        # This scalar baseline provides the sustained low-level depolarisation needed
        # to keep STN neurons near threshold between I_h-driven rebounds.
        self.baseline_drive = torch.full(
            (self.stn_size,), self.config.baseline_drive, device=self.device
        )

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(STNPopulation.STN, self.stn_neurons)

        # Ensure all tensors are on the correct device
        self.to(self.device)

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Update STN neurons based on hyperdirect cortical and GPe inputs."""
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        # =====================================================================
        # Compute conductances
        # Integrate all inputs targeting STN population
        # =====================================================================
        dendrite = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.stn_size,
            filter_by_target_population=STNPopulation.STN,
        )

        # Baseline autonomous pacemaker drive (tonic excitatory conductance)
        # I_h (voltage-dependent HCN current) is now computed inside stn_neurons.forward()
        # via enable_ih=True in the ConductanceLIF config.
        g_exc = self.baseline_drive.clone() + dendrite.g_ampa

        # AMPA-only for tonic pacemaking: tau_NMDA=100ms creates Mg²⁺-blocked bistability
        # at resting V≈0 (only 7.5% NMDA unblocked), locking STN in sub-threshold regime.
        # Biological STN tonic pacemaking is driven by T-type Ca²⁺ / persistent Na⁺,
        # represented here as pure AMPA baseline. NMDA contributes to burst responses
        # (not tonic 20 Hz pacemaking) and can be re-added once HCN/Ca dynamics are added.
        stn_spikes, _ = self.stn_neurons.forward(
            g_ampa_input=ConductanceTensor(g_exc),
            g_nmda_input=None,
            g_gaba_a_input=ConductanceTensor(dendrite.g_gaba_a),
            g_gaba_b_input=ConductanceTensor(dendrite.g_gaba_b),
        )

        region_outputs: RegionOutput = {
            STNPopulation.STN: stn_spikes,
        }

        return self._post_forward(region_outputs)

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes."""
        super().update_temporal_parameters(dt_ms)
        self.stn_neurons.update_temporal_parameters(dt_ms)
