"""Substantia Nigra pars Reticulata (SNr) - Basal Ganglia Output Nucleus.

The SNr is one of the two primary output nuclei of the basal ganglia (along with
GPi - globus pallidus internal segment). It consists of tonically-active GABAergic
neurons that provide inhibitory output to thalamus and VTA.

Biological Background:
======================
**Anatomy:**
- Location: Midbrain, ventral to VTA
- ~10,000-15,000 GABAergic neurons in humans
- Adjacent to dopamine-producing SNc (substantia nigra pars compacta)

**Function:**
- Output gate of basal ganglia motor/cognitive loop
- Tonic inhibition of thalamus (suppresses unwanted actions)
- Tonic inhibition of VTA (regulates dopamine bursting)
- Disinhibition mechanism: Striatum D1 → SNr inhibition → thalamus release

**Firing Pattern:**
- Tonic baseline: 50-70 Hz (high spontaneous rate)
- Pauses: During action selection (D1 pathway activation)
- Bursts: During action suppression (D2/indirect pathway)

**Inputs:**
- Striatum D1 (direct pathway): Inhibitory (GABAergic)
- Striatum D2 via GPe/STN (indirect pathway): Net excitatory
- STN (subthalamic nucleus): Excitatory (glutamatergic)
- GPe (globus pallidus external): Inhibitory (GABAergic)

**Outputs:**
- Thalamus (motor VA/VL nuclei): Inhibitory (motor gating)
- Superior colliculus: Inhibitory (saccade gating)
- VTA: Inhibitory (dopamine regulation)
- Pedunculopontine nucleus: Inhibitory (locomotion)

**Computational Role in Basal Ganglia:**
===========================================
SNr firing rate encodes **negative value** (inverse of state value):
- High SNr rate → action suppression (NoGo state)
- Low SNr rate → action release (Go state)

In TD learning framework:
- SNr activity ∝ 1 - V(s)  (inverse state value)
- VTA uses SNr as V(s) estimate for RPE computation:
  RPE = r + γ·V(s') - V(s)
      ≈ r - (1 - SNr_rate)

This creates a closed-loop system:
```
VTA (dopamine) → Striatum (learning) → SNr (value) → VTA (RPE feedback)
```

**Simplified Implementation (Phase 1):**
========================================
- D1 pathway: Striatum D1 MSNs directly inhibit SNr
- D2 pathway: Striatum D2 MSNs excite SNr (via GPe/STN, simplified as direct)
- Future: Add explicit GPe, STN, GPi for full basal ganglia anatomy
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from thalia.brain.configs import SubstantiaNigraConfig
from thalia.brain.regions.population_names import StriatumPopulation, SubstantiaNigraPopulation
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
    "substantia_nigra",
    aliases=["snr", "substantia_nigra_reticulata"],
    description="Substantia nigra pars reticulata - basal ganglia output nucleus",
    version="1.0",
    author="Thalia Project",
    config_class=SubstantiaNigraConfig,
)
class SubstantiaNigra(NeuralRegion[SubstantiaNigraConfig]):
    """Substantia Nigra pars Reticulata - Basal Ganglia Output Nucleus.

    Tonically-active GABAergic neurons that:
    1. Gate thalamic output (motor/cognitive actions)
    2. Provide value feedback to VTA for TD learning
    3. Integrate striatal D1 (inhibitory) and D2 (excitatory) pathways

    Input Populations:
    ------------------
    - "striatum_d1": Direct pathway (Go) - inhibits SNr
    - "striatum_d2": Indirect pathway (NoGo) - excites SNr via GPe/STN

    Output Populations:
    -------------------
    - "vta_feedback": Inhibitory projection to VTA (value signal)

    Computational Properties:
    -------------------------
    - High firing rate (50-70 Hz) = low state value (action suppression)
    - Low firing rate (<30 Hz) = high state value (action release)
    - Value estimate = inverse function of firing rate
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: SubstantiaNigraConfig, population_sizes: PopulationSizes, region_name: RegionName):
        super().__init__(config, population_sizes, region_name)

        # Store input layer sizes as attributes for connection routing
        self.vta_feedback_size = population_sizes[SubstantiaNigraPopulation.VTA_FEEDBACK.value]

        # GABAergic output neurons (tonically active)
        snr_neuron_config = ConductanceLIFConfig(
            region_name=self.region_name,
            population_name=SubstantiaNigraPopulation.VTA_FEEDBACK.value,
            device=self.device,
            tau_mem=self.config.tau_mem,  # 15ms - realistic SNr membrane tau
            v_threshold=self.config.v_threshold,  # 1.0 - standard threshold
            v_reset=0.0,
            v_rest=0.0,
            tau_ref=self.config.tau_ref,  # 2.0ms - biological refractory period
            g_L=0.10,  # Moderate leak (tau_m = C_m/g_L = 1.0/0.10 = 10ms effective)
            E_L=0.0,
            E_E=3.0,
            E_I=-0.5,
            tau_E=5.0,  # AMPA-like kinetics
            tau_I=10.0,  # GABA_A-like kinetics
            noise_std=0.007 if self.config.baseline_noise_conductance_enabled else 0.0,  # Membrane voltage noise
        )
        self.neurons = ConductanceLIF(
            n_neurons=self.vta_feedback_size,
            config=snr_neuron_config,
        )

        # Tonic drive for baseline firing
        self.baseline_drive = torch.full((self.vta_feedback_size,), config.baseline_drive, device=self.device)

        # Track firing rate for value estimation
        self._firing_rate_history: list[float] = []

        # Store current output for diagnostics (initialized as None, updated in forward pass)
        self._current_output: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(SubstantiaNigraPopulation.VTA_FEEDBACK.value, self.neurons)

        self.__post_init__()

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Update SNr neurons based on striatal input.

        Note: neuromodulator_inputs is not used - SNr is a neuromodulator source region.
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        d1_vta_feedback_synapse = SynapseId(
            source_region="striatum",
            source_population=StriatumPopulation.D1.value,
            target_region=self.region_name,
            target_population=SubstantiaNigraPopulation.VTA_FEEDBACK.value,
        )
        d2_vta_feedback_synapse = SynapseId(
            source_region="striatum",
            source_population=StriatumPopulation.D2.value,
            target_region=self.region_name,
            target_population=SubstantiaNigraPopulation.VTA_FEEDBACK.value,
        )

        d1_spikes = synaptic_inputs.get(d1_vta_feedback_synapse, None)
        d2_spikes = synaptic_inputs.get(d2_vta_feedback_synapse, None)

        # Initialize conductances
        # CRITICAL: Weights already represent conductances
        # DO NOT apply additional gain factors - they create current-like quantities!
        g_exc = torch.zeros(self.vta_feedback_size, device=self.device)
        g_inh = torch.zeros(self.vta_feedback_size, device=self.device)

        # D1 pathway (direct, Go): Inhibits SNr
        if d1_spikes is not None:
            weights_d1 = self.get_synaptic_weights(d1_vta_feedback_synapse)  # [n_snr, n_d1]
            g_inh += torch.matmul(weights_d1, d1_spikes.float())

        # D2 pathway (indirect, NoGo): Excites SNr via GPe→STN→SNr
        # Simplified: D2 spikes directly increase SNr activity
        if d2_spikes is not None:
            weights_d2 = self.get_synaptic_weights(d2_vta_feedback_synapse)  # [n_snr, n_d2]
            g_exc += torch.matmul(weights_d2, d2_spikes.float())

        # Add baseline drive as excitatory conductance to maintain tonic firing (~50-70Hz)
        g_exc_total = g_exc + self.baseline_drive
        g_inh_total = g_inh

        # Add baseline noise conductance (stochastic background activity)
        # BIOLOGY: Represents spontaneous miniature EPSPs and stochastic channel openings
        if self.config.baseline_noise_conductance_enabled:
            noise = torch.randn(self.vta_feedback_size, device=self.device) * 0.007
            g_exc_total = g_exc_total + torch.clamp(noise, min=0.0)  # Only positive noise (excitatory)

        # Split excitatory conductance: 99% AMPA (fast), 1% NMDA (slow)
        # CRITICAL: Reduced from 5% to 1% NMDA to prevent over-integration
        # (NMDA accumulates 20x more than AMPA due to tau_nmda=100ms vs tau_ampa=5ms)
        # Even 1% NMDA provides temporal integration while avoiding slow buildup
        # that creates periodic bursting and 500 Hz oscillations
        g_ampa, g_nmda = g_exc_total * 0.99, g_exc_total * 0.01

        # Update neurons using conductance-based dynamics
        vta_feedback_spikes, vta_feedback_membrane = self.neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa),
            g_gaba_a_input=ConductanceTensor(g_inh_total),
            g_nmda_input=ConductanceTensor(g_nmda),
        )

        # Store current output for diagnostics (as tuple)
        self._current_output = (vta_feedback_spikes, vta_feedback_membrane)

        region_outputs: RegionOutput = {
            SubstantiaNigraPopulation.VTA_FEEDBACK.value: vta_feedback_spikes,
        }

        return self._post_forward(region_outputs)

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes."""
        super().update_temporal_parameters(dt_ms)
        self.neurons.update_temporal_parameters(dt_ms)

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_value_estimate(self) -> float:
        """Compute state value estimate from SNr firing rate.

        Biology: SNr firing rate inversely encodes state value
        - High SNr rate (60+ Hz) = low value (action suppression)
        - Low SNr rate (<30 Hz) = high value (action release)
        - Baseline (50-60 Hz) = neutral value (~0.5)

        Returns:
            Value estimate in range [0, 1]
        """
        # Get current firing rate from stored output
        if self._current_output is not None:
            spikes = self._current_output[0]  # Extract spikes from tuple
            spike_rate = spikes.float().mean().item()
        else:
            return 0.5  # Default neutral value

        # Convert to Hz (spikes per second)
        firing_rate_hz = spike_rate * (1000.0 / self.config.dt_ms)

        # Map firing rate to value (inverse relationship)
        # Baseline: 60 Hz → 0.5 value
        # High: 80 Hz → 0.0 value (low state value)
        # Low: 20 Hz → 1.0 value (high state value)

        baseline_hz = 60.0
        value = 1.0 - (firing_rate_hz / (2 * baseline_hz))

        # Clamp to [0, 1]
        value = max(0.0, min(1.0, value))

        return value
