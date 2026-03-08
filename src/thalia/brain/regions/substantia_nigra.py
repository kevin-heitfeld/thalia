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

**Implementation:**
====================
- D1 pathway: Striatum D1 MSNs (INHIBITORY) → SNr — direct suppression of SNr, releasing cortical
  targets from tonic inhibition (Go signal).
- D2 pathway: Striatum D2 MSNs → GPe PROTOTYPIC → STN → SNr — full indirect hyperdirect circuit
  implemented in :class:`~thalia.brain.brain_builder.BrainBuilder`. This region receives STN
  excitatory input; GPe receives D2 inhibitory input which disinhibits STN.
- SNr itself is GABAergic and tonically active, suppressing thalamo-cortical activity.
"""

from __future__ import annotations

from typing import Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import TonicPacemakerConfig, get_default_snr_config
from thalia.typing import (
    NeuromodulatorInput,
    PopulationPolarity,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)

from .basal_ganglia_output_nucleus import BasalGangliaOutputNucleus
from .population_names import SubstantiaNigraPopulation
from .region_registry import register_region


@register_region(
    "substantia_nigra",
    aliases=["snr", "substantia_nigra_reticulata"],
    description="Substantia nigra pars reticulata - basal ganglia output nucleus",
    version="1.0",
    author="Thalia Project",
    config_class=get_default_snr_config,
)
class SubstantiaNigra(BasalGangliaOutputNucleus[TonicPacemakerConfig]):
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

    def __init__(
        self,
        config: TonicPacemakerConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        super().__init__(config, population_sizes, region_name, device=device)

        # Store input layer sizes as attributes for connection routing
        self.vta_feedback_size = population_sizes[SubstantiaNigraPopulation.VTA_FEEDBACK]

        # GABAergic output neurons (tonically active)
        self.neurons = self._make_bg_neurons(
            self.vta_feedback_size, SubstantiaNigraPopulation.VTA_FEEDBACK, noise_std=0.007
        )

        # Tonic drive for baseline firing (~50-70 Hz)
        self.baseline_drive = self._make_tonic_baseline(self.vta_feedback_size)

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(SubstantiaNigraPopulation.VTA_FEEDBACK, self.neurons, polarity=PopulationPolarity.ANY)

        # Ensure all tensors are on the correct device
        self.to(device)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def _step(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Update SNr neurons based on striatal D1/D2 input and STN drive.

        Uses nmda_ratio=0.01: near-soma synapses on SNr have minimal NMDA involvement.
        """
        vta_feedback_spikes = self._bg_step_single(
            synaptic_inputs,
            self.vta_feedback_size,
            SubstantiaNigraPopulation.VTA_FEEDBACK,
            self.neurons,
            self.baseline_drive,
            nmda_ratio=0.01,
        )
        return {
            SubstantiaNigraPopulation.VTA_FEEDBACK: vta_feedback_spikes,
        }
