"""Globus Pallidus Interna (GPi) - Basal Ganglia Output Nucleus for Motor / Cognitive Loops.

The GPi (also called the entopeduncular nucleus in rodents) is the primary GABAergic
output nucleus of the basal ganglia for limb movement and cognitive thalamus gating,
running in parallel with the SNr (which gates saccades and VTA dopamine output).

- **Principal neurons** (~75%): GABAergic, ~60-80 Hz tonic firing, project to thalamus
  VA/VL (motor loop) and MD (cognitive/limbic loop). Directly regulated by the direct
  pathway (D1 MSNs → GPi, "Go" = disinhibit thalamus) and hyperdirect/indirect pathways
  (STN/GPe → GPi, "Stop" = suppress thalamus).
- **Border cells** (~25%): Lower baseline, pause on unexpected reward. Proposed value-
  coding subset analogous to the reward-predictive cells in SNr.

Biological Background:
======================
**Anatomy:**
- Location: Medial segment of the globus pallidus (primates) / entopeduncular nucleus (rodents)
- ~3,000 neurons in rodents (~75% principal, ~25% border cells)
- Tonic baseline firing ~60-80 Hz (highest of all BG nuclei)
- GABAergic axons leave via internal capsule → thalamus

**Direct Pathway ("Go"):**
  D1-MSNs fire → inhibit GPi PRINCIPAL → disinhibit thalamus → promote movement/cognition

**Indirect Pathway ("No-Go"):**
  D2-MSNs → suppress GPe → disinhibit STN → STN bursts → re-excite GPi → suppress thalamus

**Hyperdirect Pathway ("Stop"):**
  Cortex L5 → STN (fast, bypasses striatum) → excites GPi → suppresses thalamus rapidly.
  Arrives at GPi ~5ms before striatal signals, enabling rapid action cancellation.

**GPe → GPi:**
  GPe PROTOTYPIC directly inhibits GPi, providing another link between indirect and
  direct loops (pallido-pallidal pathway).

**Inputs:**
- Striatum D1-MSNs: Inhibitory (direct pathway, "Go")
- STN: Excitatory (hyperdirect + indirect pathway)
- GPe PROTOTYPIC: Inhibitory (pallido-pallidal, closes indirect loop)

**Outputs:**
- Thalamus RELAY: Inhibitory (GABA_A, gates motor and cognitive thalamus relay cells)

**Key Differences from SNr:**
- GPi gates limb movement and cognitive thalamus (VA/VL/MD), NOT saccades
- GPi does NOT project to VTA; SNr does
- GPi has slightly higher baseline (~60-80 Hz vs SNr ~60-80 Hz — similar)
- GPi receives more pallidal input; SNr receives more nigral DA modulation
"""

from __future__ import annotations

from typing import Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import TonicPacemakerConfig, get_default_gpi_config
from thalia.typing import (
    NeuromodulatorInput,
    PopulationPolarity,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)

from .basal_ganglia_output_nucleus import BasalGangliaOutputNucleus
from .population_names import GPiPopulation
from .region_registry import register_region


@register_region(
    "globus_pallidus_interna",
    aliases=["gpi", "entopeduncular"],
    description="Globus pallidus interna - basal ganglia output nucleus for motor/cognitive loops",
    version="1.0",
    author="Thalia Project",
    config_class=get_default_gpi_config,
)
class GlobusPallidusInterna(BasalGangliaOutputNucleus[TonicPacemakerConfig]):
    """Globus Pallidus Interna - Motor and Cognitive Thalamus Gate.

    Contains principal neurons (→thalamus VA/VL/MD) and border cells (reward-
    sensitive, pause-on-reward subset). The highest tonically firing nucleus in
    the basal ganglia; its inhibitory hold on the thalamus is what the direct
    pathway must overcome to permit action.

    Input Populations:
    ------------------
    - striatum D1: Inhibitory (direct pathway — "Go" signal suppresses GPi)
    - STN: Excitatory (hyperdirect + indirect pathway via STN)
    - GPe PROTOTYPIC: Inhibitory (pallido-pallidal, closes indirect loop)

    Output Populations:
    -------------------
    - "principal": Projects to thalamus RELAY (inhibitory, gates thalamocortical drive)
    - "border_cells": Sub-threshold at rest; activated by combined inputs; pause on reward
    """

    def __init__(
        self,
        config: TonicPacemakerConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        super().__init__(config, population_sizes, region_name, device=device)

        self.principal_size = population_sizes[GPiPopulation.PRINCIPAL]
        self.border_cells_size = population_sizes[GPiPopulation.BORDER_CELLS]

        # Principal neurons: ~75% of GPi, ~60-80 Hz tonic, project to thalamus VA/VL/MD
        self.principal_neurons = self._make_bg_neurons(
            self.principal_size, GPiPopulation.PRINCIPAL, noise_std=0.007
        )
        # Border cells: ~25% of GPi, lower baseline (sub-threshold at rest)
        # Pause on unexpected reward; value-coding role proposed in literature.
        self.border_cells_neurons = self._make_bg_neurons(
            self.border_cells_size, GPiPopulation.BORDER_CELLS, noise_std=0.005
        )

        # Tonic drive: principal at full baseline (~60-80 Hz); border cells at 0.65×
        # With factor=0.65: g_E_ss≈0.047, V_inf≈0.95 → nearly sub-threshold; STN+D1 inhibitory balance gives ~35-45 Hz.
        # Previous 0.8× gave V_inf≈1.09 → 64.6 Hz (run-13, target 20-50 Hz); reducing factor brings border cells
        # to the lower half of the target range while principal neurons (factor=1.0) remain at ~100 Hz.
        # Previous 0.4× was fully sub-threshold (V_inf≈0.67), relying on STN excitation which is
        # chronically depleted (eff≈0.04) at STN's 20 Hz rate with U=0.50 τd=800ms.
        self.principal_baseline = self._make_tonic_baseline(self.principal_size)
        self.border_cells_baseline = self._make_tonic_baseline(self.border_cells_size, factor=0.65)

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(GPiPopulation.PRINCIPAL, self.principal_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(GPiPopulation.BORDER_CELLS, self.border_cells_neurons, polarity=PopulationPolarity.INHIBITORY)

        # Ensure all tensors are on the correct device
        self.to(device)

    def _step(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Update GPi neurons based on D1 inhibition (Go), STN excitation (Stop), and GPe pacing."""
        principal_spikes = self._bg_step_single(
            synaptic_inputs, self.principal_size, GPiPopulation.PRINCIPAL,
            self.principal_neurons, self.principal_baseline,
        )
        border_cells_spikes = self._bg_step_single(
            synaptic_inputs, self.border_cells_size, GPiPopulation.BORDER_CELLS,
            self.border_cells_neurons, self.border_cells_baseline,
        )
        return {
            GPiPopulation.PRINCIPAL: principal_spikes,
            GPiPopulation.BORDER_CELLS: border_cells_spikes,
        }
