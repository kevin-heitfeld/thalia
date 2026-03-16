"""Subiculum — Hippocampal Output Gateway.

The subiculum is the principal output nucleus of the hippocampal formation,
interposed between CA1 and the entorhinal cortex.  It receives the majority
of CA1 axon collaterals and redistributes compressed hippocampal output to a
wide set of downstream targets.

Biological Background:
======================
**Anatomy:**
- Located between CA1 (stratum oriens border) and the presubiculum
- ~75 % of CA1 projections terminate in subiculum (O'Mara et al. 2001)
- Remaining ~25 % exit directly to EC_V (the direct CA1→EC path modelled
  by the previous direct connection that this region replaces)
- ~35,000 neurons in rat; predominantly excitatory pyramidal cells

**Three Physiological Cell Types (collapsed to one population here):**
1. **Regular-spiking** (~40 %): Tonic 5-10 Hz, no bursting
2. **Burst-firing** (~40 %): Initial Ca²⁺-driven doublet/triplet, then tonic
3. **Weak-bursting** (~20 %): Single initial burst, then regular

Heterogeneous `tau_mem_ms` and `v_threshold` in `ConductanceLIF` naturally give
rise to all three modes from a single population tensor.

**Inputs:**
- ``hippocampus / HippocampusPopulation.CA1``    — excitatory (Schaffer collateral relay)
- ``medial_septum / MedialSeptumPopulation.ACH`` — optional cholinergic modulation

**Outputs (axonal targets):**
- ``EntorhinalCortexPopulation.EC_V`` — perforant path back-projection to neocortex
- ``CortexPopulation.L5_PYR``         — direct hippocampal-to-PFC report (consolidation)
- ``BLAPopulation.PRINCIPAL``         — contextual fear/safety signal to amygdala

**Why Not Model Subicular Inhibitory Interneurons Explicitly?**
PV basket cells in the subiculum (~15 % of the population) provide fast
feedback inhibition with a latency of ~1 ms.  At the simulation timestep
(dt = 1 ms) this is within-step feedback.  We approximate their net E→I→E
effect as a lateral inhibition coefficient (`lateral_inhibition_ratio`) that
scales the AMPA drive back as GABA_A input — identical to the pattern used
in EntorhinalCortex.
"""

from __future__ import annotations

from typing import ClassVar, List, Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import SubiculumConfig
from thalia.brain.neurons import (
    ConductanceLIFConfig,
    ConductanceLIF,
    heterogeneous_tau_mem,
    heterogeneous_v_threshold,
    heterogeneous_adapt_increment,
    heterogeneous_g_L,
    split_excitatory_conductance,
)
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorChannel,
    NeuromodulatorInput,
    PopulationPolarity,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)

from .neural_region import NeuralRegion
from .population_names import SubiculumPopulation
from .region_registry import register_region


@register_region(
    "subiculum",
    aliases=["sub", "hippocampal_output"],
    description="Subiculum — hippocampal output gateway (CA1 → EC / PFC / BLA relay)",
)
class Subiculum(NeuralRegion[SubiculumConfig]):
    """Subiculum — Hippocampal Output Gateway.

    Receives CA1 excitatory input and relays a regular-spiking transformed
    output to entorhinal cortex (EC_V), prefrontal cortex (L5_PYR), and
    basolateral amygdala (PRINCIPAL).

    Biologically, the subiculum *converts* CA1 complex-spike bursts into
    regular spiking: the burst-onset Ca²⁺ component is absorbed by adaptation
    (``adapt_increment``), and the subsequent regular plateau matches downstream
    expectations in EC and PFC.

    Population:
    -----------
    - ``SubiculumPopulation.PRINCIPAL``: Excitatory pyramidal cells.

    Input Populations (via SynapseId routing):
    ------------------------------------------
    - ``hippocampus / CA1``           → PRINCIPAL  (main driver)
    - ``medial_septum / ACH``         → PRINCIPAL  (optional ACh arousal)
    - ``entorhinal_cortex / EC_III``  → PRINCIPAL  (optional direct EC input)

    Output Populations:
    -------------------
    - ``SubiculumPopulation.PRINCIPAL`` — spikes forwarded to EC_V, PFC, BLA
    """

    # Subiculum is a structural relay; it does not release neuromodulators.
    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = []

    def __init__(
        self,
        config: SubiculumConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ) -> None:
        """Initialise subiculum principal population.

        Args:
            config: ``SubiculumConfig`` with neuron biophysical parameters.
            population_sizes: Must include ``SubiculumPopulation.PRINCIPAL``.
            region_name: Unique string identifier, typically ``"subiculum"``.
            device: Torch device for all tensors.
        """
        super().__init__(config, population_sizes, region_name, device=device)

        self.principal_size: int = population_sizes[SubiculumPopulation.PRINCIPAL]

        # ── Principal population: heterogeneous LIF pyramidal neurons ─────────
        self.principal_neurons: ConductanceLIF
        self.principal_neurons = self._create_and_register_neuron_population(
            population_name=SubiculumPopulation.PRINCIPAL,
            n_neurons=self.principal_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=ConductanceLIFConfig(
                g_L=heterogeneous_g_L(0.05, self.principal_size, device),
                tau_E=5.0,
                tau_I=10.0,
                v_reset=0.0,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                v_threshold=heterogeneous_v_threshold(config.v_threshold, self.principal_size, device),
                adapt_increment=heterogeneous_adapt_increment(config.adapt_increment, self.principal_size, device),
                tau_adapt=config.tau_adapt,
                tau_mem_ms=heterogeneous_tau_mem(config.tau_mem_ms, self.principal_size, device),
            ),
        )

        self.to(device)

    # =========================================================================
    # FORWARD
    # =========================================================================

    def _step(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Advance subiculum one timestep.

        Processing:
        1. Integrate all synaptic inputs targeting PRINCIPAL dendrites.
        2. Add tonic sub-threshold depolarisation.
        3. Apply implicit PV basket-cell feedback inhibition.
        4. Fire principal neurons.

        Args:
            synaptic_inputs: ``SynapseId``-keyed spike tensors.  The primary
                driver is CA1 excitatory input; optional EC_III or medial septum
                ACh inputs are also accepted if wired by the brain builder.
            neuromodulator_inputs: Not consumed directly (subiculum has no
                neuromodulator subscriptions), but passed to the base class.

        Returns:
            ``RegionOutput`` with ``SubiculumPopulation.PRINCIPAL`` spike tensor.
        """
        # ── 1. Dendritic integration ──────────────────────────────────────────
        # Sum all AMPA conductances arriving at PRINCIPAL dendrites from any
        # upstream region (CA1, EC_III, ACh arousal, etc.).
        dendrite = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.principal_size,
            filter_by_target_population=SubiculumPopulation.PRINCIPAL,
        )

        # ── 2. Combined excitatory drive ──────────────────────────────────────
        exc_drive = dendrite.g_ampa + self.config.tonic_drive
        exc_drive = torch.nn.functional.relu(exc_drive)  # Non-negative conductance

        # ── 3. AMPA / NMDA split ─────────────────────────────────────────────
        # 25 % NMDA: subicular synapses contain moderate NMDA receptor density
        # (higher than EC, lower than CA3 recurrent). This ratio matches roughly
        # the AMPA:NMDA ratio observed at CA1→subiculum synapses (Behr et al. 2000).
        g_ampa, g_nmda = split_excitatory_conductance(exc_drive, nmda_ratio=0.25)

        # ── 4. Implicit PV feedback inhibition ───────────────────────────────
        # PV basket cells respond within 1 ms — modelled as same-step fraction.
        # GABA_B input from the also-present OLM-like interneurons is omitted
        # (subiculum lacks the OLM layer; inhibition is predominantly GABA_A).
        g_inh = exc_drive * self.config.lateral_inhibition_ratio

        # ── 5. Fire principal neurons ─────────────────────────────────────────
        spikes, _ = self.principal_neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa),
            g_nmda_input=ConductanceTensor(g_nmda),
            g_gaba_a_input=ConductanceTensor(g_inh),
            g_gaba_b_input=None,
        )

        region_outputs: RegionOutput = {
            SubiculumPopulation.PRINCIPAL: spikes,
        }

        return region_outputs
