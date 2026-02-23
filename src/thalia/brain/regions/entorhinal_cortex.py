"""Entorhinal Cortex (EC) — Hippocampal Gateway Region.

The entorhinal cortex is the principal interface between the neocortex and
the hippocampal formation.  Sensory and associative cortical signals enter via
layers II and III; hippocampal outputs are relayed back to neocortex via
layer V.

Biological Background:
======================
**Anatomy:**
- Located at the medial temporal lobe, adjacent to hippocampus
- Provides ~85 % of hippocampal afferents
- Layers II and III project *in*, layer V projects *out*
- ~55,000 neurons in rat; ~5 × 10⁶ in human

**Three Functional Sub-populations:**

1. **EC_II (Layer II stellate cells — Perforant Path)**
   - Contains grid cells, border cells, head-direction cells
   - Axons cross the hippocampal fissure → perforant path
   - Project to DG outer molecular layer AND CA3 stratum lacunosum-moleculare
   - Facilitating STP — stronger signal with repeated cortical inputs
   - Primary *what/where* input to hippocampus

2. **EC_III (Layer III pyramidal cells — Temporoammonic Path)**
   - "Time cells" — ramp-like temporal coding on 100–200 ms scale
   - Project directly to CA1 stratum lacunosum-moleculare (bypassing DG/CA3)
   - Depressing STP — strong initial pulse, then adapts (novelty emphasis)
   - Critical for disambiguating retrieved memories from current perception

3. **EC_V (Layer V pyramidal cells — Back-projection)**
   - Receive CA1 / subiculum output (via synaptic projections)
   - Relay condensed memory index back to association and sensory cortices
   - Moderate sustained activity — short-term memory buffer over theta cycles

**Inputs (expected SynapseId patterns):**
- ``cortex_sensory`` / ``CortexPopulation.L5_PYR``  → EC_II (spatial context)
- ``cortex_association`` / ``CortexPopulation.L23_PYR``  → EC_II, EC_III (semantic context)
- ``hippocampus`` / ``HippocampusPopulation.CA1``  → EC_V (memory readout)

**Outputs:**
- ``ECPopulation.EC_II``  — perforant path input to DG and CA3
- ``ECPopulation.EC_III`` — temporoammonic input to CA1
- ``ECPopulation.EC_V``   — back-projection to neocortex
"""

from __future__ import annotations

from typing import ClassVar, List

import torch

from thalia.brain.configs import EntorhinalCortexConfig
from thalia.components import NeuronFactory
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorInput,
    PopulationPolarity,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)
from thalia.utils import split_excitatory_conductance

from .neural_region import NeuralRegion
from .population_names import ECPopulation
from .region_registry import register_region


@register_region(
    "entorhinal_cortex",
    aliases=["ec", "entorhinal", "hippocampal_gateway"],
    description="Entorhinal cortex — hippocampal gateway (perforant + temporoammonic paths)",
    version="1.0",
    author="Thalia Project",
    config_class=EntorhinalCortexConfig,
)
class EntorhinalCortex(NeuralRegion[EntorhinalCortexConfig]):
    """Entorhinal Cortex — Hippocampal Gateway.

    Routes cortical information into the hippocampal trisynaptic circuit via
    the perforant path (EC_II → DG, CA3) and temporoammonic direct path
    (EC_III → CA1), and relays hippocampal output back to neocortex via
    layer V (EC_V).

    Input Populations (via SynapseId routing):
    ------------------------------------------
    - Cortical inputs → EC_II (perforant path driver)
    - Cortical inputs → EC_III (temporoammonic path driver)
    - Hippocampal CA1 → EC_V (memory index back-projection)

    Output Populations:
    -------------------
    - ``ECPopulation.EC_II``  — perforant path spikes (→ hippocampus DG, CA3)
    - ``ECPopulation.EC_III`` — temporoammonic spikes (→ hippocampus CA1)
    - ``ECPopulation.EC_V``   — back-projection spikes (→ cortex)
    """

    # EC does not output neuromodulators; it is a structural relay region.
    neuromodulator_subscriptions: ClassVar[List[str]] = []

    def __init__(
        self,
        config: EntorhinalCortexConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
    ) -> None:
        """Initialise entorhinal cortex populations.

        Args:
            config: ``EntorhinalCortexConfig`` with layer-specific neuron params.
            population_sizes: Mapping of ``ECPopulation`` keys to neuron counts.
            region_name: Unique region name string (e.g. ``"entorhinal_cortex"``).
        """
        super().__init__(config, population_sizes, region_name)

        self.ec_ii_size: int = population_sizes[ECPopulation.EC_II]
        self.ec_iii_size: int = population_sizes[ECPopulation.EC_III]
        self.ec_v_size: int = population_sizes[ECPopulation.EC_V]

        # ── EC_II: Layer II stellate cells (Grid / place cells) ──────────────
        # Fast integrators (~20 ms τ_m) with moderate adaptation.
        # Facilitating synapses onto DG / CA3 are implemented in brain_builder
        # STP modules, not here.
        self.ec_ii_neurons = NeuronFactory.create_pyramidal_neurons(
            region_name=self.region_name,
            population_name=ECPopulation.EC_II,
            n_neurons=self.ec_ii_size,
            device=self.device,
            v_threshold=config.ec_ii_threshold,
            adapt_increment=config.ec_ii_adapt_increment,
            tau_adapt=config.ec_ii_adapt_tau_ms,
            tau_mem=config.ec_ii_tau_mem_ms,
        )

        # ── EC_III: Layer III pyramidal cells (Time cells) ───────────────────
        # Slower integration (~25 ms τ_m) for temporal context coding.
        self.ec_iii_neurons = NeuronFactory.create_pyramidal_neurons(
            region_name=self.region_name,
            population_name=ECPopulation.EC_III,
            n_neurons=self.ec_iii_size,
            device=self.device,
            v_threshold=config.ec_iii_threshold,
            adapt_increment=config.ec_iii_adapt_increment,
            tau_adapt=config.ec_iii_adapt_tau_ms,
            tau_mem=config.ec_iii_tau_mem_ms,
        )

        # ── EC_V: Layer V pyramidal cells (Back-projection to cortex) ────────
        # Larger, slower cells (~30 ms τ_m); long adaptation sustains activity.
        self.ec_v_neurons = NeuronFactory.create_pyramidal_neurons(
            region_name=self.region_name,
            population_name=ECPopulation.EC_V,
            n_neurons=self.ec_v_size,
            device=self.device,
            v_threshold=config.ec_v_threshold,
            adapt_increment=config.ec_v_adapt_increment,
            tau_adapt=config.ec_v_adapt_tau_ms,
            tau_mem=config.ec_v_tau_mem_ms,
        )

        # ── Register populations ──────────────────────────────────────────────
        self._register_neuron_population(
            ECPopulation.EC_II,
            self.ec_ii_neurons,
            polarity=PopulationPolarity.EXCITATORY,
        )
        self._register_neuron_population(
            ECPopulation.EC_III,
            self.ec_iii_neurons,
            polarity=PopulationPolarity.EXCITATORY,
        )
        self._register_neuron_population(
            ECPopulation.EC_V,
            self.ec_v_neurons,
            polarity=PopulationPolarity.EXCITATORY,
        )

        self.to(self.device)

    # =========================================================================
    # FORWARD
    # =========================================================================

    @torch.no_grad()
    def forward(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Route cortical → hippocampal and hippocampal → cortical signals.

        Processing order:
        1. Integrate all cortical synaptic inputs into EC_II and EC_III dendrites.
        2. Add layer-specific tonic drive.
        3. Fire EC_II and EC_III neurons (output: perforant / temporoammonic spikes).
        4. Integrate hippocampal CA1 back-projection into EC_V dendrites.
        5. Fire EC_V neurons (output: back-projection spikes to neocortex).

        Args:
            synaptic_inputs: Routed ``SynapseId``-keyed spike tensors.
            neuromodulator_inputs: Broadcast neuromodulatory signals (not used
                directly — EC does not subscribe to neuromodulators, but the
                base class may inspect them).

        Returns:
            ``RegionOutput`` mapping ``ECPopulation`` keys to boolean spike tensors.
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        cfg = self.config

        # ── 1. Integrate cortical → EC_II dendrites ──────────────────────────
        # All synaptic connections whose ``target_population == ECPopulation.EC_II``
        # are summed here (AMPA conductances from sensory + association cortex).
        ec_ii_raw = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.ec_ii_size,
            filter_by_target_population=ECPopulation.EC_II,
        ).g_ampa

        # Add tonic baseline drive (sub-threshold depolarisation from layer I)
        ec_ii_drive = ec_ii_raw + cfg.ec_ii_tonic_drive

        if cfg.baseline_noise_conductance_enabled:
            ec_ii_drive = ec_ii_drive + torch.randn_like(ec_ii_drive) * 0.005

        ec_ii_drive = torch.nn.functional.relu(ec_ii_drive)  # Non-negative conductance

        # ── 2. Fire EC_II neurons ─────────────────────────────────────────────
        ec_ii_g_ampa, ec_ii_g_nmda = split_excitatory_conductance(ec_ii_drive, nmda_ratio=0.3)
        ec_ii_spikes, _ = self.ec_ii_neurons.forward(
            g_ampa_input=ConductanceTensor(ec_ii_g_ampa),
            g_nmda_input=ConductanceTensor(ec_ii_g_nmda),
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )

        # ── 3. Integrate cortical → EC_III dendrites ─────────────────────────
        ec_iii_raw = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.ec_iii_size,
            filter_by_target_population=ECPopulation.EC_III,
        ).g_ampa

        ec_iii_drive = ec_iii_raw + cfg.ec_iii_tonic_drive

        if cfg.baseline_noise_conductance_enabled:
            ec_iii_drive = ec_iii_drive + torch.randn_like(ec_iii_drive) * 0.005

        ec_iii_drive = torch.nn.functional.relu(ec_iii_drive)

        # ── 4. Fire EC_III neurons ────────────────────────────────────────────
        ec_iii_g_ampa, ec_iii_g_nmda = split_excitatory_conductance(ec_iii_drive, nmda_ratio=0.3)
        ec_iii_spikes, _ = self.ec_iii_neurons.forward(
            g_ampa_input=ConductanceTensor(ec_iii_g_ampa),
            g_nmda_input=ConductanceTensor(ec_iii_g_nmda),
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )

        # ── 5. Integrate hippocampal CA1 → EC_V dendrites ────────────────────
        ec_v_raw = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.ec_v_size,
            filter_by_target_population=ECPopulation.EC_V,
        ).g_ampa

        ec_v_drive = ec_v_raw + cfg.ec_v_tonic_drive

        if cfg.baseline_noise_conductance_enabled:
            ec_v_drive = ec_v_drive + torch.randn_like(ec_v_drive) * 0.003

        ec_v_drive = torch.nn.functional.relu(ec_v_drive)

        # ── 6. Fire EC_V neurons ──────────────────────────────────────────────
        ec_v_g_ampa, ec_v_g_nmda = split_excitatory_conductance(ec_v_drive, nmda_ratio=0.25)
        ec_v_spikes, _ = self.ec_v_neurons.forward(
            g_ampa_input=ConductanceTensor(ec_v_g_ampa),
            g_nmda_input=ConductanceTensor(ec_v_g_nmda),
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )

        region_outputs: RegionOutput = {
            ECPopulation.EC_II: ec_ii_spikes,
            ECPopulation.EC_III: ec_iii_spikes,
            ECPopulation.EC_V: ec_v_spikes,
        }
        return self._post_forward(region_outputs)

    # =========================================================================
    # TEMPORAL PARAMETER UPDATE
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Propagate timestep change to all neuron populations and base class."""
        super().update_temporal_parameters(dt_ms)
        self.ec_ii_neurons.update_temporal_parameters(dt_ms)
        self.ec_iii_neurons.update_temporal_parameters(dt_ms)
        self.ec_v_neurons.update_temporal_parameters(dt_ms)
